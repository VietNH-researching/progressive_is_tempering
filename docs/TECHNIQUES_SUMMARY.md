# Sequential Score Matching via SMC Temperature Bridging

This document describes a sequential training algorithm for a score-based diffusion model targeting an unnormalized distribution with a tractable energy function. The key idea is to maintain a temperature ladder from an easy reference distribution to the target, iteratively generate training data at each temperature level via backward SDE sampling, correct for discretization error using importance weights and MALA rejuvenation, bridge particles to adjacent temperature levels, and jointly train a single β-conditioned score network across all levels.

---

## Introduction

Our goal is to sample from a target distribution

$$
\pi(x) = Z^{-1} \exp\{-E(x)\},
$$

where $E(x)$ is a tractable energy function (pointwise evaluable with automatic differentiation) but $\pi$ may be multimodal or otherwise intractable to sample from directly.

We train a score-based diffusion model conditioned on an inverse temperature $\beta \in [0, 1]$. A temperature ladder $\{\beta_k\}_{k=0}^{M}$ bridges an easy Gaussian reference to the target. At each outer training step, backward SDE sampling generates particles at every temperature level; these particles are importance-weighted, resampled, MCMC-refined, and fed into per-temperature replay buffers. A single inner loop then takes gradient steps on a joint denoising regression objective across all temperatures simultaneously.

---

## Geometric Temperature Path

We introduce the geometric interpolation between a Gaussian reference

$$
\rho(x) = \mathcal{N}(x;\,0,\,I)
$$

and the target $\pi$:

$$
\pi_\beta(x) = Z_\beta^{-1}\,\rho(x)^{1-\beta}\,\pi(x)^\beta
\;\propto\;
\exp\!\left\{-(1-\beta)\tfrac{1}{2}\|x\|^2 - \beta\,E(x)\right\},
\qquad \beta \in [0,1].
$$

At $\beta = 0$ this is the reference $\rho$. At $\beta = 1$ this is the target $\pi$. The score of $\pi_\beta$ satisfies

$$
\nabla_x \log \pi_\beta(x) = -(1-\beta)\,x - \beta\,\nabla E(x),
$$

and the ratio to the reference score is

$$
\nabla_x \log \frac{\pi_\beta(x)}{\rho(x)}
= \beta\,\bigl(x - \nabla E(x)\bigr).
$$

A temperature ladder

$$
0 = \beta_0 < \beta_1 < \cdots < \beta_M = 1
$$

is chosen so that adjacent distributions overlap well. The specific values are determined adaptively before training, as described in the next section.

---

## Temperature Ladder Construction via MPPT

The temperature ladder $\{\beta_k\}_{k=0}^M$ is constructed **once**, before any score-network training, using MPPT's `smc_sampler`. The problem is specified as a `PTProblem` with three callables (`rvs_ref`, `logpdf_ref`, `logpdf_final_target`); MPPT internally constructs the geometric path from these.

`smc_sampler` runs an SMC sweep from $\beta = 0$ to $\beta = 1$. The ladder is built greedily: starting from $\beta_k$, the next level $\beta_{k+1}$ is the largest value such that the normalised ESS of the IS bridge

$$
\log w_{k \to k+1}(x) = (\beta_{k+1} - \beta_k)\,\bigl(-E(x) + \log\rho(x)\bigr)
$$

satisfies $\mathrm{ESS}_{k \to k+1} / N \geq \tau$ (`local_ess_ratio`). The endpoints are clamped exactly to $\beta_0 = 0$ and $\beta_M = 1$.

The only output used in training is the ladder $\{\beta_k\}$ itself. The particle populations are not used to seed the replay buffers: buffer 0 is filled with i.i.d. draws from $\rho = \mathcal{N}(0, I)$; buffer 1 is initialized by IS-reweighting those samples to $\beta_1$; buffers $2, \ldots, M$ start empty and are populated during the training outer loop.

---

## Forward (Noising) Process

We use an Ornstein–Uhlenbeck (OU) forward SDE with drift coefficient $\alpha = 1$ and diffusion coefficient $g = \sqrt{2}$:

$$
dX_t = -X_t\,dt + \sqrt{2}\,dW_t,
\qquad X_0 \sim \pi_\beta.
$$

The transition kernel is Gaussian and available in closed form:

$$
X_t \mid X_0 \;\sim\; \mathcal{N}\!\left(e^{-t}\,X_0,\;\sigma_t^2\,I\right),
\qquad \sigma_t^2 = 1 - e^{-2t}.
$$

For large $T$, the marginal $p_T$ is approximately $\mathcal{N}(0, I) = \rho$, regardless of the starting distribution $\pi_\beta$. This makes $\rho$ a natural prior for the backward sampling pass.

Denote by $p_t^\beta(x)$ the marginal density of the forward process at time $t$ when started from $\pi_\beta$:

$$
p_t^\beta(x_t) = \int \mathcal{N}\!\left(x_t;\,e^{-t}x_0,\,\sigma_t^2 I\right)\,\pi_\beta(x_0)\,dx_0.
$$

---

## Score Network

A neural network $s_\theta(x_t, t, \beta)$ parameterizes the **excess score** of the forward-noised marginal of $\pi_\beta$ over the reference:

$$
s_\theta(x_t, t, \beta)
\;\approx\;
\nabla_{x_t} \log \frac{p_t^\beta(x_t)}{p_t^\rho(x_t)}.
$$

Here $p_t^\rho = p_t^{\beta=0}$ is the marginal under the reference. The network architecture can be any function $\mathbb{R}^d \times \mathbb{R} \times \mathbb{R} \to \mathbb{R}^d$ that is equivariant or otherwise suited to the problem geometry.

The $\beta$ conditioning is critical: it allows **one network** to jointly model the entire family $\{\pi_\beta : \beta \in [0,1]\}$. At $\beta = 0$ the excess score is zero and the network should output zero; at $\beta = 1$ it should approximate the score of the target relative to the reference.

---

## Backward SDE Sampler

To generate samples from $\pi_\beta$ we run the time-reversal of the forward OU process, starting from $x_T \sim \rho \approx \mathcal{N}(0,I)$ and integrating backward from time $T$ to $0$.

The full score of $p_t^\beta$ decomposes as

$$
\nabla_{x_t} \log p_t^\beta(x_t)
=
\underbrace{\nabla_{x_t} \log p_t^\rho(x_t)}_{\text{reference OU score}}
+
\underbrace{\nabla_{x_t} \log \frac{p_t^\beta(x_t)}{p_t^\rho(x_t)}}_{\approx\, s_\theta(x_t, t, \beta)}.
$$

The backward SDE driven by the **control** $u_\theta(x, t, \beta) = 2\,s_\theta(x, t, \beta)$ is

$$
dY_t = \bigl(-Y_t + u_\theta(Y_t, t, \beta)\bigr)\,dt + \sqrt{2}\,dW_t,
\qquad Y_T \sim \rho.
$$

When $s_\theta$ is exact, $Y_0 \sim \pi_\beta$. In practice the score is imperfect and the time discretization introduces further error, so the terminal distribution of $Y_0$ differs from $\pi_\beta$. The mismatch is corrected via **path-space importance weights** defined below.

### Exponential-Integrator Discretization

We partition $[0, T]$ into $N$ uniform steps of size $\Delta t = T / N$ and run backward from $t_N = T$ to $t_0 = 0$. A standard Euler–Maruyama step does not recover the exact OU transition when $u_\theta \equiv 0$, so it does not preserve the Gaussian reference dynamics and their reversibility at the discrete level. Instead, we freeze $u_\theta$ constant over each interval $[t_{n-1}, t_n]$ and solve the resulting linear SDE in closed form:

$$
x_{n-1} = \lambda\, x_n + (1-\lambda)\,u_\theta(x_n, t_n, \beta) + \sigma\,\varepsilon_n,
\qquad \varepsilon_n \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, I),
$$

where

$$
\lambda = e^{-\Delta t}, \qquad \sigma^2 = 1 - e^{-2\Delta t}.
$$

When $u_\theta \equiv 0$ this step reduces exactly to the OU transition kernel $x_{n-1} \sim \mathcal{N}(\lambda x_n, \sigma^2 I)$, preserving detailed balance at the discrete level for any $\Delta t$.

---

## Path-Space Importance Weights

The backward SDE defines a path measure $\mathbb{Q}_\theta$ on the discrete trajectory $(x_N, x_{N-1}, \ldots, x_0)$. The reference forward OU process defines another path measure $\mathbb{P}_\rho$ on the same trajectory. The path-space importance weight corrects for (i) the mismatch between these two path measures at every step, and (ii) the boundary terms at $t = T$ and $t = 0$.

**Step-wise log-ratio.** At each backward step, the exact forward OU transition density (in the direction $x_{n-1} \to x_n$) is

$$
p_\rho(x_n \mid x_{n-1})
=
\mathcal{N}\!\left(x_n;\;\lambda\,x_{n-1},\;\sigma^2\,I\right),
$$

while the exponential-integrator proposal density is

$$
q_\theta(x_{n-1} \mid x_n)
=
\mathcal{N}\!\left(x_{n-1};\;\lambda\,x_n + (1-\lambda)\,u_\theta,\;\sigma^2\,I\right).
$$

Both kernels share the same variance $\sigma^2$, so the normalization constant $-\tfrac{d}{2}\log(2\pi\sigma^2)$ cancels exactly from $\log p_\rho - \log q_\theta$ at every step. The per-step log-ratio is accumulated across all $N$ steps.

**Terminal correction.** At the endpoints we account for:

- $\log \pi_\beta(x_0) = -(1-\beta)\tfrac{1}{2}\|x_0\|^2 - \beta\,E(x_0) + \text{const}$, the log-density of the target at the terminal sample, and
- $-\log \rho(x_T) \propto \tfrac{1}{2}\|x_T\|^2$, the negative log-density of the prior at the initial noise.

The full path-space log-weight is therefore

$$
\boxed{
\log w_{\mathrm{path}}
=
\sum_{n=1}^{N}\bigl[\log p_\rho(x_n \mid x_{n-1}) - \log q_\theta(x_{n-1} \mid x_n)\bigr]
-\,(1-\beta)\,\tfrac{1}{2}\|x_0\|^2
-\,\beta\,E(x_0)
+\,\tfrac{1}{2}\|x_T\|^2.
}
$$

This weight is zero when the score is exact (the backward SDE perfectly reverses the forward OU) and the discretization step $\Delta t \to 0$. In general, $\exp(\log w_{\mathrm{path}})$ gives the importance weight that reweights the approximate terminal distribution of $Y_0$ to $\pi_\beta$.

---

## DDPM Regression Target

Training the score network reduces to a regression problem. We minimize the expected squared error between $s_\theta(x_t, t, \beta)$ and the true excess score. By the Tweedie identity for the OU process, the excess score satisfies

$$
\nabla_{x_t} \log \frac{p_t^\beta(x_t)}{p_t^\rho(x_t)}
=
\lambda_t \cdot \mathbb{E}_{\pi_\beta}\!\left[\,\nabla_{x_0} \log \frac{\pi_\beta(x_0)}{\rho(x_0)}\;\middle|\;x_t\right],
$$

where $\lambda_t = e^{-t}$ is the OU decay factor. Since

$$
\nabla_{x_0} \log \frac{\pi_\beta(x_0)}{\rho(x_0)}
=
\nabla_{x_0}\left[\beta\,\tfrac{1}{2}\|x_0\|^2 - \beta\,E(x_0)\right]
=
\beta\,\bigl(x_0 - \nabla E(x_0)\bigr),
$$

the true regression target (in expectation over $x_0 \sim \pi_\beta(\cdot \mid x_t)$) is

$$
\phi(x_0, t, \beta)
=
\lambda_t \cdot \beta \cdot \bigl(x_0 - \nabla E(x_0)\bigr).
$$

DDPM-style training replaces the conditional expectation over $x_0 \mid x_t$ with the draw $x_0$ that generated $x_t$ in the first place. The training objective is

$$
\mathcal{L}(\theta)
=
\mathbb{E}_\beta
\mathbb{E}_{x_0 \sim \pi_\beta}
\mathbb{E}_{t \sim \mathrm{Unif}[t_{\min}, T]}
\mathbb{E}_{x_t \sim p(\cdot \mid x_0)}
\left[
\left\|
s_\theta(x_t,\, t,\, \beta)
-
\lambda_t\,\beta\,\bigl(x_0 - \nabla E(x_0)\bigr)
\right\|^2
\right].
$$

**Key properties:**

- At $\beta = 0$: target is zero for all $(x_0, t)$. The network correctly learns no excess over the reference.
- At $\beta = 1$: target is $\lambda_t\,(x_0 - \nabla E(x_0))$, the full score correction from reference to target.
- The target requires only $\nabla E(x_0)$ evaluated at clean (unnoised) samples, which is available via automatic differentiation.
- The target does not depend on $x_t$, making it trivially computable once $x_0$ is available.

---

## MALA Rejuvenation

After resampling with path-space weights, particles are approximately i.i.d. from $\pi_\beta$, but many will be near-duplicates of a few high-weight particles. MALA (Metropolis-Adjusted Langevin Algorithm) is applied to diversify the particles while preserving the target distribution $\pi_\beta$ exactly.

### MALA Proposal

From a current state $x$, the Langevin proposal is

$$
x'
=
x
+
\frac{h^2}{2}\,\nabla_x \log \pi_\beta(x)
+
h\,\varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0, I),
$$

where $h > 0$ is the step size. Substituting the score of the geometric path:

$$
x'
=
x
-
\frac{h^2}{2}\,(1-\beta)\,x
-
\frac{h^2}{2}\,\beta\,\nabla E(x)
+
h\,\varepsilon.
$$

### Metropolis–Hastings Correction

The proposal is accepted with probability $\min(1,\, \alpha)$ where

$$
\log \alpha
=
\log \pi_\beta(x') - \log \pi_\beta(x)
+
\log q(x \mid x') - \log q(x' \mid x),
$$

and $q(x' \mid x) = \mathcal{N}(x';\;x + \tfrac{h^2}{2}\nabla_x\log\pi_\beta(x),\;h^2 I)$ is the forward Langevin proposal density. The MH correction ensures the chain has $\pi_\beta$ as its exact stationary distribution, regardless of the step size $h$.

### Per-Level Step Size Calibration

The optimal step size $h$ depends on the curvature of $\log \pi_\beta$ at the current temperature. A step that is appropriate for the near-Gaussian distribution at small $\beta$ will be too large — and give very low acceptance — for the concentrated, multimodal distribution at $\beta$ near 1, and vice versa. A single global $h$ therefore inevitably over-steps at high temperatures (wasting compute on rejections) and under-steps at low temperatures (slow mixing).

We calibrate a separate step size $h_k$ for every level $\beta_k$ **once**, before the training loop begins, by targeting an acceptance rate in the range $[60\%, 80\%]$. This range is slightly above the classical 57% optimal derived for isotropic targets in the limit $d \to \infty$; in practice on moderately-dimensional problems the efficiency curve is flat near its peak, and 60–80% leaves room for variation while keeping proposals well-correlated with the gradient direction.

**Calibration particles.** We need approximate draws from each $\pi_{\beta_k}$ before training. We draw $N_{\mathrm{cal}}$ samples from the reference $\rho = \mathcal{N}(0, I)$ and IS-resample them toward $\pi_{\beta_k}$ using the log-weights

$$
\log w(x) = \beta_k \bigl(-E(x) + \tfrac{1}{2}\|x\|^2\bigr),
$$

which is the cumulative IS weight from $\beta_0 = 0$ to $\beta_k$. For small $\beta_k$ these weights are nearly uniform and the resampled cloud closely approximates $\pi_{\beta_k}$. For large $\beta_k$ the IS estimate degrades, but the resampled cloud still captures the rough scale and geometry of the target, which is all that is needed to set the order of magnitude of $h_k$.

**Bisection procedure.** Given a particle set, the average MALA acceptance rate is a decreasing function of $h$: large steps produce proposals far from the current point, making rejection likely, while small steps are always accepted but mix slowly. We therefore bisect on $[h_{\min}, h_{\max}]$ to find $h_k$:

1. Evaluate the acceptance rate at the midpoint $h = (h_{\min} + h_{\max}) / 2$ by running a fixed number of MALA steps on the calibration particles.
2. If acceptance $< 60\%$, the step is too large: set $h_{\max} \leftarrow h$.
3. If acceptance $> 80\%$, the step is too small: set $h_{\min} \leftarrow h$.
4. If acceptance $\in [60\%, 80\%]$, stop early.
5. Repeat for a fixed number of bisection iterations.

After convergence, $h_k = (h_{\min} + h_{\max}) / 2$.

**Use in this algorithm.** After systematic resampling of backward-SDE particles at $\beta_k$, `n_mala_rejuv` MALA steps are applied using the pre-calibrated step size $h_k$. This replaces near-duplicate particles with genuinely diverse samples from $\pi_{\beta_k}$, improving the quality of the IS bridge to $\beta_{k+1}$. All levels are rejuvenated in parallel inside the outer vmap, so the per-level $h_k$ values are passed as a dynamic array rather than a static scalar.

---

## IS Bridging Between Adjacent Temperatures

After MALA rejuvenation, particles $\{x_i\}$ are approximately distributed under $\pi_{\beta_k}$. To populate the buffer at the next level $\beta_{k+1}$, we apply importance sampling (IS) reweighting using the geometric-path ratio:

$$
\frac{\pi_{\beta_{k+1}}(x)}{\pi_{\beta_k}(x)}
\propto
\exp\!\left\{(\beta_{k+1} - \beta_k)\,\bigl(-E(x) + \tfrac{1}{2}\|x\|^2\bigr)\right\}.
$$

The IS log-weights are therefore

$$
\log w_{k \to k+1}(x)
=
(\beta_{k+1} - \beta_k)\,\bigl(-E(x) + \tfrac{1}{2}\|x\|^2\bigr).
$$

These weighted particles $\{(x_i, \log w_{k \to k+1}(x_i))\}$ are appended to the replay buffer for $\beta_{k+1}$, with FIFO eviction when the buffer exceeds its capacity.

The normalized effective sample size (ESS) of the IS weights,

$$
\mathrm{ESS}_k
=
\frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2},
$$

monitors the quality of the IS bridge and diagnoses temperature gaps that are too large.

---

## Per-Temperature Replay Buffers

Each temperature level $k$ has an independent replay buffer storing pairs $(x, \log w)$:

- $x \in \mathbb{R}^d$: a sample approximately from $\pi_{\beta_k}$,
- $\log w \in \mathbb{R}$: its associated IS log-weight within the buffer.

Buffers have a fixed capacity and use FIFO eviction. When sampling a mini-batch from buffer $k$, indices are drawn via **systematic resampling** with probability proportional to $\exp(\log w)$, giving an approximately i.i.d. draw from $\pi_{\beta_k}$.

The two extreme buffers are special:

- **Buffer 0** ($\beta_0 = 0$): filled once with i.i.d. draws from $\rho = \mathcal{N}(0, I)$, with uniform weights. It is frozen for the entire training run.
- **Buffer 1** ($\beta_1$): initialized once via IS reweighting of buffer 0 samples. Also frozen after initialization. This provides a stable training signal at the first non-trivial temperature throughout training.

---

## SMC Training Loop

The full algorithm alternates between two phases: **outer SMC data generation** and **inner gradient descent**.

### Outer Loop: Parallel SMC Data Generation

At each outer step, all $M-1$ active temperature levels ($k = 1, \ldots, M-1$) are processed **simultaneously**. The four steps below are independent across levels — every level uses the same parameter snapshot $\theta$ and writes to a distinct buffer — so they are executed in parallel via a single `jax.vmap` call over the level index. Only the final buffer writes, which are Python-level mutations, are performed sequentially (negligible cost).

For each $k \in \{1, \ldots, M-1\}$ **in parallel**:

1. **Backward SDE sampling.** Run the discretized backward SDE for $N$ steps with the current parameters $\theta$, targeting $\pi_{\beta_k}$:

$$
(x_0^{(i)}, \log w_{\mathrm{path}}^{(i)})_{i=1}^{K}
\leftarrow
\text{BackwardSDE}\bigl(\theta,\,\beta_k,\,N,\,T\bigr).
$$

Compute the normalized path-space ESS to diagnose discretization quality.

2. **Systematic resampling.** Draw $K$ indices from $\{1, \ldots, K\}$ proportional to $\exp(\log w_{\mathrm{path}}^{(i)})$ using systematic resampling:

$$
\{x_0^{(i)}\}_{i=1}^K
\leftarrow
\text{SystematicResample}\!\bigl(\{(x_0^{(i)}, \log w_{\mathrm{path}}^{(i)})\}\bigr).
$$

After resampling, particles are approximately i.i.d. from $\pi_{\beta_k}$.

3. **MALA rejuvenation.** Apply `n_mala_rejuv` MALA steps to each particle at temperature $\beta_k$:

$$
\{x_0^{(i)}\}_{i=1}^K
\leftarrow
\text{MALA}\!\bigl(\{x_0^{(i)}\},\,\pi_{\beta_k},\,h_{\mathrm{rejuv}}\bigr).
$$

4. **IS bridge.** Compute the IS log-weights to the next level $\beta_{k+1}$:

$$
\log w^{(i)}
=
(\beta_{k+1} - \beta_k)\,\bigl(-E(x_0^{(i)}) + \tfrac{1}{2}\|x_0^{(i)}\|^2\bigr).
$$

5. **Buffer update.** Append $\{(x_0^{(i)}, \log w^{(i)})\}$ to buffer $k+1$.

### Inner Loop: Joint Gradient Descent

After each outer step, once every buffer holds at least $B$ samples, run `n_inner_steps` gradient updates on the joint DDPM objective.

Each inner step:

1. For each temperature $\beta_k$, draw a mini-batch of $B$ samples from buffer $k$ via systematic resampling on the buffer IS weights.
2. Sample noise times $t_i \sim \mathrm{Unif}[t_{\min}, T]$ and apply the forward OU noising: $x_{t,i} \sim \mathcal{N}(e^{-t_i}\,x_{0,i},\,\sigma_{t_i}^2 I)$.
3. Compute the regression targets $\phi_i = \lambda_{t_i}\,\beta_k\,(x_{0,i} - \nabla E(x_{0,i}))$.
4. Concatenate all $M \cdot B$ triples $(x_{t,i},\, t_i,\, \beta_k)$ into a single batch.
5. Take one Adam step on the total loss:

$$
\widehat{\mathcal{L}}(\theta)
=
\frac{1}{(M+1)B}
\sum_{k=0}^{M}
\sum_{i=1}^{B}
\bigl\|s_\theta(x_{t,i},\,t_i,\,\beta_k)
-
\lambda_{t_i}\,\beta_k\,(x_{0,i} - \nabla E(x_{0,i}))\bigr\|^2.
$$

Gradients are clipped by global norm before the Adam update.

---

## Systematic Resampling

Systematic resampling draws $M$ indices from $N$ particles with normalized weights $\{w_i\}$ using a single uniform variate $u \sim \mathrm{Unif}(0, 1/M)$:

$$
u_m = u + \frac{m-1}{M}, \qquad m = 1, \ldots, M,
$$

and each $u_m$ is mapped to the particle index via the CDF. This is a standard SMC resampling scheme with $O(1)$ variance compared to $O(N)$ for multinomial resampling, giving higher-quality resampled clouds at the same cost.

---

## Algorithm Summary

```text
Initialization:
    Construct temperature ladder {β_k}_{k=0}^M
    Initialize score network s_θ
    buffer[0] ← i.i.d. N(0,I) samples, uniform weights  [frozen]
    buffer[1] ← IS-reweighted buffer[0] to β_1          [frozen]
    Calibrate per-level MALA step sizes {h_k}_{k=0}^M   [bisection, once]

Training (num_outer_steps iterations):
    -- Outer loop (all M-1 levels run in parallel via jax.vmap) --
    For k = 1, ..., M-1  [IN PARALLEL]:
        Run backward SDE at β_k → K particles with path-space log-weights
        Systematic resample → approx. i.i.d. from π_{β_k}
        MALA rejuvenation at β_k  (n_mala_rejuv steps, step size h_k)
        IS-reweight to β_{k+1}: log w = Δβ_k · (-E(x) + ½‖x‖²)
    For k = 1, ..., M-1  [sequential — Python buffer mutations]:
        Append weighted particles to buffer[k+1]

    -- Inner loop --
    For n_inner_steps gradient steps:
        For each k: sample B particles from buffer[k] (systematic resample by IS weights)
        For each particle: sample t ~ Unif[t_min, T], noise to x_t via OU
        Compute target: λ_t · β_k · (x_0 - ∇E(x_0))
        Take Adam step on joint MSE loss across all temperatures

Sampling (after training):
    Draw x_T ~ N(0, I)
    Run backward SDE with s_θ(·, ·, β=1) from T to 0
    Return x_0 as approximate sample from π
```

---

## Overall Summary

The algorithm can be described as:

$$
\boxed{
\text{Use backward SDE + IS weights to generate training data, MALA to diversify, and joint DDPM regression to train.}
}
$$

The score network $s_\theta(x, t, \beta)$ is not a sampler in itself. It is learned as the excess score (relative to the reference OU process) for the entire family $\{\pi_\beta\}$ simultaneously. The backward SDE sampler uses this network to generate approximate samples from each $\pi_{\beta_k}$; the path-space importance weights and MALA correction ensure that training data is unbiased even when the score is imperfect. The exponential-integrator discretization further guarantees that the path-space weights are exactly 1 when $\beta = 0$ (i.e., when $u_\theta \equiv 0$), regardless of the step size $\Delta t$.
