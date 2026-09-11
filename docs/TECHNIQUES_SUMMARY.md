# Sequential Score Matching via SMC Temperature Bridging

This documentation describes a sequential training algorithm for a score-based
diffusion sampler targeting an unnormalized density with a tractable energy. The
main idea is to let an SMC sweep lay out a temperature ladder from an easy
Gaussian reference to the target, then train a single $\beta$-conditioned score
network on data the network itself generates — with a path-space importance
correction and MCMC rejuvenation standing between the network's output and the
training set, so that imperfect scores still produce usable data.

---

## Introduction:

Our goal is to sample from

$$
\pi(x)
=
Z^{-1}\exp\{-E(x)\},
$$

where $E(x)$ is a tractable energy — pointwise evaluable, differentiable by
automatic differentiation — but $\pi$ itself is multimodal or otherwise hard to
sample directly.

Instead of attacking $\pi$ directly, we introduce the geometric tempered path
between a standard Gaussian reference $\rho=\mathcal{N}(0,I)$ and the target,

$$
\pi_\beta(x)
=
Z_\beta^{-1}\,\rho(x)^{1-\beta}\pi(x)^{\beta}
\;\propto\;
\exp\!\left\{-(1-\beta)\tfrac12\|x\|^2-\beta E(x)\right\},
\qquad
\beta\in[0,1].
$$

Here $\pi_0=\rho$ is exactly samplable and $\pi_1=\pi$ is the target. Two
identities are used constantly below. The score of the path is

$$
\nabla_x\log\pi_\beta(x)
=
-(1-\beta)\,x-\beta\,\nabla E(x),
$$

and the excess of $\pi_\beta$ over the reference is

$$
\nabla_x\log\frac{\pi_\beta(x)}{\rho(x)}
=
\beta\bigl(x-\nabla E(x)\bigr).
$$

The algorithm works on a finite ladder

$$
0=\beta_0<\beta_1<\cdots<\beta_M=1,
$$

discovered once by SMC, and trains **one** network

$$
s_\theta(x_t,t,\beta)
$$

on the whole family $\{\pi_\beta:\beta\in[0,1]\}$ at once. The $\beta$
conditioning is what makes the ladder a curriculum rather than $M$ separate
problems: the easy rungs regularize the hard ones through shared weights.

The learned object is not the sampler. It is a *control* for a backward SDE, and
the sampler is that SDE plus an importance correction.

---

## Main Algorithm:

The algorithm has three phases:

- **Phase 1**: Initialization — run once, before any gradient step.
- **Phase 2**: Training — alternating data generation and gradient descent.
- **Phase 3**: Production — sampling from the trained network.

Phase 1 runs an SMC sweep that yields three things at once: the ladder
$\{\beta_k\}$, a particle population at every rung, and a per-rung MCMC kernel
adapted to that rung. All three are consumed by Phase 2; none is recomputed.

---

## The Diffusion Machinery

Both training and production run the same forward/backward pair, so it is
defined once here.

### Forward (noising) process

We use an Ornstein–Uhlenbeck forward SDE with unit drift and diffusion
$\sqrt2$:

$$
dX_t=-X_t\,dt+\sqrt2\,dW_t,
\qquad
X_0\sim\pi_\beta.
$$

Its transition kernel is Gaussian and closed-form,

$$
X_t\mid X_0\sim\mathcal{N}\!\left(e^{-t}X_0,\;\sigma_t^2 I\right),
\qquad
\sigma_t^2=1-e^{-2t},
$$

so for large $T$ the marginal $p_T$ is numerically $\mathcal{N}(0,I)=\rho$
whatever $\pi_\beta$ was. This is why $\rho$ can serve as the prior for the
backward pass. Write $p_t^\beta$ for the time-$t$ marginal started from
$\pi_\beta$,

$$
p_t^\beta(x_t)
=
\int\mathcal{N}\!\left(x_t;e^{-t}x_0,\sigma_t^2I\right)\pi_\beta(x_0)\,dx_0,
$$

and $p_t^\rho=p_t^{\beta=0}$ for the reference marginal.

### The learned object: the excess score

The network parameterizes the **excess** score of the noised target over the
noised reference,

$$
s_\theta(x_t,t,\beta)
\;\approx\;
\nabla_{x_t}\log\frac{p_t^\beta(x_t)}{p_t^\rho(x_t)}.
$$

Parameterizing the excess rather than the full score builds in the right
boundary condition: at $\beta=0$ the excess is identically zero, so the network
has nothing to learn at the bottom rung and the output layer is initialized near
zero. Anything the network outputs is a correction *away from* the reference
dynamics, which is also what makes the path weights below collapse to zero in
the reference case.

Architecturally $s_\theta$ is any map
$\mathbb{R}^d\times\mathbb{R}\times\mathbb{R}\to\mathbb{R}^d$ suited to the
problem geometry: a residual MLP on sinusoidal $(t,\beta)$ embeddings for
unstructured targets, an EGNN for particle systems where permutation and
rotation equivariance matter.

### Backward SDE

To sample $\pi_\beta$ we run the time-reversal of the OU process from
$x_T\sim\rho$ down to $t=0$. Write $\tau=T-t$ for reversed time — $\tau$ runs
$0\to T$ while the noise time $t$ runs $T\to0$ — and $Y_\tau$ for the state at
noise time $T-\tau$. The time-reversal of $dX_t=-X_t\,dt+\sqrt2\,dW_t$ flips the
sign of the forward drift and adds twice the score:

$$
dY_\tau=\bigl(Y_\tau+2\,\nabla\log p_{T-\tau}^\beta(Y_\tau)\bigr)d\tau+\sqrt2\,dB_\tau .
$$

The full score splits as

$$
\nabla_{x_t}\log p_t^\beta(x_t)
=
\underbrace{\nabla_{x_t}\log p_t^\rho(x_t)}_{\text{reference OU score}}
+
\underbrace{\nabla_{x_t}\log\frac{p_t^\beta(x_t)}{p_t^\rho(x_t)}}_{\approx\,s_\theta(x_t,t,\beta)},
$$

and the reference part is explicit: $\rho$ is stationary for the OU process, so
$p_t^\rho=\rho$ for every $t$ and $\nabla\log p_t^\rho(y)=-y$. The drift becomes
$y+2(-y+s_\theta)=-y+2s_\theta$, which leaves a backward SDE driven by the
**control** $u_\theta(x,t,\beta)=2\,s_\theta(x,t,\beta)$:

$$
dY_\tau=\bigl(-Y_\tau+u_\theta(Y_\tau,T-\tau,\beta)\bigr)d\tau+\sqrt2\,dB_\tau,
\qquad
Y_{\tau=0}\sim\rho.
$$

In reversed time this is again an OU process — the same drift $-y$ and noise
$\sqrt2$ as the forward one, only pulled towards $u_\theta$ instead of towards
$0$. That structure is what the discretization below exploits.

If $s_\theta$ were exact and the integration continuous, $Y_0\sim\pi_\beta$.
Neither holds, so $Y_0$ is only approximately right and the gap is closed by the
path-space weights below.

### Exponential-integrator discretization

Partition $[0,T]$ into $N$ steps of size $\Delta t=T/N$ on the grid
$t_n=n\,\Delta t$, and integrate backward from $t_N=T$ to $t_0=0$. Over each
interval $[t_{n-1},t_n]$ freeze the control at its value at the *start* of the
backward step, $u_\theta(x_n,t_n,\beta)$. What remains,
$dY=(-Y+u)\,d\tau+\sqrt2\,dB$ with $u$ constant, is a linear SDE — an OU process
centred at $u$ — and is solved exactly over a step of length $\Delta t$:

$$
x_{n-1}=\lambda\,x_n+(1-\lambda)\,u_\theta(x_n,t_n,\beta)+\sigma\,\varepsilon_n,
\qquad
\varepsilon_n\overset{\text{iid}}{\sim}\mathcal{N}(0,I),
$$

$$
\lambda=e^{-\Delta t},
\qquad
\sigma^2=1-e^{-2\Delta t}.
$$

At $u_\theta\equiv0$ this is exactly the OU kernel
$x_{n-1}\sim\mathcal{N}(\lambda x_n,\sigma^2I)$, for any $\Delta t$ — no small
step required. This is `backward_sde_step` in `sampler.py`.

**Relation to Euler–Maruyama.** Euler–Maruyama applied to the same SDE takes

$$
x_{n-1}=x_n+\bigl(-x_n+u_\theta(x_n,t_n,\beta)\bigr)\Delta t+\sqrt{2\Delta t}\,\varepsilon_n .
$$

Since $\lambda=1-\Delta t+O(\Delta t^2)$, $1-\lambda=\Delta t+O(\Delta t^2)$ and
$\sigma^2=2\Delta t+O(\Delta t^2)$, the two schemes agree to first order; the
exponential integrator differs only in treating the linear drift $-y$ and the
noise exactly instead of linearizing them. That difference is the whole point.
At $u_\theta\equiv0$ the Euler kernel $\mathcal N\bigl((1-\Delta t)x_n,\,2\Delta t\,I\bigr)$
has stationary law $\mathcal N\bigl(0,\tfrac{2}{2-\Delta t}I\bigr)$ rather than
$\rho$, and its variance $2\Delta t$ no longer matches the $\sigma^2$ of the exact
forward kernel $p_\rho$ used in the weights below. Even a network whose excess
score is exactly right would then produce non-zero path weights — discretization
bias that cannot be told apart from score error. The exponential kernel leaves
$\rho$ exactly invariant at every $\Delta t$.

**Time grid and where the network is queried.** In code the grid is
`linspace(T, Δt, N)`: the network is evaluated once per step, at
$t_n\in\{T,\,T-\Delta t,\,\ldots,\,\Delta t\}$, and **never at $t=0$** — the final
step from $t_1=\Delta t$ to $t_0=0$ uses $u_\theta(x_1,\Delta t,\beta)$. Training
draws $t\sim\mathrm{Unif}[t_{\min},T]$, so a grid with $\Delta t<t_{\min}$ queries
the network at roughly $t_{\min}/\Delta t$ noise times it never trained on
(about five steps for $\Delta t=0.002$, $t_{\min}=0.01$). The damage is bounded — such a
step moves $x$ by only $(1-\lambda)\,u_\theta\approx\Delta t\,u_\theta$ — but it
disappears entirely if $t_{\min}$ is set at or below the finest $\Delta t$ the
sampler will be run with.

### Path-space importance weights

The discretized backward SDE defines a path measure $\mathbb{Q}_\theta$ over
trajectories $(x_N,\ldots,x_0)$; the reference forward OU process defines
$\mathbb{P}_\rho$ over the same object. The weight corrects the mismatch step by
step, plus the two boundary terms.

**Step-wise ratio.** The exact forward OU transition, read in the direction
$x_{n-1}\to x_n$, is

$$
p_\rho(x_n\mid x_{n-1})
=
\mathcal{N}\!\left(x_n;\lambda x_{n-1},\sigma^2I\right),
$$

while the proposal actually used is

$$
q_\theta(x_{n-1}\mid x_n)
=
\mathcal{N}\!\left(x_{n-1};\lambda x_n+(1-\lambda)u_\theta,\sigma^2I\right).
$$

Both have variance $\sigma^2$, so the normalizer $-\tfrac d2\log(2\pi\sigma^2)$
cancels identically in $\log p_\rho-\log q_\theta$ and is never computed.

**Boundary terms.** At $t=0$ we credit the target log-density
$\log\pi_\beta(x_0)=-(1-\beta)\tfrac12\|x_0\|^2-\beta E(x_0)$ up to a constant;
at $t=T$ we debit the prior, $-\log\rho(x_T)\propto\tfrac12\|x_T\|^2$. Hence

$$
\boxed{
\log w_{\mathrm{path}}
=
\sum_{n=1}^{N}\bigl[\log p_\rho(x_n\mid x_{n-1})-\log q_\theta(x_{n-1}\mid x_n)\bigr]
-(1-\beta)\tfrac12\|x_0\|^2
-\beta E(x_0)
+\tfrac12\|x_T\|^2 .
}
$$

Then $\exp(\log w_{\mathrm{path}})$ reweights the terminal law of $Y_0$ back onto
$\pi_\beta$, and $\mathrm{ESS}/K$ of these weights is the run's honest report on
how good the score currently is.

**Why the exponential integrator earns its place.** Take $u_\theta\equiv0$ and
$\beta=0$. Each step contributes

$$
\log p_\rho-\log q_\theta
=
\frac{\|x_{n-1}-\lambda x_n\|^2-\|x_n-\lambda x_{n-1}\|^2}{2\sigma^2}
=
\frac{(1-\lambda^2)\bigl(\|x_{n-1}\|^2-\|x_n\|^2\bigr)}{2(1-\lambda^2)}
=
\tfrac12\bigl(\|x_{n-1}\|^2-\|x_n\|^2\bigr),
$$

because the cross terms $2\lambda\,x_n^\top x_{n-1}$ are symmetric and cancel.
The sum telescopes to $\tfrac12\|x_0\|^2-\tfrac12\|x_T\|^2$, which is exactly
cancelled by the two boundary terms at $\beta=0$. So

$$
\log w_{\mathrm{path}}=0
\quad\text{exactly, for every }\Delta t\text{ and every }N .
$$

The weights are therefore not merely *small* at the bottom of the ladder; they
are identically zero, and any non-zero weight observed at $\beta_k$ measures
genuine score error rather than discretization bias inherited from the
integrator. Euler–Maruyama loses this property, because its kernel does not leave
$\rho$ invariant (see *Relation to Euler–Maruyama* above).

---

## Phase 1: Initialization

Everything in this phase runs **once**, before the first gradient step.

### Step 1: Discover the temperature ladder by SMC

The problem is handed to `mppt.smc_sampler` as a `PTProblem` — the triple
(`rvs_ref`, `logpdf_ref`, `logpdf_final_target`) — from which mppt builds the
same geometric path defined above.

SMC sweeps $\beta=0\to1$ and grows the ladder greedily. From the current rung
$\beta_k$, the IS bridge to a candidate $\beta$ carries log-weights

$$
\log w_{k\to\beta}(x)
=
(\beta-\beta_k)\bigl(-E(x)+\tfrac12\|x\|^2\bigr),
$$

and the next rung is found by bisection on $\beta\in[\beta_k,1]$ for the point
where the normalized ESS of those weights falls to the threshold $\tau$
(`local_ess_ratio`, typically $0.5$); if the ESS at $\beta=1$ is still above
$\tau$, the sweep clamps there and terminates. Each rung is then resampled and
moved by `n_mcmc_steps` MCMC steps.

The ladder is thus *adaptive*: rungs cluster wherever $\pi_\beta$ changes
quickly, and no curvature or barrier estimate is needed, because SMC's own step
selection already encodes that information.

### Step 2: Harvest what SMC already computed

Two flags turn the sweep into a source of reusable state.

`keep_all_states=True` returns the particle positions at every rung, stacked as
`(M+1, N_smc, d)` and indexed to match $\{\beta_k\}$. These are the only samples
available before training that are actually distributed as $\pi_{\beta_k}$.

`keep_mcmc_params=True` returns the MCMC kernel SMC adapted *at* each rung. In
MALA mode the kernel is a single-leapfrog-step HMC with full velocity refresh —
i.e. preconditioned MALA — whose preconditioner is the Cholesky factor of the
empirical covariance of that rung's population and whose step size $\varepsilon$
is adapted to land the acceptance rate in $[0.4,0.65]$.

This kernel is reused verbatim for outer-loop rejuvenation in Phase 2. Nothing
is recalibrated afterwards, and that is deliberate: an earlier revision tuned a
separate isotropic MALA step size $h_k$ per level by bisection on particles
IS-resampled in one shot from $\beta_0$ straight to $\beta_k$. That single bridge
skips the whole ladder, so its ESS collapses and the "calibration set" degenerates
to a handful of configurations — $h_k$ then reflects the curvature at an atypical
point rather than at $\pi_{\beta_k}$. SMC's own per-rung adaptation has neither
problem, and it is free.

### Step 3: Check the seed before spending a training run

The top rung targets $\pi_{\beta_M}=\pi$, so it should agree with ground-truth
draws where those exist. This check puts a ceiling on what the seeded buffers can
teach the network: if SMC has not reached the target, no amount of training on its
output will repair that.

Two failure modes are distinguished, and they are not the same thing.

1. **Distributional mismatch.** The pooled summary statistic or the energy simply
   disagrees with the reference.

2. **Ancestral degeneracy.** The pooled histogram matches beautifully, yet the
   population carries far less independent information than its row count
   suggests. SMC resampling collapses ancestral lineages; the local MCMC move
   then re-equilibrates each particle *within* its basin, erasing literal
   duplicates while the basin assignment stays frozen. A distinct-row count
   reports 100% unique and misses this entirely.

The second is detected with **exchangeable features** — scalar features that are
statistically identical to one another under the target (the $d$ coordinates of a
factorized target; the $\binom n2$ pairwise distances of an $n$-particle
cluster). Every feature $f$ must share one exceedance probability
$p_f=P(\text{feature}_f>\tau)$, and from $N$ *independent* particles each
estimate $\hat p_f$ has standard error $\sqrt{p(1-p)/N}$. Comparing the observed
spread of $\{\hat p_f\}$ against that standard error gives an implied
independent-particle count,

$$
n_{\mathrm{eff}}
=
\frac{p(1-p)}{s^2},
\qquad
s=\mathrm{sd}_f\bigl(\hat p_f\bigr).
$$

A ground-truth set is pushed through the identical computation as a control:
being i.i.d., it should recover its own size, and when it does not, the features
are not exchangeable for this target and the test is uninformative. In practice
the gap is stark — a $10^4$-particle block at $\beta=1$ has been measured to
carry on the order of $10^2$ independent lineages.

### Step 4: Seed the replay buffers

Each level $k$ owns a fixed-capacity FIFO buffer of pairs $(x,\log w)$: a sample
approximately from $\pi_{\beta_k}$, and its IS log-weight *within that buffer*.
Mini-batches are drawn by systematic resampling on $\exp(\log w)$, so a draw is
approximately i.i.d. from $\pi_{\beta_k}$.

- **Buffer 0** ($\beta_0=0$) is filled to capacity with exact i.i.d. draws from
  $\rho=\mathcal{N}(0,I)$, uniform weights, and **frozen**. At $\beta_0=0$ the
  reference *is* the target, so exact draws strictly beat SMC particles.

- **Buffers $1,\ldots,M$** are seeded with that rung's **entire** SMC block —
  the particles SMC itself produced after its own reweight/resample/move at that
  rung — with no further MCMC pass applied here. The seed enters with **uniform**
  log-weights, not bridge weights: these particles are already
  $\approx\pi_{\beta_k}$, and bridge weights belong only to particles arriving
  from the rung below.

- **Buffer 1** is never written again during training and stays **frozen**. It
  holds genuine $\pi_{\beta_1}$ particles, so its ESS starts at $1.0$.

- **Buffers $2,\ldots,M$** additionally receive backward-SDE data during
  training.

SMC population, per-rung seed size, and buffer capacity are deliberately **one
number**, because they are the same particles at three points in their life: SMC
propagates $N_{\mathrm{smc}}$ particles up the ladder, every rung's block is fed
whole into that rung's buffer, and the outer loop then retires them by FIFO as
its own data arrives. Subsampling the block would only discard particles already
paid for — and Step 3 shows rows are not the scarce resource anyway.

The seed therefore drains in exactly

$$
\frac{N_{\mathrm{smc}}}{K}
\quad\text{outer steps},
$$

where $K$ is the number of backward-SDE particles per level per outer step. Keep
that comfortably below the total number of outer steps, or the run ends while the
network is still training partly on SMC output rather than on its own.

---

## Phase 2: Training

Each outer step runs a data-generation pass across the ladder, then a block of
gradient steps on the data accumulated so far.

### Step 1: Outer loop — parallel data generation

The active levels are $k=1,\ldots,M-1$. They are mutually independent — every
level reads the same parameter snapshot $\theta$ and writes to a distinct buffer
— so the four stages below run **simultaneously** under a single `jax.vmap` over
the level index. Only the buffer writes, being Python-level mutations, are
sequential, and they are negligible against the SDE and MCMC cost.

Note which levels are *absent*. No backward SDE is ever run at $\beta_M=1$ during
training: level $M-1$ generates and bridges *into* buffer $M$. Data at the target
temperature is always the image of an IS bridge, never a raw network sample.

For each $k\in\{1,\ldots,M-1\}$ **in parallel**:

**Step 1a — Backward SDE.** Integrate the discretized backward SDE for $N$ steps
at the current $\theta$, targeting $\pi_{\beta_k}$:

$$
\bigl(x_0^{(i)},\,\log w_{\mathrm{path}}^{(i)}\bigr)_{i=1}^{K}
\leftarrow
\mathrm{BackwardSDE}\bigl(\theta,\beta_k,N,T\bigr).
$$

The normalized path ESS is recorded per level; it is the primary diagnostic of
score quality during training.

**Step 1b — Systematic resampling.** Draw $K$ indices with probability
proportional to $\exp(\log w_{\mathrm{path}}^{(i)})$,

$$
\{x_0^{(i)}\}
\leftarrow
\mathrm{SystematicResample}\bigl(\{(x_0^{(i)},\log w_{\mathrm{path}}^{(i)})\}\bigr),
$$

after which the cloud is approximately i.i.d. from $\pi_{\beta_k}$ — but with
duplicates, since resampling copies high-weight particles.

**Step 1c — MALA rejuvenation.** Apply `n_mala_rejuv` steps of the *SMC-tuned*
preconditioned MALA kernel for rung $k$ (Phase 1, Step 2), which has $\pi_{\beta_k}$
as its exact stationary distribution regardless of step size, thanks to its
Metropolis correction:

$$
\{x_0^{(i)}\}
\leftarrow
\mathrm{MALA}_{k}\bigl(\{x_0^{(i)}\},\,\pi_{\beta_k},\,\varepsilon_k,\,L_k\bigr).
$$

This replaces duplicates with genuinely distinct configurations, which is what
makes the bridge in Step 1d worth anything: a bridge applied to a cloud of copies
inherits the copies. Since all levels are rejuvenated inside the outer `vmap`,
the per-rung kernel parameters are passed as a batched array rather than as
static scalars.

For targets defined on a constraint manifold — molecular systems with a
translation symmetry, where the energy is invariant to the center of mass — the
kernel itself is unaware of the constraint, so a `project_fn` hook re-projects
after *every* internal move rather than once at the end, keeping the chain on the
manifold throughout. Targets without such a symmetry leave the hook unset, which
preserves the faster unprojected code path.

**Step 1d — IS bridge to the next rung.** Reweight from $\pi_{\beta_k}$ to
$\pi_{\beta_{k+1}}$ using the geometric-path ratio

$$
\frac{\pi_{\beta_{k+1}}(x)}{\pi_{\beta_k}(x)}
\propto
\exp\!\left\{(\beta_{k+1}-\beta_k)\bigl(-E(x)+\tfrac12\|x\|^2\bigr)\right\},
$$

giving

$$
\log w^{(i)}_{k\to k+1}
=
(\beta_{k+1}-\beta_k)\bigl(-E(x_0^{(i)})+\tfrac12\|x_0^{(i)}\|^2\bigr).
$$

This is the same functional form SMC used to build the ladder in the first place,
which is why the ladder's $\tau$ directly controls how healthy these weights are.

**Step 1e — Buffer write, with block normalization.** Append
$\{(x_0^{(i)},\log w^{(i)})\}$ to buffer $k+1$ under FIFO eviction — but first
shift each block to mean weight $1$,

$$
\log w\;\leftarrow\;\log w-\operatorname{logsumexp}(\log w)+\log n .
$$

This matters because a buffer holds blocks written at different times and
`.sample()` resamples across all of them **jointly**, so the blocks' weight
*scales* must be commensurable. Raw bridge weights
$\Delta\beta(-E+\tfrac12\|x\|^2)$ are unnormalized and can sit orders of
magnitude away from the uniform zeros carried by the SMC seed; mixed raw, one
block silently captures the entire buffer, with the winner decided by nothing but
sign. Normalized, blocks combine in proportion to their particle counts.
Degenerate blocks (all weights $-\infty$) fall back to uniform.

### Step 2: Inner loop — joint gradient descent

Once every buffer holds at least $B$ samples, run `n_inner_steps` Adam updates on
the joint denoising objective.

**The regression target.** By the Tweedie identity for the OU process, the excess
score of the noised marginal is a conditional expectation of the excess score of
the *clean* one,

$$
\nabla_{x_t}\log\frac{p_t^\beta(x_t)}{p_t^\rho(x_t)}
=
\lambda_t\cdot
\mathbb{E}_{\pi_\beta}\!\left[
\nabla_{x_0}\log\frac{\pi_\beta(x_0)}{\rho(x_0)}
\;\middle|\;x_t
\right],
\qquad
\lambda_t=e^{-t}.
$$

The inner expectation is available in closed form — it is the identity from the
Introduction — so

$$
\phi(x_0,t,\beta)
=
\lambda_t\,\beta\,\bigl(x_0-\nabla E(x_0)\bigr),
$$

and DDPM-style training replaces the conditional expectation over $x_0\mid x_t$
with the single draw $x_0$ that generated $x_t$. The objective is

$$
\mathcal{L}(\theta)
=
\mathbb{E}_\beta\,
\mathbb{E}_{x_0\sim\pi_\beta}\,
\mathbb{E}_{t\sim\mathrm{Unif}[t_{\min},T]}\,
\mathbb{E}_{x_t\sim p(\cdot\mid x_0)}
\Bigl[
\bigl\|s_\theta(x_t,t,\beta)-\lambda_t\beta\bigl(x_0-\nabla E(x_0)\bigr)\bigr\|^2
\Bigr].
$$

Three properties are worth naming:

- At $\beta=0$ the target is identically zero, so the bottom rung trains the
  network to predict no excess over the reference — consistent with the exact
  path-weight identity proved above.
- At $\beta=1$ the target is $\lambda_t(x_0-\nabla E(x_0))$, the full correction
  from reference to target.
- The target needs $\nabla E$ only at **clean** samples, and does not depend on
  $x_t$ at all. It is computed once per buffer sample and reused.

**One inner step.** For each temperature $\beta_k$, $k=0,\ldots,M$:

1. Draw $B$ samples from buffer $k$ by systematic resampling on its stored IS
   weights.
2. Draw noise times $t_i\sim\mathrm{Unif}[t_{\min},T]$ and noise each sample
   through the forward OU kernel,
   $x_{t,i}\sim\mathcal{N}(e^{-t_i}x_{0,i},\sigma_{t_i}^2I)$.
3. Form the targets $\phi_i=\lambda_{t_i}\beta_k(x_{0,i}-\nabla E(x_{0,i}))$
   under `stop_gradient` — the energy Hessian is never backpropagated through.

The $(M+1)B$ triples $(x_{t,i},t_i,\beta_k)$ are concatenated into a **single**
batch and scored in one forward pass, and one Adam step is taken on

$$
\widehat{\mathcal{L}}(\theta)
=
\frac{1}{(M+1)B}
\sum_{k=0}^{M}\sum_{i=1}^{B}
\bigl\|s_\theta(x_{t,i},t_i,\beta_k)-\phi_i\bigr\|^2 .
$$

Gradients are clipped by global norm before the update. Two optional safeguards
exist for stiff energies, both disabled by default: clipping $\|\phi\|$ to a
maximum norm (direction preserved), and clipping the network output
element-wise. Targets like the GMM need neither; targets with hard-core
repulsion, where $\nabla E$ can be enormous at a bad configuration, can need
both.

---

## Phase 3: Production

Sampling needs no buffers, no ladder, and no MCMC:

$$
x_T\sim\mathcal{N}(0,I),
\qquad
\text{integrate the backward SDE with }s_\theta(\cdot,\cdot,\beta=1)\text{ from }T\text{ to }0,
$$

typically with a far finer time grid than training used, since no gradients are
needed — but refining past $\Delta t=t_{\min}$ starts querying noise times the
network never trained on (see *Time grid and where the network is queried*).
Accumulating $\log w_{\mathrm{path}}$ along the way costs one extra
scalar per trajectory and yields a corrected estimator for free.

This gives **three** distinct candidate answers to "what are this method's
samples", and reporting all three is what makes a run interpretable:

| # | Sample set | What it isolates |
|---|---|---|
| 1 | Backward-SDE output, no reweighting | the learned score alone |
| 2 | The same particles, IS-resampled on $\log w_{\mathrm{path}}$ | score **+** path-space correction |
| 3 | Buffer $M$, resampled on its stored weights | the training data at $\beta=1$ |

Sets 1 and 2 come from **one** run and differ only in whether the weights are
applied, so any gap between them is exactly the work the correction is doing. If
they agree, the network has genuinely learned the target; if only 2 matches, the
result is being carried by reweighting, and the path ESS says how few particles
are really behind it. Set 3 needs no SDE run at all and bounds the other two from
above — a network cannot beat its own training data.

Note that raw buffer contents are *not* $\pi$-distributed: blocks arriving from
the outer loop are $\approx\pi_{\beta_{M-1}}$ particles carrying bridge weights
*to* $\beta_M$, and only become $\approx\pi$ once those weights are applied.

---

## Systematic Resampling

Every resampling step above — SMC's, the path-weight correction, and buffer
mini-batching — uses the same scheme. Given normalized weights $\{w_i\}$ over $N$
particles and a single uniform variate $u\sim\mathrm{Unif}(0,1/n)$, place $n$
equally spaced probes

$$
u_m=u+\frac{m-1}{n},
\qquad
m=1,\ldots,n,
$$

and map each through the weight CDF to a particle index. The single shared
variate is the point: it gives $O(1)$ variance in the index counts against
$O(N)$ for multinomial resampling, at the same cost.

---

## Diagnostics:

The quantities worth watching, in the order they fail:

1. **SMC seed quality** at $\beta=1$ — distributional agreement *and* implied
   independent-particle count (Phase 1, Step 3). Read the reference set's own
   implied count first; if the control fails, the test is uninformative.
2. **Path-space ESS per level**, logged every outer step. This is score quality,
   and it is directly comparable across levels: the weights would be identically
   zero for a network that output zero, so what the ESS reports at $\beta_k$ is
   score error there and nothing else.
3. **Buffer ESS per level.** A buffer whose ESS is collapsing is training the
   network on a handful of distinct configurations however many rows it holds.
4. **MALA acceptance per rung** — SMC targets $[0.4,0.65]$; a rung far outside it
   means the reused kernel no longer suits the particles being fed to it.
5. **Ladder spacing.** Tight clusters of $\beta_k$ mark where $\pi_\beta$ moves
   fast; a single large gap is where the bridge weights will degenerate.
6. **Agreement among the three production sample sets** (Phase 3).
7. **Seed drain rate** $N_{\mathrm{smc}}/K$ against the total outer-step budget.

---

## Overall Summary:

```text
Initialization (once):
    run adaptive-ESS SMC from beta=0 to 1
        -> ladder {beta_k}, keeping all rung states and per-rung MALA kernels
    check seed quality at beta=1 (distribution + ancestral degeneracy)
    initialize score network s_theta
    buffer[0]     <- exact i.i.d. N(0,I), uniform weights            [frozen]
    buffer[1..M]  <- that rung's whole SMC block, uniform weights
    buffer[1] frozen; buffers 2..M also filled during training

Training (num_outer_steps iterations):
    -- outer loop: levels k = 1 .. M-1, all in parallel via jax.vmap --
        run backward SDE at beta_k  -> K particles + path log-weights
        systematic resample         -> approx. i.i.d. from pi_{beta_k}
        MALA rejuvenation with SMC's own kernel for rung k
        IS bridge to beta_{k+1}: log w = d_beta * (-E(x) + 0.5||x||^2)
    -- sequential (Python buffer mutations) --
        normalize each block to mean weight 1, append to buffer[k+1]

    -- inner loop: n_inner_steps Adam updates --
        for each k: draw B particles from buffer[k] by systematic resampling
        noise each to x_t via the OU kernel at t ~ Unif[t_min, T]
        target: lambda_t * beta_k * (x_0 - grad E(x_0))     [stop_gradient]
        one Adam step on the joint MSE over all (M+1)*B samples

Production:
    x_T ~ N(0, I)
    integrate backward SDE with s_theta(., ., beta=1) from T to 0
    report raw output, IS-corrected output, and buffer[M] side by side
```

The main idea is:

$$
\boxed{
\text{Let the network generate its own training data, but never trust it: correct by IS, diversify by MCMC.}
}
$$

The score network is not a sampler on its own. It is trained as the excess score
over the reference OU process, for the entire family $\{\pi_\beta\}$
simultaneously, and it is the backward SDE *plus* the path-space importance
weights that constitute the sampler. The ladder supplies the curriculum, SMC
supplies both its spacing and the kernels that keep the particles diverse, and
the exponential integrator guarantees the whole correction machinery reduces
exactly to the identity at the bottom rung — so every non-zero weight measured
above it is real signal about the score, not an artifact of the discretization.
