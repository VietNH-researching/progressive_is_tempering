# Sequential Score Sampler (SSS)

SSS trains one $\beta$-conditioned score network to sample from
$\pi(x)\propto e^{-E(x)}$, where $E$ is tractable. It works along a temperature
ladder from a Gaussian to the target. The network generates its own training
data, which is corrected by importance sampling and diversified by MCMC before
being trained on.

## 1. Tempered path

With the reference $\rho=\mathcal N(0,I)$,

$$
\pi_\beta(x)\propto\exp\{-(1-\beta)\tfrac12\|x\|^2-\beta E(x)\},\qquad\beta\in[0,1],
$$

so $\pi_0=\rho$ and $\pi_1=\pi$. The path is discretized as a ladder
$0=\beta_0<\beta_1<\dots<\beta_M=1$. Moving a sample from rung $k$ to rung $k+1$
uses the bridge weight

$$
\log w_{k\to k+1}(x)=(\beta_{k+1}-\beta_k)\bigl(\tfrac12\|x\|^2-E(x)\bigr).
$$

## 2. Initialization by SMC

An adaptive SMC sweep from $\beta=0$ to $1$ chooses each next $\beta$ by
bisection so that the normalized ESS of the bridge weights stays at a set
threshold. At every rung it resamples and applies a few MCMC moves. The kernel
is HMC with one leapfrog step, preconditioned by the rung's empirical
covariance, with its step size tuned to a target acceptance rate. The sweep
returns:

- the ladder $\{\beta_k\}$;
- one particle block per rung, approximately distributed as $\pi_{\beta_k}$;
- one tuned MCMC kernel per rung, reused for rejuvenation during training.

Each rung $k$ has a FIFO replay buffer of weighted samples:

- Buffer $0$ holds exact draws from $\rho$.
- Buffer $k\ge1$ starts with rung $k$'s SMC block, with uniform weights.
- Buffers $0$ and $1$ stay fixed. During training, new data replaces the SMC
  seed in buffers $2,\dots,M$.

## 3. Diffusion model

**Forward process.** The forward process is OU:
$dX_t=-X_t\,dt+\sqrt2\,dW_t$, so
$X_t\mid X_0\sim\mathcal N(e^{-t}X_0,\sigma_t^2I)$ with $\sigma_t^2=1-e^{-2t}$.
The reference $\rho$ is stationary for this process.

**Excess score.** The network learns the score of the noised $\pi_\beta$
relative to the noised reference:

$$
s_\theta(x_t,t,\beta)\approx\nabla_{x_t}\log\frac{p_t^\beta(x_t)}{p_t^\rho(x_t)} .
$$

This excess score is exactly zero at $\beta=0$.

**Backward SDE.** Sampling starts from $x_T\sim\rho$. Each step uses an
exponential integrator with the control $u_\theta=2s_\theta$:

$$
x_{n-1}=\lambda_n x_n+(1-\lambda_n)\,u_\theta(x_n,t_n,\beta)+\sigma_n\varepsilon_n,
\qquad \lambda_n=e^{-\Delta t_n},\ \sigma_n^2=1-e^{-2\Delta t_n}.
$$

The steps of the time grid shrink towards $t=0$. The grid is geometric from $T$
down to $t_{\min}$, then a last step to 0; LJ13 uses a power grid instead.

**Path weights.** Every path carries an importance weight that corrects its
endpoint towards $\pi_\beta$:

$$
\log w_{\rm path}=\sum_{n=1}^N\bigl[\log p_\rho(x_n\mid x_{n-1})-\log q_\theta(x_{n-1}\mid x_n)\bigr]
-(1-\beta)\tfrac12\|x_0\|^2-\beta E(x_0)+\tfrac12\|x_T\|^2 .
$$

Here $p_\rho$ is the exact forward OU kernel and $q_\theta$ is the backward
step above. When $u_\theta\equiv0$ and $\beta=0$, $\log w_{\rm path}=0$ exactly
for any grid. A non-zero weight therefore measures score error, not
discretization error.

## 4. Training

Each **outer step** processes the rungs $k=1,\dots,M-1$ in parallel:

1. Run the backward SDE at $\beta_k$ to get $K$ samples and their path weights.
2. Resample systematically on the path weights.
3. Rejuvenate the samples with the rung-$k$ MCMC kernel.
4. Compute bridge weights to $\beta_{k+1}$. Shift the block to mean weight 1
   and add it to buffer $k+1$, storing $\nabla E$ with each sample.

Each **inner step** draws a mini-batch from every buffer by systematic
resampling on the stored weights. It noises each $x_0$ to $x_t$ and takes one
Adam step on the joint Target Score Matching loss:

$$
\mathcal L(\theta)=\sum_{k=0}^M\mathbb E\,\bigl\|s_\theta(x_t,t,\beta_k)-e^{-t}\beta_k\bigl(x_0-\nabla E(x_0)\bigr)\bigr\|^2 .
$$

## 5. Sampling

Run the backward SDE at $\beta=1$, on a finer grid than in training. The method
reports three sample sets:

| Sample set | Description |
|---|---|
| `sde_raw` | backward-SDE output without reweighting |
| `sde_is` | the same samples, resampled on $\log w_{\rm path}$ |
| `buffer` | buffer $M$, resampled on its stored weights |

## Target-specific notes

- **Particle systems (DW4, LJ13, LJ55).** States are projected to zero centre
  of mass after every SDE step and every MCMC move. The score network is an
  EGNN.
- **Alanine dipeptide.** SMC and rejuvenation run in internal coordinates, with
  torsions treated as circles. The network, backward SDE and buffers use
  Cartesian coordinates. The loss is denoising score matching plus a small TSM
  term at small $t$. A block enters its buffer only if its path ESS passes a
  threshold. The score network is a DiT.

## Summary

```text
SMC from beta=0 to 1 -> ladder {beta_k}, per-rung particle blocks and MCMC kernels
buffer[0] <- exact N(0, I) draws; buffer[k] <- SMC block of rung k
repeat:
    for k = 1 .. M-1 (in parallel):
        backward SDE at beta_k -> samples + path weights
        resample, rejuvenate with rung-k MCMC kernel
        bridge weights to beta_{k+1}, add to buffer[k+1]
    n_inner_steps Adam steps on the TSM loss over all buffers
sample: backward SDE at beta = 1 (+ path weights)
```
