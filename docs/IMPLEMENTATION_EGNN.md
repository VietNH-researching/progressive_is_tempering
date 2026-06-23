# Implementation Notes - EGNN Architecture

This document covers the implementation details that are not described in
`docs/TECHNIQUES_SUMMARY.md` and not self-evident from reading `progressive_is_tempering/net/egnn_jax.py`.
The focus is on: how the CoM constraint is enforced throughout the computation,
how the various clipping mechanisms work and where they are applied, and how the
MALA rejuvenation is implemented exactly.

The score network architecture is implemented in JAX/Flax; see `progressive_is_tempering/net/egnn_jax.py` for
the full details.

---

## Centre-of-Mass Projection

### The `project_com` function

```python
def project_com(x, n_particles, spatial_dim):
    xr  = x.reshape(..., n_particles, spatial_dim)
    com = xr.mean(axis=-2, keepdims=True)
    return (xr - com).reshape(x.shape)
```

This works for both batched `(B, N*D)` and unbatched `(N*D,)` inputs because the
reshape/mean/reshape pattern uses `shape[:-1]` as the leading dimensions.

### Where projection is applied and in what order

**Backward SDE initialisation**

The prior noise sample is projected before the scan begins:

```python
x_init = jax.random.normal(key, (n_samples, x_dim))
x_init = project_com(x_init)
```

**Inside each backward SDE step**

The projection happens *after* the Euler step and *before* the log-weight increment:

```python
u      = control_fn(x, t_batch)             # control from projected x
mu_bwd = x + (-x + u) * dt                  # backward proposal mean

x_next = mu_bwd + sqrt(2*dt) * eps          # Euler step
x_next = project_com(x_next)                # ← project here

mu_fwd    = (1.0 - dt) * x_next             # forward mean uses projected x_next
log_ratio = (
    -sum((x - mu_fwd)**2, axis=-1) / (4*dt)           # log p_fwd
    + 0.5 * sum(((x_next - mu_bwd) / sqrt(2*dt))**2, axis=-1)  # -log q
)
```

`mu_bwd` is derived from `x` (the already-projected state entering this step).
`mu_fwd` uses the projected `x_next`.  Both terms of `log_ratio` are therefore
computed relative to projected states.

**Forward noising**

The OU transition adds noise that shifts the CoM away from zero
(`xt = λ_t x₀ + σ_t Z`, `CoM(xt) = σ_t CoM(Z) ≠ 0` in general).  Projection is
applied after noising:

```python
xt = sample_ou_transition(x0, t_batch, key)
return project_com(xt)
```

**MALA rejuvenation**

The input state is projected at the start of each MALA step (defensive, since it
should already be zero-CoM), and the proposal is projected immediately after the
Langevin move:

```python
x  = project_com(x)
g  = -(1 - β)*x - β*∇E(x)
xp = x + (h²/2)*g + h*ε
xp = project_com(xp)         # ← project proposal
gp = -(1 - β)*xp - β*∇E(xp) # gradient at projected proposal
```

Both `g` and `gp` are evaluated at projected states, and the MH log-ratio uses the
projected `x` and `xp` throughout.

**EGNN output**

Inside `EGNNDynamics`, the output velocity is projected before returning:

```python
vel = coord - coord_in
vel = vel - vel.mean(axis=1, keepdims=True)   # zero-CoM velocity
```

This ensures the learned displacement has no net translation component.

### The Q basis (MPPT only)

A separate orthonormal basis `Q` of shape `(N*D, N*D - D)` — the null space of the
CoM constraint matrix — is computed via SVD and used exclusively by MPPT, which
operates in the reduced `(N*D - D)`-dimensional space.  MPPT samples are lifted back
to the ambient space as `x = z @ Q.T`.

Training and inference never use `Q`; they operate in R^{N*D} with `project_com`.

---

## Clipping

### 1. Target norm clipping

Applied inside `compute_ddpm_targets`, before `stop_gradient`:

```python
targets = λ_t * β * (x0 - ∇E(x0))
if max_norm > 0:
    norms      = linalg.norm(targets, axis=-1, keepdims=True)
    clip_coeff = minimum(max_norm / (norms + 1e-6), 1.0)
    targets    = targets * clip_coeff
```

This rescales the full target vector when its L2 norm exceeds `max_norm`, preserving
direction.  The `1e-6` offset prevents division by zero.

The appropriate threshold depends on the target energy: DW4 uses 20.0; LJ13 uses
260.0 because the Lennard-Jones 1/r¹² repulsion produces very large gradients at
short interparticle distances.

### 2. Score output clipping

The score network output is clipped element-wise in two places:

**In the backward SDE control** (`make_control_fn`):
```python
out = score_net.apply(params, x, t, beta_batch)
if score_clip > 0:
    out = clip(out, -score_clip, score_clip)   # ← before multiplying by 2
return 2.0 * out
```

**In the training loss** (`loss_fn` inside `gradient_step_joint`):
```python
preds = score_net.apply(params, xt_all, t_all, beta_all)
if score_clip > 0:
    preds = clip(preds, -score_clip, score_clip)   # ← before MSE
per_sample_sq = sum((preds - targets)**2, axis=-1)
```

Clipping predictions in the loss ensures that runaway network outputs do not produce
runaway gradients.  The two call sites are independent: during the backward SDE (no
gradient computed through the score), clipping prevents unstable trajectories;
during the gradient step, clipping prevents large gradient magnitudes.

### 3. Gradient norm clipping

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(grad_clip),
    optax.adam(learning_rate),
)
```

`clip_by_global_norm` rescales the entire gradient vector when its L2 norm exceeds
`grad_clip = 1.0`, then Adam is applied to the clipped gradient.

---

## `stop_gradient` on Targets

```python
targets = jax.lax.stop_gradient(
    compute_ddpm_targets(x0, t_batch, beta_mini, target_clip_norm)
)
```

This prevents gradients from flowing back through `∇E(x₀)` (which is computed by
`jax.grad` of the energy) into the outer loss.  Since `x₀` comes from the replay
buffer (not from the current parameters), stopping the gradient is both correct and
efficient — otherwise JAX would build a differentiation graph through the energy
Hessian unnecessarily.

---

## MALA Rejuvenation: Exact Formulas

The step size `h` is baked into the JIT-compiled function as a Python constant
(not passed as a JAX array), so it is specialised at compile time.

**Proposal** (Langevin with MH correction):
```
xp = x + (h²/2) * ∇log π_β(x) + h * ε,   ε ~ N(0, I)
```
where `∇log π_β(x) = -(1-β)*x - β*∇E(x)`.

**Proposal log-densities** (Gaussian with variance `h²·I`):
```
log q(xp | x) = -1/(2h²) * ||xp - x - (h²/2)*g||²
log q(x | xp) = -1/(2h²) * ||x - xp - (h²/2)*gp||²
```
Note the variance is `h²` (not `2h²`), so the quadratic coefficient is `1/(2h²)`,
not `1/(4h²)` as in the backward SDE.

**MH log-acceptance ratio**:
```
log α = log π_β(xp) - log π_β(x) + log q(x|xp) - log q(xp|x)
```
where `log π_β(x) = -(1-β)*½*||x||² - β*E(x)`.

**Chain** runs for `n_mala_rejuv` steps via `jax.lax.scan`; acceptance rates are
averaged over the chain and returned for monitoring.

---

## Loss Function

The per-sample loss is a **sum** (not mean) over the spatial dimension:

```python
per_sample_sq = jnp.sum((preds - targets)**2, axis=-1)   # shape (n_temps*B,)
```

followed by a mean over the mini-batch and a mean over temperatures.  Because the
sum is over `x_dim`, the raw loss value scales with the problem dimension (8 for
DW4, 39 for LJ13); loss values are not directly comparable across targets.

---

## Target-Specific Configurations

### Shared EGNN settings

| Parameter | Value |
|---|---|
| `hidden_nf` | 128 |
| `n_layers` | 5 |
| `coords_range` | 15.0 total / 3.0 per layer |
| `beta_embedding_dim` | 8 |
| `recurrent`, `attention`, `tanh` | True, True, True |
| `t_min` | 0.01 |
| `score_clip` | 1e4 |
| `grad_clip` | 1.0 |
| `learning_rate` | 3e-4 |

### DW4

| Parameter | Value |
|---|---|
| `n_particles`, `spatial_dim`, `x_dim` | 4, 2, 8 |
| CoM-reduced dimension | 6 (Q shape 8×6) |
| Energy | Double-well; translation-invariant (`trap_scale=0`) |
| `target_clip_norm` | 20.0 |
| `buffer_capacity` | 500,000 per temperature |
| Backward steps (train / eval) | 100 / 2000 |
| Buffer 0 / Buffer 1 | i.i.d. Gaussian (exact at β=0); IS-reweighted to β₁ — both frozen |
| MALA rejuvenation | 20,000 steps, h = 0.2 per outer step |
| MPPT SMC | 100,000 particles, `local_ess_ratio=0.2`, `n_mcmc_steps=10` |

### LJ13

| Parameter | Value |
|---|---|
| `n_particles`, `spatial_dim`, `x_dim` | 13, 3, 39 |
| CoM-reduced dimension | 36 (Q shape 39×36) |
| Energy | Lennard-Jones; `ε=2.0`, `rₘ=1.0`, `trap_scale=1.0` (individual) |
| `target_clip_norm` | 260.0 |
| `buffer_capacity` | 10,000 per temperature |
| Backward steps (train / eval) | 100 / — |
| Buffer initialisation | All buffers filled directly from MPPT GT samples, log_w = 0 |
| MALA rejuvenation | Not used in the ground-truth training notebook |
| MPPT SMC | 10,000 particles, `local_ess_ratio=0.1`, `n_mcmc_steps=50` |
