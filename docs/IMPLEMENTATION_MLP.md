# Implementation Notes — MLP Architecture

This document covers implementation details that are not described in
`TECHNIQUES_SUMMARY.md` and not self-evident from reading `mlp.py` or
`embedding.py`.  The focus is on: the absence of any CoM constraint, how the
score network wires its three inputs together, and the clipping decisions.

---

## No Centre-of-Mass Constraint

Unlike the EGNN architecture, the MLP places no geometric constraint on $x$.
The backward SDE initial noise, every Euler–Maruyama step, the forward noising,
and MALA proposals all operate directly in $\mathbb{R}^d$ with no projection.

---

## Score Network: Input Wiring

`ScoreNetwork` (defined in the notebook) embeds $(x, t, \beta)$ and passes the
concatenation to `ResMLP` (see `progressive_is_tempering/net/mlp.py`):

```python
t_embed    = time_embedding(t,    freq_min=1.0, freq_max=10.0, dim=time_embedding_dim)
beta_embed = time_embedding(beta, freq_min=0.5, freq_max=5.0,  dim=beta_embedding_dim)
h = jnp.concatenate([x, t_embed, beta_embed], axis=-1)
return ResMLP(dim_output=output_dim, hidden_dim=hidden_dim, n_blocks=n_blocks)(h)
```

The sinusoidal encoding is in `progressive_is_tempering/net/embedding.py`.  The two scalars use
**different frequency ranges** for a concrete reason: $t$ sweeps $[t_{\min}, T]$
within every training step (fast variation) while $\beta$ is fixed for an entire
backward SDE trajectory and changes only across temperature levels (slow
variation).  Giving $\beta$ a lower frequency range (`0.5–5.0` vs `1.0–10.0`)
prevents the two encodings from occupying the same part of feature space and
aliasing in the first hidden layer.

For the GMM target ($d = 16$, `time_embedding_dim = 16`, `beta_embedding_dim = 8`)
the concatenated input fed to `ResMLP` is $16 + 16 + 8 = 40$ dimensional.

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

This rescales the full target vector when its L2 norm exceeds `max_norm`,
preserving direction.  The `1e-6` offset prevents division by zero.

Whether this is needed depends on the energy: the Gaussian mixture has bounded
gradients and does not require it; energies with steep repulsion (e.g.
Lennard-Jones) may.

### 2. Score output clipping

The score network output can be clipped element-wise in two independent places:

**In the backward SDE control** (`make_control_fn`):
```python
out = score_net.apply(params, x, t, beta_batch)
if score_clip > 0:
    out = clip(out, -score_clip, score_clip)   # ← before multiplying by 2
return 2.0 * out
```

**In the training loss** (`loss_fn`):
```python
preds = score_net.apply(params, xt_all, t_all, beta_all)
if score_clip > 0:
    preds = clip(preds, -score_clip, score_clip)   # ← before MSE
```

During the backward SDE, clipping prevents divergent trajectories; during the
gradient step, it prevents runaway gradient magnitudes.  Both sites are disabled
when `score_clip = 0`.

### 3. Gradient norm clipping

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(grad_clip),
    optax.adam(learning_rate),
)
```

---

## `stop_gradient` on Targets

```python
targets = jax.lax.stop_gradient(
    compute_ddpm_targets(x0, t_batch, beta_mini, target_clip_norm)
)
```

Prevents gradients from flowing back through $\nabla E(x_0)$ into the outer
loss.  Since $x_0$ comes from the replay buffer, stopping the gradient is
correct and avoids building a differentiation graph through the energy Hessian.

---

## MALA Rejuvenation

The formulas follow `docs/TECHNIQUES_SUMMARY.md` exactly.  Two MLP-specific notes:

- **No CoM projection** on proposals: `xp` is accepted or rejected in the full
  ambient space, with no subspace correction.
- **`h` is a Python constant** captured in the closure (baked into the JIT trace
  at compile time); `beta` is passed as a dynamic JAX array so a single compiled
  function serves all temperature levels without retracing.

---

## Target-Specific Configurations

### Shared MLP settings

| Parameter | Value |
|---|---|
| `hidden_dim` | 256 |
| `n_blocks` | 3 |
| `output_init_scale` | 1e-4 |
| `time_embedding_dim` | 16 (freqs: 1.0 – 10.0) |
| `beta_embedding_dim` | 8 (freqs: 0.5 – 5.0) |
| `t_min` | 0.01 |
| `grad_clip` | 1.0 |
| `learning_rate` | 3e-4 |

### GMM (Gaussian Mixture Model)

| Parameter | Value |
|---|---|
| `x_dim` | 16 |
| Energy | Bimodal Gaussian mixture; $\mu_1=-1.5$, $\mu_2=2.5$, $\sigma=0.5$, $w_1=0.2$, $w_2=0.8$ |
| `score_clip` | 0 (disabled) |
| `target_clip_norm` | 0 (disabled) |
| `buffer_capacity` | 500,000 per temperature |
| Backward steps (train / eval) | 100 / 2,000 |
| Buffer 0 / Buffer 1 | i.i.d. Gaussian (exact at β=0); IS-reweighted to β₁ — both frozen |
| MALA rejuvenation | 20,000 steps, $h = 0.2$ per outer step |
| MPPT SMC | 100,000 particles, `local_ess_ratio = 0.5`, `n_mcmc_steps = 10` |
