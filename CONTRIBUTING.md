# CONTRIBUTING.md

Development guidance for `progressive_is_tempering`.

---

## Dev setup

```bash
conda create -n pist python=3.11 && conda activate pist
pip install -e ".[test]"     # main package + pytest, pytest-xdist
pip install matplotlib       # needed by the notebooks only
python -c "import progressive_is_tempering"   # smoke check
```

The package is installed in editable mode. No `sys.path` hacks are needed or expected.

---

## Running tests

```bash
pytest                              # full suite
pytest tests/test_sampler.py        # single file
pytest -x                           # stop on first failure
pytest -n auto                      # parallel, via pytest-xdist
```

Markers registered in `pyproject.toml` (no test uses them yet, and nothing is deselected by default):

| Marker | Meaning |
|---|---|
| `slow` | End-to-end tests; deselect with `-m "not slow"` |
| `requires_x64` | Tests that need `JAX_ENABLE_X64=1` |

When adding a test that is noticeably slow (>5 s), mark it `@pytest.mark.slow` so it can be deselected.

---

## Coding patterns

### JIT-friendly loops: use `jax.lax.scan`

Python `for` loops inside a JIT region unroll at trace time, producing enormous XLA graphs. Use `jax.lax.scan` instead:

```python
# Good — traced as a single scan op
(x_final, log_w, _), _ = jax.lax.scan(step_fn, (x_T, init_lw, key), times)

# Bad — unrolls into N separate ops at trace time
for t in times:
    x, log_w = step_fn(x, log_w, t)
```

### Particle and level parallelism: use `jax.vmap`

- `backward_sde_batch` vmaps `backward_sde_single` over `n_particles`.
- The HMC kernel's `batch_move` (used by `rejuvenate`) vmaps one move over the `K` particles.
- The outer data-generation step vmaps `process_one` over temperature levels.

These three vmap axes are entirely independent and compose cleanly.

### `static_argnums` for shapes and callables

Any argument that determines the XLA graph shape (`dim`, `n_particles`, `T`, `n_steps`, `n_mala_steps`) and any Python callable (`energy_fn`, `score_apply_fn`) must appear in `static_argnums`. They must be Python-level constants at call time — not JAX arrays.

```python
@partial(jax.jit, static_argnums=(2,))
def systematic_resample(key, log_weights, n):   # n is static
    ...
```

### Unbatched network interface

`ScoreNetwork.__call__` takes `x: (d,)`, `t: scalar`, `beta: scalar`. Always vmap over batches at the call site, never inside the module:

```python
preds = jax.vmap(lambda x, t: model.apply(params, x, t, beta))(xt_batch, t_batch)
```

The exception is `EGNNDynamics` (molecular targets), which is batched: `x: (K, d)`, `t, beta: (K, 1)`.

### `stop_gradient` on TSM targets

Wrap the regression target computation in `jax.lax.stop_gradient` to prevent second-order gradients flowing through `∇E(x₀)`:

```python
targets = jax.lax.stop_gradient(jax.vmap(target_one)(x0_batch, t_batch))
```

### Excess score, not full score

The network learns `s_θ ≈ ∇ log(p_t^β / p_t^ρ)` — the score of the target relative to the Gaussian reference. It is exactly zero at `β=0`. The `output_init_scale=1e-4` in `ResMLP` enforces near-zero predictions at initialization, giving a warm start aligned with this constraint.

---

## Key gotchas

**Buffer 0 log-weights are `0`, not `-inf`.**
Buffer 0 holds i.i.d. `N(0,I)` samples with uniform weights (`jnp.zeros`). Empty buffer slots use `-inf` to be skipped by `systematic_resample`. Mixing these up corrupts sampling from buffer 0.

**`systematic_resample` requires `n` to be a static Python int.**
`@partial(jax.jit, static_argnums=(2,))` is applied. Passing a JAX scalar for `n` raises a tracing error.

**`time_embedding` requires even `embedding_dim`.**
The output is split equally into `embedding_dim // 2` sin and cos features. An odd value silently truncates to `embedding_dim - 1` output features.

**`ou_log_transition_density` is unnormalized.**
It omits `-d/2 log(2πσ²)`. This is intentional: the constant appears in both `log p` and `log q` inside `backward_sde_step` and cancels exactly. Do not use it as a standalone normalized density.

**Buffer writes are sequential, not vmapped.**
`buffers[k+1].add(...)` is a Python-level mutation that happens after `_data_gen_parallel` (the outer vmap) returns. Placing it inside the vmap silently discards the writes — JAX vmap does not support Python-level side effects.

**Inner loop is gated on all buffers being ready.**
`all_buffers_ready` blocks gradient updates until every buffer (including `buffer[M]`) holds ≥ `batch_size` samples. With SMC seeding every buffer starts full, so the gate passes from the first outer step.

**Log-weight blocks are normalized before `.add`.**
A buffer holds blocks written at different times and resamples across them jointly, so each block is shifted to mean weight 1 (`normalize_log_weights`) on the way in. See `AGENTS.md` for why.

**Per-level kernel parameters are vmapped as a stacked pytree.**
`_data_gen_parallel` receives `mcmc_params_active` (SMC's `MALAParameters`, leading axis `M_active`) and `t_idx_active = arange(1, M)`. Do not pass per-level parameters as a Python list or tuple into vmapped code, and never call `rejuvenate` with `t_idx = 0`.

**vmap over temperature levels requires uniform `n_particles`.**
The outer `jax.vmap` requires all levels to produce identically-shaped outputs. Different particle counts per temperature are not supported without restructuring the outer loop.

---

## Adding a new target distribution

A target needs a scalar energy `E(x)` and the `PTProblem` triple (`rvs_ref`, `logpdf_ref`, `logpdf_final_target`) with a standard Gaussian reference. There are two places it can live:

- **Reusable target** (as `GMMTarget`): a class in `targets.py` with
   - `energy(x: Array) -> Array` — scalar `E(x) = -log π(x) + const`
   - `log_prob(x: Array) -> Array` — `log π(x)` up to a constant
   - `grad_energy(x: Array) -> Array` — `∇E(x)` via `jax.grad`
   - `PTProblem` interface: `logpdf_ref`, `logpdf_final_target`, `rvs_ref`

   plus a `make_<name>_target(...)` factory, both listed in `targets.__all__`, and tests in `tests/test_targets.py` (shape checks, `grad_energy` against finite differences, `log_prob` consistent with `energy`).

- **Notebook-local target** (as DW4 and LJ13): wrap a bare energy (e.g. `targets.DW4`, `targets.LennardJones`) inside the experiment notebook, together with anything specific to it — for translation-invariant particle systems, the zero-CoM projection applied to reference draws, the SMC stack, forward noising, the backward SDE and `rejuvenate(..., project_fn=...)`.

---

## Adding a new experiment

1. Create a directory under `notebooks/<name>/`.
2. Write the training script as a Jupytext percent-format `.py` file (start from `notebooks/templates/minimal.py`). Notebooks are derived from the `.py` file.
3. The script should follow the algorithm phases from `AGENTS.md §Algorithm phases`: warmup (SMC ladder + per-rung kernel, buffer seeding) → outer loop → inner loop → evaluation. Keep all hyperparameters in `ModelConfig` / `TrainingConfig`.
4. Save the checkpoint, log and figures to `results/<name>/` (`checkpoint.pkl`, `log.txt`, `figures/`) and metrics records to `results/<name>/<run_id>/metrics.json`; honour `RUN_SEED` by writing under `results/<name>/seedN/`.

---

## Module dependency rules

- Leaf modules (`sde`, `targets`, `resampling`, `mala`, `mala_kernel`, `loss`, `net/embedding`, `net/mlp`, `net/egnn_jax`) must import **only** from JAX/Flax — never from each other or from higher layers.
- `progressive_is_tempering.utils` must never import from the core package.
- Cross-module imports must follow the layering diagram in `CODEMAP.md`.

---

## Commit hygiene

- Keep commits focused; one logical change per commit.
- Test before committing: `pytest`.
- Under `results/`, only `metrics.json` records are tracked; checkpoints, logs and `figures/` are gitignored — do not force-add them.
