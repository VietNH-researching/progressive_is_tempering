# CONTRIBUTING.md

Development guidance for `progressive_is_tempering`.

---

## Dev setup

```bash
conda activate pist
pip install -e ./mppt        # local mppt library (must be done first)
pip install -e ".[test]"     # main package + pytest, pytest-xdist, chex
python -c "import progressive_is_tempering"   # smoke check
```

Both packages are installed in editable mode. No `sys.path` hacks are needed or expected.

---

## Running tests

```bash
pytest                              # fast suite (default; skips slow)
pytest -m slow                      # end-to-end integration tests
pytest tests/test_sampler.py        # single file
pytest -x                           # stop on first failure
JAX_ENABLE_X64=1 pytest -m requires_x64   # precision-sensitive tests
```

Markers defined in `pyproject.toml`:

| Marker | Meaning |
|---|---|
| `slow` | End-to-end tests; excluded by default |
| `requires_x64` | Require `JAX_ENABLE_X64=1`; excluded by default |

When adding a test that is noticeably slow (>5 s), mark it `@pytest.mark.slow` so the default suite stays fast.

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
- `mala_rejuvenate` vmaps `rejuvenate_one` over the `K` particles.
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

### `stop_gradient` on DDPM targets

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
`all_buffers_ready` blocks gradient updates until every buffer (including `buffer[M]`) holds ≥ `batch_size` samples. The first `M-2` outer steps generate data but produce no parameter updates. This is expected behavior.

**Per-level MALA step sizes must be a JAX array when vmapped.**
`_data_gen_parallel` receives `mala_hs_active: (M_active,)` as a traced JAX array. Do not pass per-level step sizes as a Python list or tuple into vmapped code.

**vmap over temperature levels requires uniform `n_particles`.**
The outer `jax.vmap` requires all levels to produce identically-shaped outputs. Different particle counts per temperature are not supported without restructuring the outer loop.

---

## Adding a new target distribution

1. Create a dataclass (or plain class) in `targets.py` with the methods:
   - `energy(x: Array) -> Array` — scalar `E(x) = -log π(x) + const`
   - `log_prob(x: Array) -> Array` — `log π(x)` up to a constant
   - `grad_energy(x: Array) -> Array` — `∇E(x)` via `jax.grad`
   - MPPT interface: `logpdf_ref`, `logpdf_final_target`, `rvs_ref`

2. Add a `make_<name>_target(...)` factory function.

3. Export both from `targets.__all__` and re-export from `progressive_is_tempering/__init__.py`.

4. Add tests in `tests/test_targets.py`: shape checks, `grad_energy` correctness via finite differences, `log_prob` consistency with `energy`.

---

## Adding a new experiment

1. Create a directory under `notebooks/<name>/`.
2. Write the training script as a Jupytext percent-format `.py` file (see `notebooks/gmm/gmm.py` for a template). Notebooks are derived from the `.py` file.
3. The script should follow the algorithm phases from `AGENTS.md §Algorithm phases`: warmup (ladder construction + MALA calibration) → outer loop → inner loop → evaluation.
4. Save checkpoints to `checkpoints/<name>/` and figures to `figures/<name>/`.

---

## Module dependency rules

- Leaf modules (`sde`, `targets`, `resampling`, `mala`, `loss`, `net/embedding`, `net/mlp`, `net/egnn_jax`) must import **only** from JAX/Flax — never from each other or from higher layers.
- `progressive_is_tempering.utils` must never import from the core package.
- Cross-module imports must follow the layering diagram in `CODEMAP.md`.
- `mppt` is an external dependency; import it as a normal package. Do not vendor or modify it.

---

## Commit hygiene

- Keep commits focused; one logical change per commit.
- Test before committing: `pytest` (and `pytest -m slow` for end-to-end changes).
- Checkpoint files (`*.pkl`) and figure outputs (`*.png`, `*.gif`) are gitignored — do not force-add them.
