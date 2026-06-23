# MPPT -- Agent and Contributor Guide

A JAX library for multi-particle parallel tempering MCMC.
See `docs/architecture.md` for design rationale and `docs/codemap.md` for file layout.

## Quick Orientation

- Entry points: `parallel_tempering`, `pt_manual_runners`, `smc_sampler` in `mppt/ui/ui_main.py`
- Problem definition: `PTProblem(rvs_ref, logpdf_ref, logpdf_final_target)` in `mppt/abstract/pt_concept.py`
- Adding a new MCMC kernel: create `impl/mcmc_<name>.py` + `impl/pt_mcmc_<name>.py`, register in `ui/engine_ui.py`
- Adding a new collector: use `@add_collector` in `ui/collectors.py`

## Architecture Overview

Three-layer design:

```
abstract/   Protocols + algorithm logic    (what the algorithm does)
impl/       Concrete kernels               (how each piece works)
ui/         User API + config parsing      (how users interact)
```

Adding a new MCMC kernel requires only new files in `impl/` and a registration in `ui/engine_ui.py`. No changes to `abstract/`.

## Running Tests

Requires Python 3.12+ (uses PEP 695 type parameter syntax).

Tests use pytest with plain test functions. A `conftest.py` autouse fixture
resets `ALLOW_RUNTIME_CHECK` after each test to prevent cross-test state leakage.

```bash
# Run all tests
python -m pytest mppt/tests/ -v

# Run a single test file
python -m pytest mppt/tests/test_swappers.py -v

# Run a specific test
python -m pytest mppt/tests/test_utils.py::test_bisection -v
```

Some tests are compute-heavy (large particle counts, many iterations).
If a test seems to hang, it may just be slow on CPU.

The `TEST_MODE` flag in `mppt/tests/__init__.py` must be `False` for normal operation.
It exists to prevent JAX from loading before env vars are set.

### Which tests to run after a change

| Changed file | Run |
|---|---|
| `impl/mcmc_hmc.py` or `impl/pt_mcmc_hmc.py` | `test_mcmc_hmc.py test_pt_mcmc_hmc.py` |
| `impl/mcmc_random_walk.py` or `impl/pt_mcmc_random_walk.py` | `test_mcmc_random_walk.py test_pt_mcmc_random_walk.py` |
| `abstract/pt_exec.py` | `test_pt_exec.py` |
| `impl/swappers.py` | `test_swappers.py` |
| `impl/logZ_estimators.py` | `test_logZ.py` |
| `utils.py` | `test_utils.py` |
| Anything in `abstract/` | Run the full suite |

## Dependencies

Core: `jax`, `jaxlib`, `chex`, `numpy`, `scipy`, `tqdm`

Optional (visualization): `matplotlib`, `seaborn`

Testing: `pytest`, `statsmodels` (for KDE in test helpers)

Benchmarking (optional): `particles` (Nicolas Chopin's package, used only in `ext_interface/`)

## Conventions

### Code Style

- **NamedTuples** for all data types (particles, parameters, results). Not dataclasses, not dicts.
- **Protocols** for interfaces (structural typing, no inheritance). The `@follow_protocol` decorator is documentation only, has no runtime effect.
- **Functions over classes** for algorithm logic. Classes are used only as data containers (NamedTuples) or protocols.
- **`dummy_use()`** suppresses unused-variable warnings in JAX callbacks where arguments are structurally required but not used.

### JAX Patterns

- All randomness flows through JAX's functional PRNG (`random.split`). Never use `numpy.random` inside JIT-traced code.
- `random_split(key, N)` in `utils.py` returns `(fresh_key, subkeys_array)`, not the standard JAX split signature.
- Use `lax.cond` for conditional logic, never Python `if` on traced values.
- Use `lax.fori_loop` for fixed-iteration loops, `lax.while_loop` for dynamic ones.
- `donate_argnums` is used on the main PT loop to avoid doubling memory.

### Naming

| Symbol | Meaning |
|--------|---------|
| `t` | Temperature, float in [0,1] |
| `t_idx` | Temperature index, int |
| `K` or `T` | Number of temperature levels |
| `N` | Number of particles per temperature |
| `d` | Dimension of the sampling space |
| `X` | Flat particle pytree (length K*N) |
| `Z` | Index matrix, shape (K, N) |
| `logpdf_ref` | Log density of reference (t=0) |
| `logpdf_final_target` | Log density of target (t=1) |
| `logpdf_target` | Log density at current temperature |

### The Extend Pattern

Read `docs/extend.md` before modifying any MCMC kernel.

The key rule: MCMC `move` functions never call the target density directly.
They call `extend(x_proposed)` which bundles position with all computed fields.
This is how temperature and other PT-specific attributes propagate automatically.

For HMC, the pattern splits into three levels:

```
feed_extend(x)       -> position + logpdf + gradient (no velocity)
init_extend(key, x)  -> feed_extend(x) + random velocity
inner_extend(x, v)   -> feed_extend(x) + given velocity
```

For random walk, all three coincide.

## Common Pitfalls

1. **Forgetting contiguity after swaps**: After swaps, `Z` is shuffled but `X` is not. The exploration step assumes contiguity (Z = identity). Always call `reset_temperature` (which internally calls `make_contiguous`) before exploration.

2. **Particle shapes**: Particles are stored flat as `(K*N, ...)`. Use `PTState.reshape_to_T_N(tree, T, N)` and `PTState.flatten_first_two_dims(tree)` for reshaping. These operate on full pytrees.

3. **Temperature 0 special case**: At `t_idx=0`, the MCMC kernel is replaced by iid sampling from the reference (`rvs_ref`). Handled inside each `pt_mcmc_*.py` via `lax.cond(t_idx == 0, ...)`.

4. **Runtime checks**: `ALLOW_RUNTIME_CHECK` is `False` by default. When `True`, chex assertions fire everywhere and performance drops significantly. Use `allow_runtime_check(True)` context manager for debugging only.

5. **`verbose_scan` vs `lax.scan`**: `verbose_scan` is a Python-level loop (for progress bars). It does not trace the loop body into a single XLA program. Used when `verbose=True`.

## Sensitive Areas

Modify these with extra care:

- **`abstract/pt_concept.py`**: `PTState` and `PTProblem` are used everywhere. Changes cascade widely.
- **The extend wiring in `pt_mcmc_*.py`**: the `feed_extend` -> particle type chain is delicate. Adding/removing fields requires matching changes in the NamedTuple, the `_NoVeloc` variant (for HMC), and `reset_temperature`.
- **`utils.py:while_loops_executor`**: the vmap-over-while-loops logic is subtle and well-tested.

## Project Status

- PT with RW/HMC/MALA: stable, well-tested
- SMC sampler: beta, functional but less tested
- `ext_interface/`: benchmarking utility against Chopin's `particles` package, not part of the core library
- No packaging infrastructure yet (no pyproject.toml)

## Documentation Map

| Document | Purpose |
|----------|---------|
| `readme.md` | Usage examples, installation, quick start |
| `AGENTS.md` | This file: conventions and agent instructions |
| `docs/architecture.md` | Design decisions and trade-offs |
| `docs/extend.md` | The extend pattern (how MCMC kernels compose with PT) |
| `docs/codemap.md` | Module layout, dependency graph, key types |
| `docs/api.md` | User-facing API reference: engines, explorers, collectors |
