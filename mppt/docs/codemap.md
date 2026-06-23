# Codemap

## Directory Structure

```
mppt/
  __init__.py              # Public API re-exports (guarded by TEST_MODE)
  jax_configs.py           # JAX env setup (x64, NaN debug, GPU, runtime checks)
  utils.py                 # Core utilities: WhileLoop, bisection, pytree ops
  abstract/                # Protocols and algorithm logic
    pt_concept.py          #   PTState, PTProblem, PTParticle, geometric_anneal
    pt_exec.py             #   PT loop: initialize, explore, swap, reset
    mcmc.py                #   MCMC/PTMCMC protocols, batch_move, adapters
    smc.py                 #   SMC sampler: resampling, ESS, temperature schedule
  impl/                    # Concrete implementations
    mcmc_random_walk.py    #   RW kernel + FixedCovMVNormal proposal
    mcmc_hmc.py            #   HMC kernel (Verlet integrator, velocity refresh)
    pt_mcmc_random_walk.py #   PT wrapper for RW (adds temperature fields)
    pt_mcmc_hmc.py         #   PT wrapper for HMC (adds ref/target gradients)
    swappers.py            #   basic_swapper: pairwise Metropolis swap
    logZ_estimators.py     #   SMC_like, robust, reverse_logistic
  ui/                      # User-facing API and config parsing
    ui_main.py             #   parallel_tempering, pt_manual_runners, smc_sampler
    engine_ui.py           #   Engine registry (random_walk, hmc, mala)
    explorers.py           #   Exploration modes (adaptive, nonadaptive)
    collectors.py          #   Pluggable stat collectors (logZ, acc rates, etc.)
    swappers_api.py        #   Swapper config parsing
    frozen.py              #   FrozenDict/FrozenList for immutable defaults
    index_matrix_visualizer.py  # Particle flow visualization
  ext_interface/
    particles_interface.py #   Wrapper for Chopin's particles package (numpy)
  densities/
    eccentric.py           #   Test distribution: mixture of two Gaussians
  tests/                   # Test suite (plain functions, pytest-compatible)
    __init__.py            #   TEST_MODE flag
    conftest.py            #   Autouse fixture: resets ALLOW_RUNTIME_CHECK
    distribution_test_helper.py  # TV distance, KDE, sample validation
    pt_test_helper.py      #   Random PT problem generators
    test_*.py              #   11 test modules
docs/
  extend.md               # Design doc: the extend pattern philosophy
notebooks/
  test.ipynb              # Usage demo: SMC then PT with HMC
```

## Module Dependency Graph

```
ui/ui_main.py
  -> abstract/pt_exec.py  (initialize, swap, reset, explore)
  -> abstract/smc.py      (one_step_smc)
  -> ui/engine_ui.py       -> impl/pt_mcmc_{rw,hmc}.py -> impl/mcmc_{rw,hmc}.py
  -> ui/explorers.py       -> abstract/pt_exec.py
  -> ui/collectors.py      -> abstract/pt_exec.py, impl/logZ_estimators.py
  -> ui/swappers_api.py    -> impl/swappers.py

abstract/pt_exec.py
  -> abstract/pt_concept.py (PTState, PTProblem, geometric_anneal)
  -> abstract/mcmc.py       (PTMCMCEngine, batch ops)
  -> utils.py               (pytree_index, WhileLoop, etc.)

abstract/mcmc.py
  -> abstract/pt_concept.py
  -> utils.py

impl/pt_mcmc_*.py
  -> impl/mcmc_*.py          (delegates actual MCMC logic)
  -> abstract/pt_concept.py  (geometric_anneal, PTProblem)
  -> abstract/mcmc.py        (protocols)
```

## Key Types

| Type | Location | Role |
|------|----------|------|
| `PTProblem` | `abstract/pt_concept.py` | Problem spec: `rvs_ref`, `logpdf_ref`, `logpdf_final_target` |
| `PTState` | `abstract/pt_concept.py` | Algorithm state: `X` (flat particles), `Z` (K,N index matrix) |
| `PTParticle` | `abstract/pt_concept.py` | Protocol: position + temperature + logpdfs |
| `PTMCMC` | `abstract/mcmc.py` | Protocol: MCMC kernel with `reset_temperature` |
| `PTMCMCEngine` | `abstract/mcmc.py` | Factory: `(problem, t, t_idx) -> PTMCMC` |
| `WhileLoop` | `utils.py` | Lazy while loop as data (for adaptive nsteps) |
| `FixedCovMVNormal` | `impl/mcmc_random_walk.py` | Proposal distribution with scale + covariance |
| `HMCParameters` | `impl/mcmc_hmc.py` | HMC params: scaled covariance + refresh rate |
| `TwoLevelSwapper` | `abstract/pt_concept.py` | Protocol: swap between two adjacent temperatures |
| `LogZ_Estimator` | `abstract/pt_concept.py` | Protocol: estimate log partition function |
| `ExploreResult` | `abstract/pt_exec.py` | Output of exploration step (state + diagnostics) |
| `SwapResult` | `abstract/pt_exec.py` | Output of swap step (state + rejection rates) |

## Entry Points

| Function | Location | Purpose |
|----------|----------|---------|
| `parallel_tempering()` | `ui/ui_main.py` | Full PT with automatic adaptation |
| `pt_manual_runners()` | `ui/ui_main.py` | Returns a pre-compiled one-step callable |
| `smc_sampler()` | `ui/ui_main.py` | SMC annealing (BETA) |
| `PTProblem()` | `abstract/pt_concept.py` | Construct a problem |
| `visualize_index_matrix()` | `ui/index_matrix_visualizer.py` | Particle flow visualization |

## Patterns

- **Geometric annealing**: `logpdf_target(x;t) = (1-t)*logpdf_ref(x) + t*logpdf_final_target(x)`
- **Index-based swaps**: swaps mutate only `Z`, never `X`; `make_contiguous()` reindexes when needed
- **Extend pattern**: `feed_extend` -> `init_extend` / `inner_extend` bundles position with computed fields (see `docs/extend.md`)
- **Two-half exploration**: split particles, adapt on one half, move the other, then swap roles
- **WhileLoop as data**: adaptive nsteps across temperatures, executed via `while_loops_executor` until a proportion finish
- **Collector framework**: pluggable statistics registered with `@add_collector`
- **Engine registry**: pluggable kernels registered with `@register_engine`
- **Protocol-based design**: structural typing via `Protocol`, no inheritance; `@follow_protocol` decorator documents intent
