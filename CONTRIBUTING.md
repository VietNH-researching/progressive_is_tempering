# Contributing

## Setup and tests

```bash
pip install -e ".[test]"
pytest                  # full suite
pytest -n auto          # in parallel
```

## Conventions

- Numerical code must trace under `jax.jit`. Use `jax.lax.scan` or
  `fori_loop` inside JIT regions, not Python loops.
- Shapes and callables (`dim`, `n_particles`, `n_steps`, `energy_fn`, ...) go
  in `static_argnums`, as Python values.
- `ScoreNetwork` is unbatched: call it through `jax.vmap`. `EGNNDynamics` is
  batched.
- TSM targets are wrapped in `jax.lax.stop_gradient`.
- Buffer writes are Python-level mutations, made outside any `vmap`. Shift each
  block of log-weights to mean weight 1 before `.add`.
- Empty buffer slots have log-weight `-inf`; uniform weights are `0`.
- `systematic_resample(key, log_w, n)` needs `n` as a Python int.
- `smc.py`, `mala_kernel.py`, `hmc_kernel.py`, `periodic_hmc.py` and
  `alanine.py` are order-sensitive. Rewrites that look equivalent change the
  results for a given seed.

## Adding a target

A target needs an energy `E(x)` and a `PTProblem`
(`rvs_ref`, `logpdf_ref`, `logpdf_final_target`) with a standard Gaussian
reference. Reusable energies go in `targets.py`, with tests. Target-specific
code goes in the notebook, for example a zero-CoM projection passed to
`rejuvenate(..., project_fn=...)`.

## Adding an experiment

Start from `notebooks/templates/minimal.py`, in `notebooks/<name>/<name>.py`
(Jupytext percent format). Keep hyperparameters in `ModelConfig` and
`TrainingConfig`, and write outputs to `results/<name>/[seedN/]`. Notebooks are
self-contained: they import only from the package.

Under `results/`, only `metrics.json` files are tracked.
