# Notebook Agent Instructions

These instructions apply inside `notebooks/`.

## Jupytext Policy

- Treat percent-format `.py` files as the canonical notebook source.
- Do not edit `.ipynb` JSON directly unless the user explicitly asks for that.
- If both `.py` and `.ipynb` exist, keep them synchronized with `jupytext --sync`
  when Jupytext is available.
- If Jupytext is unavailable, edit only the `.py` source and report that the
  paired `.ipynb` was not regenerated.


## Authoring Rules

- Start new notebooks from `templates/minimal.py` unless the user requests a
  different structure.
- Keep notebook code exploratory. Number the sections to follow the steps in `docs/TECHNIQUES_SUMMARY.md`.
- Reusable algorithms, networks, and samplers belong in `progressive_is_tempering/` with tests.
  Target-specific code (the wrapper around a `targets` energy, CoM projection, plots,
  config values) stays in the notebook; see `AGENTS.md` ("Each notebook is self-contained").
- Estimation, metrics and diagnostics are kept in `progressive_is_tempering/utils/`.
- Neural network architectures are kept in `progressive_is_tempering/net/`: `ScoreNetwork` (ResMLP) for unstructured targets, `EGNNDynamics` for particle systems.
- Make sure the notebook aligns with the method description in `docs/TECHNIQUES_SUMMARY.md` and that the unit tests in `tests/` pass.
- Keep imports explicit and close to the top of the notebook.
- Use deterministic PRNG seeds in examples and diagnostics.
- Each notebook will be in charge of the sampling algorithm for one specific target distribution. All notebooks should align with each other in the main method summarized in `TECHNIQUES_SUMMARY.md`. The model configurations should stay in one place as `ModelConfig()` in the notebook so that it can easily be tuned by users. Also, the training configurations will stay in one place as `TrainingConfig()`.
- Durable notebooks should live in one folder per notebook: `notebooks/<name>/<name>.py`. For example the notebook implemented for sampling from Gaussian Mixture Model should stay as `notebooks/gmm/gmm.py`.
- Everything a run writes goes to `results/<name>/` (`results/<name>/seedN/` when `RUN_SEED=N` is set): `checkpoint.pkl`, `log.txt`, `figures/`, and one `<run_id>/metrics.json` per evaluated sample set (`sde_raw`, `sde_is`, `buffer`). Only the `metrics.json` files are tracked.
- `resume = False` starts from scratch; `resume = True` continues from `checkpoint.pkl`. `resume` sits next to `TrainingConfig()`.
- Starting from scratch appends `Attempt xxx` to `log.txt`, where `xxx` is the number of attempts.
- Scripts are run from their own folder (`cd notebooks/<name> && python <name>.py`); output and data paths are relative to it.
- Avoid relying on hidden Jupyter state. A notebook should run from top to bottom after a kernel restart.

## Commit Hygiene

- Prefer committing the `.py` notebook source.
- Commit `.ipynb` files only when they are intentionally useful to humans and
  outputs are cleared or small.
