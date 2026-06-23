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
- Keep notebook code exploratory. Numbering each Section corresponding to the `docs/TECHNIQUES_SUMMARY.md`
- Reusable algorithms, target wrappers,
  networks, and samplers belong in `progressive_is_tempering/` with tests.
- Estimation and diagnostics are kept in `progressive_is_tempering/utils/`.
- Neural Network architecture are kept in `progressive_is_tempering/net/`. More details descriptions on how to use the Neural Network architecture for specific targets are given in `docs/IMPLEMENTATION_MLP.md` and `docs/IMPLEMENTATION_EGNN.md`
- Make sure the implemented notebook align with the method descriptions in `docs/TECHNIQUES_SUMMARY.md` and pass the unit test in `tests/`.
- Keep imports explicit and close to the top of the notebook.
- Use deterministic PRNG seeds in examples and diagnostics.
- Each notebook will be in charge of the sampling algorithm for one specific target distribution. All notebooks should align with each other in the main method summarized in `TECHNIQUES_SUMMARY.md`. The model configurations should stay in one place as `ModelConfig()` in the notebook so that it can easily be tuned by users. Also, the training configurations will stay in one place as `TrainingConfig()`.
- Durable notebooks should live in one folder per notebook: `notebooks/<name>/<name>.py`. For example the notebook implemented for sampling from Gaussian Mixture Model should stay as `notebooks/gmm/gmm.py`.
- Each notebook will have one corresponding checkpoint which stays in the folder `chekpoints/`. Make sure that in each notebook, we can choose to start from scratch by setting `resume = False` and continue "exactly" where we left off the last time by setting `resume = True`. `resume` will be in the same place with `TrainingConfig()`.
- Each notebook will have one corresponding `.txt` logging file in folder `logging`. If we run the notebook from scratch (`resume = False`), then logging file will print `Attempt xxx`, where `xxx` is the number of attempts.
- Each notebook will have one corresponding folder in `figures`. In the folder will save all the plots, figures in the notebook.
- Avoid relying on hidden Jupyter state. A notebook should run from top to bottom after a kernel restart.

## Commit Hygiene

- Prefer committing the `.py` notebook source.
- Commit `.ipynb` files only when they are intentionally useful to humans and
  outputs are cleared or small.
