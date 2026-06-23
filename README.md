# progressive_is_tempering

A JAX implementation of **Sequential Score Matching via SMC Temperature Bridging** — a sequential training algorithm for a score-based diffusion model that targets an unnormalized distribution with a tractable energy function.

## What it does

The algorithm trains a single β-conditioned score network `s_θ(x, t, β)` to approximate samples from a target `π(x) ∝ exp(-E(x))`. Rather than attacking the target directly, it builds a temperature ladder

```
β₀ = 0  →  β₁  →  …  →  β_M = 1
```

that gradually anneals from an easy Gaussian reference to the hard target. At each outer training step a backward SDE generates particle data at every temperature level; those particles are importance-weighted, resampled, and MCMC-diversified before being fed into per-temperature replay buffers. An inner gradient-descent loop then trains the network jointly across all temperatures on a DDPM regression objective.

Full mathematical details are in [docs/TECHNIQUES_SUMMARY.md](docs/TECHNIQUES_SUMMARY.md).

---

## Installation

```bash
conda activate pist

# local mppt package (bundled in the repo under mppt/)
pip install -e ./mppt

# main package + test extras
pip install -e ".[test]"

# smoke check
python -c "import progressive_is_tempering"
```

**Dependencies installed automatically:** JAX, Flax, Optax, jaxtyping, and
[`jax_pdf`](https://github.com/alexxthiery/jax-pdf) (fetched from GitHub by pip).

---

## Experiments

Two ready-to-run experiments live under `notebooks/`:

| Experiment | Target | Score network | Notebook |
|---|---|---|---|
| GMM | 16-D factorized Gaussian mixture | MLP / ResMLP | `notebooks/gmm/gmm.ipynb` |
| DW4 | 4-particle double-well molecule | Equivariant EGNN | `notebooks/dw4/dw4.ipynb` |

Each notebook has a matching `.py` script (Jupytext percent format) for command-line runs.

---

## Running the tests

```bash
pytest                        # fast suite (default)
pytest -m slow                # end-to-end integration tests
pytest tests/test_sampler.py  # single file
JAX_ENABLE_X64=1 pytest -m requires_x64   # precision tests
```

---

## Project structure

```
progressive_is_tempering/      # installable package
├── sde.py                     # OU forward SDE
├── targets.py                 # GMMTarget, DW4Target
├── resampling.py              # systematic resampling + ESS
├── mala.py                    # MALA step + rejuvenation
├── sampler.py                 # backward SDE + path-space log-weights
├── buffer.py                  # per-temperature FIFO replay buffers
├── loss.py                    # DDPM regression loss
└── net/
    ├── embedding.py           # sinusoidal time + beta embeddings
    ├── mlp.py                 # MLP and ResMLP Flax modules
    ├── score_network.py       # ScoreNetwork: β-conditioned excess-score model
    └── egnn_jax.py            # EGNN backbone for molecular targets
mppt/                          # local SMC / parallel-tempering library
docs/
├── TECHNIQUES_SUMMARY.md      # full algorithm math
├── IMPLEMENTATION_MLP.md      # MLP score-network details
└── IMPLEMENTATION_EGNN.md     # EGNN score-network details
```

See [CODEMAP.md](CODEMAP.md) for a full structural reference and [CONTRIBUTING.md](CONTRIBUTING.md) for development guidance.

---

## Tech stack

JAX (jit / vmap / scan / pytrees) · Flax · Optax · jaxtyping · mppt · pytest
