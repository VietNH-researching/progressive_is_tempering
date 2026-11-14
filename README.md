# progressive_is_tempering

Code for the paper *Amortizing Tempered Sampling with Corrected Self-Training*: a JAX implementation of the **Sequential Score Sampler (SSS)**, a sequential training algorithm for a score-based diffusion model that targets an unnormalized distribution with a tractable energy function.

## What it does

The algorithm trains a single β-conditioned score network `s_θ(x, t, β)` to sample from a target `π(x) ∝ exp(-E(x))`. Rather than attacking the target directly, it works along the geometric path between a standard Gaussian reference and the target, discretized as a temperature ladder

```
β₀ = 0  →  β₁  →  …  →  β_M = 1
```

1. **Initialization.** An adaptive SMC sweep lays out the ladder, and returns a particle population and a tuned HMC kernel at every rung. The particles seed one replay buffer per rung.
2. **Training.** Each outer step runs, for every intermediate rung in parallel: a backward SDE driven by the current network, systematic resampling on path-space importance weights, HMC rejuvenation with the rung's SMC kernel, and an importance-sampling bridge to the next rung, whose buffer receives the particles. An inner loop then trains the network jointly across all rungs with a Target Score Matching (TSM) loss.
3. **Sampling.** The backward SDE at `β = 1`, optionally corrected by its path-space importance weights.

Full mathematical details are in [docs/TECHNIQUES_SUMMARY.md](docs/TECHNIQUES_SUMMARY.md).

---

## Installation

```bash
conda create -n pist python=3.11
conda activate pist

# main package + test extras
pip install -e ".[test]"

# the notebooks also need matplotlib
pip install matplotlib

# smoke check
python -c "import progressive_is_tempering"
```

**Dependencies installed automatically:** JAX, Flax, Optax, SciPy and POT. All
target energies are implemented in the package: GMM, DW4, LJ13 and LJ55 in
`progressive_is_tempering/targets.py`, alanine dipeptide (force field, internal
coordinates) in `progressive_is_tempering/alanine.py`.
Install the JAX build that matches your accelerator (see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)).

---

## Experiments

Six experiments live under `notebooks/`, one self-contained notebook per target:

| Experiment | Target | Score network | Notebook |
|---|---|---|---|
| GMM | 16-D factorized Gaussian mixture | ResMLP | `notebooks/gmm/gmm.ipynb` |
| GMM50 | the same mixture at 50-D | ResMLP | `notebooks/gmm50/gmm50.ipynb` |
| DW4 | 4-particle double-well molecule | Equivariant EGNN | `notebooks/dw4/dw4.ipynb` |
| LJ13 | 13-atom Lennard-Jones cluster | Equivariant EGNN | `notebooks/lj13/lj13.ipynb` |
| LJ55 | 55-atom Lennard-Jones cluster | Equivariant EGNN | `notebooks/lj55/lj55.ipynb` |
| ALDP | Alanine dipeptide, AMBER ff14SB + OBC1 implicit solvent, 300 K (PITA's target) | DiT (PITA's DIT3D) | `notebooks/aldp/aldp.ipynb` |

Each notebook has a matching `.py` script (Jupytext percent format), which is the canonical source. Hyperparameters live in the `ModelConfig` and `TrainingConfig` dataclasses near the top of each script.

**Reference data.** DW4, LJ13 and LJ55 are evaluated against the standard test splits from the DEM benchmark, ALDP against PITA's 300 K molecular-dynamics test split (its validation split gives the noise floor). None of the splits is tracked; fetch them before running those notebooks (PITA's ALDP force field, `data/aldp_ff14sb_obc1.npz`, is tracked):

```bash
bash data/redownload.sh
```

**Running one experiment.** Run a script from its own folder (output paths are relative):

```bash
cd notebooks/gmm
python gmm.py                # writes to results/gmm/
RUN_SEED=1 python gmm.py     # writes to results/gmm/seed1/
```

Each run writes `checkpoint.pkl`, `log.txt` and `figures/` to its results folder, plus one `metrics.json` for each of three evaluated sample sets: `sde_raw` (backward SDE, no reweighting), `sde_is` (the same samples, importance-resampled) and `buffer` (the training buffer at `β = 1`). The metrics are Distance-W2, Energy-W2 and, for DW4 / LJ13 / LJ55, Geometric-W2; ALDP reports the metrics of PITA's Table 2 instead — Rama-KL, Energy-W1 / W2 (in kT) and T-W2 — plus Distance-W2 (see `progressive_is_tempering/utils/metrics.py`). Set `resume = True` in the script to continue from the checkpoint.

**Several seeds.** Run the script once per seed:

```bash
cd notebooks/gmm
for s in 0 1 2; do RUN_SEED=$s python gmm.py; done
```

The reported results use seeds 0, 1 and 2 of every target, run in turn by `scripts/run_seeds.sh` (LJ55's and ALDP's with the same settings by the study scripts their notebooks were ported from; about 16 h of GPU time per seed for LJ55, 6 h for ALDP).

**Reproducibility.** All six scripts (GMM16, GMM50, DW4, LJ13, LJ55, ALDP) turn on deterministic GPU kernels (`--xla_gpu_deterministic_ops=true`), so `RUN_SEED=0`, `1` or `2` reproduces the committed `results/{gmm,gmm50,dw4}/seedN/*/metrics.json` exactly — on the same GPU model and software versions (NVIDIA RTX 4500 Ada, JAX 0.9.0.1). Other GPUs or JAX/CUDA versions can differ in the last bits, which SMC's adaptive temperature ladder amplifies into visibly different numbers.

---

## Running the tests

```bash
pytest                        # full suite
pytest tests/test_sampler.py  # single file
```

---

## Project structure

```
progressive_is_tempering/      # installable package
├── sde.py                     # OU forward SDE
├── targets.py                 # GMMTarget, DW4, LennardJones
├── alanine.py                 # ALDP: force field, internal + sampling coordinates, target
├── periodic_hmc.py            # HMC with radian circles + its adaptive SMC (the ALDP notebook's)
├── resampling.py              # systematic resampling + ESS
├── mala.py                    # isotropic MALA (standalone; not used by the notebooks)
├── mala_kernel.py             # PTProblem + covariance-preconditioned HMC kernel (full refresh)
├── hmc_kernel.py              # the same kernel with partial refresh (used by every notebook)
├── smc.py                     # adaptive-temperature SMC (ladder construction)
├── rejuvenate.py              # HMC rejuvenation with the SMC-adapted kernel
├── sampler.py                 # backward SDE + path-space log-weights
├── buffer.py                  # per-temperature FIFO replay buffers
├── loss.py                    # Target Score Matching (TSM) loss
├── net/
│   ├── embedding.py           # sinusoidal time + beta embeddings
│   ├── mlp.py                 # MLP and ResMLP Flax modules
│   ├── score_network.py       # ScoreNetwork: β-conditioned excess-score model
│   ├── egnn_jax.py            # EGNN backbone for molecular targets
│   └── dit.py                 # PITA's DIT3D transformer (ALDP)
└── utils/
    ├── smc_diagnostics.py     # SMC seed-quality diagnostics
    ├── metrics.py             # Distance-W2, Energy-W2, Geometric-W2; Rama-KL, Energy-W1/W2, T-W2 (ALDP)
    └── report.py              # writes metrics.json records
docs/
└── TECHNIQUES_SUMMARY.md      # full algorithm math
notebooks/                     # one folder per experiment + templates/minimal.py
scripts/run_seeds.sh           # runs seeds 0, 1, 2 of every target
data/                          # reference test splits + redownload.sh; ALDP force field
tests/                         # pytest suite
```

See [CODEMAP.md](CODEMAP.md) for a full structural reference and [CONTRIBUTING.md](CONTRIBUTING.md) for development guidance.

---

## Tech stack

JAX (jit / vmap / scan / pytrees) · Flax · Optax · SciPy · POT · pytest
