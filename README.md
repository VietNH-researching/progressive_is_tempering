# progressive_is_tempering

Code for *Amortizing Tempered Sampling with Corrected Self-Training*: a JAX
implementation of the **Sequential Score Sampler (SSS)**. SSS is a
score-based diffusion sampler for unnormalized densities
$\pi(x)\propto e^{-E(x)}$.

1. **Initialization.** An adaptive SMC sweep builds a temperature ladder from
   $\mathcal N(0,I)$ to $\pi$. Its particles seed one replay buffer per rung,
   and its tuned MCMC kernels are reused later.
2. **Training.** At every rung, a backward SDE driven by the current network
   generates samples. They are corrected by path-space importance weights and
   MCMC rejuvenation, then bridged to the next rung's buffer. One
   $\beta$-conditioned network is trained on all buffers jointly with Target
   Score Matching.
3. **Sampling.** The backward SDE at $\beta=1$, optionally reweighted.

The method is described in [docs/TECHNIQUES_SUMMARY.md](docs/TECHNIQUES_SUMMARY.md).

## Installation

```bash
conda create -n pist python=3.11 && conda activate pist
pip install -e ".[test]"
pip install matplotlib        # for the notebooks
```

Install the JAX build that matches your accelerator; see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Experiments

| Target | Notebook | Score network |
|---|---|---|
| GMM, 16-D | `notebooks/gmm/` | ResMLP |
| GMM, 50-D | `notebooks/gmm50/` | ResMLP |
| DW4 | `notebooks/dw4/` | EGNN |
| LJ13 | `notebooks/lj13/` | EGNN |
| LJ55 | `notebooks/lj55/` | EGNN |
| Alanine dipeptide (ff14SB + OBC1, 300 K) | `notebooks/aldp/` | DiT |

Each notebook is self-contained. Its `.py` file (Jupytext percent format) is
the source, and the `.ipynb` is paired with it. Hyperparameters are in
`ModelConfig` and `TrainingConfig`.

**Reference data.** DW4, LJ13 and LJ55 use the DEM test splits. Alanine
dipeptide uses PITA's 300 K MD splits; PITA's force field,
`data/aldp_ff14sb_obc1.npz`, is included. Download the splits with:

```bash
bash data/redownload.sh
```

**Running.** Run a script from its own folder:

```bash
cd notebooks/gmm
python gmm.py                 # writes to results/gmm/
RUN_SEED=1 python gmm.py      # writes to results/gmm/seed1/
```

`scripts/run_seeds.sh` runs seeds 0, 1 and 2 of every target.

A run writes `checkpoint.pkl`, `log.txt`, `figures/`, and one
`<run_id>/metrics.json` for each sample set:

- `sde_raw`: backward-SDE output;
- `sde_is`: the same output, importance-resampled;
- `buffer`: buffer $M$.

The metrics are Distance-W2 and Energy-W2, plus Geometric-W2 for particle
targets. Alanine dipeptide reports Ramachandran KL, Energy-W1/W2 and torus-W2
instead, plus Distance-W2. The reported results are in
`results/<target>/seed{0,1,2}/`.

The experiment notebooks enable deterministic GPU ops, so a seed gives the same
run on the same GPU model and software versions.

## Tests

```bash
pytest
```

## Layout

```
progressive_is_tempering/   package: SDE, SMC, MCMC kernels, buffers, loss, targets, networks
notebooks/                  one folder per experiment, plus templates/minimal.py
docs/                       method summary
scripts/run_seeds.sh        seeds 0-2 of every target
data/                       download script and the ALDP force field
results/                    metrics.json records
tests/                      pytest suite
```

See [CODEMAP.md](CODEMAP.md) for the module map and
[CONTRIBUTING.md](CONTRIBUTING.md) for development notes.
