# Code map

## Package

| Module | Contents |
|---|---|
| `sde.py` | OU forward process: transition density and forward sampling |
| `sampler.py` | Backward SDE (exponential integrator) with path-space log-weights; time grids |
| `targets.py` | `GMMTarget`, `DW4`, `LennardJones` energies |
| `alanine.py` | Alanine dipeptide: force field, internal and sampling coordinates, target |
| `resampling.py` | Systematic resampling, normalized ESS |
| `mala_kernel.py` | `PTProblem`; covariance-preconditioned HMC kernel, full velocity refresh (`"mala"`) |
| `hmc_kernel.py` | The same kernel with partial velocity refresh (`"hmc"`) |
| `mala.py` | Standalone isotropic MALA (not used by the notebooks) |
| `smc.py` | Adaptive-temperature SMC: ladder, per-rung particles and kernel parameters |
| `periodic_hmc.py` | HMC with circular coordinates and its SMC (alanine dipeptide) |
| `rejuvenate.py` | MCMC rejuvenation with the SMC-tuned kernel, optional projection |
| `buffer.py` | Per-temperature FIFO replay buffers with log-weights |
| `loss.py` | Target Score Matching loss |
| `net/score_network.py` | `ScoreNetwork`: β-conditioned ResMLP (unbatched) |
| `net/egnn_jax.py` | `EGNNDynamics`: equivariant network for particle systems (batched) |
| `net/dit.py` | `DIT3D`: transformer for alanine dipeptide |
| `net/mlp.py`, `net/embedding.py` | MLP blocks; sinusoidal time and β embeddings |
| `utils/metrics.py` | Distance-, Energy-, Geometric-W2; Ramachandran KL, torus-W2 |
| `utils/smc_diagnostics.py` | SMC seed-quality diagnostics and plot styling |
| `utils/report.py` | Writes `metrics.json` records |

## Dependencies

Arrows mean "imports from":

```
notebooks/*, tests/*
    └──► progressive_is_tempering.<module>
            sampler     ──► sde
            buffer      ──► resampling
            smc         ──► mala_kernel, hmc_kernel
            rejuvenate  ──► mala_kernel, hmc_kernel
            hmc_kernel  ──► mala_kernel
            net.score_network ──► net.embedding, net.mlp
            net.egnn_jax      ──► net.embedding
```

The other core modules import no other package module. `utils` never imports
from the core package.

## One outer step (rung k)

```
backward SDE at β_k ──► systematic resample ──► rejuvenate ──► bridge weights to β_{k+1}
                                                                   │
                                          buffer[k+1].add(x, normalized log w, ∇E(x))
```

Rungs `k = 1 … M-1` run in parallel with `jax.vmap`. Buffer writes happen after
the vmap returns.
