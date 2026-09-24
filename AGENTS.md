# AGENTS.md

Project context for coding agents (Claude Code, Codex, Cursor, Copilot, etc.).
This is the single source of truth for agent and contributor guidance; see also
`CODEMAP.md` (structure) and `CONTRIBUTING.md` (workflow, test design).

## Project Summary

**Name**: `progressive_is_tempering`

**What it is**: A JAX research implementation of the Sequential Score Sampler (SSS), the method of the paper *Amortizing Tempered Sampling with Corrected Self-Training*: a sequential training algorithm for a score-based diffusion model targeting an unnormalized distribution with a tractable energy function. The key idea is to maintain a temperature ladder from an easy reference distribution to the target, iteratively generate training data at each temperature level via backward SDE sampling, correct for discretization error using importance weights and HMC rejuvenation, bridge particles to adjacent temperature levels, and jointly train a single β-conditioned score network across all levels.

**Read first**: `docs/TECHNIQUES_SUMMARY.md` for the full mathematical
description (temperature ladder construction + data generation + data correction + Neural Network training).

**Philosophy**:

- JIT-friendly throughout. All numerical code must trace cleanly under `jax.jit`.

**External dependencies**:

- None beyond the PyPI packages in `pyproject.toml`: every target energy
  (GMM, DW4, Lennard-Jones) is implemented in `targets.py`.
- The SMC ladder construction and the HMC kernel are implemented in-package
  (`smc.py`, `mala_kernel.py`); there is no other algorithmic dependency.
- `utils/metrics.py` needs SciPy and POT (both in `pyproject.toml`). The
  notebooks additionally need matplotlib, which is not a package dependency.

## Install and dev commands

```bash
conda create -n pist python=3.11 && conda activate pist
pip install -e ".[test]"      # installs progressive_is_tempering + test extra
pip install matplotlib        # needed by the notebooks only

pytest                        # full test suite
pytest tests/test_sampler.py  # a single file
python -c "import progressive_is_tempering"        # import smoke check

# one notebook run (run from the notebook's own folder: output paths are relative)
cd notebooks/gmm && python gmm.py

# several seeds, one after another
cd notebooks/gmm && for s in 0 1 2; do RUN_SEED=$s python gmm.py; done
```

A single notebook run with `RUN_SEED=N` set writes to `results/<target>/seedN/`;
without it, to `results/<target>/`.

The DW4 and LJ13 notebooks read `data/test_split_DW4.npy` and
`data/test_split_LJ13-1000.npy`, which are not tracked; fetch them with
`bash data/redownload.sh`.

The package is installed editable and imported as `progressive_is_tempering`; there are
no `sys.path` hacks.

## Project Structure

```
progressive_is_tempering/          # repo root
├── pyproject.toml                 # build config
├── AGENTS.md                      # this file
├── docs/
│   └── TECHNIQUES_SUMMARY.md      # full algorithm math (read this first)
├── progressive_is_tempering/      # installable package
│   ├── __init__.py                # re-exports targets, sde, resampling, sampler, mala, buffer, loss, score_network
│   ├── sde.py                     # OU forward SDE: transition density + forward sampler
│   ├── targets.py                 # GMMTarget + DW4 / LennardJones particle energies
│   ├── resampling.py              # systematic resampling + normalized ESS
│   ├── mala.py                    # isotropic MALA step + step-size calibration (not used by the notebooks)
│   ├── mala_kernel.py             # PTProblem + covariance-preconditioned HMC kernel on the tempered path
│   ├── smc.py                     # adaptive-temperature SMC sampler (ladder + per-rung kernel params)
│   ├── rejuvenate.py              # HMC rejuvenation with the SMC-adapted kernel
│   ├── sampler.py                 # backward SDE with path-space log-weights
│   ├── buffer.py                  # per-temperature FIFO replay buffers
│   ├── loss.py                    # Target Score Matching (TSM) loss across temperature levels
│   ├── net/
│   │   ├── __init__.py            # re-exports MLP, ResMLP, time_embedding, ScoreNetwork
│   │   ├── embedding.py           # sinusoidal time + beta embeddings
│   │   ├── mlp.py                 # MLP and ResMLP Flax modules
│   │   ├── score_network.py       # ScoreNetwork: β-conditioned excess-score model
│   │   └── egnn_jax.py            # EGNN backbone for equivariant (molecular) targets
│   └── utils/
│       ├── __init__.py            # shared utilities (no imports from core package)
│       ├── smc_diagnostics.py     # SMC seed-quality + ancestral-degeneracy diagnostics
│       ├── metrics.py             # reported metrics: Distance-W2, Energy-W2, Geometric-W2
│       └── report.py              # writes results/<target>/<run_id>/metrics.json
├── notebooks/
│   ├── gmm/                       # GMM experiment (gmm.py + gmm.ipynb)
│   ├── gmm50/                     # GMM at 50-D (gmm50.py + gmm50.ipynb)
│   ├── dw4/                       # DW4 molecule experiment (dw4.py + dw4.ipynb)
│   ├── lj13/                      # LJ13 cluster experiment (lj13.py + lj13.ipynb)
│   └── templates/minimal.py       # minimal training loop template
├── tests/                         # pytest suite
│   ├── test_sde.py
│   ├── test_resampling.py
│   ├── test_mala.py
│   ├── test_mala_kernel.py
│   ├── test_smc.py
│   ├── test_rejuvenate.py
│   ├── test_sampler.py
│   ├── test_buffer.py
│   ├── test_loss.py
│   ├── test_targets.py
│   ├── test_score_network.py
│   ├── test_smc_diagnostics.py
│   ├── test_metrics.py
│   └── test_report.py
├── scripts/
│   └── run_seeds.sh               # runs seeds 0, 1, 2 of every target (produced the reported results)
├── data/                          # reference data for sampling benchmarks
│   └── redownload.sh              # fetches the DEM test splits (DW4, LJ13, LJ55); none are tracked
└── results/                       # per target[/seedN]: checkpoint.pkl, log.txt, figures/ (gitignored), <run_id>/metrics.json (tracked)
```

## Module dependency layering

Arrows mean "imports from".

```
notebooks/*, tests/*
        │
        ▼
progressive_is_tempering.<module>   (notebooks import each module directly;
        │                            __init__ re-exports only a subset)
        ├──► sampler  ──► sde
        ├──► buffer   ──► resampling
        ├──► mala
        ├──► smc      ──► mala_kernel
        ├──► rejuvenate ──► mala_kernel
        ├──► loss
        ├──► targets
        ├──► resampling
        ├──► sde
        └──► net.score_network ──► net.embedding
                               └──► net.mlp
             net.egnn_jax      ──► net.embedding
```

Leaf modules (`sde`, `targets`, `resampling`, `mala`, `mala_kernel`, `loss`, `net.embedding`,
`net.mlp`, `net.egnn_jax`) import only from JAX/Flax — never from each other or
from higher layers.

**Key rule**: `progressive_is_tempering.utils` never imports from the core package
(one-directional layering).

## Algorithm phases

**Warmup** (once, before any gradient steps)

1. **Temperature ladder** — call `smc.smc_sampler` with `mala_kernel.PTProblem(rvs_ref, logpdf_ref, logpdf_final_target)` to adaptively find `{β_k}` (local ESS ratio ≥ 0.5). It returns the `(M+1, N, d)` stack of particle blocks (one per rung, indexed to match the ladder), the cumulative log Z estimates, the ladder `betas`, and the stacked per-rung `MALAParameters` the kernel adapted. No parallel-tempering step follows: the ladder comes from SMC.

   The HMC kernel does not project to the zero-CoM subspace. For molecular targets, project the SMC stack once at this boundary before any downstream use.

2. **Reuse SMC's kernel — no separate calibration** — the per-rung `MALAParameters` returned by `smc_sampler` (preconditioner = Cholesky factor of the rung's empirical covariance; step size `ε` bisected to acceptance ∈ `[0.4, 0.65]`) are reused as-is for outer-loop rejuvenation. `mcmc_params_by_rung` is a *stacked* pytree with leading axis `M` (entry `k-1` belongs to rung `k`); index its leaves, not the NamedTuple. Each kernel was adapted on a population actually distributed as `π_{β_k}`, so no further calibration is needed.

3. **Buffer initialization**
   - `buffer[0]`: fill to capacity with i.i.d. `N(0, I)` samples, log-weights `= 0` (uniform). Frozen for the entire run. Use exact draws, not the SMC `β=0` block — at `β_0 = 0` the reference is the target.
   - `buffer[1..M]`: seed with that rung's whole SMC block as SMC returned it (no further HMC pass). Uniform log-weights — these particles are already `≈ π_{β_k}`, so bridge weights do **not** apply. `buffer[1]` is frozen (the outer loop writes only `2..M`).
   - `buffer[2..M]`: additionally filled during the outer loop; FIFO retires the seed as new data arrives.

**Outer loop** (one iteration = one data-generation pass)

All `M_active = M - 1` active levels (`k = 1 … M-1`) are processed in parallel inside a single `jax.vmap` call (`_data_gen_parallel`, inner function `process_one`):

1. Backward SDE at `β_k` → `(x0_batch, log_w_path)` — K particles with path-space log-weights. GMM uses `sampler.backward_sde_batch`; DW4 / LJ13 use the notebook's `backward_sde_batch_com`, which projects to zero CoM after every step and calls the EGNN once per step on the whole batch.
2. `systematic_resample(key, log_w_path, K)` → approximately i.i.d. draws from `π_{β_k}`.
3. `rejuvenate(key, particles, pt_problem, β_k, k, mcmc_params_k, n_mala_rejuv[, project_fn])` → diversified particles, acceptance rate.
4. IS log-weights to the next level, `(β_{k+1} − β_k)(−E(x) + ½‖x‖²)` (inlined in `process_one`; same formula as `ReplayBuffer.is_bridge_log_weights`).

After `jax.vmap` returns, buffer writes are sequential Python mutations: `buffers[k+1].add(x0_rejuv, normalize_log_weights(bridge_lw))`.

**Inner loop** (runs only once all buffers hold ≥ `batch_size` entries)

For each of `n_inner_steps`:

1. For every level `k`, draw a mini-batch from `buffer[k]` via `systematic_resample` on IS log-weights.
2. Sample `t ~ Unif[t_min, T]`; apply OU forward noising `x_t = e^{-t} x_0 + σ_t ε`.
3. Compute TSM targets `φ = e^{-t} β (x_0 - ∇E(x_0))` under `stop_gradient`.
4. Concatenate all `(M+1) * B` triples across temperatures; one Adam step with global-norm gradient clipping.

**Evaluation / Sampling** (after training)

Run the backward SDE at `β=1` with `n_sde_steps_eval >> n_sde_steps_train` (e.g. 2000 vs 100) and record `normalized_ess(log_w)`. Three sample sets are scored against reference samples with `utils.metrics` (Distance-W2, Energy-W2, and Geometric-W2 for particle targets) and written by `utils.report.write_record` to `results/<target>/[seedN/]<run_id>/metrics.json`:

| `run_id` | Sample set |
|---|---|
| `sde_raw` | backward-SDE output, no reweighting |
| `sde_is` | the same particles, systematically resampled on `log_w_path` |
| `buffer` | `buffer[M]`, resampled on its stored weights |

---

## Key patterns

**`jax.lax` loops inside JIT** — `backward_sde_single` scans over the `N` backward time steps; `rejuvenate` runs its `n_steps` HMC moves with `lax.fori_loop` (via `repeated_batch_move`, or its own loop when `project_fn` is set). Never use Python `for` loops inside a JIT region — they unroll at trace time.

**`jax.vmap` for particle and level parallelism** — `backward_sde_batch` vmaps `backward_sde_single` over `n_particles`; the HMC kernel's `batch_move` vmaps one move over the `K` particles; the outer data generation vmaps `process_one` over temperature levels. These three vmap axes are entirely independent.

**`static_argnums` for shapes and callables** — anything that determines the XLA graph shape (`dim`, `n_particles`, `T`, `n_steps`, `n_mala_steps`), any callable (`energy_fn`, `score_apply_fn`, `project_fn`) and the `PTProblem` must be in `static_argnums`. Values in `static_argnums` must be Python-level constants (or hashable objects), not JAX arrays. `rejuvenate` must receive the *same* `PTProblem` object SMC ran with.

**Network interfaces** — `ScoreNetwork.__call__` is unbatched: `x: (d,)`, `t: scalar`, `beta: scalar`; call it through `jax.vmap` for batched inputs. `EGNNDynamics.__call__` is batched: `x: (K, d)`, `t, beta: (K, 1)`.

**Excess score, not full score** — the network learns `s_θ ≈ ∇ log(p_t^β / p_t^ρ)`. The target is exactly zero at `β=0`. `output_init_scale=1e-4` enforces near-zero predictions at initialization, giving the network a warm start aligned with this constraint.

**`stop_gradient` on TSM targets** — `jax.lax.stop_gradient` wraps the `φ` computation in both `ddpm_loss_single_temp` and the notebook's `train_step`, preventing second-order gradients through `∇E(x_0)`.

**`ou_log_transition_density` is unnormalized** — it omits `-d/2 log(2π σ²)`. This is intentional: the same constant appears in `log_q` and cancels exactly in the per-step weight `log_p − log_q` inside `backward_sde_step`. Do not use it as a standalone normalized density.

---

## Gotchas

**Inner loop is gated on all buffers being ready** — `all_buffers_ready` blocks gradient updates until every buffer (including `buffer[M]`) holds ≥ `batch_size` samples. With SMC seeding every buffer is above `batch_size` before the first outer step, so the gate passes immediately. A notebook whose buffers `2..M` started empty would skip parameter updates until the outer loop had filled them — expected behavior, not a bug.

**Buffer writes are sequential, not vmapped** — `buffers[k+1].add(...)` is a Python-level mutation using `jnp.ndarray.at[indices].set(...)`. It happens after `_data_gen_parallel` returns, outside the vmap. Wrapping it inside a vmap will silently discard writes (JAX vmap does not support Python-level side effects).

**Buffer 0 log-weights are `0`, not `-inf`** — `buffer[0]` is seeded with `jnp.zeros(n_particles)` (uniform weights in log-space). Empty buffer slots use `-inf` to be skipped by `systematic_resample`. Confusing these two values will corrupt sampling from buffer 0.

**Log-weight blocks must be normalized before `.add`** — a buffer accumulates blocks written at different times (the SMC seed, then one bridge block per outer step) and `.sample()` resamples across all of them *jointly*, so the blocks' weight scales must be commensurable. Raw bridge weights `Δβ(-E + ½‖x‖²)` can run O(10–100) on molecular targets while seed particles carry `0`; mixing them raw lets one block capture the entire buffer regardless of particle count. Normalize every block to mean weight 1 (`lw - logsumexp(lw) + log n`) on the way in. This is a constant shift, so it leaves each block's own ESS unchanged — it only fixes the *relative* weighting between blocks.

**Each notebook is self-contained.** A notebook implements the method for one target and must not import from, read files written by, or refer to another notebook. Shared algorithm code lives in `progressive_is_tempering/`; everything target-specific (energy, projections, diagnostics, plots, config values) stays in that notebook. Start new notebooks from `notebooks/templates/minimal.py`.

**SMC seeding** — seed each buffer with its rung's **whole** SMC block. `n_smc_particles` is a single knob serving three roles — SMC population, per-rung seed size, and buffer capacity — exposed as the derived properties `buffer_capacity` and `n_seed_particles`, which are *not* dataclass fields. They are the same particles at three points in their life, so nothing is subsampled away. Translation-invariant targets project the SMC stack to zero CoM once, before any downstream use; targets without that symmetry use the coordinates as-is.

**The seed drains over `n_smc_particles / n_particles` outer steps** (`seed_flush_steps`, printed at config time). Each buffer starts exactly full, so FIFO overwrites the oldest `K` rows in place each step. Keep that count comfortably below `num_outer_steps` or the run ends while still training partly on SMC particles — which the seed-quality diagnostic can show carry far fewer independent lineages than their row count suggests. Raise `num_outer_steps` before raising `n_smc_particles`.

**`systematic_resample` requires `n` to be a static Python int** — `@partial(jax.jit, static_argnums=(2,))` is applied. Passing a JAX scalar for `n` causes a tracing error.

**`time_embedding` requires even `embedding_dim`** — the output is split equally into `embedding_dim // 2` sin and cos features. An odd value silently truncates to `embedding_dim - 1` output features.

**Per-level kernel parameters are vmapped as a stacked pytree** — `_data_gen_parallel` receives `mcmc_params_active` (the SMC `MALAParameters` with every leaf sliced to its first `M_active` entries) and `t_idx_active = arange(1, M)`, and vmaps over their leading axis. Do not pass per-level parameters as a Python list or tuple to vmapped code, and never pass `t_idx = 0` to `rejuvenate`: the kernel treats rung 0 as "draw fresh from the reference" and ignores its input.

**vmap over temperature levels requires uniform `n_particles`** — the outer `jax.vmap` in `_data_gen_parallel` requires all levels to produce identically-shaped output arrays. You cannot use different particle counts per temperature without restructuring the outer loop.

**`smc.py` / `mala_kernel.py` are order-sensitive** — keep arithmetic expression order, key splits, and `lax.cond` structure as they are when editing. Seemingly equivalent rewrites (merging the two half velocity steps, dropping the `t_idx == 0` branch in `move`, reusing `resampling.systematic_resample` in SMC) change the floating-point results or the random stream, so the same seed gives different results.

---

## Tech stack

JAX (jit/vmap/scan/pytrees), Flax (networks), Optax (Adam + gradient
clipping), SciPy + POT (metrics), pytest. Python >= 3.10.
