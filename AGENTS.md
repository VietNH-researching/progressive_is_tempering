# AGENTS.md

Project context for coding agents (Claude Code, Codex, Cursor, Copilot, etc.).
This is the single source of truth for agent and contributor guidance; see also
`CODEMAP.md` (structure) and `CONTRIBUTING.md` (workflow, test design).

## Project Summary

**Name**: `progressive_is_tempering`

**What it is**: A JAX research implementation of Sequential Score Matching via SMC Temperature Bridging, a sequential training algorithm for a score-based diffusion model targeting an unnormalized distribution with a tractable energy function. The key idea is to maintain a temperature ladder from an easy reference distribution to the target, iteratively generate training data at each temperature level via backward SDE sampling, correct for discretization error using importance weights and MALA rejuvenation, bridge particles to adjacent temperature levels, and jointly train a single β-conditioned score network across all levels.

**Read first**: `docs/TECHNIQUES_SUMMARY.md` for the full mathematical
description (temperature ladder construction + data generation + data correction + Neural Network training).

**Philosophy**:

- JIT-friendly throughout. All numerical code must trace cleanly under `jax.jit`.

**External dependencies**:

- `jax_pdf` is installed automatically from GitHub via `pyproject.toml`
  (`jax_pdf @ git+https://github.com/alexxthiery/jax-pdf.git`). Import it as a
  normal package; do not vendor or modify it.
- `mppt` is a **local package** bundled in this repo under `mppt/`. It has its
  own `pyproject.toml` and must be installed separately with
  `pip install -e ./mppt`. It is **not** fetched from GitHub.

## Install and dev commands

```bash
conda activate pist
pip install -e ./mppt         # installs local mppt package (bundled in repo)
pip install -e ".[test]"      # installs progressive_is_tempering + test extra

pytest                        # fast suite (default; excludes slow)
pytest -m slow                # end-to-end integration tests
pytest tests/test_sampler.py     # a single file
JAX_ENABLE_X64=1 pytest -m requires_x64   # precision tests
python -c "import progressive_is_tempering"        # import smoke check
```

The package is installed editable and imported as `progressive_is_tempering`; there are
no `sys.path` hacks.

## Project Structure

```
progressive_is_tempering/          # repo root
├── pyproject.toml                 # build config; pins mppt + jax_pdf from GitHub
├── AGENTS.md                      # this file
├── docs/
│   ├── TECHNIQUES_SUMMARY.md      # full algorithm math (read this first)
│   ├── IMPLEMENTATION_MLP.md      # MLP score-network details
│   └── IMPLEMENTATION_EGNN.md     # EGNN score-network details (DW4 target)
├── progressive_is_tempering/      # installable package
│   ├── __init__.py                # re-exports all public submodules
│   ├── sde.py                     # OU forward SDE: transition density + forward sampler
│   ├── targets.py                 # target distributions (GMMTarget, DW4Target)
│   ├── resampling.py              # systematic resampling + normalized ESS
│   ├── mala.py                    # MALA step + per-level step-size calibration
│   ├── sampler.py                 # backward SDE with path-space log-weights
│   ├── buffer.py                  # per-temperature FIFO replay buffers
│   ├── loss.py                    # DDPM regression loss across temperature levels
│   ├── net/
│   │   ├── __init__.py            # re-exports MLP, ResMLP, time_embedding, ScoreNetwork
│   │   ├── embedding.py           # sinusoidal time + beta embeddings
│   │   ├── mlp.py                 # MLP and ResMLP Flax modules
│   │   ├── score_network.py       # ScoreNetwork: β-conditioned excess-score model
│   │   └── egnn_jax.py            # EGNN backbone for equivariant (molecular) targets
│   └── utils/
│       └── __init__.py            # shared utilities (no imports from core package)
├── notebooks/
│   ├── gmm/                       # GMM experiment (gmm.py + gmm.ipynb)
│   ├── dw4/                       # DW4 molecule experiment (dw4.py + dw4.ipynb)
│   └── templates/minimal.py       # minimal training loop template
├── tests/                         # pytest suite (fast by default; -m slow for e2e)
│   ├── test_sde.py
│   ├── test_resampling.py
│   ├── test_mala.py
│   ├── test_sampler.py
│   ├── test_buffer.py
│   ├── test_loss.py
│   ├── test_targets.py
│   └── test_score_network.py
├── data/                          # reference data for sampling benchmarks
│   └── dw4_samples.npz            # ground-truth DW4 particle samples
├── figures/                       # saved plot outputs (gitignored binaries)
├── checkpoints/                   # saved model checkpoints (.pkl)
└── logging/                       # training log files (.txt)
```

## Module dependency layering

Arrows mean "imports from".

```
notebooks/*, tests/*
        │
        ▼
progressive_is_tempering.__init__   (re-exports everything below)
        │
        ├──► sampler  ──► sde
        ├──► buffer   ──► resampling
        ├──► mala
        ├──► loss
        ├──► targets
        ├──► resampling
        ├──► sde
        └──► net.score_network ──► net.embedding
                               └──► net.mlp
             net.egnn_jax      ──► net.embedding
```

Leaf modules (`sde`, `targets`, `resampling`, `mala`, `loss`, `net.embedding`,
`net.mlp`, `net.egnn_jax`) import only from JAX/Flax — never from each other or
from higher layers.

**Key rule**: `progressive_is_tempering.utils` never imports from the core package
(one-directional layering).

## Algorithm phases

**Warmup** (once, before any gradient steps)

1. **Temperature ladder** — call MPPT `smc_sampler` with `PTProblem(rvs_ref, logpdf_ref, logpdf_final_target)` to adaptively find `{β_k}` (local ESS ratio ≥ 0.5). Optionally follow with one round of `pt_manual_runners` to refine the particle cloud. Only the ladder values `betas` are kept; the SMC particles are discarded.

2. **Buffer initialization**
   - `buffer[0]`: fill with i.i.d. `N(0, I)` samples, log-weights `= 0` (uniform). Frozen for the entire run.
   - `buffer[1]`: IS-reweight `buffer[0]` samples to `β_1` using `ReplayBuffer.is_bridge_log_weights`. Frozen.
   - `buffer[2..M]`: start empty; filled during the outer loop.

3. **MALA step-size calibration** — for each level `k`, IS-resample `N(0, I)` particles toward `π_{β_k}`, then bisect on `h` (14 iterations, range `[1e-3, 2.0]`) until MALA acceptance ∈ `[60%, 80%]`. Stores one float `h_k` per level; passed as `mala_hs[1:-1]` to the outer vmap.

**Outer loop** (one iteration = one SMC data-generation pass)

All `M_active = M - 1` active levels (`k = 1 … M-1`) are processed in parallel inside a single `jax.vmap` call (`_data_gen_parallel`):

1. `backward_sde_batch` → `(x0_batch, log_w_path)` — K particles with path-space log-weights.
2. `systematic_resample(log_w_path, K)` → approximately i.i.d. draws from `π_{β_k}`.
3. `mala_rejuvenate(particles, β_k, h_k, n_mala_steps)` → diversified particles, acceptance rate.
4. `ReplayBuffer.is_bridge_log_weights(x0_rejuv, β_k, β_{k+1})` → IS log-weights to next level.

After `jax.vmap` returns, buffer writes are sequential Python mutations: `buffers[k+1].add(x0_rejuv, bridge_lw)`.

**Inner loop** (runs only once all buffers hold ≥ `batch_size` entries)

For each of `n_inner_steps`:

1. For every level `k`, draw a mini-batch from `buffer[k]` via `systematic_resample` on IS log-weights.
2. Sample `t ~ Unif[t_min, T]`; apply OU forward noising `x_t = e^{-t} x_0 + σ_t ε`.
3. Compute DDPM targets `φ = e^{-t} β (x_0 - ∇E(x_0))` under `stop_gradient`.
4. Concatenate all `(M+1) * B` triples across temperatures; one Adam step with global-norm gradient clipping.

**Evaluation / Sampling** (after training)

Run `backward_sde_batch` at `β=1` with `n_sde_steps_eval >> n_sde_steps_train` (e.g. 2000 vs 100). Resample by path-space weights. Use `normalized_ess(log_w)` as the primary quality metric (target: close to 1.0).

---

## Key patterns

**`jax.lax.scan` for loops inside JIT** — `backward_sde_single` scans over the `N` backward time steps; `mala_rejuvenate` scans over `n_steps` MALA steps per particle. Never use Python `for` loops inside a JIT region — they unroll at trace time.

**`jax.vmap` for particle and level parallelism** — `backward_sde_batch` vmaps `backward_sde_single` over `n_particles`; `mala_rejuvenate` vmaps `rejuvenate_one` over the `K` particles; the outer data generation vmaps `process_one` over temperature levels. These three vmap axes are entirely independent.

**`static_argnums` for shapes and callables** — anything that determines the XLA graph shape (`dim`, `n_particles`, `T`, `n_steps`, `n_mala_steps`) and any callable (`energy_fn`, `score_apply_fn`) must be in `static_argnums`. Values in `static_argnums` must be Python-level constants, not JAX arrays.

**Unbatched network interface** — `ScoreNetwork.__call__` takes `x: (d,)`, `t: scalar`, `beta: scalar`. Always call it through `jax.vmap` for batched inputs. This keeps the module itself simple and composable.

**Excess score, not full score** — the network learns `s_θ ≈ ∇ log(p_t^β / p_t^ρ)`. The target is exactly zero at `β=0`. `output_init_scale=1e-4` enforces near-zero predictions at initialization, giving the network a warm start aligned with this constraint.

**`stop_gradient` on DDPM targets** — `jax.lax.stop_gradient` wraps the `φ` computation in both `ddpm_loss_single_temp` and the notebook's `train_step`, preventing second-order gradients through `∇E(x_0)`.

**`ou_log_transition_density` is unnormalized** — it omits `-d/2 log(2π σ²)`. This is intentional: the same constant appears in `log_q` and cancels exactly in the per-step weight `log_p − log_q` inside `backward_sde_step`. Do not use it as a standalone normalized density.

---

## Gotchas

**Inner loop is gated on all buffers being ready** — `all_buffers_ready` blocks gradient updates until every buffer (including `buffer[M]`) holds ≥ `batch_size` samples. The first `M-2` outer steps generate data but produce no parameter updates. This is expected behavior, not a bug.

**Buffer writes are sequential, not vmapped** — `buffers[k+1].add(...)` is a Python-level mutation using `jnp.ndarray.at[indices].set(...)`. It happens after `_data_gen_parallel` returns, outside the vmap. Wrapping it inside a vmap will silently discard writes (JAX vmap does not support Python-level side effects).

**Buffer 0 log-weights are `0`, not `-inf`** — `buffer[0]` is seeded with `jnp.zeros(n_particles)` (uniform weights in log-space). Empty buffer slots use `-inf` to be skipped by `systematic_resample`. Confusing these two values will corrupt sampling from buffer 0.

**`systematic_resample` requires `n` to be a static Python int** — `@partial(jax.jit, static_argnums=(2,))` is applied. Passing a JAX scalar for `n` causes a tracing error.

**`time_embedding` requires even `embedding_dim`** — the output is split equally into `embedding_dim // 2` sin and cos features. An odd value silently truncates to `embedding_dim - 1` output features.

**Per-level MALA step sizes must be a JAX array when vmapped** — `_data_gen_parallel` receives `mala_hs_active: (M_active,)` as a traced JAX array and slices one scalar `mala_h_k` per level inside the vmap. Do not pass per-level step sizes as a Python list or tuple to vmapped code.

**vmap over temperature levels requires uniform `n_particles`** — the outer `jax.vmap` in `_data_gen_parallel` requires all levels to produce identically-shaped output arrays. You cannot use different particle counts per temperature without restructuring the outer loop.

**MPPT is a local package, not a GitHub download** — `mppt/` lives inside this repo and is installed with `pip install -e ./mppt`. The notebooks do `sys.path.insert(0, "../../mppt")` as a fallback for running without the editable install, but `pip install -e ./mppt` is the canonical approach and avoids the `sys.path` hack.

---

## Tech stack

JAX (jit/vmap/scan/pytrees), optax (flow training), jaxtyping (shape
annotations), mppt (ladder construction), pytest. Python >= 3.10.
