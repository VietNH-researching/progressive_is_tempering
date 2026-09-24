# CODEMAP.md — Structural overview

## Repository layout

```
progressive_is_tempering/          # repo root
├── pyproject.toml                 # build config
├── AGENTS.md                      # agent / contributor guidance
├── CODEMAP.md                     # this file
├── CONTRIBUTING.md                # dev workflow and test guidance
├── README.md                      # project overview
│
├── docs/
│   └── TECHNIQUES_SUMMARY.md      # full algorithm math (read first)
│
├── progressive_is_tempering/      # installable package
│   ├── __init__.py
│   ├── sde.py
│   ├── targets.py
│   ├── resampling.py
│   ├── mala.py
│   ├── mala_kernel.py
│   ├── smc.py
│   ├── rejuvenate.py
│   ├── sampler.py
│   ├── buffer.py
│   ├── loss.py
│   ├── net/
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   ├── mlp.py
│   │   ├── score_network.py
│   │   └── egnn_jax.py
│   └── utils/
│       ├── __init__.py
│       ├── smc_diagnostics.py     # SMC seed-quality / degeneracy diagnostics
│       ├── metrics.py             # reported metrics: Distance-W2, Energy-W2, Geometric-W2
│       └── report.py              # writes results/<target>/<run_id>/metrics.json
│
├── notebooks/
│   ├── gmm/                       # GMM experiment
│   │   ├── gmm.py                 # Jupytext percent script
│   │   └── gmm.ipynb
│   ├── gmm50/                     # GMM experiment at 50-D
│   │   ├── gmm50.py
│   │   └── gmm50.ipynb
│   ├── dw4/                       # DW4 molecule experiment
│   │   ├── dw4.py
│   │   └── dw4.ipynb
│   ├── lj13/                      # LJ13 cluster experiment
│   │   ├── lj13.py
│   │   └── lj13.ipynb
│   └── templates/minimal.py       # minimal training loop template
│
├── tests/                         # pytest suite
├── scripts/
│   └── run_seeds.sh               # runs seeds 0, 1, 2 of every target (produced the reported results)
├── data/                          # reference data for sampling benchmarks
│   └── redownload.sh              # fetches the DEM test splits (DW4, LJ13, LJ55); none are tracked
└── results/                       # per target[/seedN]: checkpoint.pkl, log.txt, figures/ (gitignored), <run_id>/metrics.json (tracked)
```

---

## Module dependency layering

Arrows mean "imports from". Leaf modules import only from JAX/Flax.

```
notebooks/*, tests/*
        │
        ▼
progressive_is_tempering.<module>   (notebooks import each module directly;
        │                            __init__ re-exports only a subset)
        ├──► sampler      ──► sde
        ├──► buffer       ──► resampling
        ├──► mala
        ├──► smc          ──► mala_kernel
        ├──► rejuvenate   ──► mala_kernel
        ├──► loss
        ├──► targets
        ├──► resampling
        ├──► sde
        └──► net.score_network ──► net.embedding
                               └──► net.mlp
             net.egnn_jax      ──► net.embedding
```

`progressive_is_tempering.utils` never imports from the core package (strictly one-directional).

---

## File-by-file reference

### `sde.py`

OU forward (noising) process.

| Symbol | Description |
|---|---|
| `ou_mean(x0, t)` | `e^{-t} x₀` |
| `ou_std(t)` | `sqrt(1 - e^{-2t})` |
| `ou_forward_sample(key, x0, t)` | draw `x_t ~ N(e^{-t} x₀, σ_t² I)` |
| `ou_log_transition_density(x_n, x_prev, dt)` | unnormalized log `p_ρ(x_n | x_{n-1})` |
| `reference_ou_score(x)` | `∇ log N(0,I) = -x` |

**Note:** `ou_log_transition_density` omits the `-d/2 log(2πσ²)` constant intentionally — it cancels in `log p - log q` inside the backward SDE.

---

### `targets.py`

Unnormalized target distributions. A target exposes `energy(x)`, `log_prob(x)`, `grad_energy(x)`, and the `PTProblem` interface (`logpdf_ref`, `logpdf_final_target`, `rvs_ref`).

| Symbol | Description |
|---|---|
| `GMMTarget` | Factorized GMM over ℝᵈ; each coordinate iid from the same 1-D mixture |
| `make_gmm_target(dim, means, weights, std)` | Constructs default GMM (d=16, μ=(-1.5, 2.5), w=(0.2, 0.8), σ=0.5) |
| `DW4(a, b, c, r0, trap_scale, trap_mode, beta)` | Double-well pair potential over 4 particles in 2D; callable, `x: (..., 4, 2)` → unnormalized log-density |
| `LennardJones(n_particles, spatial_dim, epsilon, rm, trap_scale, trap_mode, beta)` | Lennard-Jones 12-6 cluster with harmonic trap; callable, `x: (..., n, s)` → unnormalized log-density |

`DW4` and `LennardJones` are bare energies, not `GMMTarget`-style targets: each of those notebooks wraps one into `energy(x)` on flat inputs and defines the zero-CoM reference and projection itself.

---

### `resampling.py`

| Symbol | Description |
|---|---|
| `systematic_resample(key, log_weights, n)` | Draw `n` indices; JIT-compiled; `n` must be a static Python int |
| `normalized_ess(log_weights)` | Normalized ESS ∈ (0, 1]; 1 = uniform weights, 1/N = degenerate |

---

### `mala.py`

Standalone isotropic MALA (fixed scalar step size `h`, no preconditioner).

| Symbol | Description |
|---|---|
| `mala_step(key, x, beta, energy_fn, h)` | One MALA step with MH correction; returns `(x_new, accepted)` |
| `mala_rejuvenate(key, particles, beta, energy_fn, h, n_steps)` | Apply `n_steps` MALA to all `K` particles via vmap+scan; returns `(particles_new, mean_accept_rate)` |

`mala_rejuvenate` is JIT-compiled with `static_argnums=(3, 5)` (energy_fn, n_steps).
The notebooks do not call it; they use `rejuvenate.rejuvenate` below.

---

### `mala_kernel.py`

Covariance-preconditioned MALA kernel on the geometric path
`log π_t = (1-t)·logpdf_ref + t·logpdf_final_target`, written as one-step HMC with
full velocity refresh.

| Symbol | Description |
|---|---|
| `PTProblem(rvs_ref, logpdf_ref, logpdf_final_target)` | The tempering problem; passed as a static argument |
| `mala_kernel(problem, t, t_idx)` | `MALAKernel` with `init_extend`, `move`, `adapt_params`, `reset_temperature` for one rung |
| `MALAParameters(scaled_cov, refresh_rate)` | Adapted kernel parameters; `scaled_cov.eps` is the step size, `scaled_cov.chol` the preconditioner |
| `batch_init_extend`, `batch_move`, `repeated_batch_move`, `batch_reset_temperature` | vmapped helpers over a particle population |

`move` at `t_idx == 0` redraws from the reference and ignores its input.

---

### `smc.py`

| Symbol | Description |
|---|---|
| `smc_sampler(key, problem, n_particles, n_mcmc_steps, local_ess_ratio, verbose=True)` | Adaptive ladder. Returns `(particles (M+1, N, d), logZ dict, temperatures list, stacked MALAParameters (M,))` |
| `one_step_smc(key, x, problem, niter_mcmc, new_ess_ratio)` | One step: ESS-bisected next temperature → permuted systematic resampling → kernel adaptation → `niter_mcmc` moves → local log Z (SMC-like, robust, reverse-logistic) |

---

### `rejuvenate.py`

| Symbol | Description |
|---|---|
| `rejuvenate(key, positions, problem, beta_k, t_idx_k, mcmc_params_k, n_steps, project_fn=None)` | `n_steps` MALA moves with the SMC-adapted kernel for rung `k`; returns `(positions_new, mean_accept_rate)`. `project_fn` (e.g. zero-CoM) is applied after every move. JIT-compiled with `static_argnums=(2, 6, 7)` |

---

### `sampler.py`

Backward SDE with path-space importance weights.

| Symbol | Description |
|---|---|
| `backward_sde_step(key, x_n, t_n, dt, control)` | One exponential-integrator step; returns `(x_prev, log_step_weight, eps)` |
| `backward_sde_single(key, control_fn, beta, energy_fn, dim, T, n_steps)` | Single-particle trajectory via `lax.scan`; returns `(x0, log_w)` |
| `backward_sde_batch(key, control_fn, beta, energy_fn, dim, n_particles, T, n_steps)` | `n_particles` independent trajectories via vmap; returns `(x0_batch, log_w_batch)` |

Path-space log-weight formula:

```
log w = Σ_n [log p_ρ(x_n|x_{n-1}) - log q_θ(x_{n-1}|x_n)]
        - (1-β) ½‖x₀‖² - β E(x₀) + ½‖x_T‖²
```

---

### `buffer.py`

| Symbol | Description |
|---|---|
| `ReplayBuffer(capacity, dim)` | Fixed-capacity FIFO buffer with IS-weighted sampling |
| `.add(samples, log_weights)` | Append with circular FIFO eviction |
| `.sample(key, batch_size)` | Systematic resample by IS weights; returns `(samples, log_weights)` |
| `.is_ready(min_size)` | True when ≥ `min_size` entries are valid |
| `.is_bridge_log_weights(samples, energy_fn, beta_k, beta_next)` | `Δβ · (-E(x) + ½‖x‖²)`; static method |

Buffer 0 uses `log_weights = 0` (uniform); empty slots use `-inf` (skipped by resampling).

---

### `loss.py`

Target Score Matching (TSM) loss for the β-conditioned excess score network: the regression target is built from the target score `-∇E` at the clean sample `x₀`.

| Symbol | Description |
|---|---|
| `compute_ddpm_target(x0, t, beta, energy_fn, max_norm)` | `φ = e^{-t} β (x₀ - ∇E(x₀))` with optional norm clipping |
| `ddpm_loss_single_temp(params, score_apply_fn, x0_batch, t_batch, beta, energy_fn, key, ...)` | MSE at one temperature level; applies OU forward noising internally |
| `ddpm_loss(params, score_apply_fn, x0_per_temp, t_per_temp, betas, energy_fn, key, ...)` | Mean loss across all `M+1` levels |

Targets are wrapped in `jax.lax.stop_gradient` — no backprop through `∇E(x₀)`.

---

### `net/embedding.py`

| Symbol | Description |
|---|---|
| `time_embedding(t, freq_min, freq_max, embedding_dim)` | Sinusoidal embedding; `embedding_dim` must be even |

---

### `net/mlp.py`

| Symbol | Description |
|---|---|
| `MLP(dim_list)` | Simple feedforward network with ELU activations |
| `ResMLP(dim_output, hidden_dim, n_blocks, output_init_scale)` | Residual MLP with RMSNorm; small output init for training stability |
| `RMSNorm` | Root-mean-square normalization layer |
| `ResBlock(hidden_dim)` | Pre-norm residual block: `x + MLP(RMSNorm(x))` |

---

### `net/score_network.py`

| Symbol | Description |
|---|---|
| `ScoreNetwork` | Flax module: `s_θ(x, t, β) → (d,)` excess score, single-sample interface |

Architecture: `concat([x, time_embedding(t), time_embedding(β)])` → `ResMLP`. Call through `jax.vmap` for batched inputs.

---

### `net/egnn_jax.py`

Equivariant Graph Neural Network backbone for particle-system targets. Used in place of `ScoreNetwork` when SE(3) equivariance is required.

---

### `utils/smc_diagnostics.py`

SMC seed-quality diagnostics, shared by all four notebooks. Catches two failure modes of the top-rung SMC block, which are not the same thing.

| Symbol | Description |
|---|---|
| `smc_seed_quality_figure(...)` | Standard four-panel figure; returns `(fig, stats)` |
| `print_seed_quality_report(stats)` | Prints the numbers behind the figure |
| `exchangeable_degeneracy(features_test, features_ref, threshold)` | Implied independent-particle count from the spread of per-feature exceedance probabilities |
| `plot_value_comparison(ax, v_test, v_ref, ...)` | Test-vs-reference density on shared bin edges |
| `reference_anchored_window`, `honest_density_hist` | Window that always shows the reference in full; step histogram whose area is the in-window sample fraction |
| `style_axis`, `C_TEST`, `C_REF`, `C_INK`, `C_INK_2` | Shared styling; colour follows the entity, not the panel |

**Distributional mismatch** is caught by the pooled-statistic and energy panels. **Ancestral degeneracy** — a population whose histograms match perfectly while carrying far less independent information than its particle count suggests — is caught by the per-feature panels. A distinct-row count reports 100% unique and misses it entirely, because local MCMC erases literal duplicates while the basin assignment stays frozen.

The degeneracy test needs **exchangeable scalar features**: the `d` coordinates for a factorised target (i.i.d. across coordinates), the `C(n,2)` pairwise distances for an `n`-particle system (particle-permutation symmetry). The reference set is passed through the identical computation as a control — being i.i.d., its implied count should come back near its own size. **Read `n_eff_ref` before trusting `n_eff_test`:** the estimator is noisy when there are few features (a 4-particle system has only 6).

Only NumPy is needed for the statistics; matplotlib and scipy are imported lazily, so importing the module never requires them.

---

### `utils/metrics.py`

The three metrics reported for the method. Each notebook's final evaluation scores its three sample sets (SDE raw, SDE + IS, `buffer[M]`) with them.

| Symbol | Description |
|---|---|
| `distance_w2(test_stats, ref_stats)` | Distance-W2: 1-D W2 between the *pooled* summary statistics (pairwise distances; the coordinates for GMM) |
| `energy_w2(test_e, ref_e)` | Energy-W2: 1-D W2 between the energy distributions |
| `geometric_w2(x_test, x_ref, n_particles, spatial_dim, max_n, seed, n_workers)` | Geometric-W2 (particle targets only): exact OT with ground cost "assignment on unaligned coordinates, then Procrustes over O(s)", squared, outer square root |
| `wasserstein2_1d(a, b)` | Sorting-based 1-D W2; exact at equal counts, 1000 midpoint quantiles otherwise; drops non-finite entries |
| `geometric_d2_matrix(...)` | The `(n, m)` squared ground-cost matrix behind `geometric_w2` |

Geometric-W2 needs `n × m` alignments (10⁸ at 10k × 10k), so `geometric_w2` runs them on a spawn-context process pool inside a **fresh subprocess**: spawn workers re-import the parent's `__main__`, which for a notebook run as a script would re-execute the whole notebook, and forking a process with JAX loaded can deadlock. The one-shot alignment is an upper bound on the true ground distance and is asymmetric; that is the published definition, kept deliberately. Needs scipy and POT.

---

### `utils/report.py`

Writes one `metrics.json` per evaluated sample set, at `results/<target>/[seedN/]<run_id>/metrics.json` (`run_id` ∈ `sde_raw`, `sde_is`, `buffer`). The record holds `schema_version`, `method`, `target`, `run_id`, `seed`, `metrics`, plus `config` (the run's `ModelConfig` / `TrainingConfig` and ladder), `commit` (`null` when the tree is dirty — a hash that does not describe the code that ran would look authoritative), `timestamp` and `notes`. Metric names: `w2_pooled` is Distance-W2, `w2_energy` is Energy-W2, `geometric_w2` is Geometric-W2 (`null` for GMM). Non-finite values are written as `null`, not `NaN`. A new run overwrites the previous run's records.

| Symbol | Description |
|---|---|
| `write_record(results_dir, target, run_id, metrics, seed, config, notes)` | Write and return one record |
| `json_safe(x)` | Recursive conversion: NumPy/JAX scalars and arrays, dataclasses, NaN/inf → `null` |
| `git_commit()` | Short HEAD hash, or `None` if not a repo or dirty |

---

## Data flow: one outer iteration

```
x_T ~ N(0, I)                                           buffer[k+1]
    │                                                        ▲
    ▼                                                        │ .add(x_rejuv, normalize_log_weights(bridge_lw))
backward SDE(θ, β_k)                                         │
    │ x0_batch, log_w_path                                   │
    ▼                                                        │
systematic_resample(log_w_path)                              │
    │ x_resampled (≈ iid from π_{β_k})                      │
    ▼                                                        │
rejuvenate(x_resampled, β_k, k, mcmc_params_k)               │
    │ x_rejuv                                                │
    ▼                                                        │
bridge log-weights (β_{k+1} − β_k)(−E(x) + ½‖x‖²)            │
    │ bridge_lw ─────────────────────────────────────────────┘
```

All `M-1` active levels (`k = 1 … M-1`) run the above in parallel via `jax.vmap`. The backward SDE does not read `buffer[k]`; buffers are read only by the inner training loop. Buffer writes are sequential Python mutations after `vmap` returns.

---

## Key type conventions

| Name | Shape | Notes |
|---|---|---|
| `x0`, `x` | `(d,)` | Single particle; always unbatched at the leaf level |
| `x0_batch` | `(K, d)` | Particle batch |
| `log_weights` | `(N,)` | Unnormalized; `-inf` marks empty slots |
| `beta` | scalar | Inverse temperature ∈ [0, 1] |
| `t` | scalar | Noise time ∈ [0, T] |
| `betas` | `(M+1,)` | Full temperature ladder |
| `mcmc_params_by_rung` | `MALAParameters`, leaves with leading axis `M` | SMC-adapted kernel; entry `k-1` is rung `k` |
| `mcmc_params_active` | `MALAParameters`, leading axis `M-1` | Per-level kernel parameters passed to the outer vmap |
