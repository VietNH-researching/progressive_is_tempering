# CODEMAP.md — Structural overview

## Repository layout

```
progressive_is_tempering/          # repo root
├── pyproject.toml                 # build config; pins mppt + jax_pdf
├── AGENTS.md                      # agent / contributor guidance
├── CODEMAP.md                     # this file
├── CONTRIBUTING.md                # dev workflow and test guidance
├── README.md                      # project overview
│
├── docs/
│   ├── TECHNIQUES_SUMMARY.md      # full algorithm math (read first)
│   ├── IMPLEMENTATION_MLP.md      # MLP score-network details
│   └── IMPLEMENTATION_EGNN.md     # EGNN score-network details
│
├── progressive_is_tempering/      # installable package
│   ├── __init__.py
│   ├── sde.py
│   ├── targets.py
│   ├── resampling.py
│   ├── mala.py
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
│       └── __init__.py
│
├── notebooks/
│   ├── gmm/                       # GMM experiment
│   │   ├── gmm.py                 # Jupytext percent script
│   │   └── gmm.ipynb
│   ├── dw4/                       # DW4 molecule experiment
│   │   ├── dw4.py
│   │   └── dw4.ipynb
│   └── templates/minimal.py       # minimal training loop template
│
├── tests/                         # pytest suite
├── mppt/                          # local SMC / parallel-tempering library
├── data/                          # reference data for sampling benchmarks
│   └── dw4_samples.npz            # ground-truth DW4 particle samples
├── checkpoints/                   # saved model checkpoints (.pkl)
├── figures/                       # saved plot outputs
└── logging/                       # training log files
```

---

## Module dependency layering

Arrows mean "imports from". Leaf modules import only from JAX/Flax.

```
notebooks/*, tests/*
        │
        ▼
progressive_is_tempering.__init__   (re-exports everything below)
        │
        ├──► sampler      ──► sde
        ├──► buffer       ──► resampling
        ├──► mala
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

Unnormalized target distributions. Each target exposes `energy(x)`, `log_prob(x)`, `grad_energy(x)`, and the MPPT interface (`logpdf_ref`, `logpdf_final_target`, `rvs_ref`).

| Symbol | Description |
|---|---|
| `GMMTarget` | Factorized GMM over ℝᵈ; each coordinate iid from the same 1-D mixture |
| `make_gmm_target(dim, means, weights, std)` | Constructs default GMM (d=16, μ=(-1.5, 2.5), w=(0.2, 0.8), σ=0.5) |

---

### `resampling.py`

| Symbol | Description |
|---|---|
| `systematic_resample(key, log_weights, n)` | Draw `n` indices; JIT-compiled; `n` must be a static Python int |
| `normalized_ess(log_weights)` | Normalized ESS ∈ (0, 1]; 1 = uniform weights, 1/N = degenerate |

---

### `mala.py`

MALA rejuvenation for particle diversification after IS resampling.

| Symbol | Description |
|---|---|
| `mala_step(key, x, beta, energy_fn, h)` | One MALA step with MH correction; returns `(x_new, accepted)` |
| `mala_rejuvenate(key, particles, beta, energy_fn, h, n_steps)` | Apply `n_steps` MALA to all `K` particles via vmap+scan; returns `(particles_new, mean_accept_rate)` |

`mala_rejuvenate` is JIT-compiled with `static_argnums=(3, 5)` (energy_fn, n_steps).

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

DDPM regression loss for the β-conditioned excess score network.

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

Equivariant Graph Neural Network backbone for molecular targets (DW4). Used in place of `ScoreNetwork` when SE(3) equivariance is required.

---

## Data flow: one outer iteration

```
buffer[k]                          buffer[k+1]
    │                                   ▲
    │ (warm start for score eval)       │ .add(x_rejuv, bridge_lw)
    ▼                                   │
backward_sde_batch(θ, β_k)             │
    │ x0_batch, log_w_path             │
    ▼                                   │
systematic_resample(log_w_path)         │
    │ x_resampled (≈ iid from π_{β_k}) │
    ▼                                   │
mala_rejuvenate(x_resampled, β_k, h_k) │
    │ x_rejuv                           │
    ▼                                   │
is_bridge_log_weights(x_rejuv, β_k, β_{k+1})
    │ bridge_lw ─────────────────────►──┘
```

All `M-1` active levels run the above in parallel via `jax.vmap`. Buffer writes are sequential Python mutations after `vmap` returns.

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
| `mala_hs_active` | `(M-1,)` | Per-level step sizes passed to outer vmap |
