# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sequential Score Sampler (SSS) — LJ55
#
# SSS on the 55-particle Lennard-Jones cluster (LJ55, 165 coordinates). Set
# `resume = True` to continue from a checkpoint.

# %% [markdown]
# ## Imports and Seeds

# %%
from __future__ import annotations

import os
import time
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
# Allocate GPU memory on demand.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Deterministic GPU ops, so a seed reproduces its run.
if "--xla_gpu_deterministic_ops" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
    ).strip()

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from progressive_is_tempering.net.egnn_jax import EGNNDynamics
from progressive_is_tempering.resampling import systematic_resample, normalized_ess
from progressive_is_tempering.mala_kernel import PTProblem
from progressive_is_tempering.sampler import exponential_time_grid
from progressive_is_tempering.smc import smc_sampler
from progressive_is_tempering.rejuvenate import rejuvenate
from progressive_is_tempering.buffer import ReplayBuffer
from progressive_is_tempering.targets import LennardJones
from progressive_is_tempering.utils import (
    C_INK,
    C_INK_2,
    C_TEST,
    plot_value_comparison,
    print_seed_quality_report,
    smc_seed_quality_figure,
    style_axis,
)
from progressive_is_tempering.utils.metrics import distance_w2, energy_w2, geometric_w2
from progressive_is_tempering.utils.report import write_record

SEED = 0
# Seed sweeps: `RUN_SEED=N` runs seed N and writes under results/<target>/seedN/.
# Unset, the run uses SEED above and writes under results/<target>/.
if "RUN_SEED" in os.environ:
    SEED = int(os.environ["RUN_SEED"])
SEED_DIR = f"seed{SEED}" if "RUN_SEED" in os.environ else None
key = jax.random.key(SEED)
# Sanity checks and diagnostics draw from their own key, never from `key`, so
# adding, removing or re-running them leaves the training random stream as is.
_DEMO_KEY = jax.random.key(20_000 + SEED)

print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# %% [markdown]
# ## 0. Configuration

# %%
@dataclass
class ModelConfig:
    """EGNN architecture hyperparameters."""
    n_particles: int = 55
    spatial_dim: int = 3
    x_dim: int = 165         # n_particles * spatial_dim
    hidden_nf: int = 128
    n_layers: int = 5
    recurrent: bool = True
    attention: bool = True
    tanh: bool = True
    condition_time: bool = True
    condition_beta: bool = True
    beta_embedding_dim: int = 8
    coords_range: float = 15.0


@dataclass
class TrainingConfig:
    """All training / sampling hyperparameters (LJ55-specific values)."""
    # SDE. Exponential time grid t_N = T > ... > t_1 = t_min, then 0 (Section 6):
    # log-spaced levels, so the last step is exactly t_min whatever N is.
    T: float = 1.5
    t_min: float = 1e-5
    n_sde_steps_train: int = 80
    n_sde_steps_eval: int = 1_000

    # Outer loop
    num_outer_steps: int = 50
    n_particles: int = 512          # K: particles per backward SDE run, per rung

    # Inner loop. One gradient step sees batch_total states over all M+1 rungs
    # (B = batch_total // (M+1) per rung, ~4 for the ~30-rung ladder).
    n_inner_steps: int = 500
    batch_total: int = 128

    # HMC rejuvenation (the SMC-tuned kernel; see rejuvenate)
    n_mala_rejuv: int = 500         # HMC steps per particle per outer step
    # MCMC kernel, for SMC and rejuvenation alike: "hmc" (partial velocity
    # refresh) or "mala" (full refresh); see `progressive_is_tempering.smc.KERNELS`.
    mcmc_kernel: str = "hmc"

    # Clipping, all per particle (reshape to (n_particles, spatial_dim) and
    # clamp each particle's own norm), so one close contact damps only its pair.
    target_clip_quantile: float = 0.999
    # A row (x, ∇E(x)) reaches a buffer only if E and ∇E are finite and its
    # largest per-particle ‖β ∇E‖ is at most grad_cap.
    grad_cap: float = 1e4
    # SDE per-particle score clip = score_clip_factor × the largest rung clip.
    score_clip_factor: float = 1.0

    # Optimiser
    learning_rate: float = 5e-4
    grad_clip: float = 1.0

    # One number governs the SMC population, the per-rung seed, and the buffer
    # capacity: they are the same particles at three points in their life.
    n_smc_particles: int = 4_096
    smc_local_ess_ratio: float = 0.2
    smc_n_mcmc_steps: int = 200

    # Memory: the EGNN runs on chunks of this many states inside the SDE.
    net_chunk: int = 256

    # Periodic evaluation at β = 1 every quick_eval_every outer steps (0
    # disables it), on n_quick_eval particles with the training grid.
    quick_eval_every: int = 2
    n_quick_eval: int = 512
    # Final evaluation: particles in the β = 1 draw scored by the reported
    # metrics, and the subsample size for Geometric-W2.
    n_eval_samples: int = 5_000
    geometric_max_n: int = 2_000

    # --- Derived: seed size and buffer capacity are the SMC population -------
    @property
    def buffer_capacity(self) -> int:
        return self.n_smc_particles

    @property
    def n_seed_particles(self) -> int:
        return self.n_smc_particles

    @property
    def seed_flush_steps(self) -> float:
        """Outer steps until FIFO has fully retired the SMC seed."""
        return self.n_smc_particles / self.n_particles


model_cfg = ModelConfig()
train_cfg = TrainingConfig()

# The SMC seed leaves the buffers after seed_flush_steps outer steps.
_flush = train_cfg.seed_flush_steps
print(f"buffer: {train_cfg.buffer_capacity:,}/rung "
      f"= {_flush:.3g} generations at K={train_cfg.n_particles:,}")
print(f"seed retired after outer step {_flush:.3g} "
      f"of {train_cfg.num_outer_steps} "
      f"({_flush / train_cfg.num_outer_steps:.0%} of the run)")
if _flush >= train_cfg.num_outer_steps:
    print("  WARNING: seed never fully drains — raise num_outer_steps, "
          "or lower n_smc_particles")

resume: bool = False

NOTEBOOK_NAME = "lj55"
# Outputs go to results/<target>/[seedN/].
RESULTS_DIR = Path("../../results") / NOTEBOOK_NAME
if SEED_DIR is not None:
    RESULTS_DIR = RESULTS_DIR / SEED_DIR
FIGURES_DIR = RESULTS_DIR / "figures"
EVAL_FIGURES_DIR = FIGURES_DIR / "eval"
for d in [RESULTS_DIR, FIGURES_DIR, EVAL_FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.pkl"
LOG_PATH = RESULTS_DIR / "log.txt"

N_P, S_D, X_D = model_cfg.n_particles, model_cfg.spatial_dim, model_cfg.x_dim


def log(msg: str) -> None:
    """Print and append to log.txt."""
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")

# %% [markdown]
# ## 1. Target Distribution (LJ55)
#
# $$E(x)=\sum_{i<j}\varepsilon\bigl[(r_m/r_{ij})^{12}-2(r_m/r_{ij})^6\bigr]+\tfrac{s}{2}\sum_i\|x_i\|^2,\quad\varepsilon=2,\ r_m=1,\ s=1$$
#
# All states are kept at zero centre of mass (CoM), where the trap depends only
# on relative positions. Pair distances are floored at $10^{-5}$.

# %%
lj55 = LennardJones(n_particles=N_P, spatial_dim=S_D, epsilon=2.0, rm=1.0,
                    trap_scale=1.0, trap_mode="individual", beta=1.0)
_R_EPS = 1e-5
_I, _J = np.triu_indices(N_P, k=1)   # 1485 unique pairs


def energy(x: jnp.ndarray) -> jnp.ndarray:
    """E(x) = -log π(x) + const.  Expects flat (165,) input."""
    pos = x.reshape(N_P, S_D)
    diff = pos[:, None, :] - pos[None, :, :]
    r = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + _R_EPS)[_I, _J]
    inv_r = lj55.rm / r
    u_lj = jnp.sum(lj55.epsilon * (inv_r ** 12 - 2.0 * inv_r ** 6))
    return u_lj + 0.5 * lj55.trap_scale * jnp.sum(pos ** 2)


# Quick sanity check at a random point (huge: randomly scattered particles are
# far from the cluster geometry LJ favours).
_x_test = jax.random.normal(jax.random.fold_in(_DEMO_KEY, 0), (X_D,))
print(f"LJ55 dim: {lj55.dim}")
print(f"energy(random) = {energy(_x_test):.4e}")
print(f"∇E norm at random point = {jnp.linalg.norm(jax.grad(energy)(_x_test)):.4e}")

# %% [markdown]
# ## 2. Geometry Utilities, Reference Data, and Comparison Plots
#
# All particle states are kept in the zero-CoM subspace throughout.

# %%
def project_com(x: jnp.ndarray) -> jnp.ndarray:
    """Project flat (165,) or batched (..., 165) to zero centre-of-mass."""
    shape = x.shape
    xr = x.reshape(-1, N_P, S_D)
    return (xr - xr.mean(axis=-2, keepdims=True)).reshape(shape)


def per_particle_norm(v: jnp.ndarray) -> jnp.ndarray:
    """(..., 165) → (..., 55) Euclidean norm of each particle's 3-vector."""
    return jnp.linalg.norm(v.reshape(v.shape[:-1] + (N_P, S_D)), axis=-1)


def clip_per_particle(v: jnp.ndarray, max_norm) -> jnp.ndarray:
    """Clamp each particle's norm to max_norm (scalar, or one value per row of v),
    direction preserved.
    """
    vr = v.reshape(v.shape[:-1] + (N_P, S_D))
    n = jnp.linalg.norm(vr, axis=-1, keepdims=True)
    mx = jnp.asarray(max_norm)[..., None, None] if jnp.ndim(max_norm) else max_norm
    return (vr * jnp.minimum(mx / (n + 1e-6), 1.0)).reshape(v.shape)


def radius_of_gyration(x):
    """(K, 165) zero-CoM configurations → (K,) radius of gyration (reference ≈ 1.73)."""
    return jnp.sqrt(jnp.mean(jnp.sum(x.reshape(-1, N_P, S_D) ** 2, axis=-1), axis=-1))


_xp = jax.vmap(project_com)(jax.random.normal(jax.random.fold_in(_DEMO_KEY, 1), (5, X_D)))
print(f"Max |CoM| after projection: "
      f"{jnp.abs(_xp.reshape(5, N_P, S_D).mean(axis=1)).max():.2e}  (should be ≈ 0)")

# %%
# ---- Summary statistics -----------------------------------------------------
@jax.jit
def get_pairwise_distances(x_flat):
    """All 1485 unique pairwise distances for (B, 165) input → (B, 1485)."""
    pos = x_flat.reshape(-1, N_P, S_D)
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    return jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)[:, _I, _J]


@jax.jit
def get_energies(x_flat):
    """Per-sample energy E(x) for (B, 165) input → (B,)."""
    return jax.vmap(energy)(x_flat.reshape(-1, X_D))


# ---- Reference (ground-truth) samples ---------------------------------------
# Held-out LJ55 test split (DEM), zero CoM; fetched by data/redownload.sh.
_pt_ref = jnp.array(np.load(Path("../../data/test_split_LJ55-1000-part1.npy")))  # (10000, 165)
_REF_R = get_pairwise_distances(_pt_ref)      # (10000, 1485)
_REF_E = get_energies(_pt_ref)                 # (10000,)
# Per-particle size of the β = 1 TSM target on the reference: what the clips of
# Section 11 should leave alone.
_ref_g = jax.vmap(jax.grad(energy))(_pt_ref)
_ref_tsm = per_particle_norm(_pt_ref - _ref_g)
print(f"Loaded reference samples: {_pt_ref.shape}  "
      f"max|CoM|={jnp.abs(_pt_ref.reshape(-1, N_P, S_D).mean(axis=1)).max():.2e}")
print(f"reference {_pt_ref.shape}: E mean {float(_REF_E.mean()):.2f} "
      f"median {float(jnp.median(_REF_E)):.2f}; per-particle ‖x−∇E‖ "
      f"median {float(jnp.median(_ref_tsm)):.1f} q99 {float(jnp.quantile(_ref_tsm, 0.99)):.1f} "
      f"q99.9 {float(jnp.quantile(_ref_tsm, 0.999)):.1f}")

# ---- Comparison plot helpers ----
# C_TEST: the sample set under test; the reference is filled.
_style_axis = style_axis


def plot_distance_comparison(ax, r_test, label_test, r_ref=_REF_R, bins=150):
    """Pairwise-distance density: test vs reference, on shared bin edges."""
    return plot_value_comparison(
        ax, r_test, r_ref, "inter-particle distance $r_{ij}$",
        label_test, bins=bins, clip_at_zero=True,
    )


def plot_energy_comparison(ax, e_test, label_test, e_ref=_REF_E, bins=150):
    """Energy density: test vs reference. Energy weights close contacts by r^-12,
    so it is the sharper of the two statistics.
    """
    return plot_value_comparison(
        ax, e_test, e_ref, "energy $E(x)$", label_test, bins=bins,
    )

# %% [markdown]
# ## 3. Tempering Problem for LJ55
#
# Geometric path from $\mathcal N(0,I)$ (zero CoM) to $\pi$.

# %%
def logpdf_ref(x: jnp.ndarray) -> jnp.ndarray:
    """log N(x; 0, I) up to a constant (reference on the CoM-zero subspace)."""
    return -0.5 * jnp.sum(x ** 2)


def logpdf_final_target(x: jnp.ndarray) -> jnp.ndarray:
    """log π(x) = -E(x) (unnormalized, LJ55 log-density)."""
    return -energy(x)


def rvs_ref(key: jnp.ndarray) -> jnp.ndarray:
    """Draw one sample from N(0, I) projected to zero CoM."""
    return project_com(jax.random.normal(key, (X_D,)))

# %%
pt_problem = PTProblem(
    rvs_ref=rvs_ref,
    logpdf_ref=logpdf_ref,
    logpdf_final_target=logpdf_final_target,
)

# %% [markdown]
# ## 4. Forward Process
#
# OU process $dX_t=-X_t\,dt+\sqrt2\,dW_t$, close to $\mathcal N(0,I)$ at $t=T$.

# %%
def ou_forward_com(x0_all, t_all, eps):
    """Batched OU noising x_t = e^{-t} x0 + σ_t ε, projected to zero CoM."""
    sigma_t = jnp.sqrt(-jnp.expm1(-2.0 * t_all))
    return jax.vmap(project_com)(jnp.exp(-t_all)[:, None] * x0_all + sigma_t[:, None] * eps)

# %% [markdown]
# ## 5. Score Network (EGNN)
#
# $\beta$-conditioned EGNN for the excess score
# $s_\theta(x_t,t,\beta)\approx\nabla\log(p_t^\beta/p_t^\rho)$.

# %%
score_net = EGNNDynamics(
    n_particles=model_cfg.n_particles,
    spatial_dim=model_cfg.spatial_dim,
    hidden_nf=model_cfg.hidden_nf,
    n_layers=model_cfg.n_layers,
    recurrent=model_cfg.recurrent,
    attention=model_cfg.attention,
    tanh=model_cfg.tanh,
    condition_time=model_cfg.condition_time,
    condition_beta=model_cfg.condition_beta,
    beta_embedding_dim=model_cfg.beta_embedding_dim,
    coords_range=model_cfg.coords_range,
)
key, subkey = jax.random.split(key)
params = score_net.init(subkey, jnp.zeros((1, X_D)), jnp.zeros((1, 1)), jnp.zeros((1, 1)))
n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"EGNN initialized: {n_params:,} parameters")

_LOG_T_MIN = float(np.log(train_cfg.t_min))
_LOG_RANGE = float(np.log(train_cfg.T) - np.log(train_cfg.t_min))


def t_feature(t):
    """Network time input: log t mapped linearly onto [0, 1] over [t_min, T]."""
    return (jnp.log(t) - _LOG_T_MIN) / _LOG_RANGE


def apply_net_chunked(params, x, t, b, chunk):
    """EGNN on (K, 165) at scalar t, β, in sequential chunks of `chunk` states."""
    K = x.shape[0]
    C = min(chunk, K)
    n_pad = (-K) % C
    xp = jnp.pad(x, ((0, n_pad), (0, 0))).reshape(-1, C, X_D)
    tf = jnp.full((C, 1), t_feature(t))
    bf = jnp.full((C, 1), b)
    out = jax.lax.map(lambda xi: score_net.apply(params, xi, tf, bf), xp)
    return out.reshape(-1, X_D)[:K]

# %% [markdown]
# ## 6. CoM-Projected Backward SDE
#
# Backward SDE with a zero-CoM projection after every step.

# %%
def backward_sde_batch_com(key, control_fn, beta, energy_fn, n_particles, T, n_steps):
    """Batched backward SDE: all n_particles in one scan, one control call per step
    on the whole (K, 165) batch. Returns x0 (K, 165) and path log-weights (K,).
    """
    times, dts = exponential_time_grid(T, n_steps, train_cfg.t_min)
    key, subkey = jax.random.split(key)
    x_T = project_com(jax.random.normal(subkey, (n_particles, X_D)))
    init_log_w = 0.5 * jnp.sum(x_T ** 2, axis=-1)

    def step_fn(carry, t_dt):
        x_n, log_w, rng = carry
        t_n, dt = t_dt
        rng, subkey = jax.random.split(rng)
        lam = jnp.exp(-dt)
        one_minus_lam = -jnp.expm1(-dt)          # accurate for the tiny steps near 0
        sigma_sq = -jnp.expm1(-2.0 * dt)
        control = control_fn(x_n, t_n, beta)
        eps = jax.random.normal(subkey, x_n.shape)
        # Exponential-integrator backward step, then project
        mu_bwd = lam * x_n + one_minus_lam * control
        x_prev = project_com(mu_bwd + jnp.sqrt(sigma_sq) * eps)
        log_p = -0.5 * jnp.sum((x_n - lam * x_prev) ** 2, axis=-1) / sigma_sq
        log_q = -0.5 * jnp.sum((x_prev - mu_bwd) ** 2, axis=-1) / sigma_sq
        return (x_prev, log_w + log_p - log_q, rng), None

    (x0, log_w, _), _ = jax.lax.scan(step_fn, (x_T, init_log_w, key), (times, dts))
    # Terminal correction: log π_β(x_0) for each particle
    energies = jax.vmap(energy_fn)(x0)
    log_w = log_w - (1.0 - beta) * 0.5 * jnp.sum(x0 ** 2, axis=-1) - beta * energies
    log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)
    return x0, log_w


def make_control(params, score_clip):
    """Control 2·s_θ(x, t, β), the score clipped per particle at score_clip."""
    def control(x, t, b):
        s = apply_net_chunked(params, x, t, b, train_cfg.net_chunk)
        return 2.0 * clip_per_particle(s, score_clip)
    return control


@partial(jax.jit, static_argnums=(3, 4, 5))
def _run_backward_sde(key, params, beta, n_particles, T, n_steps, score_clip):
    """JIT-compiled CoM-projected backward SDE runner."""
    return backward_sde_batch_com(key, make_control(params, score_clip), beta, energy,
                                  n_particles, T, n_steps)

# %% [markdown]
# ## 7. CoM-Projected HMC Rejuvenation
#
# HMC moves at rung $k$ with the kernel SMC tuned for that rung (Section 9),
# projected to zero CoM.
#
# ## 8. IS Bridging Between Adjacent Temperatures
#
# $$\log w_{k\to k+1}(x)=(\beta_{k+1}-\beta_k)\bigl(\tfrac12\|x\|^2-E(x)\bigr)$$

# %%
@jax.jit
def is_bridge_log_weights(samples, beta_k, beta_next):
    """IS log-weights to bridge π_{β_k} → π_{β_{k+1}}."""
    energies = jax.vmap(energy)(samples)
    return (beta_next - beta_k) * (-energies + 0.5 * jnp.sum(samples ** 2, axis=-1))


@jax.jit
def normalize_block(lw):
    """Mean weight 1 over the finite entries; −inf entries stay −inf."""
    n_ok = jnp.sum(jnp.isfinite(lw))
    return lw - logsumexp(lw) + jnp.log(jnp.maximum(n_ok, 1))

# %% [markdown]
# ## 9. Temperature Ladder via SMC
#
# Adaptive SMC gives the ladder $\{\beta_k\}$, one particle block per rung (the
# buffer seeds) and one tuned HMC kernel per rung (reused for rejuvenation).

# %%
_ckpt_data = None
if resume and CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, "rb") as f:
        _ckpt_data = pickle.load(f)
    print(f"Checkpoint found: outer_step={_ckpt_data['outer_step']}, "
          f"will restore temperature ladder, buffers, clips and the HMC kernel")

if _ckpt_data is None:
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            n_attempts = f.read().count("Attempt")
    else:
        n_attempts = 0
    log(f"\nAttempt {n_attempts + 1} ({time.strftime('%F %T')})")

    key, subkey = jax.random.split(key)
    t0 = time.time()
    smc_particles, logZ_dict, betas_list, mcmc_params_by_rung = smc_sampler(
        subkey,
        pt_problem,
        n_particles=train_cfg.n_smc_particles,
        n_mcmc_steps=train_cfg.smc_n_mcmc_steps,
        local_ess_ratio=train_cfg.smc_local_ess_ratio,
        verbose=True,
        kernel=train_cfg.mcmc_kernel,
    )
    smc_time = time.time() - t0
    betas = jnp.array(betas_list)
    M = len(betas_list) - 1
    # Project once, at the boundary: the HMC kernel does not, and the individual
    # trap only confines the CoM rather than pinning it.
    smc_states = jax.vmap(jax.vmap(project_com))(smc_particles)
    log(f"SMC: {smc_time:.0f}s, {M + 1} levels: {[round(float(b), 4) for b in betas_list]}")
    log(f"SMC eps per rung: {[float(f'{e:.4g}') for e in mcmc_params_by_rung.scaled_cov.eps]}")
    log(f"logZ: {logZ_dict}")
else:
    betas_list = _ckpt_data["betas_list"]
    betas = jnp.array(betas_list)
    M = len(betas_list) - 1
    mcmc_params_by_rung = _ckpt_data["mcmc_params_by_rung"]
    print(f"Restored temperature ladder ({len(betas_list)} levels): "
          f"{[f'{b:.4f}' for b in betas_list]}")
print(f"M={M} non-reference temperature levels")

# %% [markdown]
# ## 10. SMC Seed Quality at $\beta = 1$
#
# The top SMC rung against the reference samples.

# %%
if _ckpt_data is None:
    _smc_top = smc_states[-1]
    _e_smc = get_energies(_smc_top)
    _r_smc = get_pairwise_distances(_smc_top)
    fig, _seed_stats = smc_seed_quality_figure(
        features_test=np.asarray(_r_smc), features_ref=np.asarray(_REF_R),
        energy_test=np.asarray(_e_smc), energy_ref=np.asarray(_REF_E),
        feature_label="inter-particle distance $r_{ij}$",
        title=f"LJ55 SMC seed at β=1 ({_smc_top.shape[0]:,} particles)",
        features_nonneg=True,
    )
    fig.savefig(FIGURES_DIR / "lj55_smc_seed_quality.png", dpi=120)
    plt.close(fig)
    _ns = min(_smc_top.shape[0], _pt_ref.shape[0])
    log(f"SMC seed β=1: E median {float(jnp.median(_e_smc)):.2f} (ref "
        f"{float(jnp.median(_REF_E)):.2f}); Energy-W2 "
        f"{energy_w2(np.asarray(_e_smc), np.asarray(_REF_E)):.3f}; Distance-W2 "
        f"{distance_w2(np.asarray(_r_smc[:_ns]), np.asarray(_REF_R[:_ns])):.4f}")
    print_seed_quality_report(_seed_stats)
else:
    print("Resumed from checkpoint — SMC seed diagnostic skipped (no smc_states).")

# %% [markdown]
# ## 11. Per-Temperature Replay Buffers (Seeded from SMC) and Clips
#
# Buffer 0 holds exact $\mathcal N(0,I)$ draws; buffer $k\ge1$ starts with rung
# $k$'s SMC block. Rows with non-finite or too large $\beta\nabla E$ get zero
# weight, and the TSM target is clipped per rung.

# %%
def row_ok(grads, energies, beta):
    """Rows usable as training data: finite E and ∇E, and max per-particle ‖β ∇E‖ ≤
    grad_cap (β ∇E is what enters the TSM target).
    """
    finite = jnp.isfinite(energies) & jnp.all(jnp.isfinite(grads), axis=-1)
    gmax = jnp.max(per_particle_norm(jnp.nan_to_num(grads, posinf=1e30, neginf=-1e30)), axis=-1)
    return finite & (beta * gmax <= train_cfg.grad_cap)


if _ckpt_data is None:
    buffers: list[ReplayBuffer] = [
        ReplayBuffer(capacity=train_cfg.buffer_capacity, dim=X_D, store_grads=True)
        for _ in range(M + 1)
    ]
    key, subkey = jax.random.split(key)
    _b0 = jax.vmap(project_com)(jax.random.normal(subkey, (train_cfg.buffer_capacity, X_D)))
    buffers[0].add(_b0, jnp.zeros(train_cfg.buffer_capacity), jnp.zeros_like(_b0))
    buffers[0].freeze()

    clip_by_rung = [1.0]   # rung 0: target is 0, the clip never matters
    for _k in range(1, M + 1):
        _x = smc_states[_k]
        _g = jax.vmap(jax.grad(energy))(_x)
        _ok = row_ok(_g, get_energies(_x), float(betas[_k]))
        _lw = jnp.where(_ok, 0.0, -jnp.inf)
        _g = jnp.where(_ok[:, None], _g, 0.0)
        buffers[_k].add(_x, _lw, _g)
        _n = per_particle_norm(float(betas[_k]) * (_x - _g))[_ok]
        clip_by_rung.append(float(jnp.quantile(_n, train_cfg.target_clip_quantile)))
        log(f"  buffer {_k} β={float(betas[_k]):.4f}: {int(_ok.sum())}/{_x.shape[0]} rows ok; "
            f"per-particle ‖β(x−∇E)‖ median {float(jnp.median(_n)):.1f} "
            f"→ clip {clip_by_rung[-1]:.1f}")
    # Buffer 1 is never written during training (the outer loop fills 2..M).
    if M >= 2:
        buffers[1].freeze()
else:
    buffers = _ckpt_data["buffers"]
    clip_by_rung = _ckpt_data["clip_by_rung"]
    print(f"Restored {len(buffers)} replay buffers and per-rung clips from checkpoint")

clip_by_rung_arr = jnp.array(clip_by_rung)
SCORE_CLIP = float(train_cfg.score_clip_factor * max(clip_by_rung[1:]))
B_PER = max(1, train_cfg.batch_total // (M + 1))
log(f"M={M}; per-rung batch B={B_PER} ({B_PER * (M + 1)} states/step); "
    f"SDE per-particle score clip {SCORE_CLIP:.1f}")

# %% [markdown]
# ## 12. Data Generation (Outer Loop)
#
# For every rung $k=1,\dots,M-1$, one rung at a time: backward SDE, resampling,
# HMC rejuvenation, bridge weights to rung $k+1$.

# %%
@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def _data_gen(sde_keys, resample_keys, mala_keys, n_particles, T, n_sde_steps,
              n_mala_steps, params, betas_active, betas_next, t_idx_active,
              mcmc_params_active, score_clip):
    """Full data-generation pipeline for all active temperature levels, rung by rung."""
    control = make_control(params, score_clip)

    def process_one(args):
        sde_key, resample_key, mala_key, beta_k, beta_next, t_idx_k, mcmc_params_k = args
        # 1. Backward SDE
        x0, log_w_path = backward_sde_batch_com(
            sde_key, control, beta_k, energy, n_particles, T, n_sde_steps)
        x0 = jnp.nan_to_num(x0)
        # 2. Systematic resample
        idx = systematic_resample(resample_key, log_w_path, n_particles)
        # 3. HMC rejuvenation, SMC-tuned kernel of rung k; energies from the final state
        x_rej, acc, energies = rejuvenate(
            mala_key, x0[idx], pt_problem, beta_k, t_idx_k, mcmc_params_k,
            n_mala_steps, project_fn=project_com, return_energy=True,
            kernel=train_cfg.mcmc_kernel)
        # 5. ∇E once per particle; 4. row checks and bridge weights
        grads = jax.vmap(jax.grad(energy))(x_rej)
        ok = row_ok(grads, energies, beta_k)
        bridge_lw = (beta_next - beta_k) * (-energies + 0.5 * jnp.sum(x_rej ** 2, axis=-1))
        bridge_lw = jnp.where(ok, bridge_lw, -jnp.inf)
        grads = jnp.where(ok[:, None], grads, 0.0)
        x_rej = jnp.where(ok[:, None], x_rej, 0.0)
        e0 = jnp.where(jnp.isfinite(log_w_path), jax.vmap(energy)(x0), jnp.nan)
        stats = dict(path_ess=normalized_ess(log_w_path), acc=acc, n_ok=ok.sum(),
                     e_raw_med=jnp.nanmedian(jnp.where(jnp.isfinite(e0), e0, jnp.nan)),
                     e_rej_med=jnp.median(jnp.where(ok, energies, jnp.inf)),
                     rg_raw_med=jnp.median(radius_of_gyration(x0)),
                     rg_rej_med=jnp.nanmedian(jnp.where(ok, radius_of_gyration(x_rej), jnp.nan)))
        return x_rej, bridge_lw, grads, stats, x0

    return jax.lax.map(process_one, (sde_keys, resample_keys, mala_keys, betas_active,
                                     betas_next, t_idx_active, mcmc_params_active))

# %% [markdown]
# ## 13. Training Step (Inner Loop)
#
# TSM loss on all buffers jointly, with the target clipped per particle; one
# Adam step.

# %%
optimizer = optax.chain(
    optax.clip_by_global_norm(train_cfg.grad_clip),
    optax.adam(train_cfg.learning_rate),
)
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, xg_all, t_all, beta_all, clip_all, key):
    """Loss, gradient, and Adam update."""
    x0_all, g_all = xg_all[:, :X_D], xg_all[:, X_D:]

    v = beta_all[:, None] * (x0_all - g_all)
    v_norm = per_particle_norm(v)
    clipped_frac = jnp.sum((v_norm > clip_all[:, None]) & (beta_all[:, None] > 0)) / (
        jnp.maximum(jnp.sum(beta_all > 0), 1) * N_P)
    tsm = jnp.exp(-t_all)[:, None] * clip_per_particle(v, clip_all)

    eps = jax.random.normal(key, x0_all.shape)
    xt_all = ou_forward_com(x0_all, t_all, eps)
    # stop_gradient: no backprop through the energy Hessian
    targets = jax.lax.stop_gradient(tsm)

    def loss_fn(p):
        preds = score_net.apply(p, xt_all, t_feature(t_all)[:, None], beta_all[:, None])
        return jnp.mean(jnp.sum((preds - targets) ** 2, axis=-1))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss, clipped_frac

# %% [markdown]
# ## 14. Periodic Evaluation
#
# Quick $\beta=1$ evaluation every `quick_eval_every` outer steps.

# %%
_REF_R_NP = np.asarray(_REF_R)
_DIAG_KEY = jax.random.key(10_000 + SEED)


def summarize(x, lw=None, keep_r=False):
    """Distance-W2, radius of gyration, mean distance, share with a pair < 0.7,
    median energy (and path ESS if lw is given) of a (K, 165) sample set.
    """
    e = get_energies(x)
    r = get_pairwise_distances(x)
    rmin = jnp.min(r, axis=-1)
    out = dict(n=int(x.shape[0]),
               e_med=float(jnp.nanmedian(jnp.where(jnp.isfinite(e), e, jnp.nan))),
               frac_pair_lt_07=float(jnp.mean(rmin < 0.7)),
               rg_med=float(jnp.median(radius_of_gyration(x))),
               r_mean=float(jnp.mean(r)),
               dist_w2=distance_w2(np.asarray(r), _REF_R_NP))
    if lw is not None:
        out["path_ess"] = float(normalized_ess(lw))
    if keep_r:
        out["_r"] = np.asarray(r)
    return out


def _fmt(s):
    return (f"Dist-W2 {s['dist_w2']:.4f}  Rg {s['rg_med']:.2f}  r̄ {s['r_mean']:.3f}  "
            f"pair<0.7 {s['frac_pair_lt_07']:.2f}  E med {s['e_med']:.1f}")


def plot_distances(sets, title, fname):
    """Pairwise-distance histograms of several sample sets against the reference."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    bins = np.linspace(0, 8, 241)
    for ax, lim in zip(axes, [(0, 8), (0.6, 2.2)]):
        ax.hist(_REF_R_NP.ravel(), bins=bins, density=True, color="0.8", label="reference")
        for (name, s), c in zip(sets.items(), ["C0", "C1", "C2", "C3"]):
            ax.hist(s["_r"].ravel(), bins=bins, density=True, histtype="step", lw=1.4,
                    color=c, label=f"{name} (W2 {s['dist_w2']:.3f}, Rg {s['rg_med']:.2f})")
        ax.set_xlim(*lim)
        ax.set_xlabel("inter-particle distance $r_{ij}$")
    axes[0].legend(fontsize=8)
    axes[1].set_title("zoom: first shells")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(EVAL_FIGURES_DIR / fname, dpi=110)
    plt.close(fig)


def step_distance_diag(step, x_raw_top, x_rej_top, lw_top):
    """Per outer step, no extra SDE: the top active rung's raw SDE output and its
    rejuvenated block (which goes to buffer[M]), and buffer[M] resampled.
    """
    k = jax.random.fold_in(_DIAG_KEY, step)
    ok = np.isfinite(np.asarray(lw_top))
    xb, _ = buffers[M].sample(k, 2048)
    sets = {f"raw SDE β={float(betas[M - 1]):.3f}": summarize(x_raw_top),
            "new block → buffer[M]": summarize(x_rej_top[ok]) if ok.any() else None,
            "buffer[M] resampled": summarize(xb)}
    for n, s in sets.items():
        if s is not None:
            log(f"    dist  {n:<22} {_fmt(s)}")


def quick_eval(params, step):
    """β = 1 SDE on n_quick_eval particles: raw, IS-resampled, and buffer[M]."""
    k1, k2, k3 = jax.random.split(jax.random.fold_in(_DIAG_KEY, 1_000_000 + step), 3)
    x, lw = _run_backward_sde(k1, params, betas[-1], train_cfg.n_quick_eval,
                              train_cfg.T, train_cfg.n_sde_steps_train, SCORE_CLIP)
    raw = summarize(x, lw, keep_r=True)
    xi = x[systematic_resample(k2, lw, x.shape[0])]
    isr = summarize(xi, keep_r=True)
    xb, _ = buffers[M].sample(k3, train_cfg.n_quick_eval)
    buf = summarize(xb, keep_r=True)
    log(f"  [quick eval β=1, step {step}] path_ESS={raw['path_ess']:.4f}  "
        f"(ref E med {float(jnp.median(_REF_E)):.1f}, Rg 1.73)")
    for n, s in [("raw SDE", raw), ("SDE + IS", isr), ("buffer[M]", buf)]:
        log(f"    quick {n:<10} {_fmt(s)}")
    plot_distances({"raw SDE β=1": raw, "SDE + IS": isr, "buffer[M]": buf},
                   f"LJ55 seed {SEED}, quick eval at β=1 after outer step {step} "
                   f"(path ESS {raw['path_ess']:.4f})",
                   f"lj55_distances_step{step:03d}.png")

# %% [markdown]
# ## 15. Training Loop
#
# Each outer step generates data at every rung; `n_inner_steps` Adam steps then
# train on all buffers jointly.

# %%
start_outer_step, loss_history = 0, []
if _ckpt_data is not None:
    params, opt_state = _ckpt_data["params"], _ckpt_data["opt_state"]
    start_outer_step, loss_history = _ckpt_data["outer_step"], _ckpt_data["loss_history"]
    key = jax.random.wrap_key_data(_ckpt_data["key"])
    print(f"Resumed from outer step {start_outer_step}")

M_active = M - 1
t_idx_active = jnp.arange(1, M)
# mcmc_params_by_rung[k-1] is the kernel SMC tuned for rung k: slice the first
# M_active entries to match betas[1:-1].
mcmc_params_active = jax.tree_util.tree_map(lambda l: l[:M_active], mcmc_params_by_rung)

# %%
print(f"\nStarting training: outer steps {start_outer_step} → {train_cfg.num_outer_steps}")
train_start = time.time()

for outer_step in range(start_outer_step, train_cfg.num_outer_steps):
    t_step = time.time()

    # ---- Outer loop: data generation, rung by rung ----
    if M_active > 0:
        ks = jax.random.split(key, 3 * M_active + 1)
        key = ks[0]
        x_all, lw_all, g_all, st, x_raw_all = _data_gen(
            ks[1:M_active + 1], ks[M_active + 1:2 * M_active + 1], ks[2 * M_active + 1:],
            train_cfg.n_particles, train_cfg.T, train_cfg.n_sde_steps_train,
            train_cfg.n_mala_rejuv, params, betas[1:-1], betas[2:], t_idx_active,
            mcmc_params_active, SCORE_CLIP)
        st = jax.tree_util.tree_map(np.asarray, st)
        skipped = []
        for i in range(M_active):
            if st["n_ok"][i] == 0:   # no valid row: nothing to add
                skipped.append(i + 2)
                continue
            buffers[i + 2].add(x_all[i], normalize_block(lw_all[i]), g_all[i])
    t_gen = time.time() - t_step

    # ---- Inner loop: gradient descent ----
    t_inner = time.time()
    loss_acc, clip_acc = jnp.zeros(()), jnp.zeros(())
    for _ in range(train_cfg.n_inner_steps):
        xs, ts, bs, cs = [], [], [], []
        for k in range(M + 1):
            key, k1, k2 = jax.random.split(key, 3)
            # rows are [x0 | ∇E(x0)]: one gather for both
            xg, _ = buffers[k].sample_with_grads(k1, B_PER)
            u = jax.random.uniform(k2, (B_PER,))
            xs.append(xg)
            ts.append(train_cfg.t_min * jnp.exp(u * _LOG_RANGE))   # log-uniform on [t_min, T]
            bs.append(jnp.full(B_PER, betas[k]))
            cs.append(jnp.full(B_PER, clip_by_rung_arr[k]))
        key, subkey = jax.random.split(key)
        params, opt_state, l, cf = train_step(
            params, opt_state, jnp.concatenate(xs), jnp.concatenate(ts),
            jnp.concatenate(bs), jnp.concatenate(cs), subkey)
        loss_acc, clip_acc = loss_acc + l, clip_acc + cf
    mean_loss = float(loss_acc / train_cfg.n_inner_steps)
    loss_history.append(mean_loss)
    t_inner = time.time() - t_inner

    # ---- Logging ----
    buf_ess = [float(normalized_ess(b.log_weights[:b.size])) for b in buffers]
    log(f"[step {outer_step + 1:3d}/{train_cfg.num_outer_steps}] loss={mean_loss:.2f}  "
        f"clipped={float(clip_acc / train_cfg.n_inner_steps):.4f}  "
        f"gen={t_gen:.0f}s inner={t_inner:.0f}s")
    if M_active > 0:
        log("    path_ESS " + " ".join(f"{e:.3f}" for e in st["path_ess"]))
        log("    HMC acc  " + " ".join(f"{a:.2f}" for a in st["acc"]))
        log("    rows ok  " + " ".join(f"{int(n)}" for n in st["n_ok"])
            + (f"  (skipped buffers {skipped})" if skipped else ""))
        log("    E med raw " + " ".join(f"{e:.0f}" for e in st["e_raw_med"]))
        log("    E med rej " + " ".join(f"{e:.0f}" for e in st["e_rej_med"]))
        log("    Rg raw   " + " ".join(f"{e:.2f}" for e in st["rg_raw_med"]))
        log("    Rg rej   " + " ".join(f"{e:.2f}" for e in st["rg_rej_med"]))
    log("    buf_ESS  " + " ".join(f"{e:.3f}" for e in buf_ess))
    if M_active > 0:
        step_distance_diag(outer_step + 1, x_raw_all[-1], x_all[-1], lw_all[-1])

    # ---- Checkpoint ----
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(dict(params=params, opt_state=opt_state, outer_step=outer_step + 1,
                         loss_history=loss_history, key=jax.random.key_data(key),
                         betas_list=betas_list, mcmc_params_by_rung=mcmc_params_by_rung,
                         buffers=buffers, clip_by_rung=clip_by_rung), f)

    # ---- Periodic evaluation ----
    if train_cfg.quick_eval_every and (outer_step + 1) % train_cfg.quick_eval_every == 0:
        quick_eval(params, outer_step + 1)

log(f"Training done in {time.time() - train_start:.0f}s")

# %% [markdown]
# ## 16. Final Evaluation
#
# Backward SDE at $\beta=1$ with `n_sde_steps_eval` steps, scored as three
# sample sets: raw SDE output (`sde_raw`), the same particles resampled on their
# path weights (`sde_is`), and `buffer[M]` (`buffer`).
#
# ### Three candidate definitions of "our method's samples"

# %%
t0 = time.time()
key, subkey = jax.random.split(key)
x0_eval, lw_eval = _run_backward_sde(subkey, params, betas[-1], train_cfg.n_eval_samples,
                                     train_cfg.T, train_cfg.n_sde_steps_eval, SCORE_CLIP)
ess_eval = float(normalized_ess(lw_eval))
log(f"Final SDE ({train_cfg.n_eval_samples} × {train_cfg.n_sde_steps_eval} steps): "
    f"{time.time() - t0:.0f}s, path ESS {ess_eval:.4f}")

# (2) IS-resampled SDE output and (3) buffer[M], drawn the way the inner loop
# draws it (systematic resampling on the stored weights).
key, k1, k2 = jax.random.split(key, 3)
x_resamp = x0_eval[systematic_resample(k1, lw_eval, x0_eval.shape[0])]
x_buf, _ = buffers[M].sample(k2, min(train_cfg.n_eval_samples, buffers[M].size))
buf_ess = float(normalized_ess(buffers[M].log_weights[:buffers[M].size]))

# (run_id, long label, short label, samples, subtitle)
sample_sets = [
    ("sde_raw", "Backward SDE (raw, no IS)", "SDE raw", x0_eval,
     f"unweighted, {x0_eval.shape[0]:,} particles"),
    ("sde_is", "Backward SDE + IS resample", "SDE + IS", x_resamp,
     f"path ESS={ess_eval:.4f}  (≈{ess_eval * x0_eval.shape[0]:,.1f} eff.)"),
    ("buffer", f"buffer[M]  β={float(betas[M]):.3f}", "buffer[M]", x_buf,
     f"buffer ESS={buf_ess:.3f}"),
]

# %%
# Columns: the three candidate sample sets, plus the loss curve.
# Rows: pairwise distances (geometry) and energy (physics).
fig, axes = plt.subplots(2, 4, figsize=(21, 8))
for _col, (_, _label, _short, _x, _sub) in enumerate(sample_sets):
    plot_distance_comparison(axes[0, _col], get_pairwise_distances(_x), _short)
    plot_energy_comparison(axes[1, _col], get_energies(_x), _short)
    axes[0, _col].set_title(f"{_label}\n{_sub}", color=C_INK, fontsize=10)
    axes[1, _col].set_title(f"Energy — {_short}", color=C_INK, fontsize=10)
ax_loss = axes[0, 3]
ax_loss.plot(loss_history, color=C_TEST, linewidth=1.5)
ax_loss.set_yscale("log")
ax_loss.set_xlabel("outer step", color=C_INK_2, fontsize=10)
ax_loss.set_ylabel("TSM loss (sum over dim)", color=C_INK_2, fontsize=10)
ax_loss.set_title("Training loss", color=C_INK, fontsize=11)
_style_axis(ax_loss)
axes[1, 3].set_axis_off()
fig.suptitle(f"LJ55 final evaluation — β={float(betas[-1]):.3f}, "
             f"three candidate sample sets vs ground truth", fontsize=12, color=C_INK)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "lj55_distances_energy_and_loss.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Reported metrics: Distance-𝒲₂, Energy-𝒲₂, Geometric-𝒲₂
#
# Written to `results/lj55/[seedN/]<run_id>/metrics.json`.

# %%
_ref_np, _ref_r_np, _ref_e_np = np.asarray(_pt_ref), np.asarray(_REF_R), np.asarray(_REF_E)
rng = np.random.default_rng(SEED)
log(f"{'set':<10}{'n':>7}{'Dist-W2':>10}{'Energy-W2':>12}{'Geom-W2':>10}"
    f"{'E med':>10}{'pair<0.7':>10}{'Rg':>7}")
for run_id, _label, short, x, _sub in sample_sets:
    xn = np.asarray(x)
    if xn.shape[0] > _ref_np.shape[0]:
        xn = xn[rng.choice(xn.shape[0], _ref_np.shape[0], replace=False)]
    e = np.asarray(get_energies(xn))
    s = summarize(jnp.asarray(xn))
    m = {
        # w2_pooled = Distance-W2, w2_energy = Energy-W2.
        "w2_pooled": distance_w2(np.asarray(get_pairwise_distances(xn)), _ref_r_np),
        "w2_energy": energy_w2(e, _ref_e_np),
        "geometric_w2": geometric_w2(xn, _ref_np, N_P, S_D,
                                     max_n=train_cfg.geometric_max_n, seed=SEED),
        "geometric_w2_max_n": train_cfg.geometric_max_n,
        "energy_mean": float(e[np.isfinite(e)].mean()) if np.isfinite(e).any() else None,
        "energy_median": s["e_med"],
        "energy_mean_ref": float(_ref_e_np.mean()),
        "frac_pair_lt_07": s["frac_pair_lt_07"],
        "n_eval_samples": int(xn.shape[0]), "n_ref_samples": int(_ref_np.shape[0]),
        # Path-space ESS of the one SDE run behind "SDE raw" and "SDE + IS";
        # null for buffer[M], whose block-normalised weights estimate nothing.
        "ess": None if run_id == "buffer" else ess_eval,
        "path_ess": ess_eval, "buffer_ess": buf_ess,
    }
    write_record(RESULTS_DIR, NOTEBOOK_NAME, run_id, m, seed=SEED,
                 config={"model": model_cfg, "training": train_cfg,
                         "betas": [float(b) for b in betas_list],
                         "clip_by_rung": clip_by_rung},
                 notes=f"Sample set: {_label} ({_sub}).")
    log(f"{short:<10}{xn.shape[0]:>7}{m['w2_pooled']:>10.4f}{m['w2_energy']:>12.3f}"
        f"{m['geometric_w2']:>10.4f}{s['e_med']:>10.1f}{s['frac_pair_lt_07']:>10.3f}"
        f"{s['rg_med']:>7.2f}")
log(f"reference energy median {float(np.median(_ref_e_np)):.2f}, Rg 1.73; "
    f"done {time.strftime('%F %T')}")
print(f"\nRecords written under {RESULTS_DIR}/<run_id>/metrics.json")

# %%
# Temperature ladder summary. LJ55's ladder is dense at small β, so β is shown
# on a log axis (β_0 = 0 omitted).
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, len(betas_list)), betas_list[1:], "o-", color=C_TEST, ms=4)
ax.set_yscale("log")
ax.set_xlabel("temperature level $k$", color=C_INK_2, fontsize=10)
ax.set_ylabel("$\\beta_k$", color=C_INK_2, fontsize=10)
ax.set_title(f"Temperature ladder ({len(betas_list)} levels)", color=C_INK, fontsize=11)
_style_axis(ax)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "lj55_temperature_ladder.png", dpi=150)
plt.close(fig)
print(f"Ladder: {[f'{b:.4g}' for b in betas_list]}")
