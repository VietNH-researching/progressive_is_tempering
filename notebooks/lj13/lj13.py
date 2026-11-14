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
# # Sequential Score Sampler (SSS) — LJ13
#
# SSS on the 13-particle Lennard-Jones cluster (LJ13, 39 coordinates). Set
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
from progressive_is_tempering.sde import ou_forward_sample
from progressive_is_tempering.resampling import systematic_resample, normalized_ess
from progressive_is_tempering.mala_kernel import PTProblem, batch_init_extend
from progressive_is_tempering.sampler import power_time_grid
from progressive_is_tempering.smc import KERNELS, smc_sampler
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

print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# %% [markdown]
# ## 0. Configuration

# %%
@dataclass
class ModelConfig:
    """EGNN architecture hyperparameters."""
    n_particles: int = 13
    spatial_dim: int = 3
    x_dim: int = 39          # n_particles * spatial_dim
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
    """All training / sampling hyperparameters (LJ13-specific values)."""
    # SDE: power time grid t_n = T (n/N)^grid_power, finest near t = 0.
    T: float = 1.5
    grid_power: float = 2.0
    n_sde_steps_train: int = 55
    n_sde_steps_eval: int = 2000
    # Inner-loop times t = t_min + (T - t_min) u^t_power, u ~ U(0, 1).
    t_min: float = 1e-4
    t_power: float = 4.0

    # Outer loop
    num_outer_steps: int = 50
    n_particles: int = 5_000     # K: particles per backward SDE run

    # Inner loop
    n_inner_steps: int = 1_000
    batch_size: int = 512        # B per temperature

    # HMC rejuvenation (the SMC-tuned kernel; see rejuvenate)
    n_mala_rejuv: int = 2_000       # HMC steps per particle per outer step
    # MCMC kernel, for SMC and rejuvenation alike: "hmc" (partial velocity
    # refresh) or "mala" (full refresh); see `progressive_is_tempering.smc.KERNELS`.
    mcmc_kernel: str = "hmc"

    # Clipping, per particle: the r^-12 repulsion makes ∇E blow up for close
    # pairs.
    score_clip: float = 1e4
    target_clip_norm: float = 150.0   # per-particle max norm (not global)

    # Optimiser
    learning_rate: float = 1e-3
    grad_clip: float = 1.0

    # One number governs the SMC population, the per-rung seed, and the buffer
    # capacity: they are the same particles at three points in their life.
    n_smc_particles: int = 25_000
    smc_local_ess_ratio: float = 0.2
    smc_n_mcmc_steps: int = 200

    # Checkpointing / logging
    log_every: int = 1
    save_every: int = 1
    # Outer steps between evaluation plots (None disables them).
    eval_every: int | None = None
    n_eval_particles: int = 20_000
    # Particles in the final evaluation draw, scored by the reported metrics;
    # n_eval_particles above sizes only the periodic plots.
    n_eval_samples: int = 10_000

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

NOTEBOOK_NAME = "lj13"
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

# %% [markdown]
# ## 1. Target Distribution (LJ13)
#
# $$E(x)=\sum_{i<j}\varepsilon\bigl[(r_m/r_{ij})^{12}-2(r_m/r_{ij})^6\bigr]+\tfrac{s}{2}\sum_i\|x_i\|^2,\quad\varepsilon=2,\ r_m=1,\ s=1$$
#
# All states are kept at zero centre of mass (CoM), where the trap depends only
# on relative positions.

# %%
lj13 = LennardJones(n_particles=13, spatial_dim=3, epsilon=2.0, rm=1.0,
                     trap_scale=1.0, trap_mode='individual', beta=1.0)

# Pairwise-distance floor 1e-5 (LennardJones uses 1e-8), softening the r^-12
# term for near collisions.
_R_EPS = 1e-5


def _lj13_logpdf(x: jnp.ndarray) -> jnp.ndarray:
    """log π(x) for LJ13, matching LennardJones.__call__ but with _R_EPS floor."""
    pos = x.reshape(model_cfg.n_particles, model_cfg.spatial_dim)
    diff = pos[:, None, :] - pos[None, :, :]
    r = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + _R_EPS)
    i, j = jnp.triu_indices(model_cfg.n_particles, k=1)
    r_pairs = r[i, j]

    inv_r = lj13.rm / r_pairs
    u_lj = jnp.sum(lj13.epsilon * (inv_r ** 12 - 2.0 * inv_r ** 6))
    u_trap = 0.5 * lj13.trap_scale * jnp.sum(pos ** 2)
    return -lj13.beta * (u_lj + u_trap)


def energy(x: jnp.ndarray) -> jnp.ndarray:
    """E(x) = -log π(x) + const.  Expects flat (39,) input."""
    return -_lj13_logpdf(x)


# Quick sanity check.
key, subkey = jax.random.split(key)
x_test = jax.random.normal(subkey, (model_cfg.x_dim,))
print(f"LJ13 dim: {lj13.dim}")
print(f"energy(random) = {energy(x_test):.4e}")
print(f"∇E norm at random point = {jnp.linalg.norm(jax.grad(energy)(x_test)):.4e}")

# %% [markdown]
# ## 2. Geometry Utilities, Reference Data, and Comparison Plots
#
# All particle states are kept in the zero-CoM subspace throughout.

# %%
def project_com(x: jnp.ndarray) -> jnp.ndarray:
    """Project flat (39,) or batched (..., 39) to zero centre-of-mass."""
    shape = x.shape
    xr = x.reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim)
    com = xr.mean(axis=-2, keepdims=True)   # (..., 1, D)
    return (xr - com).reshape(shape)


# Verify projection
key, subkey = jax.random.split(key)
_xr = jax.random.normal(subkey, (5, model_cfg.x_dim))
_xp = jax.vmap(project_com)(_xr)
_com = _xp.reshape(5, model_cfg.n_particles, model_cfg.spatial_dim).mean(axis=1)
print(f"Max |CoM| after projection: {jnp.abs(_com).max():.2e}  (should be ≈ 0)")

# %%
# ---- Summary statistics -----------------------------------------------------
@jax.jit
def get_pairwise_distances(x_flat: jnp.ndarray) -> jnp.ndarray:
    """All 78 unique pairwise distances for (B, 39) input → (B, 78)."""
    pos = x_flat.reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim)   # (B, 13, 3)
    diff = pos[:, :, None, :] - pos[:, None, :, :]                            # (B, 13, 13, 3)
    r = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)                         # (B, 13, 13)
    i, j = jnp.triu_indices(model_cfg.n_particles, k=1)                       # 78 unique pairs
    return r[:, i, j]                                                          # (B, 78)


@jax.jit
def get_energies(x_flat: jnp.ndarray) -> jnp.ndarray:
    """Per-sample energy E(x) for (B, 39) input → (B,)."""
    return jax.vmap(energy)(x_flat.reshape(-1, model_cfg.x_dim))


# ---- Reference (ground-truth) samples ---------------------------------------
# Held-out LJ13 test split, already projected to zero CoM (max|CoM| ≈ 4e-7).
_pt_ref = jnp.array(np.load(Path("../../data/test_split_LJ13-1000.npy")))  # (10000, 39)
_ref_com = _pt_ref.reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim).mean(axis=1)
print(f"Loaded reference samples: {_pt_ref.shape}  "
      f"max|CoM|={jnp.abs(_ref_com).max():.2e}  (should be ≈ 0)")

_REF_R = get_pairwise_distances(_pt_ref)      # (10000, 78)
_REF_E = get_energies(_pt_ref)                 # (10000,)
print(f"Reference energy: mean={float(_REF_E.mean()):.2f}  "
      f"[min={float(_REF_E.min()):.2f}, max={float(_REF_E.max()):.2f}]")

# ---- Comparison plot helpers ------------------------------------------------
# Two series everywhere: the thing under test vs. the ground-truth reference.
_style_axis = style_axis   # local alias: this notebook styles its own axes too


def plot_distance_comparison(ax, r_test, label_test, r_ref=_REF_R, bins=150):
    """Pairwise-distance density: test vs reference, on shared bin edges."""
    return plot_value_comparison(
        ax, r_test, r_ref, "inter-particle distance $r_{ij}$",
        label_test, bins=bins, clip_at_zero=True,   # distances are non-negative
    )


def plot_energy_comparison(ax, e_test, label_test, e_ref=_REF_E, bins=150):
    """Energy density: test vs reference, on shared bin edges."""
    return plot_value_comparison(
        ax, e_test, e_ref, "energy $E(x)$", label_test, bins=bins,
    )

# %% [markdown]
# ## 3. Tempering Problem for LJ13
#
# Geometric path from $\mathcal N(0,I)$ (zero CoM) to $\pi$.

# %%
def logpdf_ref(x: jnp.ndarray) -> jnp.ndarray:
    """log N(x; 0, I) up to a constant (reference on the CoM-zero subspace)."""
    return -0.5 * jnp.sum(x ** 2)


def logpdf_final_target(x: jnp.ndarray) -> jnp.ndarray:
    """log π(x) = -E(x) (unnormalized, LJ13 log-density)."""
    return -energy(x)


def rvs_ref(key: jnp.ndarray) -> jnp.ndarray:
    """Draw one sample from N(0, I) projected to zero CoM."""
    return project_com(jax.random.normal(key, (model_cfg.x_dim,)))

# %%
# pt_problem is built here (rather than at the SMC call in Section 9) so
# Section 7's HMC demo can also use it.
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
@jax.jit
def ou_forward_sample_com(key: jnp.ndarray, x0: jnp.ndarray, t: float) -> jnp.ndarray:
    """OU noising of a single sample, projected to zero CoM."""
    xt = ou_forward_sample(key, x0, t)
    return project_com(xt)

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
_x_init = jnp.zeros((1, model_cfg.x_dim))
_t_init = jnp.zeros((1, 1))
_b_init = jnp.zeros((1, 1))
params = score_net.init(subkey, _x_init, _t_init, _b_init)
n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"EGNN initialized: {n_params:,} parameters")
print(f"n_particles={model_cfg.n_particles}, spatial_dim={model_cfg.spatial_dim}, "
      f"hidden_nf={model_cfg.hidden_nf}, n_layers={model_cfg.n_layers}")

# %% [markdown]
# ## 6. CoM-Projected Backward SDE
#
# Backward SDE with a zero-CoM projection after every step.

# %%
def backward_sde_batch_com(
    key: jnp.ndarray,
    control_fn,
    beta: jnp.ndarray,
    energy_fn,
    dim: int,
    n_particles: int,
    T: float,
    n_steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batched backward SDE: all n_particles processed together in one scan."""
    times, dts = power_time_grid(T, n_steps, train_cfg.grid_power)

    key, subkey = jax.random.split(key)
    x_T = project_com(jax.random.normal(subkey, (n_particles, dim)))  # (K, dim)
    init_log_w = 0.5 * jnp.sum(x_T ** 2, axis=-1)                    # (K,)

    def step_fn(carry, t_dt):
        x_n, log_w, rng = carry  # x_n: (K, dim), log_w: (K,)
        t_n, dt = t_dt
        rng, subkey = jax.random.split(rng)
        lam = jnp.exp(-dt)
        one_minus_lam = -jnp.expm1(-dt)          # accurate for the tiny steps near 0
        sigma_sq = -jnp.expm1(-2.0 * dt)

        control = control_fn(x_n, t_n, beta)              # (K, dim)
        eps = jax.random.normal(subkey, x_n.shape)         # (K, dim)

        # Exponential-integrator backward step, then project
        mu_bwd = lam * x_n + one_minus_lam * control      # (K, dim)
        x_prev = mu_bwd + jnp.sqrt(sigma_sq) * eps         # (K, dim)
        x_prev = project_com(x_prev)                        # (K, dim)

        # Sum over dim axis to get per-particle scalar log-densities
        log_p = -0.5 * jnp.sum((x_n - lam * x_prev) ** 2, axis=-1) / sigma_sq  # (K,)
        log_q = -0.5 * jnp.sum((x_prev - mu_bwd) ** 2, axis=-1) / sigma_sq      # (K,)

        return (x_prev, log_w + log_p - log_q, rng), None

    (x0, log_w, _), _ = jax.lax.scan(step_fn, (x_T, init_log_w, key), (times, dts))

    # Terminal correction: log π_β(x_0) for each particle
    energies = jax.vmap(energy_fn)(x0)                                       # (K,)
    log_w = log_w - (1.0 - beta) * 0.5 * jnp.sum(x0 ** 2, axis=-1) - beta * energies
    return x0, log_w


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def _run_backward_sde(key, params, beta, energy_fn, dim, n_particles, T, n_steps):
    """JIT-compiled CoM-projected backward SDE runner."""
    score_clip = train_cfg.score_clip

    def control(x, t, b):
        # x: (K, dim) — full particle batch; one EGNN call per SDE step
        K = x.shape[0]
        s = score_net.apply(
            params,
            x,                                  # (K, dim)
            jnp.full((K, 1), t),               # (K, 1)
            jnp.full((K, 1), b),               # (K, 1)
        )                                       # (K, dim)
        if score_clip > 0:
            s = jnp.clip(s, -score_clip, score_clip)
        return 2.0 * s

    return backward_sde_batch_com(key, control, beta, energy_fn, dim, n_particles, T, n_steps)


# Quick sanity check with random parameters
key, subkey = jax.random.split(key)
_x0_demo, _lw_demo = _run_backward_sde(
    subkey, params, jnp.array(1.0), energy,
    model_cfg.x_dim, 50, train_cfg.T, train_cfg.n_sde_steps_train,
)
_path_ess = normalized_ess(_lw_demo)
print(f"Backward SDE demo (β=1, 50 particles, {train_cfg.n_sde_steps_train} steps):")
print(f"  log_w range: [{float(_lw_demo.min()):.2f}, {float(_lw_demo.max()):.2f}]")
print(f"  Normalized ESS: {float(_path_ess):.3f}  (1.0=perfect)")
_com_demo = _x0_demo.reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim).mean(axis=1)
print(f"  Max |CoM| in samples: {jnp.abs(_com_demo).max():.2e}  (should be ≈ 0)")

# %% [markdown]
# ## 7. CoM-Projected HMC Rejuvenation
#
# HMC moves at rung $k$ with the kernel SMC tuned for that rung (Section 9),
# projected to zero CoM.

# %%
_demo_beta, _demo_t_idx = jnp.array(0.1), 1
_demo_mcmc = KERNELS[train_cfg.mcmc_kernel](pt_problem, _demo_beta, _demo_t_idx)

key, subkey = jax.random.split(key)
_pts = jax.vmap(project_com)(jax.random.normal(subkey, (20, model_cfg.x_dim)))
key, k_init, k_adapt, k_move = jax.random.split(key, 4)
_demo_x = batch_init_extend(k_init, _pts, _demo_mcmc)
_demo_params = _demo_mcmc.adapt_params(k_adapt, _demo_x)

_pts_mala, _mala_acc, _ = rejuvenate(
    k_move, _pts, pt_problem, _demo_beta, _demo_t_idx, _demo_params, 5,
    project_fn=project_com, return_energy=True, kernel=train_cfg.mcmc_kernel,
)
print(f"HMC demo (β={float(_demo_beta):.3f}, rung 1, 5 steps): "
      f"acceptance rate = {float(_mala_acc):.3f}")

# %% [markdown]
# ## 8. IS Bridging Between Adjacent Temperatures
#
# $$\log w_{k\to k+1}(x)=(\beta_{k+1}-\beta_k)\bigl(\tfrac12\|x\|^2-E(x)\bigr)$$

# %%
@jax.jit
def is_bridge_log_weights(samples, beta_k, beta_next):
    """IS log-weights to bridge π_{β_k} → π_{β_{k+1}}."""
    delta_beta = beta_next - beta_k
    energies = jax.vmap(energy)(samples)
    norms_sq = jnp.sum(samples ** 2, axis=-1)
    return delta_beta * (-energies + 0.5 * norms_sq)


@jax.jit
def normalize_log_weights(log_w: jnp.ndarray) -> jnp.ndarray:
    """Shift a block of log-weights to mean weight 1 (i.e. logsumexp = log n)."""
    n = log_w.shape[0]
    total = logsumexp(log_w)
    return jnp.where(
        jnp.isfinite(total),
        log_w - total + jnp.log(n),
        jnp.zeros_like(log_w),
    )

# %% [markdown]
# ## 9. Temperature Ladder via SMC
#
# Adaptive SMC gives the ladder $\{\beta_k\}$, one particle block per rung (the
# buffer seeds) and one tuned HMC kernel per rung (reused for rejuvenation).

# %%
# pt_problem was built in Section 3.

# Early-load the checkpoint before SMC, so that SMC can be skipped on resume
_ckpt_data = None
if resume and CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, "rb") as f:
        _ckpt_data = pickle.load(f)
    print(f"Checkpoint found: outer_step={_ckpt_data['outer_step']}, "
          f"will restore temperature ladder, buffers, and the HMC kernel")

if _ckpt_data is None:
    key, subkey = jax.random.split(key)
    print("Running SMC to discover temperature ladder …")
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
    print(f"SMC done in {time.time()-t0:.1f}s")
    print(f"Temperature ladder ({len(betas_list)} levels): {[f'{b:.3f}' for b in betas_list]}")
    print(f"logZ estimates: {logZ_dict}")

    # Per-rung HMC step size adapted by SMC.
    _eps_by_rung = mcmc_params_by_rung.scaled_cov.eps
    print("HMC step size (ε) adapted by SMC, per rung:")
    for _k, _eps_k in enumerate(_eps_by_rung, start=1):
        print(f"  rung {_k} (β={betas_list[_k]:.3f}): ε={float(_eps_k):.5g}")

    betas = jnp.array(betas_list)
    M = len(betas_list) - 1
    print(f"M={M} non-reference temperature levels")

    # Project the SMC particles onto the zero-CoM subspace.
    smc_states = jax.vmap(jax.vmap(project_com))(smc_particles)  # (n_levels, N, 39)
    assert smc_states.shape[0] == len(betas_list), (
        f"expected one particle block per rung, got {smc_states.shape[0]} "
        f"blocks for {len(betas_list)} rungs"
    )
    _smc_com = smc_states.reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim).mean(axis=1)
    print(f"SMC particles: {smc_states.shape}  max|CoM| after projection: "
          f"{jnp.abs(_smc_com).max():.2e}  (should be ≈ 0)")
else:
    betas_list = _ckpt_data["betas_list"]
    betas = jnp.array(betas_list)
    M = len(betas_list) - 1
    mcmc_params_by_rung = _ckpt_data["mcmc_params_by_rung"]
    print(f"Restored temperature ladder ({len(betas_list)} levels): {[f'{b:.3f}' for b in betas_list]}")
    print(f"M={M} non-reference temperature levels")

# %% [markdown]
# ## 10. SMC Seed Quality at $\beta = 1$
#
# The top SMC rung against the reference samples.

# %%
if _ckpt_data is None:
    _smc_top = smc_states[-1]                      # (N, 39) at β = β_M = 1
    _r_smc = get_pairwise_distances(_smc_top)
    _e_smc = get_energies(_smc_top)

    fig, _seed_stats = smc_seed_quality_figure(
        features_test=np.asarray(_r_smc),
        features_ref=np.asarray(_REF_R),
        energy_test=np.asarray(_e_smc),
        energy_ref=np.asarray(_REF_E),
        feature_label="inter-particle distance $r_{ij}$",
        title=(f"SMC seed quality at β={float(betas[-1]):.3f} vs ground truth "
               f"({_smc_top.shape[0]:,} particles)"),
        features_nonneg=True,
    )
    fig.savefig(FIGURES_DIR / "lj13_smc_seed_quality.png", dpi=150)
    plt.close(fig)

    print(f"SMC seed at β={float(betas[-1]):.3f}:")
    print(f"  energy    mean={float(_e_smc.mean()):10.2f}   "
          f"reference mean={float(_REF_E.mean()):10.2f}")
    print(f"  energy  median={float(jnp.median(_e_smc)):10.2f}   "
          f"reference median={float(jnp.median(_REF_E)):10.2f}")
    print(f"  distance  mean={float(_r_smc.mean()):10.4f}   "
          f"reference mean={float(_REF_R.mean()):10.4f}")
    print_seed_quality_report(_seed_stats)
    print(f"  → {FIGURES_DIR / 'lj13_smc_seed_quality.png'}")
else:
    print("Resumed from checkpoint — SMC seed diagnostic skipped (no smc_states).")

# %% [markdown]
# ## 11. Per-Temperature Replay Buffers (Seeded from SMC)
#
# Buffer 0 holds exact $\mathcal N(0,I)$ draws; buffer $k\ge1$ starts with rung
# $k$'s SMC block. The outer loop refreshes buffers $2,\dots,M$ (FIFO).

# %%
if _ckpt_data is None:
    buffers: list[ReplayBuffer] = [
        ReplayBuffer(capacity=train_cfg.buffer_capacity, dim=model_cfg.x_dim,
                     store_grads=True)
        for _ in range(len(betas_list))
    ]

    # Buffer 0: exact projected N(0, I), frozen.
    key, subkey = jax.random.split(key)
    buf0_samples = jax.vmap(project_com)(
        jax.random.normal(subkey, (train_cfg.buffer_capacity, model_cfg.x_dim))
    )
    # At β = 0 the TSM target is 0 whatever ∇E is, so buffer 0 stores zeros
    # instead of evaluating it.
    buffers[0].add(buf0_samples, jnp.zeros(train_cfg.buffer_capacity),
                   jnp.zeros_like(buf0_samples))
    buffers[0].freeze()
    print(f"Buffer 0 (β=0): {buffers[0].size} samples, uniform weights [frozen]")

    # Buffers 1..M: the whole SMC block of the matching rung (zero CoM).
    assert smc_states.shape[1] == train_cfg.buffer_capacity, (
        f"SMC block {smc_states.shape[1]:,} must equal capacity "
        f"{train_cfg.buffer_capacity:,}; ReplayBuffer.add would otherwise write "
        f"duplicate indices and silently corrupt"
    )
    print("Seeding buffers directly from SMC output …")
    t0 = time.time()
    for _k in range(1, len(betas_list)):
        _seed_k = smc_states[_k]
        buffers[_k].add(_seed_k, jnp.zeros(_seed_k.shape[0]),
                        jax.vmap(jax.grad(energy))(_seed_k))
        _frozen = " [frozen]" if _k == 1 else ""
        print(f"  Buffer {_k} (β={float(betas[_k]):.3f}): {buffers[_k].size} samples{_frozen}")

    # Buffer 1 is never written during training (the outer loop fills 2..M), so
    # it stays frozen.
    if len(betas_list) >= 2:
        buffers[1].freeze()
    print(f"Seeding done in {time.time()-t0:.1f}s")
    print(f"Buffers 2..{len(betas_list)-1} additionally receive backward-SDE data "
          f"during training.")
else:
    buffers = _ckpt_data["buffers"]
    print(f"Restored {len(buffers)} replay buffers from checkpoint")
    for _k, _buf in enumerate(buffers):
        print(f"  Buffer {_k} (β={float(betas[_k]):.3f}): {_buf.size} samples")

# %% [markdown]
# ## 12. Parallel Data Generation (Outer Loop)
#
# For every rung $k=1,\dots,M-1$ in parallel: backward SDE, resampling, HMC
# rejuvenation, bridge weights to rung $k+1$.

# %%
@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def _data_gen_parallel(
    sde_keys,            # (M_active,) PRNG keys
    resample_keys,       # (M_active,) PRNG keys
    mala_keys,           # (M_active,) PRNG keys
    energy_fn,           # static callable
    dim,                 # static int
    n_particles,         # static int
    T,                   # static float
    n_sde_steps,         # static int
    n_mala_steps,        # static int
    pt_problem,          # static -- PTProblem, same object SMC ran with
    params,              # network pytree
    betas_active,        # (M_active,)
    betas_next,          # (M_active,)
    t_idx_active,        # (M_active,) rung indices k = 1 … M-1, matching betas_active
    mcmc_params_active,  # MALAParameters pytree (leading axis M_active), SMC-tuned per rung
):
    """Run full data-generation pipeline for all active temperature levels via vmap."""
    score_clip = train_cfg.score_clip

    def process_one(sde_key, resample_key, mala_key, beta_k, beta_next,
                     t_idx_k, mcmc_params_k):
        def control(x, t, b):
            # x: (K, dim) — full particle batch; one EGNN call per SDE step
            K = x.shape[0]
            s = score_net.apply(
                params,
                x,                              # (K, dim)
                jnp.full((K, 1), t),           # (K, 1)
                jnp.full((K, 1), b),           # (K, 1)
            )                                   # (K, dim)
            if score_clip > 0:
                s = jnp.clip(s, -score_clip, score_clip)
            return 2.0 * s

        # 1. Backward SDE
        x0_batch, log_w_path = backward_sde_batch_com(
            sde_key, control, beta_k, energy_fn, dim, n_particles, T, n_sde_steps
        )
        # 2. Systematic resample
        resample_idx = systematic_resample(resample_key, log_w_path, n_particles)
        x0_resampled = x0_batch[resample_idx]
        # 3. HMC rejuvenation with the SMC-tuned kernel of rung k. The trap
        # depends on the CoM, so the energy is re-evaluated after each
        # re-projection.
        x0_rejuv, mala_acc, energies = rejuvenate(
            mala_key, x0_resampled, pt_problem, beta_k, t_idx_k,
            mcmc_params_k, n_mala_steps, project_fn=project_com,
            return_energy=True, kernel=train_cfg.mcmc_kernel,
        )
        # 4. IS bridge weights; `energies` come from the final HMC state
        delta_beta = beta_next - beta_k
        norms_sq = jnp.sum(x0_rejuv ** 2, axis=-1)
        bridge_lw = delta_beta * (-energies + 0.5 * norms_sq)
        path_ess = normalized_ess(log_w_path)
        # 5. ∇E once per particle, stored with it in the buffer
        grads_rejuv = jax.vmap(jax.grad(energy_fn))(x0_rejuv)
        return x0_rejuv, bridge_lw, path_ess, mala_acc, grads_rejuv

    return jax.vmap(process_one)(
        sde_keys, resample_keys, mala_keys, betas_active, betas_next,
        t_idx_active, mcmc_params_active,
    )

# %% [markdown]
# ## 13. Training Step (Inner Loop)
#
# TSM loss on all buffers jointly, with the target clipped per particle; one
# Adam step.

# %%
def clip_per_particle_norm(x_flat: jnp.ndarray, max_norm: float) -> jnp.ndarray:
    """Clamp each particle's own vector norm to max_norm, direction preserved."""
    shape = x_flat.shape
    xr = x_flat.reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim)
    norms = jnp.linalg.norm(xr, axis=-1, keepdims=True)
    clip_coeff = jnp.minimum(max_norm / (norms + 1e-6), 1.0)
    return (xr * clip_coeff).reshape(shape)


# Diagnostic: per-particle |∇E| at equilibrium vs. at random Gaussian states.
_grad_fn_demo = jax.vmap(jax.grad(energy))
key, subkey = jax.random.split(key)
_gt_probe = jax.vmap(project_com)(jax.random.normal(subkey, (2000, model_cfg.x_dim)))
_n_probe = jnp.linalg.norm(
    _grad_fn_demo(_gt_probe).reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim), axis=-1
)
print(f"Per-particle |∇E| on random CoM-zero configs: "
      f"mean={_n_probe.mean():.2f}  p99={jnp.percentile(_n_probe, 99):.2f}  "
      f"max={_n_probe.max():.2e}  (target_clip_norm={train_cfg.target_clip_norm})")

# %%
optimizer = optax.chain(
    optax.clip_by_global_norm(train_cfg.grad_clip),
    optax.adam(train_cfg.learning_rate),
)
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, xg_all, t_all, beta_all, key):
    """Loss, gradient, and Adam update."""
    _d = xg_all.shape[1] // 2
    x0_all, g_all = xg_all[:, :_d], xg_all[:, _d:]
    target_clip = train_cfg.target_clip_norm
    score_clip = train_cfg.score_clip

    def loss_fn(p):
        B_total = x0_all.shape[0]

        # Forward OU noising, then project to zero CoM
        noise_keys = jax.random.split(key, B_total)
        sigma_t = jnp.sqrt(1.0 - jnp.exp(-2.0 * t_all))
        means = jnp.exp(-t_all[:, None]) * x0_all
        eps = jax.vmap(jax.random.normal, in_axes=(0, None))(noise_keys, (x0_all.shape[1],))
        xt_all = means + sigma_t[:, None] * eps
        xt_all = jax.vmap(project_com)(xt_all)  # project to zero CoM

        # TSM targets (stop_gradient: no backprop through energy Hessian)
        def phi_one(x0_i, grad_e, t_i, beta_i):
            lambda_t = jnp.exp(-t_i)
            phi = lambda_t * beta_i * (x0_i - grad_e)
            # Per-particle norm clipping
            if target_clip > 0:
                phi = clip_per_particle_norm(phi, target_clip)
            return phi

        targets = jax.lax.stop_gradient(
            jax.vmap(phi_one)(x0_all, g_all, t_all, beta_all)
        )

        # EGNN predictions — fully batched
        preds = score_net.apply(p, xt_all, t_all[:, None], beta_all[:, None])
        if score_clip > 0:
            preds = jnp.clip(preds, -score_clip, score_clip)

        return jnp.mean(jnp.sum((preds - targets) ** 2, axis=-1))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state_new = optimizer.update(grads, opt_state)
    params_new = optax.apply_updates(params, updates)
    return params_new, opt_state_new, loss

# %% [markdown]
# ## 14. Periodic Evaluation
#
# The final-evaluation figure, every `eval_every` outer steps.

# %%
def periodic_eval_plot(params, outer_step):
    """Plot the three evaluation sample sets (distances and energy)."""
    global key

    key, subkey = jax.random.split(key)
    x0_eval, lw_eval = _run_backward_sde(
        subkey, params, betas[-1], energy,
        model_cfg.x_dim, train_cfg.n_eval_particles,
        train_cfg.T, train_cfg.n_sde_steps_eval,
    )
    ess_eval = float(normalized_ess(lw_eval))

    # (1) Raw backward-SDE output: the network alone, no correction.
    x_raw = x0_eval

    # (2) Same particles, IS-corrected by the path-space weights.
    key, subkey = jax.random.split(key)
    resamp_idx = systematic_resample(subkey, lw_eval, len(lw_eval))
    x_resamp = x0_eval[resamp_idx]

    # (3) buffer[M]: the training data at β=1, drawn the way the inner loop
    # draws it — systematic resampling on the stored IS weights.
    if buffers[M].is_ready(1):
        key, subkey = jax.random.split(key)
        n_buf = min(train_cfg.n_eval_particles, buffers[M].size)
        x_buf, _ = buffers[M].sample(subkey, n_buf)
        buf_ess = float(normalized_ess(buffers[M].log_weights[: buffers[M].size]))
    else:
        x_buf, buf_ess = None, float("nan")

    sample_sets = [
        ("Backward SDE (raw, no IS)", "SDE raw", x_raw,
         f"unweighted, {x_raw.shape[0]:,} particles"),
        ("Backward SDE + IS resample", "SDE + IS", x_resamp,
         f"path ESS={ess_eval:.3f}  (≈{ess_eval * x_raw.shape[0]:,.0f} eff.)"),
        (f"buffer[M]  β={float(betas[M]):.3f}", "buffer[M]", x_buf,
         f"buffer ESS={buf_ess:.3f}" if x_buf is not None else "empty"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for _col, (_label, _short, _x, _sub) in enumerate(sample_sets):
        ax_r, ax_e = axes[0, _col], axes[1, _col]
        if _x is None:
            for _ax in (ax_r, ax_e):
                _ax.text(0.5, 0.5, "buffer[M] empty", ha="center", va="center",
                         transform=_ax.transAxes, color=C_INK_2, fontsize=10)
                _ax.set_axis_off()
            continue
        _r = get_pairwise_distances(_x)
        _e = get_energies(_x)
        plot_distance_comparison(ax_r, _r, _short)
        plot_energy_comparison(ax_e, _e, _short)
        ax_r.set_title(f"{_label}\n{_sub}", color=C_INK, fontsize=10)
        ax_e.set_title(f"Energy — {_short}", color=C_INK, fontsize=10)

    fig.suptitle(
        f"Step {outer_step+1}: three candidate sample sets vs ground truth  "
        f"(path ESS={ess_eval:.3f})",
        fontsize=12, color=C_INK,
    )
    plt.tight_layout()
    fname = EVAL_FIGURES_DIR / f"eval_step{outer_step+1:04d}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  [eval] step {outer_step+1}  model ESS={ess_eval:.3f}  → {fname}")

# %% [markdown]
# ## 15. Training Loop
#
# Each outer step generates data at every rung; `n_inner_steps` Adam steps then
# train on all buffers jointly.

# %%
start_outer_step = 0
loss_history = []

if _ckpt_data is not None:
    params = _ckpt_data["params"]
    opt_state = _ckpt_data["opt_state"]
    start_outer_step = _ckpt_data["outer_step"]
    loss_history = _ckpt_data.get("loss_history", [])
    key = jax.random.wrap_key_data(_ckpt_data["key"])
    print(f"Resumed from outer step {start_outer_step}")
else:
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            n_attempts = f.read().count("Attempt")
    else:
        n_attempts = 0
    with open(LOG_PATH, "a") as f:
        f.write(f"\nAttempt {n_attempts + 1}\n")
    print(f"Starting fresh (Attempt {n_attempts + 1})")

# %%
print(f"\nStarting training: outer steps {start_outer_step} → {train_cfg.num_outer_steps}")
train_start = time.time()

for outer_step in range(start_outer_step, train_cfg.num_outer_steps):
    step_start = time.time()

    # ---- Outer loop: parallel SMC data generation ----
    M_active = len(betas_list) - 2   # levels k = 1 … M-1
    outer_ess_list = []

    if M_active > 0:
        all_keys = jax.random.split(key, 3 * M_active + 1)
        key = all_keys[0]
        sde_keys    = all_keys[1            : M_active + 1]
        resamp_keys = all_keys[M_active + 1 : 2 * M_active + 1]
        mala_keys   = all_keys[2 * M_active + 1:]

        # Rungs k = 1 … M-1; mcmc_params_by_rung[k-1] belongs to rung k.
        t_idx_active = jnp.arange(1, M)
        mcmc_params_active = jax.tree_util.tree_map(
            lambda leaf: leaf[:M_active], mcmc_params_by_rung
        )

        x0_rejuv_all, bridge_lw_all, path_ess_all, _, grads_all = _data_gen_parallel(
            sde_keys, resamp_keys, mala_keys,
            energy,
            model_cfg.x_dim, train_cfg.n_particles,
            train_cfg.T, train_cfg.n_sde_steps_train,
            train_cfg.n_mala_rejuv,
            pt_problem,
            params, betas[1:-1], betas[2:],
            t_idx_active, mcmc_params_active,
        )

        outer_ess_list = path_ess_all.tolist()

        for i in range(M_active):
            # Normalize each block to mean weight 1 before adding it.
            buffers[i + 2].add(x0_rejuv_all[i], normalize_log_weights(bridge_lw_all[i]),
                               grads_all[i])

    # ---- Inner loop: gradient descent ----
    all_buffers_ready = all(
        buffers[k].is_ready(train_cfg.batch_size) for k in range(len(betas_list))
    )
    if all_buffers_ready:
        inner_loss_acc = jnp.zeros(())
        for inner_step in range(train_cfg.n_inner_steps):
            x0_batches, t_batches, beta_batches = [], [], []
            for k in range(len(betas_list)):
                key, subkey = jax.random.split(key)
                # rows are [x0 | ∇E(x0)]: one gather for both
                x0_k, _ = buffers[k].sample_with_grads(subkey, train_cfg.batch_size)

                key, subkey = jax.random.split(key)
                # t = t_min + (T - t_min) u^t_power: concentrated near t_min
                u_k = jax.random.uniform(subkey, (train_cfg.batch_size,))
                t_k = train_cfg.t_min + (train_cfg.T - train_cfg.t_min) * u_k ** train_cfg.t_power
                x0_batches.append(x0_k)
                t_batches.append(t_k)
                beta_batches.append(jnp.full(train_cfg.batch_size, betas[k]))

            x0_all = jnp.concatenate(x0_batches, axis=0)
            t_all  = jnp.concatenate(t_batches,  axis=0)
            beta_all = jnp.concatenate(beta_batches, axis=0)

            key, subkey = jax.random.split(key)
            params, opt_state, loss_val = train_step(
                params, opt_state, x0_all, t_all, beta_all, subkey
            )
            inner_loss_acc = inner_loss_acc + loss_val

        mean_loss = float(inner_loss_acc / train_cfg.n_inner_steps)
        loss_history.append(mean_loss)
    else:
        mean_loss = float("nan")

    step_time = time.time() - step_start

    # ---- Logging ----
    if (outer_step + 1) % train_cfg.log_every == 0:
        ess_str = "  ".join(
            f"β{k+1}={ess:.3f}" for k, ess in enumerate(outer_ess_list)
        ) if outer_ess_list else "—"
        buf_ess_str = "  ".join(
            f"β{k}={float(normalized_ess(buffers[k].log_weights[:buffers[k].size])):.3f}"
            if buffers[k].size > 0 else f"β{k}=—"
            for k in range(len(betas_list))
        )
        elapsed = time.time() - train_start
        log_line = (
            f"[step {outer_step+1:4d}/{train_cfg.num_outer_steps}] "
            f"loss={mean_loss:.4f}  path_ESS=[{ess_str}]  buf_ESS=[{buf_ess_str}]  "
            f"step={step_time:.1f}s  elapsed={elapsed:.1f}s"
        )
        print(log_line)
        with open(LOG_PATH, "a") as f:
            f.write(log_line + "\n")

    # ---- Checkpoint ----
    if (outer_step + 1) % train_cfg.save_every == 0:
        ckpt = {
            "params": params,
            "opt_state": opt_state,
            "outer_step": outer_step + 1,
            "loss_history": loss_history,
            "key": jax.random.key_data(key),
            "betas_list": betas_list,
            "mcmc_params_by_rung": mcmc_params_by_rung,
            "buffers": buffers,
            "_pt_ref": _pt_ref,
        }
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump(ckpt, f)
        print(f"  Checkpoint saved at step {outer_step+1}")

    # ---- Periodic evaluation ----
    if train_cfg.eval_every is not None and (outer_step + 1) % train_cfg.eval_every == 0:
        periodic_eval_plot(params, outer_step)

print(f"\nTraining complete in {time.time()-train_start:.1f}s")

# %% [markdown]
# ## 16. Final Evaluation
#
# Backward SDE at $\beta=1$ with `n_sde_steps_eval` steps, scored as three
# sample sets: raw SDE output (`sde_raw`), the same particles resampled on their
# path weights (`sde_is`), and `buffer[M]` (`buffer`).

# %%
print("Running final evaluation …")
key, subkey = jax.random.split(key)
x0_eval, lw_eval = _run_backward_sde(
    subkey, params, betas[-1], energy,
    model_cfg.x_dim, train_cfg.n_eval_samples,
    train_cfg.T, train_cfg.n_sde_steps_eval,
)
ess_eval = float(normalized_ess(lw_eval))
print(f"Evaluation path-space ESS (β=1): {ess_eval:.3f}")

# %% [markdown]
# ### Three candidate definitions of "our method's samples"

# %%
# (1) Raw backward-SDE output: the network alone, no correction.
x_raw = x0_eval

# (2) Same particles, IS-corrected by the path-space weights.
key, subkey = jax.random.split(key)
resamp_idx = systematic_resample(subkey, lw_eval, len(lw_eval))
x_resamp = x0_eval[resamp_idx]

# (3) buffer[M]: the training data at β=1, drawn the way the inner loop draws it
# — systematic resampling on the stored IS weights.
if buffers[M].is_ready(1):
    key, subkey = jax.random.split(key)
    n_buf = min(train_cfg.n_eval_samples, buffers[M].size)
    x_buf, _ = buffers[M].sample(subkey, n_buf)
    buf_ess = float(normalized_ess(buffers[M].log_weights[: buffers[M].size]))
else:
    x_buf, buf_ess = None, float("nan")

# (long label for panel headers, short label for tables, samples, subtitle)
sample_sets = [
    ("Backward SDE (raw, no IS)", "SDE raw", x_raw,
     f"unweighted, {x_raw.shape[0]:,} particles"),
    ("Backward SDE + IS resample", "SDE + IS", x_resamp,
     f"path ESS={ess_eval:.3f}  (≈{ess_eval * x_raw.shape[0]:,.0f} eff.)"),
    (f"buffer[M]  β={float(betas[M]):.3f}", "buffer[M]", x_buf,
     f"buffer ESS={buf_ess:.3f}" if x_buf is not None else "empty"),
]

# %%
# Columns: the three candidate sample sets, plus the loss curve.
fig, axes = plt.subplots(2, 4, figsize=(21, 8))

_summary = []
for _col, (_label, _short, _x, _sub) in enumerate(sample_sets):
    ax_r, ax_e = axes[0, _col], axes[1, _col]
    if _x is None:
        for _ax in (ax_r, ax_e):
            _ax.text(0.5, 0.5, "buffer[M] empty", ha="center", va="center",
                     transform=_ax.transAxes, color=C_INK_2, fontsize=10)
            _ax.set_axis_off()
        continue
    _r = get_pairwise_distances(_x)
    _e = get_energies(_x)
    plot_distance_comparison(ax_r, _r, _short)
    _frac = plot_energy_comparison(ax_e, _e, _short)
    ax_r.set_title(f"{_label}\n{_sub}", color=C_INK, fontsize=10)
    ax_e.set_title(f"Energy — {_short}", color=C_INK, fontsize=10)
    _summary.append((_short, _e, _r, _frac))

# Training loss curve (own axis — shares no units with either statistic)
ax_loss = axes[0, 3]
if loss_history:
    ax_loss.plot(loss_history, color=C_TEST, linewidth=1.5)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("outer step", color=C_INK_2, fontsize=10)
    ax_loss.set_ylabel("DDPM loss (sum over dim)", color=C_INK_2, fontsize=10)
else:
    ax_loss.text(0.5, 0.5, "No loss history", ha="center", va="center",
                 transform=ax_loss.transAxes, color=C_INK_2, fontsize=10)
ax_loss.set_title("Training loss", color=C_INK, fontsize=11)
_style_axis(ax_loss)
axes[1, 3].set_axis_off()

fig.suptitle(
    f"LJ13 final evaluation — β={float(betas[-1]):.3f}, "
    f"three candidate sample sets vs ground truth",
    fontsize=12, color=C_INK,
)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "lj13_distances_energy_and_loss.png", dpi=150)
plt.close(fig)
print(f"Figure saved to {FIGURES_DIR / 'lj13_distances_energy_and_loss.png'}")

# %%
# Numbers behind the figure. Energy is the statistic to read first.
print(f"\n── Final samples: three candidate definitions vs ground truth ──")
for _lbl, _short, _x, _sub in sample_sets:
    if _x is not None:
        print(f"    {_short:<10} {_lbl:<30} {_x.shape[0]:>7,}×{_x.shape[1]}   {_sub}")
print()
print(f"  {'set':<12}{'E mean':>13}{'E median':>13}{'r mean':>11}{'off-window':>13}")
print("  " + "─" * 62)
for _short, _e, _r, _frac in _summary:
    print(f"  {_short:<12}{float(_e.mean()):>13.2f}{float(jnp.median(_e)):>13.2f}"
          f"{float(_r.mean()):>11.4f}{_frac:>12.2%}")
print("  " + "─" * 62)
print(f"  {'reference':<12}{float(_REF_E.mean()):>13.2f}"
      f"{float(jnp.median(_REF_E)):>13.2f}{float(_REF_R.mean()):>11.4f}{'—':>13}")
print()

# %% [markdown]
# ### Reported metrics: Distance-𝒲₂, Energy-𝒲₂, Geometric-𝒲₂
#
# Written to `results/lj13/[seedN/]<run_id>/metrics.json`.

# %%
_ref_np = np.asarray(_pt_ref)
_ref_r_np, _ref_e_np = np.asarray(_REF_R), np.asarray(_REF_E)
_metric_rng = np.random.default_rng(SEED)

# Each set's record goes to results/<target>/<run_id>/metrics.json.
RUN_IDS = {"SDE raw": "sde_raw", "SDE + IS": "sde_is", "buffer[M]": "buffer"}

reported_metrics = {}
print(f"── Reported metrics vs {_ref_np.shape[0]:,} reference samples (lower is better) ──")
print(f"  {'set':<12}{'n':>8}{'Distance-W2':>14}{'Energy-W2':>14}{'Geometric-W2':>15}")
print("  " + "─" * 63)
for _lbl, _short, _x, _sub in sample_sets:
    if _x is None:
        print(f"  {_short:<12}{'empty':>8}")
        continue
    _xn = np.asarray(_x)
    if _xn.shape[0] > _ref_np.shape[0]:
        _xn = _xn[_metric_rng.choice(_xn.shape[0], _ref_np.shape[0], replace=False)]
    _t0 = time.time()
    _e = np.asarray(get_energies(_xn))
    _m = {
        # w2_pooled = Distance-W2, w2_energy = Energy-W2.
        "w2_pooled": distance_w2(np.asarray(get_pairwise_distances(_xn)), _ref_r_np),
        "w2_energy": energy_w2(_e, _ref_e_np),
        "geometric_w2": geometric_w2(
            _xn, _ref_np, model_cfg.n_particles, model_cfg.spatial_dim
        ),
        # The gate on the three above: far off means the sampler is in the
        # wrong region and the distances say little.
        "energy_mean": float(_e[np.isfinite(_e)].mean()) if np.isfinite(_e).any() else None,
        "energy_mean_ref": float(_ref_e_np[np.isfinite(_ref_e_np)].mean()),
        "n_eval_samples": int(_xn.shape[0]),
        "n_ref_samples": int(_ref_np.shape[0]),
        # Path-space ESS of the one SDE run behind "SDE raw" and "SDE + IS";
        # null for buffer[M], whose block-normalised weights estimate nothing.
        "ess": None if _short == "buffer[M]" else ess_eval,
        "path_ess": ess_eval,
        "buffer_ess": buf_ess,
    }
    reported_metrics[_short] = _m
    write_record(
        RESULTS_DIR, NOTEBOOK_NAME, RUN_IDS[_short], _m, seed=SEED,
        config={"model": model_cfg, "training": train_cfg,
                "betas": [float(b) for b in betas_list]},
        notes=f"Sample set: {_lbl} ({_sub}).",
    )
    print(f"  {_short:<12}{_xn.shape[0]:>8,}{_m['w2_pooled']:>14.4f}"
          f"{_m['w2_energy']:>14.4f}{_m['geometric_w2']:>15.4f}"
          f"   ({time.time() - _t0:.0f}s)")
print(f"\nRecords written under {RESULTS_DIR}/<run_id>/metrics.json")

# %%
# Per-coordinate statistics (should have zero CoM)
com_resamp = x_resamp.reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim).mean(axis=1)
print(f"\nCoM statistics of model samples:")
print(f"  Mean |CoM|: {jnp.abs(com_resamp).mean():.2e}  (should be ≈ 0)")

# Temperature ladder summary — one series, so one colour and no legend box;
# the title names it. Values are direct-labelled since there are only a handful.
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(betas_list)), betas_list, color=C_TEST, width=0.6)
for _k, _b in enumerate(betas_list):
    ax.text(_k, float(_b), f"{float(_b):.3f}", ha="center", va="bottom",
            fontsize=8, color=C_INK_2)
ax.set_xlabel("temperature level $k$", color=C_INK_2, fontsize=10)
ax.set_ylabel("$\\beta_k$", color=C_INK_2, fontsize=10)
ax.set_title(f"Temperature ladder ({len(betas_list)} levels)", color=C_INK, fontsize=11)
ax.set_ylim(0, 1.12)
_style_axis(ax)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "lj13_temperature_ladder.png", dpi=150)
plt.close(fig)
print(f"Temperature ladder figure saved.")
print(f"Ladder: {[f'{b:.3f}' for b in betas_list]}")
