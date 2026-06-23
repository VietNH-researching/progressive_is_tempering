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
# # Sequential Score Matching via SMC Temperature Bridging — DW4
#
# Trains a $\beta$-conditioned EGNN score network on the double-well 4-particle
# system (DW4, 8-dimensional, translation-invariant) using an SMC-discovered
# temperature ladder. All operations are projected to the zero centre-of-mass
# subspace throughout.
#
# Run top-to-bottom after a kernel restart. Set `resume = True` to continue
# from a checkpoint.

# %% [markdown]
# ## Imports and Seeds

# %%
from __future__ import annotations

import time
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from jax_pdf import DW4

from progressive_is_tempering.net.egnn_jax import EGNNDynamics
from progressive_is_tempering.sde import ou_forward_sample
from progressive_is_tempering.resampling import systematic_resample, normalized_ess
from progressive_is_tempering.buffer import ReplayBuffer

SEED = 0
key = jax.random.key(SEED)

print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# %% [markdown]
# ## 0. Configuration

# %%
@dataclass
class ModelConfig:
    """EGNN architecture hyperparameters."""
    n_particles: int = 4
    spatial_dim: int = 2
    x_dim: int = 8           # n_particles * spatial_dim
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
    """All training / sampling hyperparameters (DW4-specific values)."""
    # SDE
    T: float = 4.0
    t_min: float = 0.01
    n_sde_steps_train: int = 100
    n_sde_steps_eval: int = 2000

    # Outer loop
    num_outer_steps: int = 10
    n_particles: int = 10_000     # K: particles per backward SDE run

    # Inner loop
    n_inner_steps: int = 50_000
    batch_size: int = 512        # B per temperature

    # Buffer
    buffer_capacity: int = 50_000

    # MALA rejuvenation
    n_mala_rejuv: int = 20_000       # MALA steps per particle per outer step
    mala_h: float = 0.2

    # Clipping (DW4 values from IMPLEMENTATION_EGNN.md)
    score_clip: float = 1e4
    target_clip_norm: float = 20.0

    # Optimiser
    learning_rate: float = 3e-4
    grad_clip: float = 1.0

    # MPPT SMC (temperature ladder)
    mppt_n_particles: int = 100_000
    mppt_local_ess_ratio: float = 0.2
    mppt_n_mcmc_steps: int = 10

    # Checkpointing / logging
    log_every: int = 1
    save_every: int = 1
    eval_every: int = 10
    n_eval_particles: int = 10_000


model_cfg = ModelConfig()
train_cfg = TrainingConfig()

resume: bool = False

NOTEBOOK_NAME = "dw4"
CHECKPOINT_DIR = Path("../../checkpoints") / NOTEBOOK_NAME
LOG_DIR = Path("../../logging")
FIGURES_DIR = Path("../../figures") / NOTEBOOK_NAME
for d in [CHECKPOINT_DIR, LOG_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint.pkl"
LOG_PATH = LOG_DIR / f"{NOTEBOOK_NAME}.txt"

# %% [markdown]
# ## 1. Target Distribution (DW4)
#
# Four particles in 2D, flat representation $x \in \mathbb{R}^8$.
# Energy:
# $$U(x) = \sum_{i<j}\bigl[a(r_{ij}-r_0)^4 + b(r_{ij}-r_0)^2\bigr]$$
# with $a=0.9$, $b=-4$, $r_0=4$, `trap_scale=0` (translation-invariant).
#
# We work on the zero centre-of-mass subspace
# $\{x \in \mathbb{R}^8 : \sum_i x_{i,d} = 0 \text{ for all } d\}$.

# %%
dw4 = DW4(a=0.9, b=-4.0, c=0.0, r0=4.0, trap_scale=0.0, trap_mode='individual', beta=1.0)


def energy(x: jnp.ndarray) -> jnp.ndarray:
    """E(x) = -log π(x) + const.  Expects flat (8,) input."""
    return -dw4(x.reshape(model_cfg.n_particles, model_cfg.spatial_dim))


# Quick sanity check
x_test = jnp.zeros(model_cfg.x_dim)
print(f"DW4 dim: {dw4.dim}")
print(f"energy(0) = {energy(x_test):.4f}")
print(f"∇E norm at 0 = {jnp.linalg.norm(jax.grad(energy)(x_test)):.4f}")

# %% [markdown]
# ## 2. Centre-of-Mass Projection Utilities
#
# All particle states are kept in the zero-CoM subspace throughout.

# %%
def project_com(x: jnp.ndarray) -> jnp.ndarray:
    """Project flat (8,) or batched (..., 8) to zero centre-of-mass.

    Subtracts the per-sample mean particle position in each spatial direction.
    Works for any leading batch dimensions.
    """
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

# %% [markdown]
# ## 3. MPPT Interface for DW4
#
# The geometric path bridges
# $\rho(x) = \mathcal{N}(x;0,I)$ (restricted to CoM=0)
# to $\pi(x) \propto e^{-E(x)}$ via
# $\pi_\beta(x) \propto \exp\{-(1-\beta)\tfrac12\|x\|^2 - \beta E(x)\}$.

# %%
def logpdf_ref(x: jnp.ndarray) -> jnp.ndarray:
    """log N(x; 0, I) up to a constant (reference on the CoM-zero subspace)."""
    return -0.5 * jnp.sum(x ** 2)


def logpdf_final_target(x: jnp.ndarray) -> jnp.ndarray:
    """log π(x) = -E(x) (unnormalized, DW4 log-density)."""
    return -energy(x)


def rvs_ref(key: jnp.ndarray) -> jnp.ndarray:
    """Draw one sample from N(0, I) projected to zero CoM."""
    return project_com(jax.random.normal(key, (model_cfg.x_dim,)))


# %% [markdown]
# ## 4. Forward Process
#
# OU SDE with closed-form marginals:
# $X_t \mid X_0 \sim \mathcal{N}(e^{-t}X_0, \sigma_t^2 I)$,
# $\sigma_t^2 = 1 - e^{-2t}$.
# After noising we project back to zero CoM.

# %%
@jax.jit
def ou_forward_sample_com(key: jnp.ndarray, x0: jnp.ndarray, t: float) -> jnp.ndarray:
    """OU noising of a single sample, projected to zero CoM."""
    xt = ou_forward_sample(key, x0, t)
    return project_com(xt)


# %% [markdown]
# ## 5. Score Network (EGNN)
#
# $\beta$-conditioned $E(n)$-equivariant GNN dynamics network
# $s_\theta(x_t, t, \beta)$ approximating the excess score
# $\nabla_{x_t}\log(p_t^\beta / p_t^\rho)$.
#
# Input: flat coordinates $x_t \in \mathbb{R}^8$, time $t$, temperature $\beta$.
# Interface: **batched** — `(B, 8)`, `(B, 1)`, `(B, 1)` → `(B, 8)`.
# The EGNN output is automatically zero-CoM (velocity = displacement from input,
# CoM-projected inside `EGNNDynamics`).

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
# The backward SDE time-reversal of the OU process, with CoM projection applied
# after each exponential-integrator step. Both the forward OU density
# $p_\rho(x_n \mid x_{n-1})$ and the backward proposal
# $q_\theta(x_{n-1} \mid x_n)$ are evaluated at the projected state.
#
# The control function wraps the EGNN (batched interface) with a batch
# dimension of 1 for single-particle calls inside vmap.

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
    """Batched backward SDE: all n_particles processed together in one scan.

    control_fn(x, t, beta) receives x of shape (n_particles, dim) and returns
    the same shape — one EGNN call per time step rather than one per particle.

    Returns x0 (K, dim) and path-space log-weights (K,).
    """
    dt = T / n_steps
    times = jnp.linspace(T, dt, n_steps)

    lam = jnp.exp(-dt)
    sigma_sq = 1.0 - jnp.exp(-2.0 * dt)

    key, subkey = jax.random.split(key)
    x_T = project_com(jax.random.normal(subkey, (n_particles, dim)))  # (K, dim)
    init_log_w = 0.5 * jnp.sum(x_T ** 2, axis=-1)                    # (K,)

    def step_fn(carry, t_n):
        x_n, log_w, rng = carry  # x_n: (K, dim), log_w: (K,)
        rng, subkey = jax.random.split(rng)

        control = control_fn(x_n, t_n, beta)              # (K, dim)
        eps = jax.random.normal(subkey, x_n.shape)         # (K, dim)

        # Exponential-integrator backward step, then project
        mu_bwd = lam * x_n + (1.0 - lam) * control        # (K, dim)
        x_prev = mu_bwd + jnp.sqrt(sigma_sq) * eps         # (K, dim)
        x_prev = project_com(x_prev)                        # (K, dim)

        # Sum over dim axis to get per-particle scalar log-densities
        log_p = -0.5 * jnp.sum((x_n - lam * x_prev) ** 2, axis=-1) / sigma_sq  # (K,)
        log_q = -0.5 * jnp.sum((x_prev - mu_bwd) ** 2, axis=-1) / sigma_sq      # (K,)

        return (x_prev, log_w + log_p - log_q, rng), None

    (x0, log_w, _), _ = jax.lax.scan(step_fn, (x_T, init_log_w, key), times)

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
print(f"Backward SDE demo (β=1, 50 particles, 100 steps):")
print(f"  log_w range: [{float(_lw_demo.min()):.2f}, {float(_lw_demo.max()):.2f}]")
print(f"  Normalized ESS: {float(_path_ess):.3f}  (1.0=perfect)")
_com_demo = _x0_demo.reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim).mean(axis=1)
print(f"  Max |CoM| in samples: {jnp.abs(_com_demo).max():.2e}  (should be ≈ 0)")

# %% [markdown]
# ## 7. CoM-Projected MALA Rejuvenation
#
# MALA proposal with CoM projection applied immediately after the Langevin step.
# Both the gradient and the MH log-ratio are evaluated at projected states.

# %%
def mala_step_com(
    key: jnp.ndarray,
    x: jnp.ndarray,
    beta: jnp.ndarray,
    energy_fn,
    h: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One MALA step with CoM projection on the proposal."""
    key_prop, key_mh = jax.random.split(key)

    x = project_com(x)  # defensive

    def log_pi(z):
        return -(1.0 - beta) * 0.5 * jnp.sum(z ** 2) - beta * energy_fn(z)

    grad_x = jax.grad(log_pi)(x)

    eps = jax.random.normal(key_prop, x.shape)
    x_prop = x + 0.5 * h ** 2 * grad_x + h * eps
    x_prop = project_com(x_prop)    # project proposal

    grad_xp = jax.grad(log_pi)(x_prop)

    log_pi_x = log_pi(x)
    log_pi_xp = log_pi(x_prop)

    mean_fwd = x + 0.5 * h ** 2 * grad_x
    log_q_fwd = -0.5 * jnp.sum((x_prop - mean_fwd) ** 2) / h ** 2

    mean_bwd = x_prop + 0.5 * h ** 2 * grad_xp
    log_q_bwd = -0.5 * jnp.sum((x - mean_bwd) ** 2) / h ** 2

    log_alpha = log_pi_xp - log_pi_x + log_q_bwd - log_q_fwd

    accepted = jnp.log(jax.random.uniform(key_mh)) < log_alpha
    x_new = jnp.where(accepted, x_prop, x)
    return x_new, accepted


@partial(jax.jit, static_argnums=(3, 5))
def mala_rejuvenate_com(
    key: jnp.ndarray,
    particles: jnp.ndarray,
    beta: jnp.ndarray,
    energy_fn,
    h: jnp.ndarray,
    n_steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply n_steps of CoM-projected MALA to each particle independently."""
    K = particles.shape[0]
    keys = jax.random.split(key, K)

    def rejuvenate_one(key_i, x_init):
        step_keys = jax.random.split(key_i, n_steps)

        def scan_step(x, step_key):
            x_new, acc = mala_step_com(step_key, x, beta, energy_fn, h)
            return x_new, acc.astype(jnp.float32)

        x_final, accepts = jax.lax.scan(scan_step, x_init, step_keys)
        return x_final, jnp.mean(accepts)

    particles_new, accept_rates = jax.vmap(rejuvenate_one)(keys, particles)
    return particles_new, jnp.mean(accept_rates)


# Demonstrate MALA at β=0 (proposals should be accepted with high probability)
key, subkey = jax.random.split(key)
_pts = jax.vmap(project_com)(jax.random.normal(subkey, (20, model_cfg.x_dim)))
key, subkey = jax.random.split(key)
_pts_mala, _mala_acc = mala_rejuvenate_com(
    subkey, _pts, jnp.array(0.0), energy,
    jnp.array(train_cfg.mala_h), 5,
)
print(f"MALA demo (β=0, 5 steps): acceptance rate = {float(_mala_acc):.3f}")

# %% [markdown]
# ## 8. IS Bridging Between Adjacent Temperatures
#
# $$\log w_{k \to k+1}(x) = (\beta_{k+1}-\beta_k)\bigl(-E(x)+\tfrac12\|x\|^2\bigr)$$

# %%
@jax.jit
def is_bridge_log_weights(samples, beta_k, beta_next):
    """IS log-weights to bridge π_{β_k} → π_{β_{k+1}}."""
    delta_beta = beta_next - beta_k
    energies = jax.vmap(energy)(samples)
    norms_sq = jnp.sum(samples ** 2, axis=-1)
    return delta_beta * (-energies + 0.5 * norms_sq)


# %% [markdown]
# ## 9. Temperature Ladder via SMC (MPPT)
#
# Run SMC once before training to discover $\{\beta_k\}_{k=0}^M$ such that
# successive IS bridges have normalized ESS $\geq$ `local_ess_ratio`.

# %%
import sys
sys.path.insert(0, str(Path("../../mppt").resolve()))

from mppt import PTProblem, smc_sampler, pt_manual_runners

mppt_problem = PTProblem(
    rvs_ref=rvs_ref,
    logpdf_ref=logpdf_ref,
    logpdf_final_target=logpdf_final_target,
)

# Early-load checkpoint before SMC/PT/calibration so we can skip them on resume
_ckpt_data = None
if resume and CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, "rb") as f:
        _ckpt_data = pickle.load(f)
    print(f"Checkpoint found: outer_step={_ckpt_data['outer_step']}, "
          f"will restore temperature ladder, buffers, and MALA step sizes")

if _ckpt_data is None:
    key, subkey = jax.random.split(key)
    print("Running SMC to discover temperature ladder …")
    t0 = time.time()
    smc_particles, logZ_dict, betas_list = smc_sampler(
        subkey,
        mppt_problem,
        n_particles=train_cfg.mppt_n_particles,
        engine={"name": "hmc", "refresh_rate": 0.07},
        n_mcmc_steps=train_cfg.mppt_n_mcmc_steps,
        local_ess_ratio=train_cfg.mppt_local_ess_ratio,
        verbose=True,
        runtime_check=False,
        keep_all_states=True,
    )
    print(f"SMC done in {time.time()-t0:.1f}s")
    print(f"Temperature ladder ({len(betas_list)} levels): {[f'{b:.3f}' for b in betas_list]}")
    print(f"logZ estimates: {logZ_dict}")

    # Parallel Tempering refinement
    key, subkey = jax.random.split(key)
    print("Running Parallel Tempering refinement …")
    t0 = time.time()
    pt_runner = pt_manual_runners(
        mppt_problem,
        betas_list,
        n_particles=train_cfg.mppt_n_particles,
        engine={"name": "hmc", "refresh_rate": 0.07},
        explorer={"mode": "adaptive", "niter": 50},
        swapper={"mix": True},
    )
    pt_state, _ = pt_runner(subkey, smc_particles, full_output=False)
    print(f"PT refinement done in {time.time()-t0:.1f}s")

    betas = jnp.array(betas_list)
    M = len(betas_list) - 1
    print(f"M={M} non-reference temperature levels")
else:
    betas_list = _ckpt_data["betas_list"]
    betas = jnp.array(betas_list)
    M = len(betas_list) - 1
    print(f"Restored temperature ladder ({len(betas_list)} levels): {[f'{b:.3f}' for b in betas_list]}")
    print(f"M={M} non-reference temperature levels")

# %% [markdown]
# ## 10. Per-Temperature Replay Buffers
#
# Buffer 0 ($\beta=0$): i.i.d. $\mathcal{N}(0,I)$ projected to zero CoM, frozen.
# Buffer 1 ($\beta_1$): IS-reweighted buffer 0, frozen.
# Buffers $2,\ldots,M$: filled during the outer training loop.

# %%
if _ckpt_data is None:
    buffers: list[ReplayBuffer] = [
        ReplayBuffer(capacity=train_cfg.buffer_capacity, dim=model_cfg.x_dim)
        for _ in range(len(betas_list))
    ]

    # Buffer 0: projected N(0, I), frozen
    key, subkey = jax.random.split(key)
    buf0_samples = jax.vmap(project_com)(
        jax.random.normal(subkey, (train_cfg.n_particles, model_cfg.x_dim))
    )
    buffers[0].add(buf0_samples, jnp.zeros(train_cfg.n_particles))
    buffers[0].freeze()
    print(f"Buffer 0 (β=0): {buffers[0].size} samples, uniform weights [frozen]")

    # Buffer 1: IS-reweighted buffer 0 → β_1, frozen
    if len(betas_list) >= 2:
        lw1 = is_bridge_log_weights(buf0_samples, betas[0], betas[1])
        buffers[1].add(buf0_samples, lw1)
        buffers[1].freeze()
        ess1 = float(normalized_ess(lw1))
        print(f"Buffer 1 (β={float(betas[1]):.3f}): {buffers[1].size} samples, ESS={ess1:.3f} [frozen]")

    print(f"Buffers 2..{len(betas_list)-1} start empty; filled during training.")
else:
    buffers = _ckpt_data["buffers"]
    print(f"Restored {len(buffers)} replay buffers from checkpoint")
    for _k, _buf in enumerate(buffers):
        print(f"  Buffer {_k} (β={float(betas[_k]):.3f}): {_buf.size} samples")

# %% [markdown]
# ## 11. MALA Step Size Calibration
#
# Bisection over $h$ to achieve acceptance rate $\in [60\%,80\%]$ at each $\beta_k$.
# Calibration particles are IS-resampled from $\mathcal{N}(0,I)$ toward $\pi_{\beta_k}$.

# %%
_CALIB_N = 2_000
_CALIB_MALA_STEPS = 3_000
_CALIB_N_BISECT = 14
_CALIB_H_LO = 1e-3
_CALIB_H_HI = 2.0
_CALIB_TARGET_LO = 0.60
_CALIB_TARGET_HI = 0.80


def calibrate_mala_h_for_level(key, particles, beta_k,
                                target_lo=_CALIB_TARGET_LO,
                                target_hi=_CALIB_TARGET_HI,
                                h_lo=_CALIB_H_LO, h_hi=_CALIB_H_HI,
                                n_bisect=_CALIB_N_BISECT):
    """Bisect on h to achieve MALA acceptance in [target_lo, target_hi]."""
    beta_arr = jnp.array(beta_k)
    acc = 0.0
    for _ in range(n_bisect):
        h_mid = (h_lo + h_hi) / 2.0
        key, subkey = jax.random.split(key)
        _, acc_arr = mala_rejuvenate_com(
            subkey, particles, beta_arr, energy,
            jnp.array(h_mid), _CALIB_MALA_STEPS,
        )
        acc = float(acc_arr)
        if acc < target_lo:
            h_hi = h_mid
        elif acc > target_hi:
            h_lo = h_mid
        else:
            break
    return (h_lo + h_hi) / 2.0, acc


# %%
if _ckpt_data is None:
    print("Calibrating per-level MALA step sizes …")
    key, subkey = jax.random.split(key)
    _ref_calib = jax.vmap(project_com)(
        jax.random.normal(subkey, (_CALIB_N, model_cfg.x_dim))
    )

    mala_hs_list = []
    for _k, _beta_k in enumerate(betas_list):
        _lw_calib = is_bridge_log_weights(_ref_calib, betas[0], betas[_k])
        key, subkey = jax.random.split(key)
        _idx_calib = systematic_resample(subkey, _lw_calib, _CALIB_N)
        _particles_calib = _ref_calib[_idx_calib]

        key, subkey = jax.random.split(key)
        _h_k, _acc_k = calibrate_mala_h_for_level(
            subkey, _particles_calib, float(_beta_k)
        )
        mala_hs_list.append(_h_k)
        print(f"  β_{_k}={float(_beta_k):.3f}:  h={_h_k:.5f},  acceptance={_acc_k:.3f}")

    mala_hs = jnp.array(mala_hs_list)
    print(f"\nCalibrated step sizes: {[f'{h:.4f}' for h in mala_hs_list]}")
else:
    mala_hs_list = _ckpt_data["mala_hs_list"]
    mala_hs = jnp.array(mala_hs_list)
    print(f"Restored MALA step sizes: {[f'{h:.4f}' for h in mala_hs_list]}")

# %% [markdown]
# ## 12. Parallel Data Generation (Outer Loop)
#
# For each active temperature level $k \in \{1,\ldots,M-1\}$ **in parallel**:
# 1. Backward SDE at $\beta_k$ → $(x_0^{(i)}, \log w_\mathrm{path}^{(i)})$
# 2. Systematic resample → approx. i.i.d. from $\pi_{\beta_k}$
# 3. CoM-projected MALA rejuvenation
# 4. IS bridge weights → $\log w_{k \to k+1}$

# %%
@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def _data_gen_parallel(
    sde_keys,        # (M_active,) PRNG keys
    resample_keys,   # (M_active,) PRNG keys
    mala_keys,       # (M_active,) PRNG keys
    energy_fn,       # static callable
    dim,             # static int
    n_particles,     # static int
    T,               # static float
    n_sde_steps,     # static int
    n_mala_steps,    # static int
    params,          # network pytree
    betas_active,    # (M_active,)
    betas_next,      # (M_active,)
    mala_hs_active,  # (M_active,)
):
    """Run full data-generation pipeline for all active temperature levels via vmap."""
    score_clip = train_cfg.score_clip

    def process_one(sde_key, resample_key, mala_key, beta_k, beta_next, mala_h_k):
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
        # 3. MALA rejuvenation
        x0_rejuv, mala_acc = mala_rejuvenate_com(
            mala_key, x0_resampled, beta_k, energy_fn, mala_h_k, n_mala_steps
        )
        # 4. IS bridge weights
        delta_beta = beta_next - beta_k
        energies = jax.vmap(energy_fn)(x0_rejuv)
        norms_sq = jnp.sum(x0_rejuv ** 2, axis=-1)
        bridge_lw = delta_beta * (-energies + 0.5 * norms_sq)
        path_ess = normalized_ess(log_w_path)
        return x0_rejuv, bridge_lw, path_ess, mala_acc

    return jax.vmap(process_one)(
        sde_keys, resample_keys, mala_keys, betas_active, betas_next, mala_hs_active
    )


# %% [markdown]
# ## 13. Training Step (Inner Loop)
#
# DDPM regression target for DW4:
# $$\phi(x_0,t,\beta) = e^{-t}\,\beta\,(x_0 - \nabla E(x_0))$$
# EGNN is called on the full batch `(M*B, 8)` directly (no per-sample vmap).
# Forward-noised samples are projected to zero CoM before passing to the network.

# %%
optimizer = optax.chain(
    optax.clip_by_global_norm(train_cfg.grad_clip),
    optax.adam(train_cfg.learning_rate),
)
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, x0_all, t_all, beta_all, key):
    """Loss, gradient, and Adam update.

    x0_all: (M*B, 8), t_all: (M*B,), beta_all: (M*B,)
    """
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

        # DDPM targets (stop_gradient: no backprop through energy Hessian)
        def phi_one(x0_i, t_i, beta_i):
            lambda_t = jnp.exp(-t_i)
            grad_e = jax.grad(energy)(x0_i)
            phi = lambda_t * beta_i * (x0_i - grad_e)
            # Norm clipping
            norm = jnp.linalg.norm(phi)
            clip_coeff = jnp.where(
                target_clip > 0,
                jnp.minimum(target_clip / (norm + 1e-6), 1.0),
                1.0,
            )
            return phi * clip_coeff

        targets = jax.lax.stop_gradient(
            jax.vmap(phi_one)(x0_all, t_all, beta_all)
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
# Runs the backward SDE at $\beta=1$ and compares the distribution of pairwise
# inter-particle distances to a ground-truth reference (MPPT particles at $\beta=1$).

# %%
@jax.jit
def get_pairwise_distances(x_flat: jnp.ndarray) -> jnp.ndarray:
    """Return all 6 unique pairwise distances for (N, N*D) shaped input."""
    pos = x_flat.reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim)   # (B, 4, 2)
    diff = pos[:, :, None, :] - pos[:, None, :, :]                            # (B, 4, 4, 2)
    r = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)                         # (B, 4, 4)
    i, j = jnp.triu_indices(model_cfg.n_particles, k=1)                       # 6 unique pairs
    return r[:, i, j]                                                          # (B, 6)


def periodic_eval_plot(params, outer_step, pt_ref_samples=None):
    """Plot pairwise distance KDE vs MPPT ground truth."""
    global key

    key, subkey = jax.random.split(key)
    x0_eval, lw_eval = _run_backward_sde(
        subkey, params, betas[-1], energy,
        model_cfg.x_dim, train_cfg.n_eval_particles,
        train_cfg.T, train_cfg.n_sde_steps_eval,
    )
    ess_eval = float(normalized_ess(lw_eval))

    key, subkey = jax.random.split(key)
    resamp_idx = systematic_resample(subkey, lw_eval, len(lw_eval))
    x_resamp = x0_eval[resamp_idx]

    r_model = get_pairwise_distances(x_resamp)  # (n_eval, 6)

    fig, ax = plt.subplots(figsize=(8, 4))
    for pair_idx in range(6):
        sns.kdeplot(
            x=r_model[:, pair_idx].tolist(), ax=ax,
            alpha=0.5, label=f"pair {pair_idx}",
        )

    if pt_ref_samples is not None:
        pt_x = pt_ref_samples.reshape(-1, model_cfg.x_dim)
        r_ref = get_pairwise_distances(pt_x)
        for pair_idx in range(6):
            sns.kdeplot(
                x=r_ref[:, pair_idx].tolist(), ax=ax,
                linestyle="--", alpha=0.5, color="black",
                label="MPPT ref" if pair_idx == 0 else None,
            )

    ax.set_xlabel("inter-particle distance $r_{ij}$")
    ax.set_ylabel("density")
    ax.set_title(f"Step {outer_step+1}: pairwise distances  ESS={ess_eval:.3f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = FIGURES_DIR / f"eval_step{outer_step+1:04d}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  [eval] step {outer_step+1}  model ESS={ess_eval:.3f}  → {fname}")


# Load pre-computed reference samples
import numpy as _np
_ref_npz = _np.load(Path("../../data/dw4_samples.npz"))
_pt_ref = jnp.array(_ref_npz["particles"])  # (100000, 4, 2); reshaped to (-1, 8) when used
print(f"Loaded reference samples: {_pt_ref.shape}")

# %% [markdown]
# ## 15. Training Loop
#
# ```
# for outer_step in 1 … num_outer_steps:
#     for k = 1, …, M−1:  (parallel via jax.vmap)
#         CoM-projected backward SDE at β_k  →  (x₀, log w_path)
#         systematic resample  →  approx. i.i.d. from π_{β_k}
#         CoM-projected MALA rejuvenation
#         IS bridge weights    →  log w_{k→k+1}
#         buffer[k+1].add(x₀, log w_{k→k+1})
#     for inner_step in 1 … n_inner_steps:
#         sample mini-batches from each buffer
#         project-noised xt, compute DDPM targets
#         Adam step on joint loss via batched EGNN
# ```

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

        x0_rejuv_all, bridge_lw_all, path_ess_all, _ = _data_gen_parallel(
            sde_keys, resamp_keys, mala_keys,
            energy,
            model_cfg.x_dim, train_cfg.n_particles,
            train_cfg.T, train_cfg.n_sde_steps_train,
            train_cfg.n_mala_rejuv,
            params, betas[1:-1], betas[2:],
            mala_hs[1:-1],
        )

        outer_ess_list = path_ess_all.tolist()

        for i in range(M_active):
            buffers[i + 2].add(x0_rejuv_all[i], bridge_lw_all[i])

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
                x0_k, _ = buffers[k].sample(subkey, train_cfg.batch_size)

                key, subkey = jax.random.split(key)
                t_k = jax.random.uniform(
                    subkey, (train_cfg.batch_size,),
                    minval=train_cfg.t_min, maxval=train_cfg.T,
                )
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
            "mala_hs_list": mala_hs_list,
            "buffers": buffers,
            "_pt_ref": _pt_ref,
        }
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump(ckpt, f)
        print(f"  Checkpoint saved at step {outer_step+1}")

    # ---- Periodic evaluation ----
    if (outer_step + 1) % train_cfg.eval_every == 0:
        periodic_eval_plot(params, outer_step, _pt_ref)

print(f"\nTraining complete in {time.time()-train_start:.1f}s")

# %% [markdown]
# ### Reference Samples
#
# Pre-computed ground-truth samples are loaded from `data/dw4_samples.npz`
# (set once in section 9 above). Nothing to re-run here.

# %%
print(f"Reference samples already loaded: {_pt_ref.shape}")

# Check if reference samples are in 0-centered CoM subspace
_ref_com = _pt_ref.reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim).mean(axis=1)
print(f"Max |CoM| in reference samples: {jnp.abs(_ref_com).max():.2e}  (should be ≈ 0)")

# %% [markdown]
# ## 16. Final Evaluation
#
# Run the backward SDE at $\beta=1$ and compare the distribution of pairwise
# inter-particle distances to the MPPT ground-truth reference.

# %%
print("Running final evaluation …")
key, subkey = jax.random.split(key)
x0_eval, lw_eval = _run_backward_sde(
    subkey, params, betas[-1], energy,
    model_cfg.x_dim, train_cfg.n_eval_particles,
    train_cfg.T, train_cfg.n_sde_steps_eval,
)
ess_eval = float(normalized_ess(lw_eval))
print(f"Evaluation path-space ESS (β=1): {ess_eval:.3f}")

# %%
key, subkey = jax.random.split(key)
resamp_idx = systematic_resample(subkey, lw_eval, len(lw_eval))
x_resamp = x0_eval[resamp_idx]

r_model = get_pairwise_distances(x_resamp)   # (n_eval, 6)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: all pairwise distances flattened into one array
r_model_flat = r_model.flatten().tolist()
sns.kdeplot(x=r_model_flat, ax=axes[0], fill=True, color="blue", alpha=0.4, label="model")

if _pt_ref is not None:
    r_ref = get_pairwise_distances(_pt_ref.reshape(-1, model_cfg.x_dim))
    r_ref_flat = r_ref.flatten().tolist()
    sns.kdeplot(x=r_ref_flat, ax=axes[0], fill=False, color="red", label="MPPT ref")

axes[0].set_xlabel("inter-particle distance $r_{ij}$")
axes[0].set_ylabel("density")
axes[0].set_title(f"Pairwise distances (ESS={ess_eval:.3f})")
axes[0].legend()

# Right: training loss curve
if loss_history:
    axes[1].plot(loss_history)
    axes[1].set_xlabel("outer step")
    axes[1].set_ylabel("DDPM loss (sum over dim)")
    axes[1].set_title("Training loss")
    axes[1].set_yscale("log")
else:
    axes[1].text(0.5, 0.5, "No loss history", ha="center", va="center",
                 transform=axes[1].transAxes)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "dw4_distances_and_loss.png", dpi=150)
plt.close(fig)
print(f"Figure saved to {FIGURES_DIR / 'dw4_distances_and_loss.png'}")

# %%
# Per-coordinate statistics (should have zero CoM)
com_resamp = x_resamp.reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim).mean(axis=1)
print(f"\nCoM statistics of model samples:")
print(f"  Mean |CoM|: {jnp.abs(com_resamp).mean():.2e}  (should be ≈ 0)")

# Temperature ladder summary
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(betas_list)), betas_list, alpha=0.7)
ax.set_xlabel("Temperature level $k$")
ax.set_ylabel("$\\beta_k$")
ax.set_title(f"Temperature ladder ({len(betas_list)} levels)")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "dw4_temperature_ladder.png", dpi=150)
plt.close(fig)
print(f"Temperature ladder figure saved.")
print(f"Ladder: {[f'{b:.3f}' for b in betas_list]}")
