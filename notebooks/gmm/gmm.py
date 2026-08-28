# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sequential Score Matching via SMC Temperature Bridging — GMM
#
# Trains a $\beta$-conditioned score network on a 16-dimensional factorised GMM
# using an SMC-discovered temperature ladder. Run top-to-bottom after a kernel
# restart. Set `resume = True` to continue from a checkpoint.

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
from scipy import stats

from progressive_is_tempering.targets import make_gmm_target
from progressive_is_tempering.net.score_network import ScoreNetwork
from progressive_is_tempering.sde import ou_forward_sample
from progressive_is_tempering.resampling import systematic_resample, normalized_ess
from progressive_is_tempering.sampler import backward_sde_batch
from progressive_is_tempering.mala import mala_rejuvenate
from progressive_is_tempering.buffer import ReplayBuffer
from progressive_is_tempering.loss import ddpm_loss_single_temp

# Deterministic seeds
SEED = 0
key = jax.random.key(SEED)

print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# %% [markdown]
# ## 0. Configuration
#
# `ModelConfig` holds score network architecture parameters; `TrainingConfig`
# holds all training and sampling hyperparameters.

# %%
@dataclass
class ModelConfig:
    """Score network architecture and embedding hyperparameters."""
    x_dim: int = 16
    hidden_dim: int = 256
    n_blocks: int = 3
    output_init_scale: float = 1e-4
    time_embedding_dim: int = 16
    beta_embedding_dim: int = 8
    time_freq_min: float = 1.0
    time_freq_max: float = 10.0
    beta_freq_min: float = 0.5
    beta_freq_max: float = 5.0


@dataclass
class TrainingConfig:
    """All training / sampling hyperparameters."""
    # SDE
    T: float = 4.0
    t_min: float = 0.01
    n_sde_steps_train: int = 100    # backward SDE steps during training
    n_sde_steps_eval: int = 2000    # backward SDE steps during evaluation

    # Outer loop
    num_outer_steps: int = 50
    n_particles: int = 10_000       # K: particles per backward SDE run

    # Inner loop
    n_inner_steps: int = 500
    batch_size: int = 512           # B per temperature

    # Buffer
    buffer_capacity: int = 50_000

    # MALA rejuvenation
    n_mala_rejuv: int = 20          # MALA steps per outer step
    mala_h: float = 0.2

    # IS clipping / score clipping (setting = 0: disabled for GMM)
    score_clip: float = 0.0
    target_clip_norm: float = 0.0

    # Optimiser
    learning_rate: float = 1e-3
    grad_clip: float = 1.0

    # MPPT SMC (temperature ladder)
    mppt_n_particles: int = 10_000
    mppt_local_ess_ratio: float = 0.5
    mppt_n_mcmc_steps: int = 10

    # Checkpointing / logging
    log_every: int = 1              # outer steps between log lines
    save_every: int = 10            # outer steps between checkpoint saves
    eval_every: int = 10            # outer steps between evaluation runs
    n_eval_particles: int = 100_000 # particles for evaluation plots


# Instantiate configs — edit these to tune the experiment
model_cfg = ModelConfig()
train_cfg = TrainingConfig()

# Resume or start fresh
resume: bool = False

# Paths
NOTEBOOK_NAME = "gmm"
CHECKPOINT_DIR = Path("../../checkpoints") / NOTEBOOK_NAME
LOG_DIR = Path("../../logging")
FIGURES_DIR = Path("../../figures") / NOTEBOOK_NAME
EVAL_FIGURES_DIR = FIGURES_DIR / "eval"
for d in [CHECKPOINT_DIR, LOG_DIR, FIGURES_DIR, EVAL_FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint.pkl"
LOG_PATH = LOG_DIR / f"{NOTEBOOK_NAME}.txt"

# %% [markdown]
# ## 1. Target Distribution
#
# Factorised 16-dimensional GMM:
# $$\pi(x)=\prod_{i=1}^{d}\bigl[0.2\,\mathcal{N}(x_i;\,-1.5,\,0.5^2)
#   +0.8\,\mathcal{N}(x_i;\,2.5,\,0.5^2)\bigr]$$
# Equivalently $\pi(x)\propto e^{-E(x)}$ with
# $E(x)=-\sum_i\log[\cdots]$.

# %%
target = make_gmm_target(
    dim=model_cfg.x_dim,
    means=(-1.5, 2.5),
    weights=(0.2, 0.8),
    std=0.5,
)

# Quick sanity check
key, subkey = jax.random.split(key)
x_test = target.rvs_ref(subkey)
print(f"target dim: {target.dim}")
print(f"log_prob(0) = {target.log_prob(jnp.zeros(target.dim)):.4f}")
print(f"energy(0)   = {target.energy(jnp.zeros(target.dim)):.4f}")

# %% [markdown]
# ## 2. Geometric Temperature Path
#
# $$\pi_\beta(x)\propto\exp\!\bigl\{-(1-\beta)\tfrac12\|x\|^2-\beta\,E(x)\bigr\},
#   \quad\beta\in[0,1]$$
#
# $\beta=0$: standard Gaussian $\mathcal{N}(0,I)$. $\;\beta=1$: target $\pi$.
# Score: $\nabla_x\log\pi_\beta(x)=-(1-\beta)\,x-\beta\,\nabla_x E(x)$.

# %%
def log_pi_beta(x: jnp.ndarray, beta: float) -> jnp.ndarray:
    """log π_β(x) = -(1-β)½‖x‖² - β E(x)  (unnormalized)."""
    return -(1.0 - beta) * 0.5 * jnp.sum(x ** 2) - beta * target.energy(x)


# Verify: at β=0 it is the standard Gaussian log-prob (up to constant)
x0 = jnp.zeros(target.dim)
print(f"log_pi_0(0) should be 0: {log_pi_beta(x0, 0.0):.4f}")
print(f"log_pi_1(0): {log_pi_beta(x0, 1.0):.4f} (= log_prob(0) = {target.log_prob(x0):.4f})")

# %% [markdown]
# ## 3. Temperature Ladder via SMC
#
# `smc_sampler` (MPPT) runs SMC to find $\{\beta_k\}_{k=0}^M$ such that
# successive IS weights stay well-behaved (local ESS ratio $\geq0.5$).
# One round of Parallel Tempering then refines the population.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("../../mppt").resolve()))

from mppt import PTProblem, smc_sampler, pt_manual_runners

mppt_problem = PTProblem(
    rvs_ref=target.rvs_ref,
    logpdf_ref=target.logpdf_ref,
    logpdf_final_target=target.logpdf_final_target,
)

# Run SMC to discover the temperature ladder
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

# %%
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
# One PT iteration to refine SMC output
pt_state, _ = pt_runner(subkey, smc_particles, full_output=False)
print(f"PT refinement done in {time.time()-t0:.1f}s")

betas = jnp.array(betas_list)
M = len(betas_list) - 1  # number of non-reference levels (β_1, …, β_M = 1)
print(f"M={M} non-reference temperature levels")

# %% [markdown]
# ## 4. Forward Process
#
# Ornstein–Uhlenbeck SDE with closed-form marginals:
# $$dX_t=-X_t\,dt+\sqrt{2}\,dW_t,\quad
#   X_t\mid X_0\sim\mathcal{N}\!\bigl(e^{-t}X_0,\,(1-e^{-2t})I\bigr)$$
#
# At $t=T=4$ the marginal is numerically $\mathcal{N}(0,I)$ (verified by Shapiro–Wilk).

# %%
# Noise N target samples to time T and verify marginal is N(0,I) via Shapiro-Wilk (per dim)
key, subkey = jax.random.split(key)
_x0s = jax.vmap(lambda k: target.rvs_ref(k))(jax.random.split(subkey, 500))
key, subkey = jax.random.split(key)
_xTs = jax.vmap(lambda k, x: ou_forward_sample(k, x, train_cfg.T))(
    jax.random.split(subkey, 500), _x0s
)
_sw_stats = [stats.shapiro(_xTs[:, d].tolist()) for d in range(target.dim)]
_p_vals = [r.pvalue for r in _sw_stats]
print(f"Shapiro-Wilk at t=T={train_cfg.T} (500 samples, {target.dim} dims): "
      f"min p={min(_p_vals):.3f}, median p={sorted(_p_vals)[len(_p_vals)//2]:.3f}")

# %% [markdown]
# ## 5. Score Network
#
# A $\beta$-conditioned residual MLP $s_\theta(x_t,t,\beta)$ approximates
# the excess score over the reference Gaussian:
# $$s_\theta(x_t,t,\beta)\approx\nabla_{x_t}\log\frac{p_t^\beta(x_t)}{p_t^\rho(x_t)}$$
#
# Input: $[x_t,\,\mathrm{emb}(t),\,\mathrm{emb}(\beta)]\in\mathbb{R}^{16+16+8}$;
# 3-block ResMLP with hidden dim 256.

# %%
score_net = ScoreNetwork(
    x_dim=model_cfg.x_dim,
    hidden_dim=model_cfg.hidden_dim,
    n_blocks=model_cfg.n_blocks,
    output_init_scale=model_cfg.output_init_scale,
    time_embedding_dim=model_cfg.time_embedding_dim,
    beta_embedding_dim=model_cfg.beta_embedding_dim,
    time_freq_min=model_cfg.time_freq_min,
    time_freq_max=model_cfg.time_freq_max,
    beta_freq_min=model_cfg.beta_freq_min,
    beta_freq_max=model_cfg.beta_freq_max,
)

key, subkey = jax.random.split(key)
params = score_net.init(subkey, jnp.zeros(model_cfg.x_dim), 1.0, 0.5)
n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Score network initialized: {n_params:,} parameters")
print(f"Input dim: {model_cfg.x_dim} + {model_cfg.time_embedding_dim} + {model_cfg.beta_embedding_dim} = "
      f"{model_cfg.x_dim + model_cfg.time_embedding_dim + model_cfg.beta_embedding_dim}")

# %% [markdown]
# ## 6. Backward SDE Sampler
#
# Time-reversal of the OU process:
# $$dY_t=\bigl(-Y_t+2\,s_\theta(Y_t,t,\beta)\bigr)dt+\sqrt{2}\,dW_t,
#   \quad Y_T\sim\mathcal{N}(0,I)$$
#
# Discretised via Euler–Maruyama from $T$ to $0$.

# %%
@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def _run_backward_sde(key, params, beta, energy_fn, dim, n_particles, T, n_steps):
    """JIT-compiled backward SDE runner.

    params and beta are JAX arrays — their values can change between calls
    without triggering a retrace.  energy_fn, dim, n_particles, T, and n_steps
    are static: they determine the compiled XLA graph shape and must be
    Python-level constants.
    """
    score_clip = train_cfg.score_clip
    def control(x, t, b):
        s = score_net.apply(params, x, t, b)
        if score_clip > 0:  # Python-level branch; evaluated at trace time
            s = jnp.clip(s, -score_clip, score_clip)
        return 2.0 * s
    return backward_sde_batch(key, control, beta, energy_fn, dim, n_particles, T, n_steps)


# %% [markdown]
# ## 7. Path-Space Importance Weights
#
# Log-weight correcting the mismatch between the backward path measure
# $Q_\theta$ and the true OU forward measure $P_\rho$:
# $$\log w=\sum_n\bigl[\log p_\rho(x_n\mid x_{n-1})
#   -\log q_\theta(x_{n-1}\mid x_n)\bigr]
#   -(1-\beta)\tfrac12\|x_0\|^2-\beta\,E(x_0)+\tfrac12\|x_T\|^2$$
# Zero when $s_\theta$ is exact and $\Delta t\to0$.

# %%
# Demonstrate path-space IS weight computation (at random params, weights will be non-zero)
key, subkey = jax.random.split(key)
x0_demo_batch, lw_demo = _run_backward_sde(
    subkey, params, betas[-1], target.energy,
    model_cfg.x_dim, 50,
    train_cfg.T, train_cfg.n_sde_steps_train,
)
path_ess = normalized_ess(lw_demo)
print(f"Backward SDE demo (β=1, 50 particles, 100 steps):")
print(f"  log_w range: [{float(lw_demo.min()):.2f}, {float(lw_demo.max()):.2f}]")
print(f"  Normalized ESS: {float(path_ess):.3f}  (1.0=perfect)")

# %% [markdown]
# ## 8. DDPM Regression Loss
#
# The score network is trained by denoising regression against:
# $$\phi(x_0,t,\beta)=e^{-t}\,\beta\,\bigl(x_0-\nabla_x E(x_0)\bigr)$$
# Loss: $\mathbb{E}\bigl[\|s_\theta(x_t,t,\beta)-\phi(x_0,t,\beta)\|^2\bigr]$,
# with $x_t$ drawn from the OU marginal. At $\beta=0$ the target is zero.

# %%
# Demonstrate target computation
key, subkey = jax.random.split(key)
x0_ex = target.rvs_ref(subkey)
t_ex = 1.0
beta_ex = 0.7

grad_e = target.grad_energy(x0_ex)
lambda_t = jnp.exp(-jnp.array(t_ex))
phi = lambda_t * beta_ex * (x0_ex - grad_e)
print(f"Demo DDPM target at β={beta_ex}, t={t_ex}:")
print(f"  ‖x0‖={jnp.linalg.norm(x0_ex):.3f}, ‖∇E‖={jnp.linalg.norm(grad_e):.3f}")
print(f"  ‖φ‖={jnp.linalg.norm(phi):.3f},  λ_t={float(lambda_t):.3f}")

# %% [markdown]
# ## 9. MALA Rejuvenation
#
# Diversifies near-duplicate particles after resampling while preserving
# $\pi_\beta$ as the exact stationary distribution:
# $$x'=x+\tfrac{h^2}{2}\nabla_x\log\pi_\beta(x)+h\,\varepsilon,
#   \quad\varepsilon\sim\mathcal{N}(0,I)$$
# followed by Metropolis–Hastings acceptance.

# %%
# Demonstrate MALA on reference Gaussian samples at β=0
key, subkey = jax.random.split(key)
x_init_demo = jax.random.normal(subkey, (20, model_cfg.x_dim))
key, subkey = jax.random.split(key)
x_mala_demo, mala_acc = mala_rejuvenate(
    subkey, x_init_demo, jnp.array(0.0), target.energy,
    h=train_cfg.mala_h, n_steps=5
)
print(f"MALA demo (β=0, 5 steps): acceptance rate = {float(mala_acc):.3f}")

# %% [markdown]
# ## 10. IS Bridging Between Adjacent Temperatures
#
# Reweight particles from $\pi_{\beta_k}$ to $\pi_{\beta_{k+1}}$:
# $$\log w_{k\to k+1}(x)=(\beta_{k+1}-\beta_k)\bigl(\tfrac12\|x\|^2-E(x)\bigr)$$

# %%
# Demonstrate IS bridging
@jax.jit
def is_bridge_log_weights(samples, beta_k, beta_next):
    """IS log-weights to bridge from π_{β_k} to π_{β_{k+1}}."""
    delta_beta = beta_next - beta_k
    energies = jax.vmap(target.energy)(samples)
    norms_sq = jnp.sum(samples ** 2, axis=-1)
    return delta_beta * (-energies + 0.5 * norms_sq)


# %%
@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def _data_gen_parallel(
    sde_keys,        # (M_active,) typed PRNG keys, one per level
    resample_keys,   # (M_active,) typed PRNG keys
    mala_keys,       # (M_active,) typed PRNG keys
    energy_fn,       # static callable — E(x) → scalar
    dim,             # static int
    n_particles,     # static int
    T,               # static float
    n_sde_steps,     # static int
    n_mala_steps,    # static int
    params,          # network pytree — shared across levels, not batched
    betas_active,    # (M_active,)  β_k  for k = 1 … M-1
    betas_next,      # (M_active,)  β_{k+1}
    mala_hs_active,  # (M_active,)  per-level MALA step sizes
):
    """Run the full data-generation pipeline for all active temperature levels
    simultaneously via jax.vmap.

    Each level is independent: same params, different β_k and PRNG keys.
    The buffer writes (Python-level mutations) are handled by the caller.

    Returns
    -------
    x0_rejuv_all  : (M_active, n_particles, dim)
    bridge_lw_all : (M_active, n_particles)
    path_ess_all  : (M_active,)
    mala_acc_all  : (M_active,)
    """
    score_clip = train_cfg.score_clip  # Python constant; branch resolved at trace time

    def process_one(sde_key, resample_key, mala_key, beta_k, beta_next, mala_h_k):
        def control(x, t, b):
            s = score_net.apply(params, x, t, b)
            if score_clip > 0:
                s = jnp.clip(s, -score_clip, score_clip)
            return 2.0 * s

        # 1. Backward SDE
        x0_batch, log_w_path = backward_sde_batch(
            sde_key, control, beta_k, energy_fn, dim, n_particles, T, n_sde_steps
        )
        # 2. Systematic resample
        resample_idx = systematic_resample(resample_key, log_w_path, n_particles)
        x0_resampled = x0_batch[resample_idx]
        # 3. MALA rejuvenation
        x0_rejuv, mala_acc = mala_rejuvenate(
            mala_key, x0_resampled, beta_k, energy_fn, mala_h_k, n_mala_steps
        )
        # 4. IS bridge weights (inlined to avoid passing through a nested-JIT boundary)
        delta_beta = beta_next - beta_k
        energies = jax.vmap(energy_fn)(x0_rejuv)
        norms_sq = jnp.sum(x0_rejuv ** 2, axis=-1)
        bridge_lw = delta_beta * (-energies + 0.5 * norms_sq)
        # Path-space ESS
        path_ess = normalized_ess(log_w_path)
        return x0_rejuv, bridge_lw, path_ess, mala_acc

    return jax.vmap(process_one)(
        sde_keys, resample_keys, mala_keys, betas_active, betas_next, mala_hs_active
    )


# %% [markdown]
# ## 11. Per-Temperature Replay Buffers
#
# Each level $k$ maintains a fixed-capacity FIFO buffer of $(x,\log w)$ pairs.
# Buffer 0 ($\beta=0$): i.i.d. $\mathcal{N}(0,I)$, frozen.
# Buffer 1 ($\beta_1$): IS-reweighted buffer 0 via $w_{0\to1}$, frozen.
# Buffers $2,\ldots,M$: filled during the outer loop.

# %%
# Initialize all replay buffers
buffers: list[ReplayBuffer] = [
    ReplayBuffer(capacity=train_cfg.buffer_capacity, dim=model_cfg.x_dim)
    for _ in range(len(betas_list))
]

# Buffer 0: i.i.d. N(0, I) (frozen)
key, subkey = jax.random.split(key)
buf0_samples = jax.random.normal(subkey, (train_cfg.n_particles, model_cfg.x_dim))
buffers[0].add(buf0_samples, jnp.zeros(train_cfg.n_particles))
buffers[0].freeze()
print(f"Buffer 0 (β=0): {buffers[0].size} samples, uniform weights [frozen]")

# Buffer 1: IS-reweighted buffer 0 → β_1 (frozen)
if len(betas_list) >= 2:
    lw1 = is_bridge_log_weights(buf0_samples, betas[0], betas[1])
    buffers[1].add(buf0_samples, lw1)
    buffers[1].freeze()
    ess1 = float(normalized_ess(lw1))
    print(f"Buffer 1 (β={float(betas[1]):.3f}): {buffers[1].size} samples, ESS={ess1:.3f} [frozen]")

print(f"Buffers 2..{len(betas_list)-1} start empty; filled during training.")

# %% [markdown]
# ## 12. MALA Step Size Calibration
#
# Before training, bisection finds a per-level step size $h_k$ such that
# the MALA acceptance rate at $\pi_{\beta_k}$ lies in $[60\%,80\%]$.
# Calibration particles are obtained by IS-resampling $\mathcal{N}(0,I)$
# toward $\pi_{\beta_k}$.

# %%
_CALIB_N = 2_000          # particles used per level during calibration
_CALIB_MALA_STEPS = 30    # MALA steps per bisection trial
_CALIB_N_BISECT = 14      # bisection iterations (2^-14 ≈ 6e-5 resolution)
_CALIB_H_LO = 1e-3
_CALIB_H_HI = 2.0
_CALIB_TARGET_LO = 0.60
_CALIB_TARGET_HI = 0.80


def calibrate_mala_h_for_level(key, particles, beta_k, energy_fn,
                                target_lo=_CALIB_TARGET_LO,
                                target_hi=_CALIB_TARGET_HI,
                                h_lo=_CALIB_H_LO, h_hi=_CALIB_H_HI,
                                n_bisect=_CALIB_N_BISECT):
    """Bisect on h to achieve MALA acceptance in [target_lo, target_hi] at β_k.

    Larger h → lower acceptance, so:
      acc < target_lo  →  h is too big  →  tighten upper bound
      acc > target_hi  →  h is too small →  raise lower bound
    """
    beta_arr = jnp.array(beta_k)
    acc = 0.0
    for _ in range(n_bisect):
        h_mid = (h_lo + h_hi) / 2.0
        key, subkey = jax.random.split(key)
        _, acc_arr = mala_rejuvenate(
            subkey, particles, beta_arr, energy_fn,
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
print("Calibrating per-level MALA step sizes …")
key, subkey = jax.random.split(key)
_ref_calib = jax.random.normal(subkey, (_CALIB_N, model_cfg.x_dim))

mala_hs_list = []
for _k, _beta_k in enumerate(betas_list):
    # IS-resample from N(0,I) toward π_{β_k} for approximate calibration particles
    _lw_calib = is_bridge_log_weights(_ref_calib, betas[0], betas[_k])
    key, subkey = jax.random.split(key)
    _idx_calib = systematic_resample(subkey, _lw_calib, _CALIB_N)
    _particles_calib = _ref_calib[_idx_calib]

    key, subkey = jax.random.split(key)
    _h_k, _acc_k = calibrate_mala_h_for_level(
        subkey, _particles_calib, float(_beta_k), target.energy
    )
    mala_hs_list.append(_h_k)
    print(f"  β_{_k}={float(_beta_k):.3f}:  h={_h_k:.5f},  acceptance={_acc_k:.3f}")

mala_hs = jnp.array(mala_hs_list)
print(f"\nCalibrated step sizes (all {len(mala_hs_list)} levels): "
      f"{[f'{h:.4f}' for h in mala_hs_list]}")

# %% [markdown]
# ## 13. Training Loop
#
# Alternates an outer SMC data-generation pass (all levels in parallel via
# `jax.vmap`) with inner Adam gradient steps on the joint DDPM loss.

# %%
# Optimiser with gradient norm clipping
optimizer = optax.chain(
    optax.clip_by_global_norm(train_cfg.grad_clip),
    optax.adam(train_cfg.learning_rate),
)
opt_state = optimizer.init(params)

# JIT-compiled train step: forward pass, gradient, and Adam update fused into one call
@jax.jit
def train_step(params, opt_state, x0_all, t_all, beta_all, key):
    """Loss, gradient, and Adam update in a single JIT-compiled call.
    x0_all: (M*B, d), t_all: (M*B,), beta_all: (M*B,)
    """
    def loss_fn(p):
        # Noise all samples
        B_total = x0_all.shape[0]
        noise_keys = jax.random.split(key, B_total)
        sigma_t = jnp.sqrt(1.0 - jnp.exp(-2.0 * t_all))
        means = jnp.exp(-t_all[:, None]) * x0_all
        eps = jax.vmap(jax.random.normal, in_axes=(0, None))(noise_keys, (x0_all.shape[1],))
        xt_all = means + sigma_t[:, None] * eps

        # Targets (stop_gradient: no backprop through energy)
        def phi_one(x0_i, t_i, beta_i):
            lambda_t = jnp.exp(-t_i)
            grad_e = jax.grad(target.energy)(x0_i)
            return lambda_t * beta_i * (x0_i - grad_e)

        targets = jax.lax.stop_gradient(
            jax.vmap(phi_one)(x0_all, t_all, beta_all)
        )

        # Predictions
        preds = jax.vmap(lambda x, t, b: score_net.apply(p, x, t, b))(xt_all, t_all, beta_all)

        return jnp.mean(jnp.sum((preds - targets) ** 2, axis=-1))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state_new = optimizer.update(grads, opt_state)
    params_new = optax.apply_updates(params, updates)
    return params_new, opt_state_new, loss


# %%
def periodic_eval_plot(params, outer_step):
    """Plot model 1-D marginal and buffer[M] quality every eval_every outer steps."""
    global key

    # Run backward SDE at β=1 (same settings as end-of-training eval)
    key, subkey = jax.random.split(key)
    x0_eval, lw_eval = _run_backward_sde(
        subkey, params, betas[-1], target.energy,
        model_cfg.x_dim, train_cfg.n_eval_particles,
        train_cfg.T, train_cfg.n_sde_steps_eval,
    )
    ess_eval = float(normalized_ess(lw_eval))
    key, subkey = jax.random.split(key)
    resamp_idx = systematic_resample(subkey, lw_eval, len(lw_eval))
    x_resamp = x0_eval[resamp_idx]

    # True GMM density (matches target definition)
    xs_1d = jnp.linspace(-4, 6, 300)
    _w1, _w2, _mu1, _mu2, _sigma = 0.2, 0.8, -1.5, 2.5, 0.5
    pdf_true = (
        _w1 * jnp.exp(-0.5 * ((xs_1d - _mu1) / _sigma) ** 2) / (_sigma * (2 * jnp.pi) ** 0.5)
        + _w2 * jnp.exp(-0.5 * ((xs_1d - _mu2) / _sigma) ** 2) / (_sigma * (2 * jnp.pi) ** 0.5)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: backward SDE model samples
    axes[0].hist(x_resamp[:, 0].tolist(), bins=60, density=True, alpha=0.4, label="Model (IS resampled)")
    axes[0].plot(xs_1d, pdf_true, "r-", lw=2, label="True GMM")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"Step {outer_step+1}: model 1-D marginal  ESS={ess_eval:.3f}")
    axes[0].legend()

    # Right: buffer[M] weighted samples
    if buffers[M].is_ready(1):
        n_buf = min(train_cfg.n_eval_particles, buffers[M].size)
        key, subkey = jax.random.split(key)
        x_buf, _ = buffers[M].sample(subkey, n_buf)
        axes[1].hist(x_buf[:, 0].tolist(), bins=60, density=True, alpha=0.4, label="buffer[M] (IS resampled)")
        axes[1].plot(xs_1d, pdf_true, "r-", lw=2, label="True GMM")
        axes[1].set_title(f"Step {outer_step+1}: buffer[M] 1-D marginal  β={float(betas[M]):.3f}")
    else:
        axes[1].text(0.5, 0.5, "buffer[M] not yet filled", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_title(f"Step {outer_step+1}: buffer[M] (empty)")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("density")
    axes[1].legend()

    plt.tight_layout()
    fname = EVAL_FIGURES_DIR / f"eval_step{outer_step+1:04d}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  [eval] step {outer_step+1}  model ESS={ess_eval:.3f}  → {fname}")


# %% [markdown]
# ```
# for outer_step in 1 … num_outer_steps:
#     for k = 1, …, M−1:  (parallel via jax.vmap)
#         backward SDE at β_k  →  (x₀, log w_path)
#         systematic resample  →  approx. i.i.d. from π_{β_k}
#         MALA rejuvenation
#         IS bridge weights    →  log w_{k→k+1}
#         buffer[k+1].add(x₀, log w_{k→k+1})
#     for inner_step in 1 … n_inner_steps:
#         sample mini-batches from each buffer
#         concatenate across temperatures
#         Adam step on joint DDPM loss
# ```

# %%
# Resume or start fresh
start_outer_step = 0
loss_history = []

if resume and CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    opt_state = optimizer.init(params)  # re-init opt state from checkpoint
    opt_state = ckpt["opt_state"]
    start_outer_step = ckpt["outer_step"]
    loss_history = ckpt.get("loss_history", [])
    print(f"Resumed from outer step {start_outer_step}")
else:
    # Count attempts in log file
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            content = f.read()
        n_attempts = content.count("Attempt")
    else:
        n_attempts = 0
    with open(LOG_PATH, "a") as f:
        f.write(f"\nAttempt {n_attempts + 1}\n")
    print(f"Starting fresh (Attempt {n_attempts + 1})")

# %%
# ---- Main training loop ----
print(f"\nStarting training: outer steps {start_outer_step} → {train_cfg.num_outer_steps}")
train_start = time.time()

for outer_step in range(start_outer_step, train_cfg.num_outer_steps):
    step_start = time.time()

    # ---- Outer loop: parallel SMC data generation across all temperature levels ----
    # All M_active levels are independent (same params, different β and keys),
    # so we run them simultaneously with jax.vmap inside _data_gen_parallel.
    M_active = len(betas_list) - 2  # levels k = 1 … M-1
    outer_ess_list = []

    if M_active > 0:
        all_keys    = jax.random.split(key, 3 * M_active + 1)
        key         = all_keys[0]
        sde_keys    = all_keys[1            : M_active + 1]
        resamp_keys = all_keys[M_active + 1 : 2 * M_active + 1]
        mala_keys   = all_keys[2 * M_active + 1:]

        x0_rejuv_all, bridge_lw_all, path_ess_all, _ = _data_gen_parallel(
            sde_keys, resamp_keys, mala_keys,
            target.energy,
            model_cfg.x_dim, train_cfg.n_particles,
            train_cfg.T, train_cfg.n_sde_steps_train,
            train_cfg.n_mala_rejuv,
            params, betas[1:-1], betas[2:],
            mala_hs[1:-1],
        )

        outer_ess_list = path_ess_all.tolist()

        # Buffer writes are Python-level mutations; negligible vs. SDE/MALA compute.
        for i in range(M_active):
            buffers[i + 2].add(x0_rejuv_all[i], bridge_lw_all[i])

    # ---- Inner loop: gradient descent ----
    all_buffers_ready = all(
        buffers[k].is_ready(train_cfg.batch_size)
        for k in range(len(betas_list))
    )
    if all_buffers_ready:
        inner_loss_acc = jnp.zeros(())
        for inner_step in range(train_cfg.n_inner_steps):
            # Sample mini-batches from each temperature buffer
            x0_batches, t_batches, beta_batches = [], [], []
            for k in range(len(betas_list)):
                key, subkey = jax.random.split(key)
                x0_k, _ = buffers[k].sample(subkey, train_cfg.batch_size)

                key, subkey = jax.random.split(key)
                t_k = jax.random.uniform(
                    subkey, (train_cfg.batch_size,),
                    minval=train_cfg.t_min, maxval=train_cfg.T
                )
                x0_batches.append(x0_k)
                t_batches.append(t_k)
                beta_batches.append(jnp.full(train_cfg.batch_size, betas[k]))

            # Concatenate across temperatures
            x0_all = jnp.concatenate(x0_batches, axis=0)
            t_all = jnp.concatenate(t_batches, axis=0)
            beta_all = jnp.concatenate(beta_batches, axis=0)

            key, subkey = jax.random.split(key)
            params, opt_state, loss_val = train_step(params, opt_state, x0_all, t_all, beta_all, subkey)
            inner_loss_acc = inner_loss_acc + loss_val

        mean_loss = float(inner_loss_acc / train_cfg.n_inner_steps)
        loss_history.append(mean_loss)
    else:
        mean_loss = float("nan")

    step_time = time.time() - step_start

    # ---- Logging ----
    if (outer_step + 1) % train_cfg.log_every == 0:
        ess_per_level = "  ".join(
            f"β{k+1}={ess:.3f}" for k, ess in enumerate(outer_ess_list)
        ) if outer_ess_list else "—"
        buf_ess_parts = []
        for k in range(len(betas_list)):
            if buffers[k].is_ready(1):
                b_ess = float(normalized_ess(buffers[k].log_weights[: buffers[k].size]))
                buf_ess_parts.append(f"β{k}={b_ess:.3f}")
            else:
                buf_ess_parts.append(f"β{k}=—")
        buf_ess_str = "  ".join(buf_ess_parts)
        elapsed = time.time() - train_start
        log_line = (
            f"[step {outer_step+1:4d}/{train_cfg.num_outer_steps}] "
            f"loss={mean_loss:.4f}  path_ESS=[{ess_per_level}]  "
            f"buf_ESS=[{buf_ess_str}]  "
            f"Elapsed={elapsed:.1f}s"
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
        }
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump(ckpt, f)
        print(f"  Checkpoint saved at step {outer_step+1}")

    # ---- Periodic evaluation ----
    if (outer_step + 1) % train_cfg.eval_every == 0:
        periodic_eval_plot(params, outer_step)

print(f"\nTraining complete in {time.time()-train_start:.1f}s")

# %% [markdown]
# ## 14. Evaluation
#
# Runs the backward SDE at $\beta=1$ with $N=100\,000$ particles and 2000
# steps. Path-space IS weights are used to resample, then the 1-D marginal
# of $x_1$ is compared to the true GMM density via KDE.

# %%
def eval_samples(params, n_eval=None):
    """Draw samples from π via the trained backward SDE at β=1."""
    if n_eval is None:
        n_eval = train_cfg.n_eval_particles
    global key
    key, subkey = jax.random.split(key)
    x0_eval, lw_eval = _run_backward_sde(
        subkey, params, betas[-1], target.energy,
        model_cfg.x_dim, n_eval,
        train_cfg.T, train_cfg.n_sde_steps_eval,
    )
    ess_eval = float(normalized_ess(lw_eval))
    return x0_eval, lw_eval, ess_eval


print("Running evaluation …")
x_eval, lw_eval, ess_eval = eval_samples(params)
print(f"Evaluation ESS (path-space, β=1): {ess_eval:.3f}")

# %%
# 1-D marginal KDE for first coordinate
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Weighted KDE of model samples
key, subkey = jax.random.split(key)
resamp_idx = systematic_resample(subkey, lw_eval, len(lw_eval))
x_resamp = x_eval[resamp_idx]

axes[0].hist(x_resamp[:, 0].tolist(), bins=60, density=True, alpha=0.4, label="Model samples (IS)")
# True GMM marginal
xs_1d = jnp.linspace(-4, 6, 300)
w1, w2 = 0.2, 0.8
mu1, mu2, sigma = -1.5, 2.5, 0.5
pdf_true = (w1 * jnp.exp(-0.5*((xs_1d-mu1)/sigma)**2) / (sigma*(2*jnp.pi)**0.5)
           + w2 * jnp.exp(-0.5*((xs_1d-mu2)/sigma)**2) / (sigma*(2*jnp.pi)**0.5))
axes[0].plot(xs_1d, pdf_true, "r-", lw=2, label="True GMM")
axes[0].set_xlabel("$x_1$")
axes[0].set_ylabel("density")
axes[0].set_title(f"1-D marginal: model vs target (GMM in {model_cfg.x_dim} dimensions)")
axes[0].legend()

# Training loss curve
if loss_history:
    axes[1].plot(loss_history)
    axes[1].set_xlabel("outer step")
    axes[1].set_ylabel("DDPM loss")
    axes[1].set_title("Training loss")
    axes[1].set_yscale("log")
else:
    axes[1].text(0.5, 0.5, "No loss history", ha="center", va="center",
                 transform=axes[1].transAxes)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "gmm_marginal_and_loss.png", dpi=150)
plt.close(fig)
print(f"Figure saved to {FIGURES_DIR / 'gmm_marginal_and_loss.png'}")

# %%
# Per-dimension mean and std
means_eval = jnp.mean(x_resamp, axis=0)
stds_eval = jnp.std(x_resamp, axis=0)
# True GMM mean and std
true_mean = w1 * mu1 + w2 * mu2
true_var = w1*(mu1**2 + sigma**2) + w2*(mu2**2 + sigma**2) - true_mean**2
true_std = true_var ** 0.5

print(f"\nTrue GMM per-dim: mean={true_mean:.4f}, std={true_std:.4f}")
print(f"Model per-dim (mean over 16 dims): mean={float(jnp.mean(means_eval)):.4f}, "
      f"std={float(jnp.mean(stds_eval)):.4f}")

# %%
# Temperature ladder ESS diagnostics
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(betas_list)), betas_list, alpha=0.7)
ax.set_xlabel("Temperature level k")
ax.set_ylabel("β_k")
ax.set_title(f"Temperature ladder ({len(betas_list)} levels)")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "gmm_temperature_ladder.png", dpi=150)
plt.close(fig)
print(f"Temperature ladder figure saved.")
print(f"\nTemperature ladder: {[f'{b:.3f}' for b in betas_list]}")

# %% [markdown]
# ## 15. Extended Diagnostics
#
# Estimates several scalar functionals of $X\sim\pi$ from IS-resampled model
# samples and compares to ground-truth values computed analytically or via
# 1-D numerical integration / root-finding.
#
# | Functional | Why it matters |
# |---|---|
# | $\mathbb{E}[X]$ | Basic mode coverage |
# | $\sqrt{\mathrm{Var}[X]}$ | Spread / weight balance |
# | $\mathbb{E}[X^2]$ | Second moment |
# | $\mathbb{E}[\lvert X\rvert]$ | Asymmetric around zero; catches mode collapse |
# | $\mathbb{E}[\sin(\pi X)]$ | Oscillatory nonlinear functional |
# | Median, Q1, Q3 | Quantiles are especially sensitive to the 0.2/0.8 weight split |

# %%
import numpy as np
from scipy.integrate import quad as _quad
from scipy.optimize import brentq as _brentq

_w1, _w2, _mu1, _mu2, _sig = 0.2, 0.8, -1.5, 2.5, 0.5

def _pdf(x):
    return _w1 * stats.norm.pdf(x, _mu1, _sig) + _w2 * stats.norm.pdf(x, _mu2, _sig)

def _cdf(x):
    return _w1 * stats.norm.cdf(x, _mu1, _sig) + _w2 * stats.norm.cdf(x, _mu2, _sig)

# ── Ground truth ──────────────────────────────────────────────────────────────
_gt_mean    = _w1 * _mu1 + _w2 * _mu2                                       # analytical
_gt_ex2     = _w1*(_mu1**2 + _sig**2) + _w2*(_mu2**2 + _sig**2)             # analytical
_gt_std     = (_gt_ex2 - _gt_mean**2) ** 0.5                                # analytical
_gt_eabs,_  = _quad(lambda x: abs(x) * _pdf(x), -20, 20)
_gt_esinpix,_ = _quad(lambda x: np.sin(np.pi * x) * _pdf(x), -20, 20)
_gt_median  = _brentq(lambda x: _cdf(x) - 0.50, -10, 10)
_gt_q1      = _brentq(lambda x: _cdf(x) - 0.25, -10, 10)
_gt_q3      = _brentq(lambda x: _cdf(x) - 0.75, -10, 10)

# ── Model estimates (IS-resampled samples, averaged over all 16 i.i.d. dims) ──
_x = np.array(x_resamp).ravel()   # (n_eval_particles * dim,)
_m_mean    = float(_x.mean())
_m_std     = float(_x.std())
_m_ex2     = float((_x**2).mean())
_m_eabs    = float(np.abs(_x).mean())
_m_esinpix = float(np.sin(np.pi * _x).mean())
_m_median  = float(np.median(_x))
_m_q1      = float(np.percentile(_x, 25))
_m_q3      = float(np.percentile(_x, 75))

# ── Print table ───────────────────────────────────────────────────────────────
_rows = [
    ("E[X]  (mean)",       _m_mean,    _gt_mean),
    ("std(X)",             _m_std,     _gt_std),
    ("E[X²]",              _m_ex2,     _gt_ex2),
    ("E[|X|]",             _m_eabs,    _gt_eabs),
    ("E[sin(πX)]",         _m_esinpix, _gt_esinpix),
    ("Median",             _m_median,  _gt_median),
    ("Q1 (25th pct)",      _m_q1,      _gt_q1),
    ("Q3 (75th pct)",      _m_q3,      _gt_q3),
]
print(f"\n── Diagnostics ({x_resamp.shape[0]}×{x_resamp.shape[1]} samples, averaged over dims) ──")
print(f"  {'Functional':<20}  {'Model':>10}  {'Ground truth':>12}  {'|error|':>9}")
print("  " + "─" * 57)
for _name, _mv, _tv in _rows:
    print(f"  {_name:<20}  {_mv:>10.4f}  {_tv:>12.4f}  {abs(_mv-_tv):>9.4f}")
print()

