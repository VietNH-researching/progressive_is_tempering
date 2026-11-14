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
# # Sequential Score Sampler (SSS) — GMM
#
# SSS on a 16-D factorised Gaussian mixture. Set `resume = True` to continue
# from a checkpoint.

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
import numpy as np
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
from progressive_is_tempering.mala_kernel import PTProblem
from progressive_is_tempering.smc import smc_sampler
from progressive_is_tempering.rejuvenate import rejuvenate
from progressive_is_tempering.buffer import ReplayBuffer
from progressive_is_tempering.loss import ddpm_loss_single_temp
from progressive_is_tempering.utils import (
    C_INK,
    C_INK_2,
    C_REF,
    C_TEST,
    REF_FILL_ALPHA,
    print_seed_quality_report,
    smc_seed_quality_figure,
    style_axis,
)
from progressive_is_tempering.utils.metrics import distance_w2, energy_w2, geometric_w2
from progressive_is_tempering.utils.report import write_record

# Deterministic seeds
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
    # Smallest training time; also the last time of the backward SDE grid.
    t_min: float = 0.001
    n_sde_steps_train: int = 100    # backward SDE steps during training
    n_sde_steps_eval: int = 2000    # backward SDE steps during evaluation

    # Outer loop
    num_outer_steps: int = 10
    n_particles: int = 30_000       # K: particles per backward SDE run

    # Inner loop
    n_inner_steps: int = 1_000
    batch_size: int = 512           # B per temperature

    # HMC rejuvenation (the SMC-tuned kernel; see rejuvenate)
    n_mala_rejuv: int = 10          # HMC steps per outer step

    # IS clipping / score clipping (setting = 0: disabled for GMM)
    score_clip: float = 0.0
    target_clip_norm: float = 0.0

    # Optimiser
    learning_rate: float = 1e-3
    grad_clip: float = 1.0

    # SMC (temperature ladder). n_smc_particles is also the per-rung buffer seed
    # and the buffer capacity.
    n_smc_particles: int = 81_920
    smc_local_ess_ratio: float = 0.5
    smc_n_mcmc_steps: int = 3
    # MCMC kernel for SMC and rejuvenation: "hmc" (partial refresh) or "mala"
    # (full refresh).
    mcmc_kernel: str = "hmc"

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

    # Checkpointing / logging
    log_every: int = 1              # outer steps between log lines
    save_every: int = 1             # outer steps between checkpoint saves
    # Outer steps between evaluation plots (None disables them).
    eval_every: int | None = None
    n_eval_particles: int = 100_000 # particles for evaluation plots
    # Particles in the final evaluation draw, scored by the reported metrics;
    # n_eval_particles above sizes only the periodic plots.
    n_eval_samples: int = 100_000


# Instantiate configs — edit these to tune the experiment
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

# Resume or start fresh
resume: bool = False

# Paths
NOTEBOOK_NAME = "gmm"
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
# ## 1. Target Distribution
#
# $$\pi(x)=\prod_{i=1}^{16}\bigl[0.2\,\mathcal{N}(x_i;-1.5,0.5^2)+0.8\,\mathcal{N}(x_i;2.5,0.5^2)\bigr]$$

# %%
# Mixture parameters live in one place: the target and the exact reference
# draws in the SMC seed diagnostic below must not be able to drift apart.
GMM_MEANS = (-1.5, 2.5)
GMM_WEIGHTS = (0.2, 0.8)
GMM_STD = 0.5

target = make_gmm_target(
    dim=model_cfg.x_dim,
    means=GMM_MEANS,
    weights=GMM_WEIGHTS,
    std=GMM_STD,
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
# $$\pi_\beta(x)\propto\exp\{-(1-\beta)\tfrac12\|x\|^2-\beta E(x)\},\quad\beta\in[0,1]$$

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
# Adaptive SMC gives the ladder $\{\beta_k\}$, one particle block per rung (the
# buffer seeds) and one tuned HMC kernel per rung (reused for rejuvenation).

# %%
pt_problem = PTProblem(
    rvs_ref=target.rvs_ref,
    logpdf_ref=target.logpdf_ref,
    logpdf_final_target=target.logpdf_final_target,
)

# Run SMC to discover the temperature ladder
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

# mcmc_params_by_rung is a stacked pytree: index rung k through its leaves.
def _mcmc_params_for_rung(k):
    return jax.tree_util.tree_map(lambda leaf: leaf[k - 1], mcmc_params_by_rung)

# %%
betas = jnp.array(betas_list)
M = len(betas_list) - 1  # number of non-reference levels (β_1, …, β_M = 1)
print(f"M={M} non-reference temperature levels")

# Per-rung particle blocks, indexed to match betas_list. No CoM projection:
# the GMM target has no translation symmetry, so coordinates are used as-is.
smc_states = smc_particles  # (n_levels, n_smc_particles, 16)
assert smc_states.shape[0] == len(betas_list), (
    f"expected one particle block per rung, got {smc_states.shape[0]} "
    f"blocks for {len(betas_list)} rungs"
)
print(f"SMC particles kept: {smc_states.shape}")

# %% [markdown]
# ### SMC seed quality at $\beta = 1$
#
# The top SMC rung against reference samples.

# %%
def gmm_reference_samples(key: jnp.ndarray, n: int) -> jnp.ndarray:
    """Exact i.i.d. draws from the factorised GMM target, shape (n, x_dim)."""
    k_comp, k_norm = jax.random.split(key)
    heavy = jax.random.bernoulli(
        k_comp, p=GMM_WEIGHTS[1], shape=(n, model_cfg.x_dim)
    )
    means = jnp.where(heavy, GMM_MEANS[1], GMM_MEANS[0])
    return means + GMM_STD * jax.random.normal(k_norm, (n, model_cfg.x_dim))


_smc_top = smc_states[-1]                       # (N, 16) at β = β_M = 1
key, subkey = jax.random.split(key)
_ref_x = gmm_reference_samples(subkey, train_cfg.n_seed_particles)

# Exchangeable features: the 16 coordinates, i.i.d. under a factorised target.
_fig, _seed_stats = smc_seed_quality_figure(
    features_test=np.asarray(_smc_top),
    features_ref=np.asarray(_ref_x),
    energy_test=np.asarray(jax.vmap(target.energy)(_smc_top)),
    energy_ref=np.asarray(jax.vmap(target.energy)(_ref_x)),
    feature_label="$x_i$ (pooled over coordinates)",
    title=(f"SMC seed quality at β={float(betas[-1]):.3f} vs exact draws "
           f"({_smc_top.shape[0]:,} particles)"),
)
_fig.savefig(FIGURES_DIR / "gmm_smc_seed_quality.png", dpi=150)
plt.close(_fig)

print(f"\nSMC seed quality at β={float(betas[-1]):.3f}:")
print_seed_quality_report(_seed_stats)
print(f"  → {FIGURES_DIR / 'gmm_smc_seed_quality.png'}")

# %% [markdown]
# ## 4. Forward Process
#
# OU process $dX_t=-X_t\,dt+\sqrt2\,dW_t$, close to $\mathcal N(0,I)$ at $t=T$.

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
# $\beta$-conditioned ResMLP for the excess score
# $s_\theta(x_t,t,\beta)\approx\nabla\log(p_t^\beta/p_t^\rho)$.

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
# Exponential integrator for the time reversal, on an exponential time grid from
# $T$ down to $t_{\min}$.

# %%
@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def _run_backward_sde(key, params, beta, energy_fn, dim, n_particles, T, n_steps):
    """JIT-compiled backward SDE runner."""
    score_clip = train_cfg.score_clip
    def control(x, t, b):
        s = score_net.apply(params, x, t, b)
        if score_clip > 0:  # Python-level branch; evaluated at trace time
            s = jnp.clip(s, -score_clip, score_clip)
        return 2.0 * s
    return backward_sde_batch(key, control, beta, energy_fn, dim, n_particles, T, n_steps,
                              t_end=train_cfg.t_min)

# %% [markdown]
# ## 7. Path-Space Importance Weights
#
# Log-weights $\log w$ that correct backward-SDE samples towards $\pi_\beta$.

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
# ## 8. Target Score Matching (TSM) Loss
#
# $$\phi(x_0,t,\beta)=e^{-t}\beta\bigl(x_0-\nabla E(x_0)\bigr),\qquad\mathcal L=\mathbb E\|s_\theta(x_t,t,\beta)-\phi\|^2$$

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
# ## 9. HMC Rejuvenation
#
# HMC moves at rung $k$ with the kernel SMC tuned for that rung.

# %%
# Demonstrate HMC rejuvenation at rung 1, using SMC's own tuned kernel
key, subkey = jax.random.split(key)
x_init_demo = smc_particles[1][:20]
key, subkey = jax.random.split(key)
x_mala_demo, mala_acc, _ = rejuvenate(
    subkey, x_init_demo, pt_problem, betas[1], 1,
    _mcmc_params_for_rung(1), 5, return_energy=True,
    kernel=train_cfg.mcmc_kernel,
)
print(f"HMC demo (β={float(betas[1]):.3f}, rung 1, 5 steps): "
      f"acceptance rate = {float(mala_acc):.3f}")

# %% [markdown]
# ## 10. IS Bridging Between Adjacent Temperatures
#
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
# ## 11. Parallel Data Generation (Outer Loop)
#
# For every rung $k=1,\dots,M-1$ in parallel: backward SDE, resampling, HMC
# rejuvenation, bridge weights to rung $k+1$.

# %%
@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def _data_gen_parallel(
    sde_keys,            # (M_active,) typed PRNG keys, one per level
    resample_keys,       # (M_active,) typed PRNG keys
    mala_keys,           # (M_active,) typed PRNG keys
    energy_fn,           # static callable — E(x) → scalar
    dim,                 # static int
    n_particles,         # static int
    T,                   # static float
    n_sde_steps,         # static int
    n_mala_steps,        # static int
    pt_problem,          # static — PTProblem, same object SMC ran with
    params,              # network pytree — shared across levels, not batched
    betas_active,        # (M_active,)  β_k  for k = 1 … M-1
    betas_next,          # (M_active,)  β_{k+1}
    t_idx_active,        # (M_active,)  rung indices k = 1 … M-1, matching betas_active
    mcmc_params_active,  # MALAParameters pytree (leading axis M_active), SMC-tuned per rung
):
    """Run the full data-generation pipeline for all active temperature levels
    simultaneously via jax.vmap.
    """
    score_clip = train_cfg.score_clip  # Python constant; branch resolved at trace time

    def process_one(sde_key, resample_key, mala_key, beta_k, beta_next,
                     t_idx_k, mcmc_params_k):
        def control(x, t, b):
            s = score_net.apply(params, x, t, b)
            if score_clip > 0:
                s = jnp.clip(s, -score_clip, score_clip)
            return 2.0 * s

        # 1. Backward SDE
        x0_batch, log_w_path = backward_sde_batch(
            sde_key, control, beta_k, energy_fn, dim, n_particles, T, n_sde_steps,
            t_end=train_cfg.t_min,
        )
        # 2. Systematic resample
        resample_idx = systematic_resample(resample_key, log_w_path, n_particles)
        x0_resampled = x0_batch[resample_idx]
        # 3. HMC rejuvenation, using the same SMC-tuned kernel from Section 3
        x0_rejuv, mala_acc, energies = rejuvenate(
            mala_key, x0_resampled, pt_problem, beta_k, t_idx_k,
            mcmc_params_k, n_mala_steps, return_energy=True,
            kernel=train_cfg.mcmc_kernel,
        )
        # 4. IS bridge weights (inlined to avoid passing through a nested-JIT
        #    boundary); `energies` come from the final HMC state.
        delta_beta = beta_next - beta_k
        norms_sq = jnp.sum(x0_rejuv ** 2, axis=-1)
        bridge_lw = delta_beta * (-energies + 0.5 * norms_sq)
        # Path-space ESS
        path_ess = normalized_ess(log_w_path)
        # 5. ∇E once per particle, stored with it in the buffer
        grads_rejuv = jax.vmap(jax.grad(energy_fn))(x0_rejuv)
        return x0_rejuv, bridge_lw, path_ess, mala_acc, grads_rejuv

    return jax.vmap(process_one)(
        sde_keys, resample_keys, mala_keys, betas_active, betas_next,
        t_idx_active, mcmc_params_active,
    )

# %% [markdown]
# ## 12. Per-Temperature Replay Buffers (Seeded from SMC)
#
# Buffer 0 holds exact $\mathcal N(0,I)$ draws; buffer $k\ge1$ starts with rung
# $k$'s SMC block. The outer loop refreshes buffers $2,\dots,M$ (FIFO).

# %%
# Initialize all replay buffers
buffers: list[ReplayBuffer] = [
    ReplayBuffer(capacity=train_cfg.buffer_capacity, dim=model_cfg.x_dim,
                 store_grads=True)
    for _ in range(len(betas_list))
]

# Buffer 0: exact i.i.d. N(0, I) (frozen).
key, subkey = jax.random.split(key)
buf0_samples = jax.random.normal(
    subkey, (train_cfg.buffer_capacity, model_cfg.x_dim)
)
# At β = 0 the TSM target is 0 whatever ∇E is, so buffer 0 stores zeros
# instead of evaluating it.
buffers[0].add(buf0_samples, jnp.zeros(train_cfg.buffer_capacity),
               jnp.zeros_like(buf0_samples))
buffers[0].freeze()
print(f"Buffer 0 (β=0): {buffers[0].size} samples, uniform weights [frozen]")

# Buffers 1..M: the whole SMC block of the matching rung, in random order.
_n_seed = train_cfg.n_seed_particles
assert _n_seed == smc_states.shape[1], (
    f"seed size {_n_seed:,} must equal the SMC block size "
    f"{smc_states.shape[1]:,} — both derive from n_smc_particles"
)
assert _n_seed <= train_cfg.buffer_capacity, (
    f"seed {_n_seed:,} exceeds capacity {train_cfg.buffer_capacity:,}; "
    f"ReplayBuffer.add would write duplicate indices and silently corrupt"
)
print(f"Seeding buffers directly from SMC output ({_n_seed:,} particles/rung) …")
t0 = time.time()
for _k in range(1, len(betas_list)):
    key, subkey = jax.random.split(key)
    _perm_seed = jax.random.permutation(subkey, smc_states.shape[1])[:_n_seed]
    _seed_k = smc_states[_k][_perm_seed]
    buffers[_k].add(_seed_k, jnp.zeros(_seed_k.shape[0]),
                    jax.vmap(jax.grad(target.energy))(_seed_k))
    _frozen = " [frozen]" if _k == 1 else ""
    print(f"  Buffer {_k} (β={float(betas[_k]):.3f}): {buffers[_k].size} samples{_frozen}")

# Buffer 1 is never written during training (the outer loop fills 2..M), so it
# stays frozen. It holds genuine π_{β_1} particles from SMC, so its ESS is 1.0.
if len(betas_list) >= 2:
    buffers[1].freeze()
print(f"Seeding done in {time.time()-t0:.1f}s")
print(f"Buffers 2..{len(betas_list)-1} additionally receive backward-SDE data "
      f"during training.")

# %% [markdown]
# ## 13. Training Loop
#
# Each outer step generates data at every rung; `n_inner_steps` Adam steps then
# train on all buffers jointly.

# %%
# Optimiser with gradient norm clipping
optimizer = optax.chain(
    optax.clip_by_global_norm(train_cfg.grad_clip),
    optax.adam(train_cfg.learning_rate),
)
opt_state = optimizer.init(params)

# JIT-compiled train step: forward pass, gradient, and Adam update fused into one call
@jax.jit
def train_step(params, opt_state, xg_all, t_all, beta_all, key):
    """Loss, gradient and Adam update; xg_all rows are [x0 | ∇E(x0)]."""
    _d = xg_all.shape[1] // 2
    x0_all, g_all = xg_all[:, :_d], xg_all[:, _d:]

    def loss_fn(p):
        # Noise all samples
        B_total = x0_all.shape[0]
        noise_keys = jax.random.split(key, B_total)
        sigma_t = jnp.sqrt(1.0 - jnp.exp(-2.0 * t_all))
        means = jnp.exp(-t_all[:, None]) * x0_all
        eps = jax.vmap(jax.random.normal, in_axes=(0, None))(noise_keys, (x0_all.shape[1],))
        xt_all = means + sigma_t[:, None] * eps

        # Targets (stop_gradient: no backprop through energy); ∇E(x0) is read
        # from the buffer row, not recomputed
        def phi_one(x0_i, grad_e, t_i, beta_i):
            lambda_t = jnp.exp(-t_i)
            return lambda_t * beta_i * (x0_i - grad_e)

        targets = jax.lax.stop_gradient(
            jax.vmap(phi_one)(x0_all, g_all, t_all, beta_all)
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
    """Plot the three evaluation sample sets."""
    global key

    # Run backward SDE at β=1 (same settings as end-of-training eval)
    key, subkey = jax.random.split(key)
    x0_eval, lw_eval = _run_backward_sde(
        subkey, params, betas[-1], target.energy,
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

    # Shared bin edges and y-limit so the three panels are read against each
    # other, not each against its own auto-scaled axis.
    _bins = np.linspace(-4, 6, 61)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (_label, _short, _x, _sub) in zip(axes, sample_sets):
        if _x is None:
            ax.text(0.5, 0.5, "buffer[M] empty", ha="center", va="center",
                    transform=ax.transAxes, color=C_INK_2, fontsize=10)
            ax.set_title(_label, color=C_INK, fontsize=10)
            continue
        ax.hist(np.asarray(_ref_x[:, 0]), bins=_bins, density=True,
                histtype="stepfilled", color=C_REF, alpha=REF_FILL_ALPHA,
                label="reference", zorder=1)
        ax.hist(np.asarray(_x[:, 0]), bins=_bins, density=True,
                histtype="step", linewidth=1.5, color=C_TEST, label=_short,
                zorder=2)
        ax.set_xlabel("$x_1$", color=C_INK_2, fontsize=10)
        ax.set_ylabel("density", color=C_INK_2, fontsize=10)
        ax.set_title(f"{_label}\n{_sub}", color=C_INK, fontsize=10)
        ax.set_xlim(-4, 6)
        ax.legend(fontsize=8, frameon=False, labelcolor=C_INK)
        style_axis(ax)

    _ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, _ymax)

    fig.suptitle(f"Step {outer_step+1}: three candidate sample sets vs true GMM  "
                 f"(path ESS={ess_eval:.3f})", color=C_INK, fontsize=12)
    plt.tight_layout()
    fname = EVAL_FIGURES_DIR / f"eval_step{outer_step+1:04d}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  [eval] step {outer_step+1}  model ESS={ess_eval:.3f}  → {fname}")

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

    # ---- Outer loop: data generation at all rungs (jax.vmap) ----
    M_active = len(betas_list) - 2  # levels k = 1 … M-1
    outer_ess_list = []

    if M_active > 0:
        all_keys    = jax.random.split(key, 3 * M_active + 1)
        key         = all_keys[0]
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
            target.energy,
            model_cfg.x_dim, train_cfg.n_particles,
            train_cfg.T, train_cfg.n_sde_steps_train,
            train_cfg.n_mala_rejuv,
            pt_problem,
            params, betas[1:-1], betas[2:],
            t_idx_active, mcmc_params_active,
        )

        outer_ess_list = path_ess_all.tolist()

        # Buffer writes are Python-level mutations; negligible vs. SDE/HMC compute.
        for i in range(M_active):
            # Normalize each block to mean weight 1 before adding it.
            buffers[i + 2].add(x0_rejuv_all[i], normalize_log_weights(bridge_lw_all[i]),
                               grads_all[i])

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
                # rows are [x0 | ∇E(x0)]: one gather for both
                x0_k, _ = buffers[k].sample_with_grads(subkey, train_cfg.batch_size)

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
    if train_cfg.eval_every is not None and (outer_step + 1) % train_cfg.eval_every == 0:
        periodic_eval_plot(params, outer_step)

print(f"\nTraining complete in {time.time()-train_start:.1f}s")

# %% [markdown]
# ## 14. Evaluation
#
# Backward SDE at $\beta=1$, scored as three sample sets: raw SDE output
# (`sde_raw`), the same particles resampled on their path weights (`sde_is`),
# and `buffer[M]` (`buffer`).

# %%
def eval_samples(params, n_eval=None):
    """Draw samples from π via the trained backward SDE at β=1."""
    if n_eval is None:
        n_eval = train_cfg.n_eval_samples
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
# ---- The three candidate sample sets ----
# (1) and (2) share the single SDE run above; only (2) applies log w_path.

# (1) Raw backward-SDE output: the network alone, no correction whatsoever.
x_raw = x_eval

# (2) Same particles, IS-corrected by the path-space weights.
key, subkey = jax.random.split(key)
resamp_idx = systematic_resample(subkey, lw_eval, len(lw_eval))
x_resamp = x_eval[resamp_idx]

# (3) buffer[M], resampled on its stored weights.
if buffers[M].is_ready(1):
    key, subkey = jax.random.split(key)
    n_buf = min(train_cfg.n_eval_samples, buffers[M].size)
    x_buf, _ = buffers[M].sample(subkey, n_buf)
    buf_ess = float(normalized_ess(buffers[M].log_weights[: buffers[M].size]))
else:
    x_buf, buf_ess = None, float("nan")

# (long label for the figure, short label for the table, samples, subtitle)
sample_sets = [
    ("Backward SDE (raw, no IS)", "SDE raw", x_raw,
     f"unweighted, {x_raw.shape[0]:,} particles"),
    ("Backward SDE + IS resample", "SDE + IS", x_resamp,
     f"path ESS={ess_eval:.3f}  (≈{ess_eval * x_raw.shape[0]:,.0f} eff.)"),
    (f"buffer[M]  β={float(betas[M]):.3f}", "buffer[M]", x_buf,
     f"buffer ESS={buf_ess:.3f}" if x_buf is not None else "empty"),
]

# %%
# 1-D marginal of x_1 for each sample set, plus the training loss curve.
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

# Shared bin edges and y-limit so the three panels are read against each other,
# not each against its own auto-scaled axis.
_bins = np.linspace(-4, 6, 61)
for ax, (_label, _short, _x, _sub) in zip(axes, sample_sets):
    if _x is None:
        ax.text(0.5, 0.5, "buffer[M] empty", ha="center", va="center",
                transform=ax.transAxes, color=C_INK_2, fontsize=10)
        ax.set_title(_label, color=C_INK, fontsize=10)
        continue
    # Ground truth: exact i.i.d. GMM draws, filled, behind every panel.
    ax.hist(np.asarray(_ref_x[:, 0]), bins=_bins, density=True,
            histtype="stepfilled", color=C_REF, alpha=REF_FILL_ALPHA,
            label="reference", zorder=1)
    ax.hist(np.asarray(_x[:, 0]), bins=_bins, density=True,
            histtype="step", linewidth=1.5, color=C_TEST, label=_short,
            zorder=2)
    ax.set_xlabel("$x_1$", color=C_INK_2, fontsize=10)
    ax.set_ylabel("density", color=C_INK_2, fontsize=10)
    ax.set_title(f"{_label}\n{_sub}", color=C_INK, fontsize=10)
    ax.set_xlim(-4, 6)
    ax.legend(fontsize=8, frameon=False, labelcolor=C_INK)
    style_axis(ax)

_ymax = max(ax.get_ylim()[1] for ax in axes[:3])
for ax in axes[:3]:
    ax.set_ylim(0, _ymax)

# Training loss curve (own axis — shares no units with the marginals)
if loss_history:
    axes[3].plot(loss_history, color=C_TEST, linewidth=1.5)
    axes[3].set_xlabel("outer step", color=C_INK_2, fontsize=10)
    axes[3].set_ylabel("DDPM loss", color=C_INK_2, fontsize=10)
    axes[3].set_title("Training loss", color=C_INK, fontsize=10)
    axes[3].set_yscale("log")
    style_axis(axes[3])
else:
    axes[3].text(0.5, 0.5, "No loss history", ha="center", va="center",
                 transform=axes[3].transAxes, color=C_INK_2, fontsize=10)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "gmm_marginal_and_loss.png", dpi=150)
plt.close(fig)
print(f"Figure saved to {FIGURES_DIR / 'gmm_marginal_and_loss.png'}")

# %%
# Per-dimension mean and std, for each of the three sample sets
(w1, w2), (mu1, mu2), sigma = GMM_WEIGHTS, GMM_MEANS, GMM_STD
true_mean = w1 * mu1 + w2 * mu2
true_var = w1*(mu1**2 + sigma**2) + w2*(mu2**2 + sigma**2) - true_mean**2
true_std = true_var ** 0.5

print(f"\nTrue GMM per-dim: mean={true_mean:.4f}, std={true_std:.4f}")
for _label, _short, _x, _ in sample_sets:
    if _x is None:
        continue
    print(f"  {_label:<32} mean={float(jnp.mean(jnp.mean(_x, axis=0))):>8.4f}  "
          f"std={float(jnp.mean(jnp.std(_x, axis=0))):>8.4f}")

# %% [markdown]
# ### Reported metrics: Distance-𝒲₂, Energy-𝒲₂
#
# Written to `results/gmm/[seedN/]<run_id>/metrics.json`.

# %%
_metric_ref = gmm_reference_samples(
    jax.random.fold_in(jax.random.key(SEED), 1), train_cfg.n_eval_samples
)
_metric_ref_np = np.asarray(_metric_ref)
_metric_ref_e = np.asarray(jax.vmap(target.energy)(_metric_ref))

# Each set's record goes to results/<target>/<run_id>/metrics.json.
RUN_IDS = {"SDE raw": "sde_raw", "SDE + IS": "sde_is", "buffer[M]": "buffer"}

reported_metrics = {}
print(f"── Reported metrics vs {_metric_ref_np.shape[0]:,} exact draws (lower is better) ──")
print(f"  {'set':<12}{'n':>9}{'Distance-W2':>14}{'Energy-W2':>14}")
print("  " + "─" * 49)
for _label, _short, _x, _ in sample_sets:
    if _x is None:
        print(f"  {_short:<12}{'empty':>9}")
        continue
    _e = np.asarray(jax.vmap(target.energy)(_x))
    _m = {
        # w2_pooled = Distance-W2, w2_energy = Energy-W2.
        "w2_pooled": distance_w2(np.asarray(_x), _metric_ref_np),
        "w2_energy": energy_w2(_e, _metric_ref_e),
        "geometric_w2": None,        # undefined: no particle symmetry
        # The gate on the two above: far off means the sampler is in the
        # wrong region and the distances say little.
        "energy_mean": float(_e[np.isfinite(_e)].mean()) if np.isfinite(_e).any() else None,
        "energy_mean_ref": float(_metric_ref_e.mean()),
        "n_eval_samples": int(_x.shape[0]),
        "n_ref_samples": int(_metric_ref_np.shape[0]),
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
        notes=f"Sample set: {_label}.",
    )
    print(f"  {_short:<12}{_x.shape[0]:>9,}{_m['w2_pooled']:>14.4f}"
          f"{_m['w2_energy']:>14.4f}")
print(f"\nRecords written under {RESULTS_DIR}/<run_id>/metrics.json")

# %%
# Temperature ladder ESS diagnostics
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(betas_list)), betas_list, color=C_TEST, alpha=0.85)
ax.set_xlabel("Temperature level k", color=C_INK_2, fontsize=10)
ax.set_ylabel("β_k", color=C_INK_2, fontsize=10)
ax.set_title(f"Temperature ladder ({len(betas_list)} levels)", color=C_INK, fontsize=11)
style_axis(ax)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "gmm_temperature_ladder.png", dpi=150)
plt.close(fig)
print(f"Temperature ladder figure saved.")
print(f"\nTemperature ladder: {[f'{b:.3f}' for b in betas_list]}")

# %% [markdown]
# ## 15. Extended Diagnostics
#
# Moments and quantiles of the samples against their exact values.

# %%
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

# ── Functionals, evaluated on any sample set (all 16 dims pooled — they are
#    i.i.d. under this factorised target, so pooling is legitimate) ───────────
_FUNCTIONALS = [
    ("E[X]  (mean)",  lambda v: float(v.mean()),                 _gt_mean),
    ("std(X)",        lambda v: float(v.std()),                  _gt_std),
    ("E[X²]",         lambda v: float((v**2).mean()),            _gt_ex2),
    ("E[|X|]",        lambda v: float(np.abs(v).mean()),         _gt_eabs),
    ("E[sin(πX)]",    lambda v: float(np.sin(np.pi*v).mean()),   _gt_esinpix),
    ("Median",        lambda v: float(np.median(v)),             _gt_median),
    ("Q1 (25th pct)", lambda v: float(np.percentile(v, 25)),     _gt_q1),
    ("Q3 (75th pct)", lambda v: float(np.percentile(v, 75)),     _gt_q3),
]

_active = [(short, np.asarray(x).ravel())
           for _, short, x, _ in sample_sets if x is not None]

# ── Print table: one column per sample set, ground truth on the right ─────────
print(f"\n── Diagnostics: three candidate sample sets vs ground truth ──")
for _lbl, _short, _x, _sub in sample_sets:
    if _x is not None:
        print(f"    {_short:<10} {_lbl:<30} {_x.shape[0]:>7,}×{_x.shape[1]}   {_sub}")
print()
_hdr = "  " + f"{'Functional':<16}" + "".join(f"{lbl:>24}" for lbl, _ in _active)
print(_hdr + f"{'Ground truth':>14}")
print("  " + "─" * (16 + 24 * len(_active) + 14))
for _name, _fn, _tv in _FUNCTIONALS:
    _cells = ""
    for _lbl, _v in _active:
        _mv = _fn(_v)
        _cells += f"{_mv:>14.4f}{'(' + format(abs(_mv - _tv), '.4f') + ')':>10}"
    print(f"  {_name:<16}" + _cells + f"{_tv:>14.4f}")
print(f"\n  Each cell: estimate (|error| vs ground truth).")
print()

