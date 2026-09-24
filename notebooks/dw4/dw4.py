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
# # Sequential Score Sampler (SSS) — DW4
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

import os
import time
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
# Allocate GPU memory on demand. By default JAX preallocates ~75% of the
# device's memory at backend init and holds it for the life of the process;
# on-demand allocation can cost a little speed but lets other processes use the
# GPU alongside this one. Must precede `import jax`, since the value is read
# once at backend init; `setdefault` lets the shell override it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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
from progressive_is_tempering.mala_kernel import PTProblem, batch_init_extend, mala_kernel
from progressive_is_tempering.smc import smc_sampler
from progressive_is_tempering.rejuvenate import rejuvenate
from progressive_is_tempering.buffer import ReplayBuffer
from progressive_is_tempering.targets import DW4
from progressive_is_tempering.utils import (
    C_INK,
    C_INK_2,
    C_REF,
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
    n_particles: int = 30_000     # K: particles per backward SDE run

    # Inner loop
    n_inner_steps: int = 1_000
    batch_size: int = 512        # B per temperature

    # MALA rejuvenation (the SMC-tuned kernel; see rejuvenate)
    n_mala_rejuv: int = 10        # MALA steps per particle per outer step

    # Clipping (empirically tuned for DW4's EGNN score network)
    score_clip: float = 1e4
    target_clip_norm: float = 20.0

    # Optimiser
    learning_rate: float = 1e-3
    grad_clip: float = 1.0

    # SMC (temperature ladder)
    #
    # One number governs the SMC population, the per-rung seed, and the buffer
    # capacity: they are the same particles at three points in their life. SMC
    # propagates N particles up the ladder, each rung's block is fed whole into
    # that rung's buffer, and the outer loop evicts them gradually (FIFO) as
    # backward-SDE data arrives.
    #
    # The seed drains in n_smc_particles / n_particles ≈ 2.7 outer steps, about a quarter
    # of num_outer_steps — comfortably inside the run, so the network spends
    # most of the run training on its own generated data. Raising n_smc_particles
    # pushes that fraction up; raise num_outer_steps to match.
    n_smc_particles: int = 81_920
    smc_local_ess_ratio: float = 0.2
    smc_n_mcmc_steps: int = 10

    # Checkpointing / logging
    log_every: int = 1
    save_every: int = 1
    # Outer steps between periodic evaluation plots; None disables them (e.g.
    # for seed sweeps). Each periodic evaluation draws from the training key,
    # so a run with None does not reproduce one with plots draw-for-draw.
    eval_every: int | None = None
    n_eval_particles: int = 10_000
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

# The seed fills every buffer to capacity, so FIFO needs `seed_flush_steps`
# outer steps to retire it. If that exceeds num_outer_steps the run finishes
# while still training partly on SMC particles, which are far more degenerate
# than their row count suggests (see the seed-quality diagnostic).
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

NOTEBOOK_NAME = "dw4"
# Everything a run writes lives in results/<target>/ (results/<target>/seedN/
# for a seed sweep): checkpoint.pkl, log.txt and figures/ (gitignored) and one
# <run_id>/metrics.json per evaluated sample set (tracked).
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
# ## 2. Geometry Utilities, Reference Data, and Summary Statistics
#
# All particle states are kept in the zero-CoM subspace throughout.
#
# This section also holds the ground-truth reference samples and the two
# summary statistics every comparison in this notebook uses — **pairwise
# distances** (cluster geometry) and **energy** (how physically plausible a
# configuration is). They live here, ahead of SMC, because three separate
# places compare against them: the SMC seed diagnostic (Section 10), the
# periodic eval during training (Section 14), and the final eval (Section 16).
# One definition, one styling, so the three are directly comparable.

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


# %%
# ---- Summary statistics -----------------------------------------------------
@jax.jit
def get_pairwise_distances(x_flat: jnp.ndarray) -> jnp.ndarray:
    """All 6 unique pairwise distances for (B, 8) input → (B, 6)."""
    pos = x_flat.reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim)   # (B, 4, 2)
    diff = pos[:, :, None, :] - pos[:, None, :, :]                            # (B, 4, 4, 2)
    r = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)                         # (B, 4, 4)
    i, j = jnp.triu_indices(model_cfg.n_particles, k=1)                       # 6 unique pairs
    return r[:, i, j]                                                          # (B, 6)


@jax.jit
def get_energies(x_flat: jnp.ndarray) -> jnp.ndarray:
    """Per-sample energy E(x) for (B, 8) input → (B,)."""
    return jax.vmap(energy)(x_flat.reshape(-1, model_cfg.x_dim))


# ---- Reference (ground-truth) samples ---------------------------------------
# Standard DW4 test split, data/test_split_DW4.npy (data/redownload.sh
# re-fetches it). The raw samples are not CoM-centred; project
# them onto the zero-CoM subspace the rest of the notebook works in. Both
# summary statistics are translation invariant, so this changes neither.
_pt_ref = project_com(
    jnp.array(np.load(Path("../../data/test_split_DW4.npy"))).reshape(-1, model_cfg.x_dim)
)  # (10000, 8)
print(f"Loaded reference samples: {_pt_ref.shape}")

_REF_R = get_pairwise_distances(_pt_ref)      # (10000, 6)
_REF_E = get_energies(_pt_ref)                 # (10000,)
print(f"Reference energy: mean={float(_REF_E.mean()):.2f}  "
      f"[min={float(_REF_E.min()):.2f}, max={float(_REF_E.max()):.2f}]")

# ---- Comparison plot helpers ------------------------------------------------
# From `progressive_is_tempering/utils/smc_diagnostics.py`. Colour follows the
# entity, never the panel — C_TEST is always whatever is under evaluation,
# C_REF is always ground truth — and the reference is additionally a filled
# histogram, so identity never rests on colour alone.
_style_axis = style_axis   # local alias: this notebook styles its own axes too


def plot_distance_comparison(ax, r_test, label_test, r_ref=_REF_R, bins=150):
    """Pairwise-distance density: test vs reference, on shared bin edges."""
    return plot_value_comparison(
        ax, r_test, r_ref, "inter-particle distance $r_{ij}$",
        label_test, bins=bins, clip_at_zero=True,   # distances are non-negative
    )


def plot_energy_comparison(ax, e_test, label_test, e_ref=_REF_E, bins=150):
    """Energy density: test vs reference, on shared bin edges.

    Energy is the sharper of the two statistics: a single bad pair moves it far
    more than it perturbs the distance histogram. The off-window fraction
    reported here is the real diagnostic.
    """
    return plot_value_comparison(
        ax, e_test, e_ref, "energy $E(x)$", label_test, bins=bins,
    )

# %% [markdown]
# ## 3. Tempering Problem for DW4
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


# %%
# pt_problem is built here (rather than at the SMC call in Section 9) so
# Section 7's MALA demo can also use it.
pt_problem = PTProblem(
    rvs_ref=rvs_ref,
    logpdf_ref=logpdf_ref,
    logpdf_final_target=logpdf_final_target,
)

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
# All particles advance together: the EGNN (batched interface) is called once
# per time step on the whole `(K, 8)` batch.

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
# Rejuvenation uses the covariance-preconditioned MALA kernel (`rejuvenate`,
# from `progressive_is_tempering/rejuvenate.py`) with the parameters SMC
# (Section 9) tunes per rung. The kernel has no CoM-awareness at all, so
# `rejuvenate`'s `project_fn` hook re-projects to zero-CoM after *every*
# internal move, never leaving the constraint manifold.
#
# This demo runs before SMC has produced a real ladder, so it adapts a
# throwaway kernel with the kernel's own `adapt_params` at an illustrative
# rung. Training does not use it: training uses the kernels SMC adapts in
# Section 9.

# %%
_demo_beta, _demo_t_idx = jnp.array(0.1), 1
_demo_mcmc = mala_kernel(pt_problem, _demo_beta, _demo_t_idx)

key, subkey = jax.random.split(key)
_pts = jax.vmap(project_com)(jax.random.normal(subkey, (20, model_cfg.x_dim)))
key, k_init, k_adapt, k_move = jax.random.split(key, 4)
_demo_x = batch_init_extend(k_init, _pts, _demo_mcmc)
_demo_params = _demo_mcmc.adapt_params(k_adapt, _demo_x)

_pts_mala, _mala_acc = rejuvenate(
    k_move, _pts, pt_problem, _demo_beta, _demo_t_idx, _demo_params, 5,
    project_fn=project_com,
)
print(f"MALA demo (β={float(_demo_beta):.3f}, rung 1, 5 steps): "
      f"acceptance rate = {float(_mala_acc):.3f}")

# %% [markdown]
# ## 8. IS Bridging Between Adjacent Temperatures
#
# $$\log w_{k \to k+1}(x) = (\beta_{k+1}-\beta_k)\bigl(-E(x)+\tfrac12\|x\|^2\bigr)$$

# %%
@jax.jit
def is_bridge_log_weights(samples, beta_k, beta_next):
    """IS log-weights to bridge π_{β_k} → π_{β_{k+1}}.

    Reference definition of the formula above, kept for interactive diagnostics.
    The training path does not call it: `_data_gen_parallel` (Section 12)
    recomputes it inline so the whole pipeline stays inside one jit region.
    """
    delta_beta = beta_next - beta_k
    energies = jax.vmap(energy)(samples)
    norms_sq = jnp.sum(samples ** 2, axis=-1)
    return delta_beta * (-energies + 0.5 * norms_sq)


@jax.jit
def normalize_log_weights(log_w: jnp.ndarray) -> jnp.ndarray:
    """Shift a block of log-weights to mean weight 1 (i.e. logsumexp = log n).

    A buffer holds several blocks written at different times, and `.sample()`
    resamples across all of them jointly — so the blocks' weight *scales* must
    be commensurable. Raw bridge weights Δβ(-E + ½‖x‖²) are unnormalized,
    whereas the SMC seed particles are already ≈ π_{β_k} and so carry weight 0.
    Mixing those two raw would let one block dominate the buffer outright,
    depending only on sign. Normalizing every block on the way in makes blocks
    combine in proportion to their particle counts, which is what we want.

    Degenerate blocks (all weights -inf) fall back to uniform.
    """
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
# Run SMC once before training to discover $\{\beta_k\}_{k=0}^M$ such that
# successive IS bridges have normalized ESS $\geq$ `local_ess_ratio`.
#
# `smc_sampler` returns the particle positions at **every** rung,
# stacked as `(len(betas_list), n_smc_particles, 8)` and indexed in the same
# order as `betas_list`. These are the only samples available before training
# that are actually distributed according to $\pi_{\beta_k}$ at each level,
# and are used to seed the replay buffers (Section 11).
#
# It additionally returns the per-rung MALA kernel
# (covariance + step size) SMC itself adapted while building the ladder. That
# kernel is reused as-is for outer-loop rejuvenation (Section 12); see
# `rejuvenate`. The buffer seed (Section 11) uses SMC's particle output
# directly, with no further MALA pass. SMC moves with the same MALA kernel
# `rejuvenate` uses, so the captured kernel is calibrated for exactly that
# acceptance regime.

# %%
# pt_problem was built in Section 3.

# Early-load the checkpoint before SMC, so that SMC can be skipped on resume
_ckpt_data = None
if resume and CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, "rb") as f:
        _ckpt_data = pickle.load(f)
    print(f"Checkpoint found: outer_step={_ckpt_data['outer_step']}, "
          f"will restore temperature ladder, buffers, and the MALA kernel")

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
    )
    print(f"SMC done in {time.time()-t0:.1f}s")
    print(f"Temperature ladder ({len(betas_list)} levels): {[f'{b:.3f}' for b in betas_list]}")
    print(f"logZ estimates: {logZ_dict}")

    # Scalar MALA step size (ε), adapted per rung by SMC's own acceptance-rate
    # bisection (MALAParameters.scaled_cov.eps) — the same value reused wherever
    # mcmc_params_by_rung is indexed below.
    _eps_by_rung = mcmc_params_by_rung.scaled_cov.eps
    print("MALA step size (ε) adapted by SMC, per rung:")
    for _k, _eps_k in enumerate(_eps_by_rung, start=1):
        print(f"  rung {_k} (β={betas_list[_k]:.3f}): ε={float(_eps_k):.5g}")

    betas = jnp.array(betas_list)
    M = len(betas_list) - 1
    print(f"M={M} non-reference temperature levels")

    # Project the SMC particles onto the zero-CoM subspace. The MALA kernel does
    # not project (nothing in mala_kernel.py touches the CoM). DW4 uses
    # trap_scale=0, so the energy is exactly translation-invariant and the CoM
    # is an entirely free direction under MCMC — only the reference term in the
    # geometric path confines it at all, and not at β=1. Everything downstream
    # assumes CoM-zero states, so project once, at the boundary.
    smc_states = jax.vmap(jax.vmap(project_com))(smc_particles)  # (n_levels, N, 8)
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
# The top rung of the ladder targets $\pi_{\beta_M} = \pi$ — the same
# distribution the reference set is drawn from — so the two should agree. This
# is the one check worth running *before* committing to a training run: it puts
# a ceiling on what the seeded buffers can teach the network. If SMC has not
# reached the target at $\beta=1$, no amount of training on its output will.
#
# Two failure modes, and they are not the same thing:
#
# 1. **Distributional mismatch** (top row). Both statistics are shown because
#    they fail differently. Pairwise distances describe cluster *geometry* and
#    are forgiving — a badly-strained cluster can still produce a plausible
#    distance histogram. Energy is the sharper test. Agreement on distances but
#    not energy means the geometry is roughly right and the fine structure is
#    not.
# 2. **Ancestral degeneracy** (bottom row). The histograms match beautifully,
#    yet the population carries far less independent information than its
#    particle count suggests. SMC resampling collapses the ancestral lineages;
#    local MCMC then re-equilibrates each particle *within* its basin, erasing
#    literal duplicates while the basin assignment stays frozen. A distinct-row
#    count reports 100% unique and misses this entirely.
#
# The degeneracy test uses the 6 pairwise distances as exchangeable features:
# particle-permutation symmetry means every pair must give the same exceedance
# probability $P(r_{ij} > \tau)$. The reference set goes through the identical
# computation as a control — being i.i.d., it should recover its own size, which
# is what makes the test's verdict on the SMC block trustworthy. See
# `progressive_is_tempering/utils/smc_diagnostics.py`.
#
# Skipped on resume, where `smc_states` does not exist.

# %%
if _ckpt_data is None:
    _smc_top = smc_states[-1]                      # (N, 8) at β = β_M = 1

    _fig, _seed_stats = smc_seed_quality_figure(
        features_test=np.asarray(get_pairwise_distances(_smc_top)),
        features_ref=np.asarray(_REF_R),
        energy_test=np.asarray(get_energies(_smc_top)),
        energy_ref=np.asarray(_REF_E),
        feature_label="inter-particle distance $r_{ij}$",
        title=(f"SMC seed quality at β={float(betas[-1]):.3f} vs ground truth "
               f"({_smc_top.shape[0]:,} particles)"),
        features_nonneg=True,
    )
    _fig.savefig(FIGURES_DIR / "dw4_smc_seed_quality.png", dpi=150)
    plt.close(_fig)

    print(f"\nSMC seed quality at β={float(betas[-1]):.3f}:")
    print_seed_quality_report(_seed_stats)
    print(f"  → {FIGURES_DIR / 'dw4_smc_seed_quality.png'}")
else:
    print("Resumed from checkpoint — SMC seed diagnostic skipped (no smc_states).")

# %% [markdown]
# ## 11. Per-Temperature Replay Buffers (Seeded from SMC)
#
# Buffer 0 ($\beta=0$): i.i.d. $\mathcal{N}(0,I)$ projected to zero CoM, frozen.
# The reference *is* the $\beta=0$ target, so exact draws beat SMC particles here.
#
# Buffers $1,\ldots,M$: seeded directly with this rung's **entire** SMC block —
# the particles SMC itself already produced after its own reweight/resample/MALA
# move at that rung (Section 9), with no further rejuvenation applied here.
# Seed size, SMC population and buffer capacity are one number
# (`n_smc_particles`), so each buffer starts exactly full and nothing is
# subsampled away.
# Buffers $2,\ldots,M$ additionally receive backward-SDE data during training,
# and FIFO eviction retires the seed as that arrives — over
# `n_smc_particles / n_particles` outer steps, since each step overwrites the
# oldest $K$ rows in place.
#
# Seed particles are already $\approx \pi_{\beta_k}$, so they enter with
# **uniform** log-weights — no bridge weights (those are for particles arriving
# from the rung below). See `normalize_log_weights` in Section 8 for why every
# block is normalized on the way in.

# %%
if _ckpt_data is None:
    buffers: list[ReplayBuffer] = [
        ReplayBuffer(capacity=train_cfg.buffer_capacity, dim=model_cfg.x_dim)
        for _ in range(len(betas_list))
    ]

    # Buffer 0: exact projected N(0, I), frozen. Filled to capacity rather than
    # to n_particles: it is never evicted, and exact draws are free, so leaving
    # slots empty would only narrow the β=0 training data for no reason.
    key, subkey = jax.random.split(key)
    buf0_samples = jax.vmap(project_com)(
        jax.random.normal(subkey, (train_cfg.buffer_capacity, model_cfg.x_dim))
    )
    buffers[0].add(buf0_samples, jnp.zeros(train_cfg.buffer_capacity))
    buffers[0].freeze()
    print(f"Buffer 0 (β=0): {buffers[0].size} samples, uniform weights [frozen]")

    # Buffers 1..M: the whole SMC block at the matching rung, no subsampling and
    # no further rejuvenation — SMC's own reweight/resample/MALA move (Section 9)
    # already produced these, already projected to zero CoM. The permutation
    # below is not selection (it selects everything) but ordering insurance:
    # FIFO evicts in index order, so if the SMC block ever carried lineage
    # structure in its row order, the first K rows retired would be a
    # correlated subset rather than a random one.
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
        buffers[_k].add(_seed_k, jnp.zeros(_seed_k.shape[0]))
        _frozen = " [frozen]" if _k == 1 else ""
        print(f"  Buffer {_k} (β={float(betas[_k]):.3f}): {buffers[_k].size} samples{_frozen}")

    # Buffer 1 is never written during training (the outer loop fills 2..M), so
    # it stays frozen. It holds genuine π_{β_1} particles from SMC, so its ESS
    # is 1.0.
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
# For each active temperature level $k \in \{1,\ldots,M-1\}$ **in parallel**:
# 1. Backward SDE at $\beta_k$ → $(x_0^{(i)}, \log w_\mathrm{path}^{(i)})$
# 2. Systematic resample → approx. i.i.d. from $\pi_{\beta_k}$
# 3. CoM-projected MALA rejuvenation
# 4. IS bridge weights → $\log w_{k \to k+1}$

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
        # 3. MALA rejuvenation, using the same SMC-tuned kernel from Section 9
        x0_rejuv, mala_acc = rejuvenate(
            mala_key, x0_resampled, pt_problem, beta_k, t_idx_k,
            mcmc_params_k, n_mala_steps, project_fn=project_com,
        )
        # 4. IS bridge weights
        delta_beta = beta_next - beta_k
        energies = jax.vmap(energy_fn)(x0_rejuv)
        norms_sq = jnp.sum(x0_rejuv ** 2, axis=-1)
        bridge_lw = delta_beta * (-energies + 0.5 * norms_sq)
        path_ess = normalized_ess(log_w_path)
        return x0_rejuv, bridge_lw, path_ess, mala_acc

    return jax.vmap(process_one)(
        sde_keys, resample_keys, mala_keys, betas_active, betas_next,
        t_idx_active, mcmc_params_active,
    )


# %% [markdown]
# ## 13. Training Step (Inner Loop)
#
# Target Score Matching (TSM) target for DW4, built from the score of the
# clean sample $x_0$:
# $$\phi(x_0,t,\beta) = e^{-t}\,\beta\,(x_0 - \nabla E(x_0))$$
# EGNN is called on the full batch `((M+1)*B, 8)` directly (no per-sample vmap).
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

    x0_all: ((M+1)*B, 8), t_all: ((M+1)*B,), beta_all: ((M+1)*B,)
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

        # TSM targets (stop_gradient: no backprop through energy Hessian)
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
# Same three candidate sample sets and two summary statistics as the final
# evaluation (Section 16) — pairwise distances and energy, for raw backward-SDE
# output, IS-resampled backward-SDE output, and buffer[M] — run every
# `eval_every` outer steps so the training-time figures are directly comparable
# to the end-of-run one (minus its loss-curve column).

# %%
def periodic_eval_plot(params, outer_step):
    """Same three candidate sample sets and two summary statistics as the final
    evaluation (Section 16) — distances (top row) and energy (bottom row) for
    raw backward-SDE output, IS-resampled backward-SDE output, and buffer[M] —
    minus the loss-curve column, which belongs to the end-of-run figure only.
    """
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


# Reference samples (_pt_ref, _REF_R, _REF_E) were loaded in Section 2, ahead of
# SMC, so the seed diagnostic and both evals compare against the same data.

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
#         project-noised xt, compute TSM targets
#         Adam step on joint loss via batched EGNN
# ```

# %%
start_outer_step = 0
loss_history = []
ess_history: list[list[float]] = []  # one entry per outer step: path ESS per active rung

if _ckpt_data is not None:
    params = _ckpt_data["params"]
    opt_state = _ckpt_data["opt_state"]
    start_outer_step = _ckpt_data["outer_step"]
    loss_history = _ckpt_data.get("loss_history", [])
    ess_history = _ckpt_data.get("ess_history", [])
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

        # Active rungs are k = 1 … M-1 (M_active of them); mcmc_params_by_rung[k-1]
        # is the MALAParameters SMC tuned for rung k (Section 9), so slice its
        # first M_active entries to match betas[1:-1].
        t_idx_active = jnp.arange(1, M)
        mcmc_params_active = jax.tree_util.tree_map(
            lambda leaf: leaf[:M_active], mcmc_params_by_rung
        )

        x0_rejuv_all, bridge_lw_all, path_ess_all, _ = _data_gen_parallel(
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
        ess_history.append(outer_ess_list)

        for i in range(M_active):
            # Normalize each block on the way in so it is commensurable with
            # the uniform-weight SMC seed already in the buffer, and with
            # blocks written at earlier outer steps (Section 8).
            buffers[i + 2].add(x0_rejuv_all[i], normalize_log_weights(bridge_lw_all[i]))

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
            "ess_history": ess_history,
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
# ### Reference Samples
#
# The standard DW4 test split is loaded from `data/test_split_DW4.npy`
# (set once in Section 2 above). Nothing to re-run here.

# %%
print(f"Reference samples already loaded: {_pt_ref.shape}")

# Check if reference samples are in 0-centered CoM subspace
_ref_com = _pt_ref.reshape(-1, model_cfg.n_particles, model_cfg.spatial_dim).mean(axis=1)
print(f"Max |CoM| in reference samples: {jnp.abs(_ref_com).max():.2e}  (should be ≈ 0)")

# %% [markdown]
# ## 16. Final Evaluation
#
# Run the backward SDE at $\beta=1$ and compare the distribution of pairwise
# inter-particle distances to the ground-truth reference samples.

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
#
# | # | Sample set | What it isolates |
# |---|---|---|
# | 1 | Backward SDE output, **no** reweighting | the learned score alone |
# | 2 | Same run, IS-resampled on the path weights | score **+** path-space correction |
# | 3 | `buffer[M]`, IS-resampled on its stored weights | the training data at $\beta=1$ |
#
# (1) and (2) come from the **single** SDE run above — the same particles,
# differing only in whether `log w_path` is applied — so any gap between them is
# exactly the work the importance correction is doing. If they agree, the
# network learned the target; if only (2) matches, the reweighting is carrying
# the result and `ess_eval` says how few particles are really behind it.
#
# (3) needs no SDE run: it is what the outer loop deposited at the top rung,
# i.e. what the inner loop actually trained on. It bounds what the network could
# have learned — a network cannot beat its own training data.

# %%
# (1) Raw backward-SDE output: the network alone, no correction.
x_raw = x0_eval

# (2) Same particles, IS-corrected by the path-space weights.
key, subkey = jax.random.split(key)
resamp_idx = systematic_resample(subkey, lw_eval, len(lw_eval))
x_resamp = x0_eval[resamp_idx]

# (3) buffer[M]: the training data at β=1, drawn the way the inner loop draws
# it — systematic resampling on the stored IS weights. Raw buffer contents
# would NOT be π-distributed: the blocks arriving from the outer loop are
# ≈ π_{β_{M-1}} particles carrying bridge weights to β_M, and only become ≈ π
# once those weights are applied. `.sample()` applies them.
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
# Rows: the two summary statistics. Distances say whether the *geometry* is
# right, energy says whether the *physics* is right — energy is the sharper
# test. Every panel carries the same filled ground-truth reference, so the
# three sets are read against each other rather than each against its own axis.
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
    f"DW4 final evaluation — β={float(betas[-1]):.3f}, "
    f"three candidate sample sets vs ground truth",
    fontsize=12, color=C_INK,
)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "dw4_distances_and_loss.png", dpi=150)
plt.close(fig)
print(f"Figure saved to {FIGURES_DIR / 'dw4_distances_and_loss.png'}")

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
# The three numbers reported for the method, computed by
# `progressive_is_tempering.utils.metrics`; lower
# is better for all three.
#
# - **Distance-𝒲₂** — 𝒲₂ between the pooled pairwise distances (6 per
#   configuration), all thrown into one 1-D distribution.
# - **Energy-𝒲₂** — 𝒲₂ between the energy distributions. Non-finite energies
#   are dropped first.
# - **Geometric-𝒲₂** — exact optimal transport between configurations, with the
#   ground distance taken modulo particle permutation and rotation/reflection:
#   linear assignment on unaligned coordinates first, then orthogonal
#   Procrustes over O(2). That one-shot alignment is an upper bound on the true
#   ground distance, and matches the published definition.
#
# Every sample set is scored against the whole reference set
# (10,000 configurations); a set larger than that is first subsampled to the
# same size, without replacement and with a fixed seed. Geometric-𝒲₂ needs one
# alignment per (sample, reference) pair — 10⁸ at 10k × 10k — and runs on all
# CPU cores but one: about 1.5 min on 48 cores per sample set.
#
# Each set is also written as its own record,
# `results/dw4/[seedN/]<run_id>/metrics.json` with `run_id` ∈ {`sde_raw`, `sde_is`,
# `buffer`}, under the metric names `w2_pooled` (Distance-𝒲₂), `w2_energy`
# (Energy-𝒲₂) and `geometric_w2` (Geometric-𝒲₂). A
# record also carries the energy means, the ESS and the full configuration.
# A new run overwrites the previous run's records in the same folder.

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

# %% [markdown]
# ## 17. Post-Training Diagnostics
#
# Two additional diagnostics, computed after training has finished.

# %% [markdown]
# ### 17.1 Path-space ESS per rung, over outer steps
#
# `ess_history[step]` is the list of per-rung path-space ESS values
# (`outer_ess_list`, Section 15) recorded each outer step during the loop —
# one entry per active rung $k = 1, \ldots, M-1$, in that fixed order. A
# rung's ESS collapsing over the run (rather than settling) means its
# backward-SDE proposal is drifting away from $\pi_{\beta_k}$ as the network
# changes.

# %%
if ess_history:
    _ess_arr = np.array(ess_history)          # (n_outer_steps_logged, M_active)
    _steps = np.arange(1, _ess_arr.shape[0] + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for _i in range(_ess_arr.shape[1]):
        _rung = _i + 1                          # ess_history columns are rungs 1..M-1, in order
        ax.plot(_steps, _ess_arr[:, _i], marker="o", markersize=3, linewidth=1.2,
                label=f"rung {_rung} (β={float(betas[_rung]):.3f})")
    ax.set_xlabel("outer step", color=C_INK_2, fontsize=10)
    ax.set_ylabel("normalized path ESS", color=C_INK_2, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Path-space ESS per rung over outer steps", color=C_INK, fontsize=11)
    ax.legend(fontsize=8, ncol=min(_ess_arr.shape[1], 4), frameon=False, labelcolor=C_INK)
    _style_axis(ax)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "dw4_path_ess_history.png", dpi=150)
    plt.close(fig)
    print(f"Path ESS history figure saved → {FIGURES_DIR / 'dw4_path_ess_history.png'}")
else:
    print("No path-ESS history recorded (no active rungs during training) — "
          "skipping ESS-over-steps diagnostic.")

# %% [markdown]
# ### 17.2 Merged comparison: all three sample sets vs ground truth
#
# Section 16 compares each candidate sample set to the reference in its own
# column. This overlays all three (`_summary`, computed there) on top of the
# reference in a single pair of axes — one for pairwise distances, one for
# energy — each sample set in its own colour, so the three can be read
# against each other directly rather than only against ground truth.

# %%
from progressive_is_tempering.utils import honest_density_hist, reference_anchored_window

_MERGED_COLORS = {
    "SDE raw": C_TEST,
    "SDE + IS": "#2ca02c",
    "buffer[M]": "#9467bd",
}

if _summary:
    fig, (ax_r, ax_e) = plt.subplots(1, 2, figsize=(14, 5.5))

    _all_r = jnp.concatenate([_REF_R] + [_r for _, _, _r, _ in _summary])
    _all_e = jnp.concatenate([_REF_E] + [_e for _, _e, _, _ in _summary])
    _r_lo, _r_hi = reference_anchored_window(_all_r, _REF_R)
    _r_lo = max(_r_lo, 0.0)
    _e_lo, _e_hi = reference_anchored_window(_all_e, _REF_E)

    honest_density_hist(ax_r, _REF_R, _r_lo, _r_hi, 150, "reference", C_REF, filled=True)
    honest_density_hist(ax_e, _REF_E, _e_lo, _e_hi, 150, "reference", C_REF, filled=True)
    for _short, _e, _r, _frac in _summary:
        _color = _MERGED_COLORS.get(_short, C_INK_2)
        honest_density_hist(ax_r, _r, _r_lo, _r_hi, 150, _short, _color)
        honest_density_hist(ax_e, _e, _e_lo, _e_hi, 150, _short, _color)

    ax_r.set_xlabel("inter-particle distance $r_{ij}$", color=C_INK_2, fontsize=10)
    ax_e.set_xlabel("energy $E(x)$", color=C_INK_2, fontsize=10)
    for _ax, _title in ((ax_r, "Pairwise distances"), (ax_e, "Energy")):
        _ax.set_ylabel("density", color=C_INK_2, fontsize=10)
        _ax.set_title(_title, color=C_INK, fontsize=11)
        _ax.legend(fontsize=9, frameon=False, labelcolor=C_INK)
        _style_axis(_ax)

    fig.suptitle(
        f"DW4 — all candidate sample sets vs ground truth, merged (β={float(betas[-1]):.3f})",
        fontsize=12, color=C_INK,
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "dw4_merged_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Merged comparison figure saved → {FIGURES_DIR / 'dw4_merged_comparison.png'}")
else:
    print("No sample sets available — skipping merged comparison diagnostic.")
