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
# # Sequential Score Sampler (SSS) — Alanine Dipeptide
#
# Trains a $\beta$-conditioned score network on alanine dipeptide (ALDP, 22
# atoms, 66 Cartesian coordinates) in implicit solvent: AMBER ff14SB + OBC1 at
# 300 K, the target of PITA (*Progressive Inference-Time Annealing*), scored
# against PITA's molecular-dynamics test set with the metrics of its Table 2.
#
# The method is the one of the other notebooks, with the MCMC and the neural
# sampler working in **different coordinates**:
#
# | | other notebooks | ALDP | why |
# |---|---|---|---|
# | annealing path | $\rho = \mathcal N(0, I)$ in $x$ | $\rho$ = $\mathcal N(0,I)$ × uniform torsions in sampling coordinates $z$ (Section 3) | internal coordinates make a reference with every basin in reach |
# | SMC + rejuvenation | `smc.py`, `rejuvenate.py` in $x$ | `periodic_hmc` in $z$: torsions are circles, step size tuned per rung | Cartesian HMC cannot cross torsional barriers |
# | network, SDE, buffers | $x$ | Cartesian $y = x/S$, zero centre of mass | |
# | score network | ResMLP / EGNN | PITA's DiT (`net/dit.py`), random-rotation augmentation | |
# | learned object | excess score vs $\rho$ (0 at $\beta=0$) | excess score vs $\gamma = \mathcal N(0, I)$ (not 0 at $\beta=0$, Section 5) | $\rho \neq \gamma$ |
# | loss | TSM | DSM at every $t$ + 0.01 · TSM for $t < 0.0184$ (Section 13) | the stiff Cartesian TSM target is too noisy alone |
# | bridge blocks | always written | written only if path ESS · K ≥ `ess_gate` (Section 12) | a block of a few distinct paths would replace the seed |
#
# Run top-to-bottom after a kernel restart. Set `resume = True` to continue
# from a checkpoint.
#
# **About the committed results.** `results/aldp/seed{0,1,2}/` were trained by
# the study script this notebook was ported from, with the libraries that
# `progressive_is_tempering.alanine` and `progressive_is_tempering.periodic_hmc`
# reduce: same code path and random stream (checked bit for bit, including a
# short end-to-end run), but the training has not been rerun by the notebook
# itself. The final evaluation was: run from those checkpoints with
# `resume = True`, it reproduced the evaluation samples behind the committed
# records bit for bit. The training SDE keeps the float32 step coefficients
# those runs used (Section 6); the final evaluation uses accurate ones (Section 16).

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
# Allocate GPU memory on demand (must precede `import jax`; the shell may override).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Deterministic GPU kernels: a given RUN_SEED then reproduces a run's numbers
# exactly (same hardware and software versions). Without this, XLA may sum
# GPU reductions and scatters in a different order in each process, and SMC's
# adaptive ladder turns those last-bit differences into different results.
if "--xla_gpu_deterministic_ops" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
    ).strip()

import jax

# The force field, the coordinate maps and the MCMC run in float64; the network,
# the SDE and the buffers in float32.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from progressive_is_tempering import alanine
from progressive_is_tempering.periodic_hmc import (
    PTProblem, batch_init_extend, make_engine, repeated_batch_move, smc_sampler,
)
from progressive_is_tempering.net.dit import DIT3D
from progressive_is_tempering.sampler import exponential_time_grid
from progressive_is_tempering.resampling import systematic_resample, normalized_ess
from progressive_is_tempering.buffer import ReplayBuffer
from progressive_is_tempering.utils import (
    C_INK, C_INK_2, C_REF, C_TEST, plot_value_comparison, style_axis,
)
from progressive_is_tempering.utils.metrics import (
    distance_w2, ramachandran_kl, torus_w2, wasserstein_1d_exact,
)
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

F32 = jnp.float32

# %% [markdown]
# ## 0. Configuration

# %%
@dataclass
class ModelConfig:
    """PITA's DIT3D for ALDP (3,179,267 parameters)."""
    hidden_size: int = 192
    cond_dim: int = 64
    n_blocks: int = 6
    n_heads: int = 6
    dropout: float = 0.1
    # Training rotates (y0, ∇ log π_β(y0)) by one uniform rotation per row.
    rotation_augment: bool = True


@dataclass
class TrainingConfig:
    """All training / sampling hyperparameters (ALDP-specific values)."""
    temperature: float = 300.0       # K
    # Loss (Section 13): σ_t² [ |s − DSM target|² + tsm_weight · 1{t < tsm_t_max} |s − TSM target|² ].
    # tsm_t_max = 0.0184 is σ_t = 0.19, PITA's TSM cutoff (time 0.2 of its EDM schedule).
    tsm_weight: float = 0.01
    tsm_t_max: float = 0.0184

    # SDE. Exponential time grid t_N = T > ... > t_1 = t_min, then 0 (Section 6).
    T: float = 2.0
    t_min: float = 1e-5
    n_sde_steps_train: int = 100
    n_sde_steps_eval: int = 1_000

    # Outer loop
    num_outer_steps: int = 100
    n_particles: int = 1_024         # K: backward-SDE paths per rung per outer step
    # Rung k's block goes to buffer[k+1] only if its path ESS · K ≥ ess_gate
    # (effective paths; 0 disables the gate). 10.24 = 1% of K.
    ess_gate: float = 10.24
    rung_chunk: int = 4              # rungs per jax.lax.map chunk in data generation

    # Inner loop
    n_inner_steps: int = 500
    batch_size: int = 64             # per rung, so (M+1)·64 rows per gradient step
    learning_rate: float = 5e-4
    grad_clip: float = 1.0

    # Clips: the network's score per coordinate in the SDE; the TSM target per
    # atom at this quantile of ‖∇ log π_β + y0‖ over the rung's SMC seed.
    score_clip: float = 1e4
    target_clip_quantile: float = 0.999

    # HMC in sampling coordinates (periodic_hmc): partial velocity refresh,
    # step size adapted per rung.
    refresh_rate: float = 0.07
    n_rejuv: int = 50                # HMC moves per particle per outer step

    # One number governs the SMC population, the per-rung seed and the buffer
    # capacity (the same particles at three points in their life). The seed
    # drains in n_smc_particles / n_particles = 4 outer steps — on rungs whose
    # blocks pass the ESS gate.
    n_smc_particles: int = 4_096
    smc_moves: int = 50
    smc_local_ess: float = 0.8

    # Quick evaluation every quick_eval_every outer steps (not after the last):
    # n_quick_eval paths at β = 1 on the training grid. It draws from the
    # training key stream (as the runs behind results/aldp did).
    quick_eval_every: int = 10
    n_quick_eval: int = 2_048

    # Final evaluation: n_eval_samples β = 1 paths in chunks of eval_chunk, on
    # their own key stream jax.random.key(eval_seed).
    n_eval_samples: int = 10_000
    eval_chunk: int = 2_500
    eval_seed: int = 7

    # --- Derived: seed size and buffer capacity are the SMC population -------
    @property
    def buffer_capacity(self) -> int:
        return self.n_smc_particles

    @property
    def n_seed_particles(self) -> int:
        return self.n_smc_particles

    @property
    def seed_flush_steps(self) -> float:
        """Outer steps until FIFO has fully retired the SMC seed (gate permitting)."""
        return self.n_smc_particles / self.n_particles


model_cfg = ModelConfig()
train_cfg = TrainingConfig()
print(f"buffer: {train_cfg.buffer_capacity:,}/rung = {train_cfg.seed_flush_steps:.3g} "
      f"generations at K={train_cfg.n_particles:,} (only blocks that pass the ESS gate count)")

resume: bool = False

NOTEBOOK_NAME = "aldp"
# Everything a run writes lives in results/<target>/ (results/<target>/seedN/
# for a seed sweep): checkpoint.pkl, log.txt and figures/ (gitignored) and one
# <run_id>/metrics.json per evaluated sample set (tracked).
RESULTS_DIR = Path("../../results") / NOTEBOOK_NAME
if SEED_DIR is not None:
    RESULTS_DIR = RESULTS_DIR / SEED_DIR
FIGURES_DIR = RESULTS_DIR / "figures"
for d in [RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.pkl"
LOG_PATH = RESULTS_DIR / "log.txt"
DATA_DIR = Path("../../data")

N_ATOMS = alanine.N_ATOMS
DIM = 3 * N_ATOMS


def log(msg: str) -> None:
    """Print and append to log.txt."""
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")

# %% [markdown]
# ## 1. Target Distribution (ALDP, ff14SB + OBC1, 300 K)
#
# $\log p(x) = -U(x)/k_BT$ on the L-alanine sector (−∞ for the D enantiomer),
# positions in nm, float64 (`progressive_is_tempering/alanine.py`). The force
# field is PITA's (AMBER ff14SB + OBC1 implicit solvent), stored in
# `data/aldp_ff14sb_obc1.npz`.

# %%
FF_PARAMS = alanine.load_force_field(DATA_DIR / "aldp_ff14sb_obc1.npz")
KT = alanine.KB * train_cfg.temperature      # kJ/mol

# %% [markdown]
# ## 2. Geometry Utilities, Reference Data, and Diagnostics
#
# The network works in $y = x / S$ with zero centre of mass, where the scale
# $S$ is the centred position standard deviation of the SMC block at $\beta=1$
# (data-free; about 0.165 nm). Random uniform rotations implement the rotation
# invariance (Sections 7 and 13).
#
# Ground truth is PITA's 300 K MD test split (10,000 frames, `data/redownload.sh`);
# its validation split, scored the same way, gives the noise floor of every
# metric. Besides the reported metrics (Section 16) the notebook tracks coarse
# basin populations ($\alpha_R$ / $\beta$ / $\alpha_L$ / remainder on the
# Ramachandran plane) and the three methyl-hydrogen label parities, which are
# exactly 1/2 under the target (swapping two identical hydrogens maps the
# target onto itself); MD cannot equilibrate them, so they are compared with
# 0.5, not with MD.

# %%
def center(x):  # (..., 22, 3)
    return x - x.mean(-2, keepdims=True)


def project_com(y):  # flat (..., 66)
    s = y.shape
    return center(y.reshape(*s[:-1], N_ATOMS, 3)).reshape(s)


def random_rotations(key, n, dtype=jnp.float64):
    """n uniform rotations (Haar): QR of a Gaussian matrix, signs fixed, det +1."""
    A = jax.random.normal(key, (n, 3, 3), dtype)
    Q, R = jnp.linalg.qr(A)
    Q = Q * jnp.sign(jnp.diagonal(R, axis1=-2, axis2=-1))[:, None, :]
    return Q.at[:, :, 0].multiply(jnp.linalg.det(Q)[:, None])


def rotate(Rm, v):  # Rm (n,3,3), v (n,22,3)
    return jnp.einsum("bij,baj->bai", Rm, v)


REGION_NAMES = ("alpha_R", "beta", "alpha_L", "remainder")


def region_labels(angles):
    """Coarse Ramachandran regions (reporting convention, not basin boundaries).

    Degrees: the negative-phi side also includes phi >= 150 so the extended
    region crosses the seam. There alpha_R has -100 <= psi < 50 and beta is the
    rest; otherwise alpha_L has -100 <= psi < 100 and the rest is remainder.
    """
    angles = np.asarray(angles)
    phi, psi = np.moveaxis(np.degrees((angles+np.pi) % (2*np.pi)-np.pi), -1, 0)
    negative_side = (phi < 0) | (phi >= 150)
    return np.where(negative_side, np.where((psi >= -100) & (psi < 50), 0, 1),
                    np.where((psi >= -100) & (psi < 100), 2, 3)).astype(int)


def methyl_label_parities(positions):
    """Signed ordering of the three H labels of each methyl (CB, ACE, NME),
    shape (..., 3); each parity is 1/2 under the target."""
    x = np.asarray(positions)
    values = []
    for carbon, hydrogens in ((10, (11, 12, 13)), (1, (0, 2, 3)), (18, (19, 20, 21))):
        a, b, c = (x[..., h, :]-x[..., carbon, :] for h in hydrogens)
        values.append(np.sum(np.cross(a, b)*c, axis=-1) > 0)
    return np.stack(values, axis=-1)


_IU, _JU = np.triu_indices(N_ATOMS, 1)
_MASSES = np.asarray(FF_PARAMS.masses)


def pairwise_distances(x):
    """(n, 22, 3) -> (n, 231) interatomic distances (nm)."""
    return np.linalg.norm(x[:, _IU] - x[:, _JU], axis=-1)


def radius_of_gyration(x):
    """Mass-weighted radius of gyration (nm), (n, 22, 3) -> (n,)."""
    c = (x * _MASSES[None, :, None]).sum(1, keepdims=True) / _MASSES.sum()
    return np.sqrt(((x - c) ** 2).sum(-1) @ _MASSES / _MASSES.sum())

# %% [markdown]
# ## 3. Tempering Problem: the Path in Internal Coordinates
#
# **Coordinates.** `alanine.aldp_zmatrix` gives 60 internal coordinates $q$
# (21 bonds, 20 angles, 19 torsions), rooted on the rigid C–CA–N backbone, so
# that torsion 0 is the CA improper whose sign is the handedness and $\phi$,
# $\psi$ are torsions 12 and 5. The sampling coordinates $z$ map bonds and
# angles to the real line (log / logit maps centred on a reference structure,
# local scales 0.005 nm and 0.15 rad) and keep the 18 other torsions as radian
# circles; the improper is restricted to $(0, \pi)$, which selects L-alanine
# exactly.
#
# **Path.** In $z$: reference $\rho_Z = \mathcal N(0, I_{42}) \times
# \mathrm{Unif}[-\pi,\pi)^{18}$, target $p_Z$ = the Boltzmann factor pulled back
# to $z$ (physical conformational volume and the chart Jacobian included once),
# and $\pi_\beta = \rho^{1-\beta} p^\beta$. The geometric path is
# coordinate-free: a Jacobian $J$ enters as $J^{1-\beta}J^\beta = J$. So the same
# path in Cartesian coordinates (zero CoM, uniform orientation) is
# $$\log\rho_X(x) = \log\rho_Z(z(x)) - \log|\partial q/\partial z| - \log\mathrm{vol}(q(x)),
# \qquad \log p_X(x) = -U(x)/k_BT,$$
# up to one common constant, and bridge weights and the ladder are the same in
# both coordinates. Both densities depend on $x$ only through $q(x)$: they are
# rotation- and translation-invariant.

# %%
def sample_reference(key, width=1.):
    """N(0, width² I) on R^42 and uniform on 18 radian circles."""
    normal_key, circle_key = jax.random.split(key)
    return jnp.concatenate((width*jax.random.normal(normal_key, (42,)),
                            jax.random.uniform(circle_key, (18,), minval=-jnp.pi, maxval=jnp.pi)))


def reference_logpdf(z, width=1.):
    """Normalized reference density in sampling coordinates, shape (..., 60)."""
    z = jnp.asarray(z)
    value = -.5*jnp.sum((z[..., :42]/width)**2, axis=-1)
    value -= 42*(jnp.log(width)+.5*jnp.log(2*jnp.pi))+18*jnp.log(2*jnp.pi)
    valid = jnp.all(jnp.isfinite(z), axis=-1) & jnp.all(
        (z[..., 42:] >= -jnp.pi) & (z[..., 42:] < jnp.pi), axis=-1)
    return jnp.where(valid, value, -jnp.inf)


class HybridALDP:
    """The ALDP annealing path in sampling coordinates z (for the MCMC) and in
    Cartesian coordinates x (for the network), with the maps between them."""

    def __init__(self, params, temperature=300.0, reference_width=1.0):
        self.cartesian = alanine.AlanineDipeptide(params, temperature)
        self.internal = alanine.AlanineDipeptideInternal(self.cartesian)
        geometry = self.internal.transform
        q0 = geometry.to_internal(self.cartesian.reference_positions)
        lower, upper = geometry.bounds
        # Torsion 0 (the CA improper) on (0, pi): the L sector. All bonds and
        # angles still fluctuate; the other torsions are circles.
        ci = geometry.torsion_slice.start
        lower, upper = lower.at[ci].set(0.), upper.at[ci].set(jnp.pi)
        scales = jnp.concatenate((jnp.full(21, alanine.BOND_SCALE),
                                  jnp.full(21, alanine.ANGLE_SCALE), jnp.ones(18)))
        self.coords = alanine.SamplingCoordinates(q0, scales, lower, upper,
                                                  tuple(range(ci + 1, geometry.dim)))
        self.problem = PTProblem(lambda key: sample_reference(key, reference_width),
                                 lambda z: reference_logpdf(z, reference_width),
                                 lambda z: self.coords.pullback(self.internal, z), exact_ref=True)
        self.transform = self.internal.transform
        self.periodic_indices = self.coords.periodic_indices  # z axes that are circles

    # ---- coordinate maps (single event; vmap for batches) -------------------
    def x_to_z(self, x):
        return self.coords.to_sampling(self.transform.to_internal(x))

    def z_to_x(self, z):
        """Canonical-frame positions (22, 3) of sampling coordinates z (60,)."""
        return self.transform.to_cartesian(self.coords.to_physical(z))

    # ---- Cartesian log densities on the shared path ---------------------------
    def log_p_x(self, x):
        """-U(x)/kT; -inf outside the L sector."""
        return self.cartesian(x)

    def log_rho_x(self, x):
        """The reference pushed to Cartesian coordinates (up to the path-wide
        constant); -inf off the chart (non-finite internals, D sector)."""
        q = self.transform.to_internal(x)
        ok_q = jnp.all(jnp.isfinite(q)) & self.transform.in_support(q)
        # the chirality torsion is an open interval (0, pi) in the chart
        lo, hi = self.coords.lower, self.coords.upper
        ok_q = ok_q & jnp.all(jnp.where(jnp.isinf(hi), q > lo, (q > lo) & (q < hi)) |
                              jnp.zeros_like(q, bool).at[jnp.asarray(self.periodic_indices)].set(True))
        q_safe = jnp.where(ok_q, q, self.coords.reference)
        z = self.coords.to_sampling(q_safe)
        val = (self.problem.logpdf_ref(z) - self.coords.log_abs_det_jacobian(z)
               - self.transform.log_volume(q_safe))
        return jnp.where(ok_q & jnp.isfinite(val), val, -jnp.inf)

    def log_pi_x(self, x, beta):
        lr, lp = self.log_rho_x(x), self.log_p_x(x)
        val = (1.0 - beta) * lr + beta * lp
        # β = 0 must not see −∞·0 from p (and vice versa)
        val = jnp.where(beta == 0, lr, jnp.where(beta == 1, lp, val))
        return jnp.where(jnp.isfinite(val), val, -jnp.inf)


TGT = HybridALDP(FF_PARAMS, train_cfg.temperature)
problem = TGT.problem
engine = make_engine(train_cfg.refresh_rate, TGT.periodic_indices)

# %% [markdown]
# **Cartesian ↔ sampling coordinates.** `z_to_y` builds the canonical-frame
# structure of $z$, applies a fresh uniform rotation, removes the centre of
# mass and divides by $S$; `y_to_x` undoes the scaling. Gradients of
# $\log\pi_\beta$ in $y$ are $S\,\nabla_x\log\pi_\beta(Sy)$; a non-finite
# gradient entry is set to 0.

# %%
def z_to_y(key, z, S):
    """z (n, 60) float64 -> y (n, 66) float32: canonical frame, uniform rotation, zero CoM, / S."""
    x = center(jax.vmap(TGT.z_to_x)(z))
    x = rotate(random_rotations(key, z.shape[0]), x)
    return (x / S).reshape(-1, DIM).astype(F32)


def y_to_x(y, S):
    return (y.astype(jnp.float64) * S).reshape(-1, N_ATOMS, 3)


def log_pi_y(y, beta, S):
    """log π_β in y-units (the constant 63 log S dropped). y flat (n, 66) -> (n,)."""
    x = y_to_x(y, S)
    return jax.vmap(lambda xx: TGT.log_pi_x(xx, beta))(x)


def grad_log_pi_y(y, beta, S):
    """∇_y log π_β(y) = S ∇_x log π_β(S y), float32 (n, 66)."""
    x = y_to_x(y, S)
    g = jax.vmap(jax.grad(lambda xx: TGT.log_pi_x(xx, beta)))(x)
    g = jnp.where(jnp.isfinite(g), g, 0.0)
    return (S * g).reshape(-1, DIM).astype(F32)


# Host-side callers (buffer seeding) use jitted versions; inside jit regions the
# plain functions are traced as part of the enclosing program.
grad_log_pi_y_jit = jax.jit(grad_log_pi_y, static_argnums=2)
z_to_y_jit = jax.jit(lambda key, z, S: z_to_y(key, z, S), static_argnums=2)

# %%
# ---- Reference (ground-truth) samples: PITA's 300 K MD splits ---------------
_log_p_x_batch = jax.jit(jax.vmap(TGT.log_p_x))
_to_internal_batch = jax.jit(jax.vmap(TGT.transform.to_internal))


def md_features(x):
    """Energies (kT), (phi, psi) and pairwise distances of (n, 22, 3) frames."""
    e = -np.asarray(_log_p_x_batch(jnp.asarray(x)))
    a = alanine.angles_from_internal(np.asarray(_to_internal_batch(jnp.asarray(x))))
    return e, a, pairwise_distances(x)


X_MD = np.load(DATA_DIR / "test_split_AL22-10000_T300.npy").reshape(-1, N_ATOMS, 3).astype(np.float64)
X_MD_VAL = np.load(DATA_DIR / "val_split_AL22-10000_T300.npy").reshape(-1, N_ATOMS, 3).astype(np.float64)
E_MD, A_MD, PD_MD = md_features(X_MD)
REG_MD = np.bincount(region_labels(A_MD), minlength=4) / len(A_MD)
print(f"MD test split: n={len(X_MD)}, αR/β/αL/rem = {' / '.join(f'{p:.4f}' for p in REG_MD)}, "
      f"<E> = {E_MD.mean() * KT:.2f} kJ/mol, Rg = {radius_of_gyration(X_MD).mean():.4f} nm")

# %% [markdown]
# ## 4. Forward Process
#
# OU SDE in $y$: $Y_t \mid Y_0 \sim \mathcal N(e^{-t}Y_0, \sigma_t^2 I)$,
# $\sigma_t^2 = 1-e^{-2t}$, applied in the training step (Section 13) with the
# noise and the result projected to zero CoM.
#
# ## 5. Score Network (PITA's DiT)
#
# **What is learned.** The reference $\rho$ is not the OU-stationary
# $\gamma = \mathcal N(0, I)$, so the network learns the excess score against
# $\gamma$: $s_\theta(y_t, t, \beta) \approx \nabla\log p^\beta_t(y_t) + y_t$.
# Writing $\pi_\beta = \gamma\,g$,
# $$s(y_t) = e^{-t}\,\mathbb E\bigl[\nabla\log\pi_\beta(Y_0) + Y_0 \mid Y_t = y_t\bigr],$$
# which reduces to the other notebooks' $e^{-t}\beta(y_0 - \nabla E)$ when
# $\rho = \gamma$. It is not zero at $\beta = 0$; buffer 0 holds exact $\rho$
# draws with their gradients.
#
# **Network.** `DIT3D` (`progressive_is_tempering/net/dit.py`): one token per
# atom, adaLN conditioning on the noise level and $\beta$, 6 transformer blocks
# with rotary position embeddings. With $F$ the DiT, the score is
# $$s = \frac{e^{-t}}{\sigma_t}\,F\Bigl(\tfrac18\log\sigma_t^2,\; y_t,\; \beta\Bigr),$$
# projected to zero CoM and clipped at `score_clip` per coordinate. Dropout is
# on in training only.

# %%
dit = DIT3D(n_particles=N_ATOMS, vocab_size=3, hidden_size=model_cfg.hidden_size, cond_dim=model_cfg.cond_dim,
            n_blocks=model_cfg.n_blocks, n_heads=model_cfg.n_heads, dropout=model_cfg.dropout)


def score_apply(params, y, t, b, deterministic=True, rng=None):
    """Excess score vs N(0, I): s(y_t, t, β), y (B, 66) f32, t, b (B,) f32 -> (B, 66) f32."""
    sig2 = 1.0 - jnp.exp(-2.0 * t)
    F = dit.apply(params, 0.125 * jnp.log(sig2), y, b, deterministic=deterministic,
                  rngs=None if rng is None else {"dropout": rng})
    s = (jnp.exp(-t) / jnp.sqrt(sig2))[:, None] * F
    sc = jnp.asarray(train_cfg.score_clip, F32)
    return jnp.clip(project_com(s), -sc, sc)

# %% [markdown]
# ## 6. CoM-Projected Backward SDE
#
# Exponential-integrator steps of the time-reversed OU process on the
# exponential grid ($N$ log-spaced levels from $T = 2$ down to
# $t_{\min} = 10^{-5}$, then 0), each followed by the zero-CoM projection, with
# the path log-weight $\log p - \log q$ per step and $\log\pi_\beta(y_0)$ at the
# end. A path whose weight is not finite gets $-\infty$.
#
# **Step coefficients.** `backward_sde` (training and quick evaluation)
# computes $\lambda = e^{-\Delta t}$, $1-\lambda$ and $\sigma^2 = 1-e^{-2\Delta t}$
# in float32, as the runs behind `results/aldp` did. Near $t_{\min}$ the steps
# are so small that $1-\lambda$ and $\sigma^2$ lose most of their digits to
# cancellation (about 20% error on the last of 1000 steps; at 5000 steps some
# drifts round to exactly 0). `backward_sde_eval` (Section 16) computes them as
# `-expm1(-Δt)` and `-expm1(-2Δt)`, which keeps them accurate: with it the
# $\beta=1$ samples no longer change between 1000, 2000 and 5000 steps.

# %%
def backward_sde(key, params, beta, n, n_steps):
    """K paths at β; returns y0 (n, 66) f32 and path log-weights (n,) f64."""
    times, dts = exponential_time_grid(train_cfg.T, n_steps, train_cfg.t_min)
    key, sk = jax.random.split(key)
    yT = project_com(jax.random.normal(sk, (n, DIM), F32))
    bvec = jnp.full((n,), beta, F32)

    def step(c, tdt):
        t, dt = tdt
        y, lw, rng = c
        rng, sk = jax.random.split(rng)
        lam, s2 = jnp.exp(-dt).astype(F32), (1.0 - jnp.exp(-2.0 * dt)).astype(F32)
        mu = lam * y + (1.0 - lam) * 2.0 * score_apply(params, y, jnp.full((n,), t, F32), bvec).astype(F32)
        yp = project_com(mu + jnp.sqrt(s2) * jax.random.normal(sk, y.shape, F32))
        lp = -0.5 * jnp.sum((y - lam * yp) ** 2, -1) / s2
        lq = -0.5 * jnp.sum((yp - mu) ** 2, -1) / s2
        return (yp, lw + (lp - lq).astype(jnp.float64), rng), None

    (y0, lw, _), _ = jax.lax.scan(step, (yT, 0.5 * jnp.sum(yT.astype(jnp.float64) ** 2, -1), key), (times, dts))
    lw = lw + log_pi_y(y0, beta, S)
    return y0, jnp.where(jnp.isfinite(lw), lw, -jnp.inf)


@partial(jax.jit, static_argnums=(3, 4))
def run_sde(key, params, beta, n, n_steps):
    return backward_sde(key, params, beta, n, n_steps)

# %% [markdown]
# ## 7. Rejuvenation in Sampling Coordinates
#
# A resampled Cartesian block is mapped to $z$ (orientation dropped), moved by
# `n_rejuv` HMC steps of rung $k$'s kernel (`periodic_hmc`, step size adapted on
# the rung's SMC block, Section 9), mapped back to the canonical frame and given
# a fresh uniform rotation. HMC leaves $\pi_{\beta,Z}$ invariant, and
# $\pi_{\beta,X}$ is rotation-invariant with the orientation independent of the
# conformation, so redrawing the rotation is an exact Gibbs step.
#
# ## 8. IS Bridging Between Adjacent Temperatures
#
# $$\log w_{k\to k+1} = (\beta_{k+1}-\beta_k)\,\bigl(\log p - \log\rho\bigr),$$
# read off the final HMC state (identical in $z$ and $x$). Each block is
# normalized to mean weight 1 before it enters a buffer.

# %%
@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def data_gen(keys, n, n_steps, n_rejuv, chunk, params, betas_act, betas_next, t_idx, mp_act):
    """Sections 6–8 for every active rung (jax.lax.map in chunks of `chunk` rungs)."""
    def one(args):
        k, b, bn, i, mp = args
        k_sde, k_res, k_ext, k_mc, k_rot = jax.random.split(k, 5)
        y0, lw = backward_sde(k_sde, params, b, n, n_steps)
        ess = normalized_ess(lw)
        any_ok = jnp.any(jnp.isfinite(lw))
        idx = systematic_resample(k_res, jnp.where(any_ok, lw, 0.0), n)
        yr = y0[idx]
        # to z; invalid rows (only possible if no finite weight) -> reference centre
        x = y_to_x(yr, S)
        z = jax.vmap(TGT.x_to_z)(x)
        okz = jnp.all(jnp.isfinite(z), -1) & jnp.isfinite(lw[idx])
        z = jnp.where(okz[:, None], z, TGT.coords.to_sampling(TGT.coords.reference))
        kern = engine(problem, b, i)
        st = batch_init_extend(k_ext, z, kern)
        st = repeated_batch_move(k_mc, st, mp, kern, n_rejuv)
        acc = jnp.mean(st.accepted)
        bridge = (bn - b) * (st.logpdf_final_target - st.logpdf_ref)
        y_new = z_to_y(k_rot, st.position, S)
        g_next = grad_log_pi_y(y_new, bn, S)          # target of buffer k+1 at its own β
        return y_new, bridge, ess, acc, g_next, any_ok

    return jax.lax.map(one, (keys, betas_act, betas_next, t_idx, mp_act), batch_size=chunk)


@jax.jit
def normalize_lw(lw):
    """Mean weight 1 (uniform if no weight is finite)."""
    tot = logsumexp(lw)
    return jnp.where(jnp.isfinite(tot), lw - tot + jnp.log(lw.shape[0]), jnp.zeros_like(lw))

# %% [markdown]
# ## 9. Temperature Ladder via SMC (in Sampling Coordinates)
#
# `periodic_hmc.smc_sampler` anneals `n_smc_particles` from $\rho_Z$ to $p_Z$,
# choosing each $\beta$ so that the reweighting ESS drops to `smc_local_ess`
# (about 40 levels), with `smc_moves` HMC moves per stage. It returns the
# particle block at every rung. The scale $S$ is read off the $\beta=1$ block,
# and every rung's HMC step size is adapted on that rung's block; those kernels
# are used as-is for rejuvenation.

# %%
_ckpt = None
if resume and CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, "rb") as f:
        _ckpt = pickle.load(f)
    log(f"resuming after outer step {_ckpt['outer_step']}")

if _ckpt is None:
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            n_attempts = f.read().count("Attempt")
    else:
        n_attempts = 0
    log(f"\nAttempt {n_attempts + 1} ({time.strftime('%F %T')})")
log(f"seed={SEED} devices={jax.devices()}\nmodel: {model_cfg}\ntrain: {train_cfg}")

if _ckpt is None:
    key, k_smc = jax.random.split(key)
    t0 = time.time()
    pops, logz_smc, ladder = smc_sampler(
        k_smc, problem, n_particles=train_cfg.n_smc_particles, engine=engine,
        n_mcmc_steps=train_cfg.smc_moves, local_ess_ratio=train_cfg.smc_local_ess,
        verbose=True, keep_all_states=True)
    pops = jnp.asarray(pops)  # (levels, N, 60), float64
    betas_list = [float(b) for b in ladder]
    log(f"SMC: {len(betas_list)} levels in {time.time() - t0:.0f}s; logZ {dict((k, float(v)) for k, v in logz_smc.items())}")
    log("ladder: " + " ".join(f"{b:.3g}" for b in betas_list))
    # data-free scale: centred position std of the β=1 block
    x_top = center(jax.vmap(TGT.z_to_x)(pops[-1]))
    S = float(jnp.std(x_top))
    log(f"S = {S:.4f} nm (centred std of the SMC beta=1 block)")

    # per-rung HMC step size, adapted on each level's SMC population
    @jax.jit
    def calibrate(k, z_levels, temps):
        L = z_levels.shape[0]

        def one(kk, i, t, z):
            kern = engine(problem, t, i)
            ka, kb = jax.random.split(kk)
            return kern.adapt_params(kb, batch_init_extend(ka, z, kern))
        return jax.vmap(one)(jax.random.split(k, L), jnp.arange(L), temps, z_levels)

    key, k_cal = jax.random.split(key)
    mcmc_params = calibrate(k_cal, pops, jnp.asarray(betas_list))
    log("HMC eps per level (every 5th): " + " ".join(f"{float(e):.3g}" for e in np.asarray(mcmc_params.scaled_cov.eps)[::5]))
else:
    betas_list, S, mcmc_params = _ckpt["betas_list"], _ckpt["S"], _ckpt["mcmc_params"]
betas = jnp.asarray(betas_list)
M = len(betas_list) - 1
M_active = M - 1
log(f"M = {M}, active rungs 1..{M - 1}")

# %% [markdown]
# ## 10. Diagnostics of a Sample Set
#
# `describe` logs the invalid fraction (D enantiomer or non-finite energy),
# basin populations, methyl parities, mean energy and, given path weights,
# the path ESS. The SMC block at $\beta = 1$ is the first set it sees.

# %%
def describe(y, name, lw=None):
    """Region populations, methyl parities, energy (kJ/mol), invalid fraction for y (n, 66)."""
    x = np.asarray(y_to_x(jnp.asarray(y), S))
    lp = np.asarray(jax.jit(jax.vmap(TGT.log_p_x))(jnp.asarray(x)))
    ok = np.isfinite(lp)
    xv = x[ok]
    q = np.asarray(jax.jit(jax.vmap(TGT.transform.to_internal))(jnp.asarray(xv)))
    regions = region_labels(alanine.angles_from_internal(q))
    pr = np.bincount(regions, minlength=len(REGION_NAMES)) / max(len(regions), 1)
    par = methyl_label_parities(xv).mean(0) if len(xv) else np.full(3, np.nan)
    e = -lp[ok] * KT
    m = {"n": int(len(x)), "invalid_frac": float(1 - ok.mean()),
         "region_probabilities": pr.tolist(), "methyl_positive": np.asarray(par).tolist(),
         "energy_mean_kJmol": float(e.mean()) if len(e) else None}
    if lw is not None:
        m["path_ess"] = float(normalized_ess(jnp.asarray(lw)))
    log(f"  {name:<10} n={m['n']:6d} invalid={m['invalid_frac']:.3f}  "
        f"αR/β/αL/rem = {' / '.join(f'{p:.4f}' for p in pr)}  "
        f"methyl+ = {' '.join(f'{p:.3f}' for p in m['methyl_positive'])}  "
        f"<E> = {m['energy_mean_kJmol'] if m['energy_mean_kJmol'] is None else round(m['energy_mean_kJmol'], 2)} kJ/mol"
        + (f"  path ESS {m['path_ess']:.4f}" if lw is not None else ""))
    return m

# %% [markdown]
# ## 11. Per-Temperature Replay Buffers (Seeded from SMC) and Clips
#
# Each row stores $(y, \log w)$ and $\nabla_y\log\pi_{\beta_k}(y)$ at the rung's
# own $\beta$, computed once when the row is written.
#
# - Buffer 0 ($\beta=0$): `n_smc_particles` exact $\rho$ draws, uniform
#   weights, frozen. Their gradients are not zero ($\rho \ne \gamma$).
# - Buffers $1,\ldots,M$: the rung's whole SMC block (mapped to $y$, one random
#   rotation each), uniform weights; buffer 1 is frozen, buffers $2,\ldots,M$
#   also receive bridge blocks that pass the ESS gate.
#
# **TSM target clip.** Per rung, the 99.9% quantile of the per-atom norm of
# $\nabla\log\pi_\beta + y_0$ over the rung's seed; it trims only outliers.

# %%
if _ckpt is None:
    buffers = [ReplayBuffer(train_cfg.buffer_capacity, DIM, store_grads=True) for _ in range(M + 1)]
    cap = train_cfg.buffer_capacity
    # buffer 0: exact ρ draws (frozen). Targets are NOT zero here: ∇ log ρ.
    key, k0, k1 = jax.random.split(key, 3)
    z0 = jax.vmap(problem.rvs_ref)(jax.random.split(k0, cap))
    y0 = z_to_y_jit(k1, z0, S)
    buffers[0].add(y0, jnp.zeros(cap, F32), grad_log_pi_y_jit(y0, 0.0, S))
    buffers[0].freeze()
    # buffers 1..M: whole SMC block at that level, uniform weights
    for k in range(1, M + 1):
        key, kk = jax.random.split(key)
        yk = z_to_y_jit(kk, pops[k], S)
        buffers[k].add(yk, jnp.zeros(cap, F32), grad_log_pi_y_jit(yk, betas_list[k], S))
    # Buffer 1 is never written during training (the outer loop fills 2..M).
    buffers[1].freeze()
    log(f"SMC seed at beta=1 (all {cap} particles):")
    describe(buffers[M].data[: buffers[M].size], "SMC seed")
    del pops
else:
    buffers = _ckpt["buffers"]


def _clip_norm(buf, q):
    rows = buf.rows[: buf.size]
    v = (rows[:, DIM:] + rows[:, :DIM]).reshape(-1, N_ATOMS, 3)
    return float(jnp.quantile(jnp.linalg.norm(v, axis=-1), q))


if _ckpt is None:
    clip_norms = jnp.asarray([_clip_norm(b, train_cfg.target_clip_quantile) for b in buffers], F32)
else:
    clip_norms = _ckpt["clip_norms"]
log("per-atom target clip (every 5th rung): " + " ".join(f"{float(c):.1f}" for c in np.asarray(clip_norms)[::5]))

# %%
if _ckpt is None:
    key, kp = jax.random.split(key)
    params = dit.init(kp, jnp.zeros(1, F32), jnp.zeros((1, DIM), F32), jnp.zeros(1, F32))
    # x64 mode makes some initializers (e.g. LayerNorm ones) float64: keep the network float32
    params = jax.tree_util.tree_map(lambda a: a.astype(F32), params)
else:
    params = _ckpt["params"]
_np = sum(x.size for x in jax.tree_util.tree_leaves(params))
_dt = {str(x.dtype) for x in jax.tree_util.tree_leaves(params)}
log(f"DiT params: {_np:,} (PITA DIT3D 3,179,267), dtypes {_dt}")

# %% [markdown]
# ## 12. Data Generation (Outer Loop) and the ESS Gate
#
# `data_gen` (Section 8) runs, for every active rung $k \in \{1,\ldots,M-1\}$:
# backward SDE at $\beta_k$ (K paths, training grid) → systematic resampling →
# rejuvenation → bridge weights and gradients at $\beta_{k+1}$. The rungs run
# with `jax.lax.map` in chunks of `rung_chunk` (memory).
#
# **ESS gate.** Rung $k$'s block is added to buffer $k+1$ only if its path ESS
# $\times K \geq$ `ess_gate`. With path ESS near $1/K$ a block is a few distinct
# paths replicated $K$ times; writing it would replace a quarter of the
# buffer's seed with near-copies.
#
# ## 13. Training Step (Inner Loop): DSM + TSM
#
# Per step, `batch_size` rows from every buffer by systematic resampling on the
# stored weights, $t$ log-uniform on $[t_{\min}, T]$, one random rotation per
# row applied to $y_0$ and its gradient alike, $y_t = e^{-t}y_0 + \sigma_t\varepsilon$
# (zero CoM). Two regression targets for the excess score:
#
# - DSM: $\nabla\log p(y_t \mid y_0) + y_t = y_t - \varepsilon/\sigma_t$, at every $t$;
# - TSM: $e^{-t}\,\mathrm{clip}_k\bigl(\nabla\log\pi_\beta(y_0) + y_0\bigr)$ (per-atom clip, Section 11).
#
# $$\mathcal L = \mathbb E\,\sigma_t^2\Bigl[\lVert s_\theta - \mathrm{DSM}\rVert^2
#   + 0.01\,\mathbb 1\{t < 0.0184\}\,\lVert s_\theta - \mathrm{TSM}\rVert^2\Bigr].$$
# Both targets have the same conditional expectation, so the mixture is
# unbiased; the TSM term only acts below $\sigma_t = 0.19$, as in PITA. All
# `n_inner_steps` steps of one outer iteration run as one compiled
# `jax.lax.scan`.

# %%
optimizer = optax.chain(optax.clip_by_global_norm(train_cfg.grad_clip), optax.adam(train_cfg.learning_rate))
opt_state = optimizer.init(params) if _ckpt is None else _ckpt["opt_state"]


def clip_per_atom(v, cn):  # v (B,66), cn (B,)
    vr = v.reshape(-1, N_ATOMS, 3)
    nr = jnp.linalg.norm(vr, axis=-1, keepdims=True)
    return (vr * jnp.minimum(1.0, cn[:, None, None] / (nr + 1e-6))).reshape(v.shape)


@jax.jit
def train_step(params, opt_state, rows, t, beta, cn, key):
    y0, g = rows[:, :DIM], rows[:, DIM:]
    k_rot, k_noise, k_drop = jax.random.split(key, 3)
    if model_cfg.rotation_augment:   # π_β rotation-invariant: rotate y0 and its score together
        Rm = random_rotations(k_rot, y0.shape[0], F32)
        y0 = rotate(Rm, y0.reshape(-1, N_ATOMS, 3)).reshape(y0.shape)
        g = rotate(Rm, g.reshape(-1, N_ATOMS, 3)).reshape(g.shape)
    sig = jnp.sqrt(1.0 - jnp.exp(-2.0 * t))
    eps = project_com(jax.random.normal(k_noise, y0.shape, F32))
    yt = project_com(jnp.exp(-t)[:, None] * y0) + sig[:, None] * eps
    # TSM target for the excess score (vs N(0, I)): e^{-t} (∇ log π_β(y0) + y0)
    tgt_tsm = jax.lax.stop_gradient(project_com(jnp.exp(-t)[:, None] * clip_per_atom(g + y0, cn)))
    # DSM target for the excess score: ∇ log p(y_t | y0) + y_t = y_t − ε/σ_t
    tgt_dsm = jax.lax.stop_gradient(yt - eps / sig[:, None])
    w_tsm = train_cfg.tsm_weight * (t < train_cfg.tsm_t_max).astype(F32)

    def loss_fn(p):
        pred = score_apply(p, yt, t, beta, deterministic=False, rng=k_drop)
        # both terms on the same σ_t² weighting; TSM only below tsm_t_max (PITA-style)
        l_dsm = jnp.sum((pred - tgt_dsm) ** 2, -1)
        l_tsm = jnp.sum((pred - tgt_tsm) ** 2, -1)
        return jnp.mean(sig ** 2 * (l_dsm + w_tsm * l_tsm))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    upd, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, upd), opt_state, loss


@partial(jax.jit, static_argnums=(6, 7))
def inner_loop(params, opt_state, rows_all, lw_all, betas_l, clip_l, n_steps, B, key):
    """All inner steps in one program. rows_all (L, cap, 2·DIM), lw_all (L, cap).

    Per step and rung: B rows by systematic resampling on that buffer's log
    weights (as ReplayBuffer.sample_with_grads), t log-uniform on [t_min, T].
    """
    L = rows_all.shape[0]

    def body(carry, k):
        p, o = carry
        k_s, k_t, k_step = jax.random.split(k, 3)
        idx = jax.vmap(lambda kk, lw: systematic_resample(kk, lw, B))(jax.random.split(k_s, L), lw_all)
        rows = jnp.take_along_axis(rows_all, idx[:, :, None], axis=1).reshape(L * B, -1)
        t = jnp.exp(jax.random.uniform(k_t, (L * B,), F32, np.log(train_cfg.t_min), np.log(train_cfg.T)))
        beta = jnp.repeat(betas_l, B)
        cn = jnp.repeat(clip_l, B)
        p, o, lv = train_step(p, o, rows, t, beta, cn, k_step)
        return (p, o), lv

    (params, opt_state), losses = jax.lax.scan(body, (params, opt_state), jax.random.split(key, n_steps))
    return params, opt_state, jnp.mean(losses)

# %% [markdown]
# ## 14. Periodic Evaluation
#
# Every `quick_eval_every` outer steps (except after the last): a $\beta = 1$
# backward SDE on `n_quick_eval` paths on the training grid, described as in
# Section 10. Unlike the other notebooks it draws from the training key
# stream, as the runs behind `results/aldp` did.
#
# ## 15. Training Loop
#
# ```
# for outer_step in 1 … num_outer_steps:
#     for k = 1, …, M−1:  (jax.lax.map, chunks of rung_chunk)
#         backward SDE at β_k → (y₀, log w_path); systematic resample
#         y → z, n_rejuv HMC moves at rung k, z → y with a fresh rotation
#         bridge weights to β_{k+1}, ∇ log π_{β_{k+1}}
#     buffer[k+1].add(block, normalize(log w), ∇)   if path ESS·K ≥ ess_gate
#     n_inner_steps Adam steps on the DSM + TSM loss over all M+1 buffers
# ```
#
# Each outer step logs the loss, the path ESS over rungs (min / median / max
# and the top three), how many blocks were written, and the median HMC
# acceptance.

# %%
start = 0
loss_history, ess_history = [], []
if _ckpt is not None:
    start, loss_history, ess_history = _ckpt["outer_step"], _ckpt["loss_history"], _ckpt["ess_history"]
    key = jax.random.wrap_key_data(_ckpt["key"])
mp_active = jax.tree_util.tree_map(lambda leaf: leaf[1:M], mcmc_params)   # levels 1..M-1
t_run = time.time()
for outer in range(start, train_cfg.num_outer_steps):
    t0 = time.time()
    key, kg = jax.random.split(key)
    ys, bridge, pess, acc, gs, any_ok = data_gen(
        jax.random.split(kg, M_active), train_cfg.n_particles, train_cfg.n_sde_steps_train,
        train_cfg.n_rejuv, train_cfg.rung_chunk, params,
        betas[1:-1], betas[2:], jnp.arange(1, M), mp_active)
    pess, any_ok = np.asarray(pess), np.asarray(any_ok)
    written = []
    for i in range(M_active):
        if not any_ok[i] or (train_cfg.ess_gate > 0 and pess[i] * train_cfg.n_particles < train_cfg.ess_gate):
            continue
        buffers[i + 2].add(ys[i], normalize_lw(bridge[i]).astype(F32), gs[i])
        written.append(i + 2)
    ess_history.append(pess.tolist())
    t_gen = time.time() - t0

    assert all(b.size == b.capacity for b in buffers), "buffers are seeded full; partial buffers unsupported"
    t_tr = time.time()
    key, ki = jax.random.split(key)
    params, opt_state, mean_loss = inner_loop(
        params, opt_state, jnp.stack([b.rows for b in buffers]), jnp.stack([b.log_weights for b in buffers]),
        jnp.asarray(betas_list, F32), clip_norms, train_cfg.n_inner_steps, train_cfg.batch_size, ki)
    loss_history.append(float(mean_loss))
    t_train = time.time() - t_tr
    q = np.quantile(pess, [0, .5, 1])
    log(f"[outer {outer + 1:3d}/{train_cfg.num_outer_steps}] loss={loss_history[-1]:.2f}  "
        f"path ESS min/med/max={q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}  top3={np.round(pess[-3:], 4).tolist()}  "
        f"written={len(written)}/{M_active} (max buffer {max(written) if written else '-'})  "
        f"hmc acc med={float(np.median(np.asarray(acc))):.2f}  gen={t_gen:.0f}s train={t_train:.0f}s step={time.time() - t0:.0f}s "
        f"total={time.time() - t_run:.0f}s")
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump({"params": params, "opt_state": opt_state, "outer_step": outer + 1,
                     "loss_history": loss_history, "ess_history": ess_history,
                     "key": jax.random.key_data(key), "betas_list": betas_list, "S": S,
                     "mcmc_params": mcmc_params, "buffers": buffers, "clip_norms": clip_norms}, f)
    if (train_cfg.quick_eval_every and (outer + 1) % train_cfg.quick_eval_every == 0
            and outer + 1 < train_cfg.num_outer_steps):
        key, ke = jax.random.split(key)
        y_q, lw_q = run_sde(ke, params, betas[-1], train_cfg.n_quick_eval, train_cfg.n_sde_steps_train)
        log(f"  [quick eval, {train_cfg.n_quick_eval} paths, {train_cfg.n_sde_steps_train} steps]")
        describe(y_q, "SDE raw", lw_q)

# %% [markdown]
# ## 16. Final Evaluation
#
# The backward SDE at $\beta = 1$ on the fine grid (`n_sde_steps_eval` = 1000
# steps, same $T$ and $t_{\min}$) with the accurate step coefficients of
# Section 6, `n_eval_samples` = 10,000 paths (PITA's evaluation size) in chunks,
# on its own key stream `jax.random.key(eval_seed)`.
#
# | # | Sample set (`run_id`) | What it isolates |
# |---|---|---|
# | 1 | Backward SDE output, **no** reweighting (`sde_raw`) | the learned score alone |
# | 2 | Same run, systematically resampled on the path weights (`sde_is`) | score **+** path-space correction |
# | 3 | `buffer[M]`, resampled on its stored weights (`buffer`) | the training data at $\beta=1$ (4096 rows) |
#
# (1) and (2) are the same particles. On ALDP the $\beta=1$ path ESS is
# $3\cdot10^{-4}$ to $2\cdot10^{-2}$ over the committed seeds, so (2) is built
# from a few to a few hundred paths and varies a lot between seeds.

# %%
def backward_sde_eval(key, params, n, n_steps):
    """β = 1 backward SDE with accurate step coefficients (expm1); y0 (n, 66) f32
    and path log-weights (n,) f64. log π_1 = log p, evaluated directly."""
    times, dts = exponential_time_grid(train_cfg.T, n_steps, train_cfg.t_min)
    key, sk = jax.random.split(key)
    yT = project_com(jax.random.normal(sk, (n, DIM), F32))
    bvec = jnp.ones((n,), F32)

    def step(c, tdt):
        t, dt = tdt
        y, lw, rng = c
        rng, sk = jax.random.split(rng)
        c, s2 = (-jnp.expm1(-dt)).astype(F32), (-jnp.expm1(-2.0 * dt)).astype(F32)
        lam = (1.0 - c).astype(F32)
        mu = lam * y + c * 2.0 * score_apply(params, y, jnp.full((n,), t, F32), bvec).astype(F32)
        yp = project_com(mu + jnp.sqrt(s2) * jax.random.normal(sk, y.shape, F32))
        lp = -0.5 * jnp.sum((y - lam * yp) ** 2, -1) / s2
        lq = -0.5 * jnp.sum((yp - mu) ** 2, -1) / s2
        return (yp, lw + (lp - lq).astype(jnp.float64), rng), None

    (y0, lw, _), _ = jax.lax.scan(step, (yT, 0.5 * jnp.sum(yT.astype(jnp.float64) ** 2, -1), key), (times, dts))
    lw = lw + jax.vmap(TGT.log_p_x)(y_to_x(y0, S))
    return y0, jnp.where(jnp.isfinite(lw), lw, -jnp.inf)


@partial(jax.jit, static_argnums=(2, 3))
def run_sde_eval(key, params, n, n_steps):
    return backward_sde_eval(key, params, n, n_steps)


def systematic_np(lw, n, seed=0):
    """Systematic resampling of n indices on log-weights lw (NumPy, fixed seed):
    the resampling that produced the committed `sde_is` records."""
    w = np.exp(lw - lw.max())
    w /= w.sum()
    u = (np.random.default_rng(seed).random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), u), len(w) - 1)


log("final evaluation")
t0 = time.time()
eval_key = jax.random.key(train_cfg.eval_seed)
ys_e, lws_e = [], []
for _ in range(train_cfg.n_eval_samples // train_cfg.eval_chunk):
    eval_key, ke = jax.random.split(eval_key)
    y_c, lw_c = run_sde_eval(ke, params, train_cfg.eval_chunk, train_cfg.n_sde_steps_eval)
    ys_e.append(np.asarray(y_c))
    lws_e.append(np.asarray(lw_c))
y_ev, lw_ev = np.concatenate(ys_e), np.concatenate(lws_e)
ess_eval = float(normalized_ess(jnp.asarray(lw_ev)))
log(f"final SDE ({len(y_ev)} × {train_cfg.n_sde_steps_eval} steps): {time.time() - t0:.0f}s, "
    f"path ESS {ess_eval:.4f}")
y_is = y_ev[systematic_np(lw_ev, len(y_ev))]
# buffer[M]: the key schedule of the runs behind results/aldp, whose final
# evaluation split the training key once per evaluation chunk before drawing
# (kr, kb); kr is unused here.
for _ in range(train_cfg.n_eval_samples // train_cfg.eval_chunk):
    key, _ = jax.random.split(key)
key, kr, kb = jax.random.split(key, 3)
y_buf, _ = buffers[M].sample(kb, min(train_cfg.n_eval_samples, buffers[M].size))
y_buf = np.asarray(y_buf)
buf_ess = float(normalized_ess(buffers[M].log_weights[:buffers[M].size]))

sample_sets = [
    ("sde_raw", "Backward SDE (raw, no IS)", "SDE raw", y_ev,
     f"unweighted, {len(y_ev):,} paths"),
    ("sde_is", "Backward SDE + IS resample", "SDE + IS", y_is,
     f"path ESS={ess_eval:.4f} (≈{ess_eval * len(y_ev):,.1f} eff.)"),
    ("buffer", f"buffer[M]  β={float(betas[M]):.3f}", "buffer[M]", y_buf,
     f"{len(y_buf):,} rows, buffer ESS={buf_ess:.3f}"),
]
for run_id, _label, short, y, _sub in sample_sets:
    describe(y, short, lw_ev if run_id == "sde_raw" else None)

# %% [markdown]
# ### Reported metrics (PITA's Table 2)
#
# Computed on the valid samples (finite energy: L-alanine, no blow-up) of each
# set against the 10,000 MD test frames, with `progressive_is_tempering.utils.metrics`:
#
# - **Rama-KL** — KL(MD ‖ sample) of 100 × 100 $(\phi,\psi)$ histograms
#   (`ramachandran_kl`). PITA does not release its definition; compare with
#   the MD floor below, which also shows the sample-size dependence (a
#   4096-sample set has a floor about twice that of 10,000).
# - **Energy-𝒲₁, Energy-𝒲₂** — exact 1-D Wasserstein distances between the
#   energy distributions, in $k_BT$ (`wasserstein_1d_exact`).
# - **T-𝒲₂** — exact 2-Wasserstein on the $(\phi,\psi)$ torus (`torus_w2`).
# - **Distance-𝒲₂** (`w2_pooled`) — pooled interatomic distances (nm), as for
#   the other particle targets.
#
# PITA's TICA-𝒲₂ is not reported: its TICA model needs time-ordered
# trajectories, and the released MD splits are shuffled frames.
#
# $(\phi,\psi)$ are the biochemical backbone dihedrals, from the internal
# coordinates (`alanine.angles_from_internal`). Each set is written as its own
# record, `results/aldp/[seedN/]<run_id>/metrics.json`.

# %%
def aldp_metrics(x):
    """Reported metrics of (n, 22, 3) positions (nm) against the MD test split."""
    e = -np.asarray(_log_p_x_batch(jnp.asarray(x)))
    ok = np.isfinite(e)
    x, e = x[ok], e[ok]
    a = alanine.angles_from_internal(np.asarray(_to_internal_batch(jnp.asarray(x))))
    reg = np.bincount(region_labels(a), minlength=4) / len(a)
    return {
        "rama_kl": ramachandran_kl(A_MD, a),
        "energy_w1_kT": wasserstein_1d_exact(E_MD, e, 1),
        "energy_w2_kT": wasserstein_1d_exact(E_MD, e, 2),
        "t_w2": torus_w2(a, A_MD),
        "w2_pooled": distance_w2(pairwise_distances(x), PD_MD),
        "n": int(ok.size), "n_valid": int(ok.sum()), "invalid_frac": float(1 - ok.mean()),
        "region_probabilities": reg.tolist(), "region_probabilities_md": REG_MD.tolist(),
        "region_tv": float(0.5 * np.abs(reg - REG_MD).sum()),
        "methyl_positive": methyl_label_parities(x).mean(0).tolist(),
        "energy_mean_kJmol": float(e.mean() * KT), "energy_mean_md_kJmol": float(E_MD.mean() * KT),
        "rg_mean_nm": float(radius_of_gyration(x).mean()),
        "n_ref_samples": int(len(X_MD)),
    }


def _fmt(m):
    return (f"{m['n']:>6}{m['rama_kl']:>9.3f}{m['energy_w1_kT']:>9.3f}{m['energy_w2_kT']:>9.3f}"
            f"{m['t_w2']:>8.3f}{m['w2_pooled']:>10.5f}{m['region_tv']:>8.4f}{m['invalid_frac']:>8.3f}")


log(f"{'set':<12}{'n':>6}{'RamaKL':>9}{'E-W1':>9}{'E-W2':>9}{'T-W2':>8}{'Dist-W2':>10}{'reg TV':>8}{'invalid':>8}")
_rng = np.random.default_rng(0)
for _name, _x in [("MD val", X_MD_VAL), ("MD val 4096", X_MD_VAL[_rng.choice(len(X_MD_VAL), 4096, replace=False)])]:
    log(f"{_name:<12}" + _fmt(aldp_metrics(_x)))
for run_id, _label, short, y, _sub in sample_sets:
    m = aldp_metrics(np.asarray(y_to_x(jnp.asarray(y), S)))
    m.update(path_ess=ess_eval, buffer_ess=buf_ess,
             ess=None if run_id == "buffer" else ess_eval)
    write_record(RESULTS_DIR, NOTEBOOK_NAME, run_id, m, seed=SEED,
                 config={"model": model_cfg, "training": train_cfg, "betas": betas_list, "S": S},
                 notes=f"Sample set: {_label} ({_sub}).")
    log(f"{short:<12}" + _fmt(m))
log(f"done {time.strftime('%F %T')}")
print(f"\nRecords written under {RESULTS_DIR}/<run_id>/metrics.json")

# %% [markdown]
# ### Figures
#
# Ramachandran plots of the MD test set and the three sample sets, energy
# densities against MD, and the training curves (loss and per-rung path ESS),
# then the temperature ladder.

# %%
_sets = [("MD test", X_MD)] + [(short, np.asarray(y_to_x(jnp.asarray(y), S)))
                               for _, _, short, y, _ in sample_sets]
fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
for ax, (name, x) in zip(axes[0], _sets):
    e = -np.asarray(_log_p_x_batch(jnp.asarray(x)))
    a = alanine.angles_from_internal(np.asarray(_to_internal_batch(jnp.asarray(x[np.isfinite(e)]))))
    ax.hist2d(np.degrees(a[:, 0]), np.degrees(a[:, 1]), bins=90, range=[[-180, 180], [-180, 180]],
              norm=matplotlib.colors.LogNorm(), cmap="magma")
    ax.set_xlabel("φ (deg)", color=C_INK_2)
    ax.set_ylabel("ψ (deg)", color=C_INK_2)
    ax.set_title(f"{name} (n={len(a):,})", color=C_INK, fontsize=11)
for ax, (name, x) in zip(axes[1, 1:], _sets[1:]):
    e = -np.asarray(_log_p_x_batch(jnp.asarray(x)))
    plot_value_comparison(ax, e[np.isfinite(e)] * KT, E_MD * KT, "energy U (kJ/mol)", name)
    ax.set_title(f"Energy — {name}", color=C_INK, fontsize=11)
ax = axes[1, 0]
ax.plot(np.arange(1, len(loss_history) + 1), loss_history, color=C_TEST, lw=1.5, label="loss")
ax.set_yscale("log")
ax.set_xlabel("outer step", color=C_INK_2)
ax.set_ylabel("DSM + TSM loss", color=C_INK_2)
ax2 = ax.twinx()
_eh = np.asarray(ess_history)
ax2.plot(np.arange(1, len(_eh) + 1), np.median(_eh, 1), color=C_REF, lw=1.2, label="path ESS (median rung)")
ax2.axhline(train_cfg.ess_gate / train_cfg.n_particles, color=C_INK_2, ls=":", lw=1, label="ESS gate")
ax2.set_yscale("log")
ax2.legend(fontsize=8, frameon=False, loc="upper right")
ax.set_title("Training", color=C_INK, fontsize=11)
style_axis(ax)
fig.suptitle(f"ALDP seed {SEED}: final evaluation at β=1 vs MD (ff14SB + OBC1, 300 K)", color=C_INK, fontsize=12)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "aldp_ramachandran_energy_and_loss.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, len(betas_list)), betas_list[1:], "o-", color=C_TEST, ms=4)
ax.set_yscale("log")
ax.set_xlabel("temperature level $k$", color=C_INK_2, fontsize=10)
ax.set_ylabel("$\\beta_k$", color=C_INK_2, fontsize=10)
ax.set_title(f"Temperature ladder ({len(betas_list)} levels)", color=C_INK, fontsize=11)
style_axis(ax)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "aldp_temperature_ladder.png", dpi=150)
plt.close(fig)
print(f"Ladder: {[f'{b:.4g}' for b in betas_list]}")
