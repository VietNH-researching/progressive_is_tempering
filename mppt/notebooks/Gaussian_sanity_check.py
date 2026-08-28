# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Sanity Check: Sampling a Mixture of Two Gaussians with Parallel Tempering
#
# Target is a two-component mixture:
#
# $$
# p_{\text{target}}(x) \;=\; w_1\,\mathcal{N}(x;\mu_1, C_1) + w_2\,\mathcal{N}(x;\mu_2, C_2),
# \qquad w_1 = 0.7,\; w_2 = 0.3.
# $$
#
# - $\mu_1 = 5\cdot\mathbf{1}_d$, $C_1 = \mathrm{diag}(1, 2, \dots, d)$.
# - $\mu_2 = -5\cdot\mathbf{1}_d$, $C_2 = P\,\mathrm{diag}(1,2,\dots,d)\,P^\top$
#   with $P$ a fixed random orthogonal matrix (same spectrum as $C_1$, random axes).
#
# Reference is the standard normal $\mathcal{N}(0, I_d)$. Both components are written in
# fully normalized form and the weights sum to one, so $Z_{\text{target}} = Z_{\text{ref}} = 1$
# exactly and the SMC estimate of $\log Z$ should be $\approx 0$.
#
# Analytical truth used in the sanity-check section:
# - Global mean: $\bar\mu = w_1 \mu_1 + w_2 \mu_2 = 2\cdot\mathbf{1}_d$.
# - Per-coordinate variance: $w_1(C_{1,ii} + 9) + w_2(C_{2,ii} + 49)$ (derived from the
#   law of total variance using $(\mu_1 - \bar\mu)_i = 3$ and $(\mu_2 - \bar\mu)_i = -7$).
# - Empirical component weights should concentrate near $(0.7, 0.3)$.
#
# With $\mu_1, \mu_2$ separated by $\|\mu_1 - \mu_2\| = 10\sqrt{d}$ and max component
# std $\sqrt{d}$, the modes are well-separated, so this is a genuinely multimodal
# sampling problem where PT does real work.

# %%
from jax import config
config.update("jax_enable_x64", True)

import sys
from pathlib import Path

parent = Path().resolve().parent
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))

# %%
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt

from mppt import PTProblem

# %% [markdown]
# ## Target and reference densities
#
# - **Target**: 0.7 N(mu_1, C_1) + 0.3 N(mu_2, C_2) with parameters defined above.
# - **Reference**: $\mathcal{N}(0, I_d)$.
#
# Both components are fully normalized and the weights sum to 1, so $\log Z = 0$
# is exact.

# %%
import jax.scipy.linalg as jsl
from jax.scipy.special import logsumexp

dim = 50
log_2pi = jnp.log(2.0 * jnp.pi)

# Component 1: mean +5, axis-aligned covariance diag(1..d)
mu1 = 5.0 * jnp.ones(dim)
eigs = jnp.arange(1, dim + 1, dtype=jnp.float64)      # 1, 2, ..., d
C1 = jnp.diag(eigs)
L1 = jnp.linalg.cholesky(C1)
log_det_C1 = jnp.sum(jnp.log(eigs))

# Component 2: mean -5, same spectrum rotated by a fixed random orthogonal P
mu2 = -5.0 * jnp.ones(dim)
A = random.normal(random.key(0), (dim, dim))
P, _ = jnp.linalg.qr(A)                               # Haar-distributed orthogonal
C2 = (P * eigs) @ P.T                                 # P @ diag(eigs) @ P.T
L2 = jnp.linalg.cholesky(C2)
log_det_C2 = jnp.sum(jnp.log(eigs))                   # same spectrum

# Mixture weights
w1, w2 = 0.7, 0.3
log_w1 = jnp.log(w1)
log_w2 = jnp.log(w2)

def _log_normal(z, mu, L, log_det):
    """log N(z; mu, C) with C = L L^T, L lower-triangular."""
    y = jsl.solve_triangular(L, z - mu, lower=True)
    return -0.5 * jnp.sum(y**2) - 0.5 * log_det - 0.5 * dim * log_2pi

def logpdf_ref(z):
    return -0.5 * jnp.sum(z**2) - 0.5 * dim * log_2pi

def rvs_ref(key):
    return random.normal(key, shape=(dim,))

def logpdf_final_target(z):
    lp1 = log_w1 + _log_normal(z, mu1, L1, log_det_C1)
    lp2 = log_w2 + _log_normal(z, mu2, L2, log_det_C2)
    return logsumexp(jnp.stack([lp1, lp2]))

problem = PTProblem(
    rvs_ref=rvs_ref,
    logpdf_ref=logpdf_ref,
    logpdf_final_target=logpdf_final_target,
)

# Orthogonality check on P (sanity)
Perr = jnp.linalg.norm(P @ P.T - jnp.eye(dim))
print(f"||P P^T - I|| = {float(Perr):.2e}   (should be ~0)")
print(f"log det C1    = {float(log_det_C1):.4f}   (= sum log 1..{dim} = log d!)")
print(f"log det C2    = {float(log_det_C2):.4f}   (same spectrum)")
print(f"||mu1 - mu2|| = {float(jnp.linalg.norm(mu1 - mu2)):.4f}")

# Evaluate the target at each mode center
print(f"logpdf_target(mu1) = {logpdf_final_target(mu1):.4f}   (near mode 1)")
print(f"logpdf_target(mu2) = {logpdf_final_target(mu2):.4f}   (near mode 2)")
print(f"logpdf_target(0)   = {logpdf_final_target(jnp.zeros(dim)):.4f}   (between modes)")
print(f"logpdf_ref(0)      = {logpdf_ref(jnp.zeros(dim)):.4f}")

# %% [markdown]
# ## Sampling strategy
#
# Same two-phase approach as the LJ notebooks:
#
# 1. **SMC sampler** anneals from the reference to the target, discovering a temperature
#    schedule adaptively.
# 2. **Parallel tempering** uses those temperatures, initialized from the SMC output
#    particles.

# %%
from mppt.ui.ui_main import smc_sampler

key = random.key(42)
n_particles = 1_000

smc_output, logZ_smc, temperatures = smc_sampler(
    key, problem,
    n_particles=n_particles,
    engine={'name': 'hmc', 'refresh_rate': 0.07},
    n_mcmc_steps=50,
    local_ess_ratio=0.5,
    verbose=True,
    runtime_check=False,
    keep_all_states=True,
)

temperatures = np.array(temperatures)
print(f"\n{len(temperatures)} temperatures: " + " < ".join(f"{t:.3f}" for t in temperatures))
print("SMC logZ estimates (truth: 0.0):")
for name, val in logZ_smc.items():
    print(f"  {name}: {float(val):.4f}")

# %%
from mppt.ui.ui_main import pt_manual_runners

pt_runner = pt_manual_runners(
    problem, temperatures,
    n_particles=n_particles,
    engine={'name': 'hmc', 'refresh_rate': 0.07},
    explorer={'mode': 'adaptive', 'niter': 10},
    swapper={'mix': True},
)

# %%
from tqdm import trange

key = random.key(1)
pt_state = smc_output[:, :n_particles, :]  # shape (n_temps, n_particles, dim)

n_pt_iters = 1_000

diagnostics = []
for i in trange(n_pt_iters):
    key, subkey = random.split(key)
    full = (i == n_pt_iters - 1)
    pt_state, diag = pt_runner(subkey, pt_state, full_output=full)
    diagnostics.append(diag)

# %%
import jax.tree_util as jtu

# Collect the last n_last_collect states to pool more samples
n_last_collect = 50
n_pt_iters = 50
pt_states_collected = []

for i in trange(n_pt_iters):
    key, subkey = random.split(key)
    full = (i == n_pt_iters - 1)
    pt_state, diag = pt_runner(subkey, pt_state, full_output=full)
    if i >= n_pt_iters - n_last_collect:
        pt_states_collected.append(jtu.tree_map(np.array, pt_state))

# %% [markdown]
# ## Diagnostics
#
# Key quantities to monitor:
#
# - **Swap rejection rates** (one per adjacent temperature pair).
# - **MCMC acceptance rates** (one per temperature).
# - **logZ estimates**: for this problem the truth is $\log Z = 0$ exactly.

# %%
last_diag = diagnostics[-1]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(range(len(temperatures) - 1), last_diag['swap_rejection_rates'])
axes[0].set_xlabel('Temperature pair index')
axes[0].set_ylabel('Rejection rate')
axes[0].set_title('Swap rejection rates')
axes[0].set_ylim(0, 1)

axes[1].bar(range(len(temperatures)), last_diag['mcmc_acc_rates'])
axes[1].set_xlabel('Temperature index')
axes[1].set_ylabel('Acceptance rate')
axes[1].set_title('MCMC acceptance rates')
axes[1].set_ylim(0, 1)

for name in ['logZ_SMC_like', 'logZ_reverse_logistic', 'logZ_robust']:
    if name in last_diag:
        axes[2].bar(name.replace('logZ_', ''), last_diag[name])
axes[2].axhline(0.0, color='red', linestyle='--', alpha=0.6, label='truth (0)')
axes[2].set_title('logZ estimates (last iteration)')
axes[2].legend()
axes[2].grid(True)

print("\nlogZ estimates at last iteration (truth = 0):")
for name in ['logZ_SMC_like', 'logZ_reverse_logistic', 'logZ_robust']:
    if name in last_diag:
        print(f"  {name}: {last_diag[name]:.4f}")

# %%
fig2, ax2 = plt.subplots(figsize=(6, 4))
for name in ['logZ_SMC_like', 'logZ_reverse_logistic', 'logZ_robust']:
    if name in diagnostics[0]:
        ax2.plot([diag[name] for diag in diagnostics], label=name.replace('logZ_', ''))
ax2.axhline(0.0, color='red', linestyle='--', alpha=0.6, label='truth (0)')
ax2.set_title('logZ estimates across iterations')
ax2.set_xlabel('PT iteration')
ax2.set_ylabel('logZ estimate')
ax2.legend()
ax2.grid(True)

# %% [markdown]
# ## Comparison with the analytical truth
#
# Per-coordinate moments of the mixture:
# $\bar\mu_i = w_1(5) + w_2(-5) = 2$ and
# $\mathrm{Var}(x_i) = w_1(C_{1,ii} + 9) + w_2(C_{2,ii} + 49)$.
# Empirical component weights, obtained by assigning each sample to the component
# with highest posterior responsibility, should concentrate near $(w_1, w_2) = (0.7, 0.3)$.

# %%
# Pool samples at the final temperature (t=1) across all collected states
all_samples = np.concatenate(
    [s.X.position[s.Z[-1]] for s in pt_states_collected], axis=0
)  # shape (n_total, dim)
print(f"Pooled samples: {all_samples.shape}")

# Component-assignment via log responsibility (argmax of log w_k + log N_k)
log_N1_batched = jax.vmap(lambda z: _log_normal(z, mu1, L1, log_det_C1))
log_N2_batched = jax.vmap(lambda z: _log_normal(z, mu2, L2, log_det_C2))
lp1_all = log_w1 + log_N1_batched(all_samples)
lp2_all = log_w2 + log_N2_batched(all_samples)
assignment = np.array(lp2_all > lp1_all).astype(int)   # 0 = comp 1, 1 = comp 2

emp_w1 = float((assignment == 0).mean())
emp_w2 = float((assignment == 1).mean())
print(f"Empirical component weights: ({emp_w1:.3f}, {emp_w2:.3f})   (truth: (0.7, 0.3))")

# Global moments
emp_mean = all_samples.mean(axis=0)
emp_var = all_samples.var(axis=0)

mean_truth = w1 * 5.0 + w2 * (-5.0)                    # = 2 for every coord
var_truth = w1 * (np.array(jnp.diag(C1)) + 9.0) \
          + w2 * (np.array(jnp.diag(C2)) + 49.0)

print(f"Empirical mean -- min: {emp_mean.min():.4f}  max: {emp_mean.max():.4f}  "
      f"mean: {emp_mean.mean():.4f}    (truth: {mean_truth:.1f})")
print(f"Max |var diff|:  {np.max(np.abs(emp_var - var_truth)):.4f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) per-coordinate empirical mean vs truth (= 2)
axes[0].plot(emp_mean, 'o', label='empirical', markersize=4)
axes[0].axhline(mean_truth, color='red', linestyle='--',
                label=r'truth $\bar\mu = 2$')
axes[0].set_xlabel('Coordinate index')
axes[0].set_ylabel('Mean')
axes[0].set_title('Per-coordinate mean')
axes[0].legend()

# (b) per-coordinate empirical variance vs truth
axes[1].plot(emp_var, 'o', markersize=4, label='empirical')
axes[1].plot(var_truth, 'r--', label='truth')
axes[1].set_xlabel('Coordinate index')
axes[1].set_ylabel('Variance')
axes[1].set_title('Per-coordinate variance')
axes[1].legend()

# (c) empirical vs true component weights
axes[2].bar([0, 1], [emp_w1, emp_w2], width=0.4, alpha=0.7, label='empirical')
axes[2].bar([0 + 0.4, 1 + 0.4], [w1, w2], width=0.4, alpha=0.7, label='truth')
axes[2].set_xticks([0.2, 1.2])
axes[2].set_xticklabels(['comp 1', 'comp 2'])
axes[2].set_ylabel('Weight')
axes[2].set_title('Component weights')
axes[2].legend()

fig.tight_layout()
plt.show()

# %%
# 1D marginal of x_0: exact mixture of N(5, C1[0,0]=1) and N(-5, C2[0,0]).
c1_00 = float(C1[0, 0])
c2_00 = float(C2[0, 0])
xs = np.linspace(all_samples[:, 0].min() - 0.5, all_samples[:, 0].max() + 0.5, 600)
pdf_mix = (
    w1 * np.exp(-0.5 * (xs - 5.0)**2 / c1_00) / np.sqrt(2 * np.pi * c1_00)
    + w2 * np.exp(-0.5 * (xs + 5.0)**2 / c2_00) / np.sqrt(2 * np.pi * c2_00)
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(all_samples[:, 0], bins=80, density=True, alpha=0.6, label='PT samples')
ax.plot(xs, pdf_mix, 'r--', label='exact mixture marginal')
ax.set_xlabel(r'$x_0$')
ax.set_ylabel('Density')
ax.set_title(
    r'First marginal: $0.7\,\mathcal{N}(5, ' + f'{c1_00:.2f}' + r')'
    r' + 0.3\,\mathcal{N}(-5, ' + f'{c2_00:.2f}' + r')$'
)
ax.legend()
plt.show()

# %%
# 2D marginal (x_0, x_1) colored by assigned component
fig, ax = plt.subplots(figsize=(6, 5))
mask1 = assignment == 0
ax.scatter(all_samples[mask1, 0], all_samples[mask1, 1], s=4, alpha=0.3,
           label=f'comp 1 ({mask1.mean():.2f})', color='C0')
ax.scatter(all_samples[~mask1, 0], all_samples[~mask1, 1], s=4, alpha=0.3,
           label=f'comp 2 ({(~mask1).mean():.2f})', color='C1')
ax.plot(5.0, 5.0, marker='x', color='red', markersize=12, label=r'$\mu_1$')
ax.plot(-5.0, -5.0, marker='x', color='darkred', markersize=12, label=r'$\mu_2$')
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$x_1$')
ax.set_title(r'2D marginal $(x_0, x_1)$, colored by assigned component')
ax.set_aspect('equal')
ax.legend()
plt.show()
