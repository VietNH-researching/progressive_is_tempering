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
# # Minimal progressive_is_tempering notebook
#
# The method end to end on a 4-D GMM, as small as possible: a starting point for
# a new target. Rungs are processed in a Python loop here; `jax.vmap` over rungs
# is faster. Translation-invariant targets also need a zero-CoM projection
# (`project_fn` in `rejuvenate`).

# %%
import os

# Allocate GPU memory on demand instead of preallocating ~75% of the device at
# backend init, so the GPU stays shareable. Must precede `import jax`.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Uncomment to pin a device and to make runs reproducible (deterministic GPU ops).
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
#     + " --xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0").strip()

import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import logsumexp

from progressive_is_tempering.buffer import ReplayBuffer
from progressive_is_tempering.loss import ddpm_loss
from progressive_is_tempering.mala_kernel import PTProblem
from progressive_is_tempering.net.score_network import ScoreNetwork
from progressive_is_tempering.rejuvenate import rejuvenate
from progressive_is_tempering.resampling import normalized_ess, systematic_resample
from progressive_is_tempering.sampler import backward_sde_batch
from progressive_is_tempering.smc import smc_sampler
from progressive_is_tempering.targets import make_gmm_target

SEED = 0
key = jax.random.key(SEED)
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
# ## 0. Configuration

# %%
DIM = 4
T = 4.0                    # terminal time of the OU / backward SDE
T_MIN = 0.01               # training never draws a noise time below this
N_SDE_STEPS = 50           # backward SDE steps during training
N_SDE_STEPS_EVAL = 500     # more steps at evaluation: less discretisation bias
N_SMC_PARTICLES = 4096     # SMC population = per-rung seed = buffer capacity
N_PARTICLES = 1024         # K: particles per backward SDE run, per level
NUM_OUTER_STEPS = 5           # > N_SMC_PARTICLES/N_PARTICLES, so the seed drains
N_INNER_STEPS = 100
BATCH_SIZE = 128
N_MALA_REJUV = 5
SMC_LOCAL_ESS_RATIO = 0.5
SMC_N_MCMC_STEPS = 3
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0

print(f"seed drains after {N_SMC_PARTICLES / N_PARTICLES:.3g} of "
      f"{NUM_OUTER_STEPS} outer steps")

# %% [markdown]
# ## 1. Target
#
# Replace with your own target: the notebook uses `target.energy` and the three
# `PTProblem` methods.

# %%
target = make_gmm_target(dim=DIM)
print(f"target dim={target.dim}  energy(0)={target.energy(jnp.zeros(DIM)):.4f}")

pt_problem = PTProblem(
    rvs_ref=target.rvs_ref,
    logpdf_ref=target.logpdf_ref,
    logpdf_final_target=target.logpdf_final_target,
)

# %% [markdown]
# ## 2. Temperature ladder via SMC
#
# SMC gives the ladder, a particle block per rung and a tuned HMC kernel per
# rung.

# %%
key, subkey = jax.random.split(key)
smc_particles, logZ, betas_list, mcmc_params_by_rung = smc_sampler(
    subkey, pt_problem, n_particles=N_SMC_PARTICLES,
    n_mcmc_steps=SMC_N_MCMC_STEPS, local_ess_ratio=SMC_LOCAL_ESS_RATIO,
    verbose=False,
)
betas = jnp.array(betas_list)
M = len(betas_list) - 1          # number of non-reference rungs; β_M = 1
print(f"ladder ({len(betas_list)} levels): {[f'{b:.3f}' for b in betas_list]}")


def mcmc_params_for_rung(k):
    """MALAParameters SMC adapted for rung k (k = 1 … M)."""
    return jax.tree_util.tree_map(lambda leaf: leaf[k - 1], mcmc_params_by_rung)

# %% [markdown]
# ## 3. Score network
#
# Excess score over the reference (zero at β=0); unbatched, so call it through
# `jax.vmap`.

# %%
score_net = ScoreNetwork(x_dim=DIM, hidden_dim=128, n_blocks=2)
key, subkey = jax.random.split(key)
params = score_net.init(subkey, jnp.zeros(DIM), 1.0, 0.5)
print(f"{sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")

optimizer = optax.chain(optax.clip_by_global_norm(GRAD_CLIP),
                        optax.adam(LEARNING_RATE))
opt_state = optimizer.init(params)


def make_control(p):
    """Backward-SDE drift correction u_θ(x, t, β) = 2 s_θ(x, t, β)."""
    return lambda x, t, b: 2.0 * score_net.apply(p, x, t, b)

# %% [markdown]
# ## 4. Replay buffers, seeded from SMC
#
# Buffer 0: exact N(0, I) draws; buffers 1…M: that rung's SMC block. The outer
# loop writes buffers 2…M.

# %%
buffers = [ReplayBuffer(capacity=N_SMC_PARTICLES, dim=DIM)
           for _ in range(len(betas_list))]

key, subkey = jax.random.split(key)
buffers[0].add(jax.random.normal(subkey, (N_SMC_PARTICLES, DIM)),
               jnp.zeros(N_SMC_PARTICLES))
buffers[0].freeze()

for k in range(1, len(betas_list)):
    buffers[k].add(smc_particles[k], jnp.zeros(N_SMC_PARTICLES))
buffers[1].freeze()
print(f"buffers seeded: {[b.size for b in buffers]}")


def normalize_log_weights(log_w):
    """Shift a block of log-weights to mean weight 1."""
    total = logsumexp(log_w)
    return jnp.where(jnp.isfinite(total),
                     log_w - total + jnp.log(log_w.shape[0]),
                     jnp.zeros_like(log_w))

# %% [markdown]
# ## 5. Training

# %%
@jax.jit
def train_step(params, opt_state, x0_per_temp, t_per_temp, key):
    def loss_fn(p):
        return ddpm_loss(p, score_net.apply, x0_per_temp, t_per_temp,
                         betas, target.energy, key)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss

# %%
for outer_step in range(NUM_OUTER_STEPS):
    # ---- outer: generate data at every active level (k = 1 … M-1) ----
    path_ess = []
    for k in range(1, M):
        key, k_sde, k_res, k_mala = jax.random.split(key, 4)

        x0, log_w_path = backward_sde_batch(
            k_sde, make_control(params), betas[k], target.energy,
            DIM, N_PARTICLES, T, N_SDE_STEPS,
        )
        x0 = x0[systematic_resample(k_res, log_w_path, N_PARTICLES)]
        x0, _ = rejuvenate(k_mala, x0, pt_problem, betas[k], k,
                           mcmc_params_for_rung(k), N_MALA_REJUV)

        # IS bridge to the next rung: Δβ(-E(x) + ½‖x‖²)
        d_beta = betas[k + 1] - betas[k]
        bridge_lw = d_beta * (-jax.vmap(target.energy)(x0)
                              + 0.5 * jnp.sum(x0 ** 2, axis=-1))
        buffers[k + 1].add(x0, normalize_log_weights(bridge_lw))
        path_ess.append(float(normalized_ess(log_w_path)))

    # ---- inner: gradient steps, once every buffer can fill a batch ----
    if not all(b.is_ready(BATCH_SIZE) for b in buffers):
        print(f"[step {outer_step + 1}] buffers not ready, no update")
        continue

    loss_acc = 0.0
    for _ in range(N_INNER_STEPS):
        x0_per_temp, t_per_temp = [], []
        for k in range(len(betas_list)):
            key, k_buf, k_t = jax.random.split(key, 3)
            x0_k, _ = buffers[k].sample(k_buf, BATCH_SIZE)
            x0_per_temp.append(x0_k)
            t_per_temp.append(jax.random.uniform(
                k_t, (BATCH_SIZE,), minval=T_MIN, maxval=T))

        key, k_step = jax.random.split(key)
        params, opt_state, loss = train_step(
            params, opt_state, x0_per_temp, t_per_temp, k_step)
        loss_acc += float(loss)

    ess_str = " ".join(f"{e:.3f}" for e in path_ess)
    print(f"[step {outer_step + 1}/{NUM_OUTER_STEPS}] "
          f"loss={loss_acc / N_INNER_STEPS:.4f}  path_ESS=[{ess_str}]")

# %% [markdown]
# ## 6. Evaluation
#
# Backward SDE at β=1 on a finer grid, corrected with the path-space weights.

# %%
key, subkey = jax.random.split(key)
x_raw, log_w = backward_sde_batch(
    subkey, make_control(params), betas[-1], target.energy,
    DIM, 4096, T, N_SDE_STEPS_EVAL,
)
key, subkey = jax.random.split(key)
x_is = x_raw[systematic_resample(subkey, log_w, x_raw.shape[0])]

w1, w2, mu1, mu2, sigma = 0.2, 0.8, -1.5, 2.5, 0.5
true_mean = w1 * mu1 + w2 * mu2
true_std = (w1 * (mu1 ** 2 + sigma ** 2)
            + w2 * (mu2 ** 2 + sigma ** 2) - true_mean ** 2) ** 0.5

print(f"path ESS (β=1): {float(normalized_ess(log_w)):.4f}")
print(f"{'':<10}{'mean':>10}{'std':>10}")
print(f"{'truth':<10}{true_mean:>10.4f}{true_std:>10.4f}")
for name, x in (("SDE raw", x_raw), ("SDE + IS", x_is)):
    print(f"{name:<10}{float(jnp.mean(x)):>10.4f}{float(jnp.std(x)):>10.4f}")
