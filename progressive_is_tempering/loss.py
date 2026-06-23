"""
DDPM regression loss for the β-conditioned excess score network.

From TECHNIQUES_SUMMARY.md §8 (DDPM Regression Target):

    φ(x_0, t, β) = λ_t · β · (x_0 - ∇E(x_0))

where λ_t = e^{-t} is the OU decay factor.

The training objective is the mean squared error between the score network
output s_θ(x_t, t, β) and the regression target φ, averaged over temperature
levels and mini-batches.

From IMPLEMENTATION_MLP.md:
  - Target norm clipping (per-sample): clip ‖φ‖ to max_norm when max_norm > 0.
  - Score output clipping: clip s_θ element-wise to [-score_clip, score_clip]
    when score_clip > 0.
  - stop_gradient on targets: targets are not differentiated through.
  - Gradient norm clipping via optax.clip_by_global_norm.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["compute_ddpm_target", "ddpm_loss_single_temp", "ddpm_loss"]


def compute_ddpm_target(
    x0: Array,
    t: Array,
    beta: Array,
    energy_fn: Callable[[Array], Array],
    max_norm: float = 0.0,
) -> Array:
    """
    Compute DDPM regression target for a single clean sample.

    φ(x_0, t, β) = λ_t · β · (x_0 - ∇E(x_0))

    Parameters
    ----------
    x0 : Array, shape (d,)
        Clean (unnoised) sample from the replay buffer.
    t : Array or float
        Noise time.
    beta : Array or float
        Inverse temperature.
    energy_fn : callable
        E(x) → scalar (with jax.grad available).
    max_norm : float
        If > 0, rescale φ so ‖φ‖ ≤ max_norm (direction preserved).

    Returns
    -------
    Array, shape (d,)
        Regression target (stop_gradient should be applied by the caller).
    """
    lambda_t = jnp.exp(-t)
    grad_e = jax.grad(energy_fn)(x0)
    target = lambda_t * beta * (x0 - grad_e)

    # Norm clipping: rescale when ‖target‖ > max_norm (no-op when max_norm=0)
    norm = jnp.linalg.norm(target)
    clip_coeff = jnp.where(
        max_norm > 0,
        jnp.minimum(max_norm / (norm + 1e-6), 1.0),
        1.0,
    )
    return target * clip_coeff


def ddpm_loss_single_temp(
    params,
    score_apply_fn: Callable,
    x0_batch: Array,
    t_batch: Array,
    beta: Array,
    energy_fn: Callable[[Array], Array],
    key: Array,
    max_norm: float = 0.0,
    score_clip: float = 0.0,
) -> Array:
    """
    DDPM MSE loss at a single temperature level β.

    The forward OU noising x_t ~ N(e^{-t} x_0, σ_t² I) is applied inside
    this function; the caller only needs to supply clean samples x_0 from
    the buffer.

    Parameters
    ----------
    params : PyTree
        Score network parameters.
    score_apply_fn : callable
        score_apply_fn(params, x, t, beta) → (d,), single-sample.
    x0_batch : Array, shape (B, d)
        Clean samples from the buffer.
    t_batch : Array, shape (B,)
        Noise times, one per sample.
    beta : Array
        Inverse temperature scalar (same for all samples in this batch).
    energy_fn : callable
    key : Array
        PRNG key for the noising step.
    max_norm : float
        Target norm clipping threshold (0 = disabled).
    score_clip : float
        Score output clipping threshold (0 = disabled).

    Returns
    -------
    Array
        Scalar MSE loss.
    """
    B, d = x0_batch.shape

    # Forward OU noising: x_t = e^{-t} x0 + σ_t ε
    noise_keys = jax.random.split(key, B)
    sigma_t = jnp.sqrt(1.0 - jnp.exp(-2.0 * t_batch))  # (B,)
    means = jnp.exp(-t_batch[:, None]) * x0_batch  # (B, d)
    eps = jax.vmap(jax.random.normal, in_axes=(0, None))(noise_keys, (d,))  # (B, d)
    xt_batch = means + sigma_t[:, None] * eps  # (B, d)

    # Regression targets (stop_gradient: no backprop through energy Hessian)
    def target_one(x0_i, t_i):
        return compute_ddpm_target(x0_i, t_i, beta, energy_fn, max_norm)

    targets = jax.lax.stop_gradient(
        jax.vmap(target_one)(x0_batch, t_batch)  # (B, d)
    )

    # Score predictions
    def score_one(xt_i, t_i):
        s = score_apply_fn(params, xt_i, t_i, beta)
        if score_clip > 0:
            s = jnp.clip(s, -score_clip, score_clip)
        return s

    preds = jax.vmap(score_one)(xt_batch, t_batch)  # (B, d)

    return jnp.mean(jnp.sum((preds - targets) ** 2, axis=-1))


def ddpm_loss(
    params,
    score_apply_fn: Callable,
    x0_per_temp: list[Array],
    t_per_temp: list[Array],
    betas: Array,
    energy_fn: Callable[[Array], Array],
    key: Array,
    max_norm: float = 0.0,
    score_clip: float = 0.0,
) -> Array:
    """
    Joint DDPM loss across all temperature levels.

    From TECHNIQUES_SUMMARY.md §12 (Inner Loop):

        L̂(θ) = (1/M) Σ_k (1/B) Σ_i ‖s_θ(x_{t,i}, t_i, β_k) - φ_i‖²

    Parameters
    ----------
    params : PyTree
        Score network parameters.
    score_apply_fn : callable
        Single-sample score fn: score_apply_fn(params, x, t, beta) → (d,).
    x0_per_temp : list of Array, each shape (B, d)
        Mini-batch of clean samples, one list entry per temperature.
    t_per_temp : list of Array, each shape (B,)
        Noise times, one list entry per temperature.
    betas : Array, shape (M+1,)
        Full temperature ladder.
    energy_fn : callable
    key : Array
        PRNG key (split internally per temperature).
    max_norm : float
    score_clip : float

    Returns
    -------
    Array
        Scalar mean loss across all temperature levels.
    """
    M = len(x0_per_temp)
    keys = jax.random.split(key, M)

    total = jnp.zeros(())
    for k, (x0_k, t_k, beta_k, key_k) in enumerate(
        zip(x0_per_temp, t_per_temp, betas, keys)
    ):
        total = total + ddpm_loss_single_temp(
            params,
            score_apply_fn,
            x0_k,
            t_k,
            beta_k,
            energy_fn,
            key_k,
            max_norm=max_norm,
            score_clip=score_clip,
        )
    return total / M