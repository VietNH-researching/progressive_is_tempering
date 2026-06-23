"""
Systematic resampling for SMC / importance sampling.

Systematic resampling draws n indices from N particles with normalized
weights using a single uniform variate u ~ Unif(0, 1/n):

    u_m = u + (m-1)/n,   m = 1, ..., n

Each u_m is mapped to a particle index via the CDF of the weight distribution.
This gives O(1) variance per index (vs O(N) for multinomial), making it the
preferred resampling scheme for SMC.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["systematic_resample", "normalized_ess"]


@partial(jax.jit, static_argnums=(2,))
def systematic_resample(key: Array, log_weights: Array, n: int) -> Array:
    """
    Systematic resampling: draw n indices from the weight distribution.

    Parameters
    ----------
    key : Array
        JAX PRNG key.
    log_weights : Array, shape (N,)
        Unnormalized log-weights.  May contain -inf for empty/masked slots.
    n : int
        Number of indices to draw.

    Returns
    -------
    Array, shape (n,), dtype int32
        Resampled indices into [0, N).
    """
    N = log_weights.shape[0]

    # Numerically stable normalization
    log_w_norm = log_weights - jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_w_norm)

    # CDF
    cumsum = jnp.cumsum(weights)
    # Clamp to [0, 1] to absorb floating-point rounding
    cumsum = jnp.clip(cumsum, 0.0, 1.0)

    # Single uniform variate for the whole batch
    u0 = jax.random.uniform(key, shape=()) / n
    points = u0 + jnp.arange(n, dtype=jnp.float32) / n  # (n,)

    # Vectorized searchsorted: map each point to a particle index
    indices = jnp.searchsorted(cumsum, points, side="left")
    return jnp.clip(indices, 0, N - 1).astype(jnp.int32)


@jax.jit
def normalized_ess(log_weights: Array) -> Array:
    """
    Normalized effective sample size (ESS / N) for a weight vector.

    ESS = (Σ w_i)² / Σ w_i²

    Normalized by N so the result is in [1/N, 1]:
      - 1.0  → all weights equal (ideal)
      - 1/N  → single dominant particle (degenerate)

    Parameters
    ----------
    log_weights : Array, shape (N,)
        Unnormalized log-weights.

    Returns
    -------
    Array
        Scalar in (0, 1].
    """
    N = log_weights.shape[0]
    log_w_norm = log_weights - jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_w_norm)
    ess = 1.0 / jnp.sum(weights ** 2)
    return ess / N