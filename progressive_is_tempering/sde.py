"""
Ornstein–Uhlenbeck forward (noising) process.

The SDE is:
    dX_t = -X_t dt + sqrt(2) dW_t,   X_0 ~ π_β

Closed-form marginal:
    X_t | X_0 ~ N(e^{-t} X_0, σ_t² I),   σ_t² = 1 - e^{-2t}

For large T the marginal is ≈ N(0, I) regardless of the starting distribution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "ou_mean",
    "ou_std",
    "ou_forward_sample",
    "ou_log_transition_density",
    "reference_ou_score",
]


def ou_mean(x0: Array, t: Array) -> Array:
    """Mean of the OU marginal: e^{-t} x_0."""
    return jnp.exp(-t) * x0


def ou_std(t: Array) -> Array:
    """Standard deviation of the OU marginal: sqrt(1 - e^{-2t})."""
    return jnp.sqrt(1.0 - jnp.exp(-2.0 * t))


def ou_forward_sample(key: Array, x0: Array, t: Array) -> Array:
    """
    Draw x_t ~ N(e^{-t} x_0, σ_t² I).

    Parameters
    ----------
    key : Array
        JAX PRNG key.
    x0 : Array, shape (d,)
        Clean sample from π_β.
    t : Array or float
        Time in [0, T].

    Returns
    -------
    Array, shape (d,)
        Noised sample at time t.
    """
    mean = ou_mean(x0, t)
    std = ou_std(t)
    eps = jax.random.normal(key, x0.shape)
    return mean + std * eps


def ou_log_transition_density(x_n: Array, x_prev: Array, dt: Array) -> Array:
    """
    Exact log p_ρ(x_n | x_{n-1}) for the forward OU process over one step dt.

    x_n | x_{n-1} ~ N(λ x_{n-1}, σ² I),   λ = e^{-dt},   σ² = 1 - e^{-2dt}

    Parameters
    ----------
    x_n : Array, shape (d,)
        Next state (forward direction).
    x_prev : Array, shape (d,)
        Previous state.
    dt : float or Array
        Step size.

    Returns
    -------
    Array
        Log-density (scalar), up to the constant -d/2 log(2π σ²).
    """
    lam = jnp.exp(-dt)
    sigma_sq = 1.0 - jnp.exp(-2.0 * dt)
    mean = lam * x_prev
    return -0.5 * jnp.sum((x_n - mean) ** 2) / sigma_sq


def reference_ou_score(x: Array) -> Array:
    """
    Score of the stationary distribution ρ = N(0, I):
        ∇_x log ρ(x) = -x.

    Used as the reference score component in the backward SDE.
    """
    return -x