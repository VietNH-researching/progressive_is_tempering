"""
MALA (Metropolis-Adjusted Langevin Algorithm) rejuvenation.

After importance resampling produces near-duplicate particles, MALA diversifies
them while preserving the exact target π_β as the stationary distribution.

MALA proposal from TECHNIQUES_SUMMARY.md §9:

    x' = x + (h²/2) ∇_x log π_β(x) + h ε,   ε ~ N(0, I)

where
    ∇_x log π_β(x) = -(1-β) x - β ∇E(x)

Accepted with MH probability min(1, α):

    log α = log π_β(x') - log π_β(x) + log q(x|x') - log q(x'|x)

Step size h is a hyperparameter; optimal acceptance ≈ 57% in high dimensions.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["mala_step", "mala_rejuvenate"]


def _log_pi_beta(x: Array, beta: Array, energy_fn: Callable[[Array], Array]) -> Array:
    """log π_β(x) = -(1-β)½‖x‖² - β E(x)  (up to const)."""
    return -(1.0 - beta) * 0.5 * jnp.sum(x ** 2) - beta * energy_fn(x)


def _grad_log_pi_beta(x: Array, beta: Array, energy_fn: Callable[[Array], Array]) -> Array:
    """∇_x log π_β(x) = -(1-β) x - β ∇E(x)."""
    grad_e = jax.grad(energy_fn)(x)
    return -(1.0 - beta) * x - beta * grad_e


def mala_step(
    key: Array,
    x: Array,
    beta: Array,
    energy_fn: Callable[[Array], Array],
    h: float,
) -> tuple[Array, Array]:
    """
    One MALA step with Metropolis–Hastings correction.

    Parameters
    ----------
    key : Array
        PRNG key.
    x : Array, shape (d,)
        Current particle position.
    beta : Array
        Inverse temperature (scalar).
    energy_fn : callable
        E(x) → scalar.
    h : float
        Langevin step size.

    Returns
    -------
    x_new : Array, shape (d,)
        Updated position (proposal or original, after MH).
    accepted : Array
        Boolean scalar, True if the proposal was accepted.
    """
    key_prop, key_mh = jax.random.split(key)

    grad_x = _grad_log_pi_beta(x, beta, energy_fn)

    # Langevin proposal
    eps = jax.random.normal(key_prop, x.shape)
    x_prop = x + 0.5 * h ** 2 * grad_x + h * eps

    grad_x_prop = _grad_log_pi_beta(x_prop, beta, energy_fn)

    # Log-densities at proposal and current
    log_pi_x = _log_pi_beta(x, beta, energy_fn)
    log_pi_xp = _log_pi_beta(x_prop, beta, energy_fn)

    # Forward and backward proposal log-densities (Gaussian, h² variance)
    mean_fwd = x + 0.5 * h ** 2 * grad_x
    log_q_fwd = -0.5 * jnp.sum((x_prop - mean_fwd) ** 2) / h ** 2

    mean_bwd = x_prop + 0.5 * h ** 2 * grad_x_prop
    log_q_bwd = -0.5 * jnp.sum((x - mean_bwd) ** 2) / h ** 2

    log_alpha = log_pi_xp - log_pi_x + log_q_bwd - log_q_fwd

    # Accept / reject
    u = jax.random.uniform(key_mh)
    accepted = jnp.log(u) < log_alpha
    x_new = jnp.where(accepted, x_prop, x)

    return x_new, accepted


@partial(jax.jit, static_argnums=(3, 5))
def mala_rejuvenate(
    key: Array,
    particles: Array,
    beta: Array,
    energy_fn: Callable[[Array], Array],
    h: Array,
    n_steps: int,
) -> tuple[Array, Array]:
    """
    Apply n_steps of MALA to each particle independently.

    Uses jax.lax.scan for JIT-friendly iteration and jax.vmap over particles.

    Parameters
    ----------
    key : Array
        PRNG key.
    particles : Array, shape (K, d)
        Initial particle positions.
    beta : Array
        Inverse temperature scalar.
    energy_fn : callable
        E(x) → scalar.
    h : float
        Langevin step size.
    n_steps : int
        Number of MALA steps per particle.

    Returns
    -------
    particles_new : Array, shape (K, d)
        Updated particles.
    mean_accept_rate : Array
        Mean acceptance rate across all particles and steps.
    """
    K = particles.shape[0]
    keys = jax.random.split(key, K)

    def rejuvenate_one(key_i: Array, x_init: Array) -> tuple[Array, Array]:
        """Apply n_steps MALA to a single particle."""
        step_keys = jax.random.split(key_i, n_steps)

        def scan_step(x: Array, step_key: Array) -> tuple:
            x_new, acc = mala_step(step_key, x, beta, energy_fn, h)
            return x_new, acc.astype(jnp.float32)

        x_final, accepts = jax.lax.scan(scan_step, x_init, step_keys)
        return x_final, jnp.mean(accepts)

    particles_new, accept_rates = jax.vmap(rejuvenate_one)(keys, particles)
    return particles_new, jnp.mean(accept_rates)