"""
HMC rejuvenation with the SMC-adapted kernel.

Reuses the exact covariance-preconditioned HMC kernel that
`progressive_is_tempering.smc.smc_sampler` already adapts per temperature
rung (returned as its `mcmc_params`), instead of maintaining a second,
independently-calibrated isotropic MALA implementation.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array

from progressive_is_tempering.mala_kernel import (
    PTProblem,
    batch_init_extend,
    batch_move,
    mala_kernel,
    repeated_batch_move,
)

__all__ = ["rejuvenate"]


@partial(jax.jit, static_argnums=(2, 6, 7))
def rejuvenate(
    key: Array,
    positions: Array,
    problem: PTProblem,
    beta_k: Array,
    t_idx_k: Array,
    mcmc_params_k: Any,
    n_steps: int,
    project_fn: Optional[Callable[[Array], Array]] = None,
) -> tuple[Array, Array]:
    """
    Rejuvenate `positions` at rung (beta_k, t_idx_k) with the HMC kernel,
    using the `MALAParameters` SMC already adapted for this rung.

    Returns ``(positions_new, mean_accept_rate)``.

    t_idx_k must never be 0 -- the kernel treats t_idx==0 as "draw fresh
    from the reference" and ignores `positions` entirely (see
    `progressive_is_tempering.mala_kernel.mala_kernel`). Rung 0 (the frozen
    reference buffer) is never rejuvenated, so this holds at every real call
    site.

    Parameters
    ----------
    key : Array
        PRNG key.
    positions : Array, shape (K, d)
        Particle positions to rejuvenate.
    problem : PTProblem
        (rvs_ref, logpdf_ref, logpdf_final_target) triple defining pi_beta.
        Static: must be the same object (by identity) SMC was run with.
    beta_k : Array
        Inverse-temperature scalar for this rung.
    t_idx_k : Array
        Rung index (>= 1) for this beta_k -- must match the index SMC used
        when adapting `mcmc_params_k`.
    mcmc_params_k : MALAParameters
        Step size / covariance preconditioner adapted for this rung.
    n_steps : int
        Number of HMC moves to apply. Static.
    project_fn : callable, optional
        If given (e.g. a zero-center-of-mass projector), applied to the
        particle positions after *every* internal move, with the particle's
        density/gradient fields then re-derived at the projected position
        (via `batch_init_extend`, which also draws a fresh velocity -- lossless
        here specifically because the kernel uses `refresh_rate=1`, so
        `move()` already fully re-randomizes velocity every step regardless).
        This keeps the chain on the constraint manifold throughout. A target
        with no such constraint should leave this as None, which uses the
        plain `repeated_batch_move` path. Static.

    Returns
    -------
    positions_new : Array, shape (K, d)
    mean_accept_rate : Array
    """
    kernel = mala_kernel(problem, beta_k, t_idx_k)
    key_init, key_move = jax.random.split(key)
    x = batch_init_extend(key_init, positions, kernel)

    if project_fn is None:
        x = repeated_batch_move(key_move, x, mcmc_params_k, kernel, n_steps)
    else:
        def body(_i, carry):
            k, xi = carry
            k, k_move, k_reinit = jax.random.split(k, 3)
            xi = batch_move(k_move, xi, mcmc_params_k, kernel)
            proj_pos = jax.vmap(project_fn)(xi.position)
            xi = batch_init_extend(k_reinit, proj_pos, kernel)
            return k, xi

        _, x = jax.lax.fori_loop(0, n_steps, body, (key_move, x))

    return x.position, jnp.mean(x.accepted)
