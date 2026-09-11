"""
MALA rejuvenation backed by mppt's own SMC-tuned MALA-mode HMC kernel.

Reuses the exact covariance-preconditioned kernel that `mppt.smc_sampler`
already adapts per temperature rung (captured via `keep_mcmc_params=True`),
instead of maintaining a second, independently-calibrated isotropic MALA
implementation.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["mppt_rejuvenate"]


@partial(jax.jit, static_argnums=(2, 6, 7))
def mppt_rejuvenate(
    key: Array,
    positions: Array,
    mppt_problem: Any,
    beta_k: Array,
    t_idx_k: Array,
    mcmc_params_k: Any,
    n_steps: int,
    project_fn: Optional[Callable[[Array], Array]] = None,
) -> tuple[Array, Array]:
    """
    Rejuvenate `positions` at rung (beta_k, t_idx_k) using mppt's own
    MALA-mode HMC kernel, with the HMCParameters SMC already tuned for this
    rung (captured via ``mppt.smc_sampler(..., keep_mcmc_params=True)``).

    Same call shape/contract as the retired
    ``progressive_is_tempering.mala.mala_rejuvenate``: returns
    ``(positions_new, mean_accept_rate)``.

    t_idx_k must never be 0 -- mppt's engine treats t_idx==0 as "draw fresh
    from the reference" and ignores `positions` entirely (see
    `mppt.impl.pt_mcmc_hmc.hmc_pt_mcmc`). Rung 0 (the frozen reference buffer)
    is never rejuvenated, so this holds at every real call site.

    Parameters
    ----------
    key : Array
        PRNG key.
    positions : Array, shape (K, d)
        Particle positions to rejuvenate.
    mppt_problem : mppt.PTProblem
        (rvs_ref, logpdf_ref, logpdf_final_target) triple defining pi_beta.
        Static: must be the same object (by identity) SMC was run with.
    beta_k : Array
        Inverse-temperature scalar for this rung.
    t_idx_k : Array
        Rung index (>= 1) for this beta_k -- must match the index mppt used
        when adapting `mcmc_params_k` during SMC.
    mcmc_params_k : mppt HMCParameters
        Step size / covariance preconditioner adapted for this rung.
    n_steps : int
        Number of MALA-mode HMC moves to apply. Static.
    project_fn : callable, optional
        If given (e.g. a zero-center-of-mass projector), applied to the
        particle positions after *every* internal move, with the particle's
        density/gradient fields then re-derived at the projected position
        (via `batch_init_extend`, which also draws a fresh velocity -- lossless
        here specifically because MALA mode forces `refresh_rate=1`, so
        `move()` already fully re-randomizes velocity every step regardless).
        This keeps the chain on the constraint manifold throughout, matching
        the safety margin a per-step-projecting sampler (e.g. the retired
        `mala_step_com` in the DW4/LJ13 notebooks) already provides -- a
        symmetric target (e.g. GMM) has no such constraint and should leave
        this as None, which preserves the exact prior fast path
        (`repeated_batch_move`). Static.

    Returns
    -------
    positions_new : Array, shape (K, d)
    mean_accept_rate : Array
    """
    from mppt.abstract.mcmc import batch_init_extend, batch_move, repeated_batch_move
    from mppt.impl.pt_mcmc_hmc import hmc_pt_mcmc

    mcmc = hmc_pt_mcmc(mppt_problem, beta_k, t_idx_k, mala=True)
    key_init, key_move = jax.random.split(key)
    x = batch_init_extend(key_init, positions, mcmc)

    if project_fn is None:
        x = repeated_batch_move(key_move, x, mcmc_params_k, mcmc, n_steps)
    else:
        def body(_i, carry):
            k, xi = carry
            k, k_move, k_reinit = jax.random.split(k, 3)
            xi = batch_move(k_move, xi, mcmc_params_k, mcmc)
            proj_pos = jax.vmap(project_fn)(xi.position)
            xi = batch_init_extend(k_reinit, proj_pos, mcmc)
            return k, xi

        _, x = jax.lax.fori_loop(0, n_steps, body, (key_move, x))

    return x.position, jnp.mean(x.accepted)
