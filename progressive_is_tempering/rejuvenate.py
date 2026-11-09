"""
HMC rejuvenation with the SMC-adapted kernel.

`kernel` selects the kernel SMC ran with: "mala" (full refresh) or "hmc"
(partial refresh); see `progressive_is_tempering.smc.KERNELS`.

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
    geometric_anneal,
    mala_kernel,
    repeated_batch_move,
)
from progressive_is_tempering.hmc_kernel import hmc_kernel

__all__ = ["rejuvenate"]

_KERNELS = {"mala": mala_kernel, "hmc": hmc_kernel}


@partial(jax.jit, static_argnums=(2, 6, 7, 8, 9, 10))
def rejuvenate(
    key: Array,
    positions: Array,
    problem: PTProblem,
    beta_k: Array,
    t_idx_k: Array,
    mcmc_params_k: Any,
    n_steps: int,
    project_fn: Optional[Callable[[Array], Array]] = None,
    project_invariant: bool = False,
    return_energy: bool = False,
    kernel: str = "mala",
) -> tuple[Array, ...]:
    """
    Rejuvenate `positions` at rung (beta_k, t_idx_k) with the HMC kernel,
    using the `MALAParameters` SMC already adapted for this rung.

    Returns ``(positions_new, mean_accept_rate)``, plus the energies at
    `positions_new` when `return_energy` is set.

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
        for kernel="mala", whose `refresh_rate=1` re-randomizes the velocity
        every move anyway; for kernel="hmc" the pre-projection velocity is
        carried over instead, so momentum persists across moves).
        This keeps the chain on the constraint manifold throughout. A target
        with no such constraint should leave this as None, which uses the
        plain `repeated_batch_move` path. Static.
    project_invariant : bool
        Declare that `problem.logpdf_final_target` is invariant under
        `project_fn` (e.g. a pair-distance energy under the zero-CoM
        projection). The re-derivation after each projection then keeps the
        particle's target log-density and recomputes only the reference term,
        saving one target evaluation per particle per step. Ignored when
        `project_fn` is None. Static.
    return_energy : bool
        Also return E(x) = -logpdf_final_target(x) at `positions_new`. It is
        read from the final particle state, so it costs no extra evaluation.
        Static.
    kernel : {"mala", "hmc"}
        The kernel `mcmc_params_k` was adapted for (the `kernel` passed to
        `smc_sampler`). Static.

    Returns
    -------
    positions_new : Array, shape (K, d)
    mean_accept_rate : Array
    energies : Array, shape (K,)
        Only when `return_energy` is set.
    """
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {tuple(_KERNELS)} (got {kernel!r})")
    keep_veloc = kernel == "hmc"
    kernel = _KERNELS[kernel](problem, beta_k, t_idx_k)
    key_init, key_move = jax.random.split(key)
    x = batch_init_extend(key_init, positions, kernel)

    if project_fn is None:
        x = repeated_batch_move(key_move, x, mcmc_params_k, kernel, n_steps)
    else:
        def body(_i, carry):
            k, xi = carry
            k, k_move, k_reinit = jax.random.split(k, 3)
            xi = batch_move(k_move, xi, mcmc_params_k, kernel)
            old_veloc = xi.veloc
            proj_pos = jax.vmap(project_fn)(xi.position)
            if project_invariant:
                # Same state as `batch_init_extend` (same per-particle velocity
                # draw from k_reinit, accepted reset to True, gradient fields
                # NaN), but the target log-density is carried over instead of
                # re-evaluated.
                subkeys = jax.random.split(k_reinit, proj_pos.shape[0])
                veloc = jax.vmap(
                    lambda kk, xx: jax.random.normal(kk, shape=xx.shape)
                )(subkeys, proj_pos)
                lref = jax.vmap(problem.logpdf_ref)(proj_pos)
                nan = jnp.full_like(proj_pos, jnp.nan)
                xi = xi._replace(
                    position=proj_pos, veloc=veloc,
                    logpdf_ref=lref, gradlogpdf_ref=nan,
                    gradlogpdf_final_target=nan,
                    logpdf_target=geometric_anneal(beta_k, lref, xi.logpdf_final_target),
                    gradlogpdf_target=nan,
                    accepted=jnp.ones_like(xi.accepted),
                )
            else:
                xi = batch_init_extend(k_reinit, proj_pos, kernel)
            if keep_veloc:
                # Partial refresh: keep the momentum across the projection.
                xi = xi._replace(veloc=old_veloc)
            return k, xi

        _, x = jax.lax.fori_loop(0, n_steps, body, (key_move, x))

    if return_energy:
        # The final particle carries log pi(x) = -E(x) at its position (the
        # accepted endpoint or the kept state).
        return x.position, jnp.mean(x.accepted), -x.logpdf_final_target
    return x.position, jnp.mean(x.accepted)
