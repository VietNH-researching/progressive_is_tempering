"""
Covariance-preconditioned HMC kernel with partial velocity refresh.

Same tempered target, particle state, preconditioner and single-leapfrog move as
`progressive_is_tempering.mala_kernel` (whose types and batch helpers it
reuses), with three settings changed:

* refresh rate 0.07 instead of 1: after each move the velocity is
  ``v <- -sqrt(1 - 0.07) v + sqrt(0.07) z``, so momentum persists across
  successive moves instead of being redrawn;
* step size bisected towards an acceptance rate in [0.8, 0.95] instead of
  [0.4, 0.65];
* initial upper bracket ``1.65 / d**(1/6)`` instead of ``1.25 * 1.65 / d**(1/6)``.

`mala_kernel` (full refresh) is the special case refresh 1: each move is then
one preconditioned Langevin step. With partial refresh the chain moves
ballistically over several steps, at the price of a smaller step size.

Select it with ``kernel="hmc"`` in `smc.smc_sampler` and `rejuvenate.rejuvenate`.
"""

from __future__ import annotations

import jax.numpy as jnp

from progressive_is_tempering.mala_kernel import (
    MALAKernel,
    MALAParameters,
    MALAParticle,
    PTProblem,
    _scaled_cov,
    _vmapped_move,
    bisection,
    mala_kernel,
)

__all__ = ["hmc_kernel", "REFRESH_RATE", "TARGET_ACC_RANGE"]

# Partial refresh and its step-size bisection settings: target acceptance in
# [0.8, 0.95], initial upper bracket 1.65 / d**(1/6), at most 10 bisection updates.
REFRESH_RATE = 0.07
TARGET_ACC_RANGE = (0.8, 0.95)
_STEPSIZE_NITER_MAX = 10


def hmc_kernel(problem: PTProblem, t, t_idx) -> MALAKernel:
    """
    Build the partial-refresh HMC kernel for rung `(t, t_idx)` of `problem`.

    Identical to `mala_kernel(problem, t, t_idx)` except `adapt_params`, which
    returns parameters with refresh rate `REFRESH_RATE` and a step size
    bisected towards `TARGET_ACC_RANGE`. The move itself reads the refresh
    rate from those parameters, so it is `mala_kernel`'s move unchanged.
    """
    base = mala_kernel(problem, t, t_idx)

    def adapt_params(key, x: MALAParticle) -> MALAParameters:
        """
        Preconditioner = empirical covariance of `x.position`; step size
        bisected (reusing `key` at every probe, for a smooth objective) until
        the mean acceptance of one move is within `TARGET_ACC_RANGE`.
        """
        N, d = x.position.shape
        emp_cov = jnp.cov(x.position, rowvar=False)
        emp_cov = _scaled_cov(eps=1, cov=emp_cov)

        def stepsize_to_params(eps):
            cov_obj = emp_cov._replace(eps=eps)
            return MALAParameters(cov_obj, REFRESH_RATE)

        eps_upper = 1.65 / d**(1/6)
        acc_default = (TARGET_ACC_RANGE[0] + TARGET_ACC_RANGE[1])/2
        acc_delta = abs(TARGET_ACC_RANGE[1] - TARGET_ACC_RANGE[0])/2

        def f(eps):
            params = stepsize_to_params(eps)
            acc = _vmapped_move(base.move, key, x, params).accepted
            return jnp.mean(acc) - acc_default

        # At eps = 0 every proposal is accepted, so f(0) is known exactly.
        f0 = jnp.mean(jnp.ones((N,), dtype=bool)) - acc_default
        eps_star = bisection(f, 0., eps_upper, acc_delta, _STEPSIZE_NITER_MAX,
                             fleft_=f0)
        return stepsize_to_params(eps_star)

    return base._replace(adapt_params=adapt_params)
