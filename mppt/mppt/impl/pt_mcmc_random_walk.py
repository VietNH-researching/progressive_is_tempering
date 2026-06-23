r"""This module implements the Parallel Tempering version of the generic random
walk algorithm in ``mcmc_random_walk.py``. The main difference is that particles
in the PT version have several additional attributes relating to the current
temperature. As such, this module is just a wrapper which delegates real tasks
to ``mcmc_random_walk.py``.
"""
from __future__ import annotations
from typing import NamedTuple, Callable, Union
from mppt.abstract.mcmc import PTMCMCEngine, PTMCMC
from mppt.abstract.pt_concept import PTProblem, PTParticle, geometric_anneal
from mppt.impl.mcmc_random_walk import FixedCovMVNormal, RW_MCMC
from mppt.jax_configs import ALLOW_RUNTIME_CHECK, chex_runtime_assert
from mppt.utils import follow_protocol, WhileLoop
from jax import Array, lax, numpy as jnp
from jax.typing import ArrayLike

RdVector = Array
Float = Union[float, ArrayLike]

@follow_protocol(PTParticle)
class RW_PT_Particle(NamedTuple):
    r"""Random walk particle with PT temperature fields. Carries both ref
    and target logpdfs so temperature changes don't require re-evaluation.
    """
    position: RdVector
    temperature: Float
    temperature_idx: int
    logpdf_ref: Float
    logpdf_final_target: Float
    logpdf_target: Float
    accepted: bool

RW_PT_Particles = RW_PT_Particle  # multiple particles are represented
# by the same pytree structure.

# noinspection DuplicatedCode
@follow_protocol(PTMCMC)
class RW_PT_MCMC(NamedTuple):
    r"""A random walk MCMC algorithm in the context of Parallel Tempering.
    The parameters of this MCMC algorithm are the proposal scale and the proposal
    covariance matrix. They are encoded by the ``FixedCovMVNormal`` class.
    """
    extend: Callable[[RdVector], RW_PT_Particle]
    move: Callable[[Key, RW_PT_Particle, FixedCovMVNormal], RW_PT_Particle]
    adapt_params: Callable[[Key, RW_PT_Particles], FixedCovMVNormal]
    adapt_nsteps: Callable[[Key, RW_PT_Particles, FixedCovMVNormal], WhileLoop]
    reset_temperature: Callable[[RW_PT_Particle], RW_PT_Particle]
    init_extend: Callable[[Key, RdVector], RW_PT_Particle]

@follow_protocol(PTMCMCEngine)
def rw_pt_mcmc(pt_prob: PTProblem, t: float, t_idx: int,
               nsteps_max=jnp.iinfo(jnp.int32).max, identity_kernel=False,
               use_bisection=False) -> RW_PT_MCMC:
    r"""Creates an MCMC kernel for the temperature ``t`` with temperature index
    ``t_idx``. It is assumed that if t_idx=0 then t=0 and the MCMC kernel is reduced
    to iid sampling from the reference.

    :param nsteps_max: max MCMC steps in adaptive nsteps. To fix at 50:
        ``partial(rw_pt_mcmc, nsteps_max=50)``
    :param identity_kernel: if True, the move function returns the input
        unchanged (no MCMC). Debugging tool for testing swap mechanics
        without MCMC interference.
    :param use_bisection: tune step size by bisection (True) or use
        2.38/sqrt(d) rule (False).
    """
    def extend(x: RdVector) -> RW_PT_Particle:
        logpdf_ref = pt_prob.logpdf_ref(x)
        logpdf_final_target = pt_prob.logpdf_final_target(x)
        # logpdf_target = (1-t) * logpdf_ref + t * logpdf_final_target
        logpdf_target = geometric_anneal(t, logpdf_ref, logpdf_final_target)
        return RW_PT_Particle(
            x, t, t_idx, logpdf_ref, logpdf_final_target, logpdf_target, True)

    # most of the heavy lifting is delegated to this engine.
    random_walk_engine = RW_MCMC(
        extend, adapt_nsteps_max=lax.cond(
            t_idx == 0, lambda:1, lambda: nsteps_max),
        use_bisection_for_step_size=use_bisection
    )

    def move(key, x: RW_PT_Particle, params: FixedCovMVNormal) -> RW_PT_Particle:
        assert x.temperature.shape == ()
        if ALLOW_RUNTIME_CHECK:  # do not act on particles from different t
            chex_runtime_assert(jnp.allclose(x.temperature, t))
        return lax.cond(
            t_idx == 0,
            lambda: extend(pt_prob.rvs_ref(key)),
            lambda: (random_walk_engine.move(key, x, params) if not identity_kernel else x)
        )

    def adapt_params(key, x: RW_PT_Particles) -> FixedCovMVNormal:
        # noinspection PyTypeChecker
        return random_walk_engine.adapt_params(key, x)

    def adapt_nsteps(key, x: RW_PT_Particles, params: FixedCovMVNormal) -> WhileLoop:
        # noinspection PyTypeChecker
        return random_walk_engine.adapt_nsteps(key, x, params)

    def reset_temperature(x: RW_PT_Particle) -> RW_PT_Particle:
        if ALLOW_RUNTIME_CHECK:
            chex_runtime_assert(jnp.allclose(
                x.logpdf_target,  # (1-x.temperature)*x.logpdf_ref+x.temperature*x.logpdf_final_target
                geometric_anneal(x.temperature, x.logpdf_ref, x.logpdf_final_target)
            ))
        # new_logpdf_target = (1-t) * x.logpdf_ref + t * x.logpdf_final_target
        new_logpdf_target = geometric_anneal(t, x.logpdf_ref, x.logpdf_final_target)
        return RW_PT_Particle(
            x.position, t, t_idx, x.logpdf_ref, x.logpdf_final_target,
            new_logpdf_target, True)

    return RW_PT_MCMC(extend, move, adapt_params, adapt_nsteps, reset_temperature,
                      init_extend=random_walk_engine.init_extend)