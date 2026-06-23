r"""Parallel tempering wrapper for HMC Kernel.
Heavy lifting is done in impl/mcmc_hmc.py
"""
from __future__ import annotations
from collections import namedtuple
from typing import NamedTuple, Callable

from jax import value_and_grad, numpy as jnp, lax, random

from mppt.abstract.mcmc import PTMCMCEngine, PTMCMC
from mppt.abstract.pt_concept import PTProblem, Float, Key, RdVector, PTParticle, geometric_anneal
from mppt.impl.mcmc_hmc import HMCParticle, Bool, HMCParameters, HMC
from mppt.jax_configs import ALLOW_RUNTIME_CHECK, chex_runtime_assert
from mppt.utils import follow_protocol, WhileLoop


@follow_protocol(PTParticle, HMCParticle)
class HMC_PT_Particle(NamedTuple):
    r"""HMC particle with PT temperature fields. Carries both ref and target
    logpdfs/gradients so temperature changes don't require re-evaluation.
    """
    position: RdVector

    logpdf_ref: Float
    logpdf_final_target: Float
    gradlogpdf_ref: RdVector
    gradlogpdf_final_target: RdVector

    temperature: Float
    temperature_idx: int

    logpdf_target: Float
    gradlogpdf_target: RdVector
    veloc: RdVector
    accepted: Bool

HMC_PT_Particles = HMC_PT_Particle  # the class to represent a system of
# multiple particles has the same structure as the original class.

# Create an internal type like HMC_PT_Particle but w/o velocity
_all_fields = list(HMC_PT_Particles._fields)
del _all_fields[_all_fields.index('veloc')]
assert ('veloc' not in _all_fields) and ('accepted' in _all_fields)
_HMC_PT_Particle_NoVeloc = namedtuple('_HMC_PT_Particle_NoVeloc', _all_fields)

@follow_protocol(PTMCMC)
class HMC_PT_MCMC(NamedTuple):
    r"""HMC in the context of parallel tempering.
    See documentation of the protocol PTMCMC and the function
    `hmc_pt_mcmc` below.
    """
    init_extend: Callable[[Key, RdVector], HMC_PT_Particle]
    move: Callable[[Key, HMC_PT_Particle, HMCParameters], HMC_PT_Particle]
    adapt_params: Callable[[Key, HMC_PT_Particles], HMCParameters]
    adapt_nsteps: Callable[[Key, HMC_PT_Particles, HMCParameters], WhileLoop]
    reset_temperature: Callable[[HMC_PT_Particle], HMC_PT_Particle]

@follow_protocol(PTMCMCEngine)
def hmc_pt_mcmc(pt_problem: PTProblem, t: Float, t_idx: int, default_refresh_rate=None, mala=False) -> HMC_PT_MCMC:
    r"""Creates an HMC kernel for a given temperature `t` and a parallel tempering
    problem `pt_problem`. See `docs/extend.md` for the philosophy; see
    `impl/mcmc_hmc.py` for the main workhorse. This function simply creates
    a feeder extend to wrap around `impl/mcmc_hmc.py`

    :param default_refresh_rate: the default value of the refreshment rate, used
    in the adaptation process
    :param mala: whether the adaptation should target acc rate of MALA (0.5-0.65)
    instead of HMC (0.8-0.95). In this case the refresh rate is set to 1 and
    the user-given value is ignored.
    """
    def feed_extend(x: RdVector) -> _HMC_PT_Particle_NoVeloc:
        assert jnp.ndim(x) == 1
        logpdf_ref, gradlogpdf_ref = value_and_grad(pt_problem.logpdf_ref)(x)
        logpdf_final_target, gradlogpdf_final_target =\
            value_and_grad(pt_problem.logpdf_final_target)(x)
        logpdf_target = geometric_anneal(t, logpdf_ref, logpdf_final_target)
        gradlogpdf_target = geometric_anneal(t, gradlogpdf_ref, gradlogpdf_final_target)
        if ALLOW_RUNTIME_CHECK:
            chex_runtime_assert(jnp.all(jnp.isfinite(gradlogpdf_target)))
        # noinspection PyArgumentList
        return _HMC_PT_Particle_NoVeloc(
            position=x, logpdf_ref=logpdf_ref, gradlogpdf_ref=gradlogpdf_ref,
            logpdf_final_target=logpdf_final_target,
            gradlogpdf_final_target=gradlogpdf_final_target,
            temperature=t, temperature_idx=t_idx,
            logpdf_target=logpdf_target, gradlogpdf_target=gradlogpdf_target,
            accepted=True
        )

    if default_refresh_rate is None:
        default_refresh_rate = HMC._field_defaults['default_refresh_rate']

    # Main object that does all the heavylifting.
    hmc_core = HMC(feed_extend=feed_extend, HMCParticleType=HMC_PT_Particle,
                   default_refresh_rate=default_refresh_rate, mala_mode=mala)

    def new_from_ref(key):
        key_ref, key_extend = random.split(key)
        x = pt_problem.rvs_ref(key_ref)
        return hmc_core.init_extend(key_extend, x)

    def move(key, x: HMC_PT_Particle, params: HMCParameters) -> HMC_PT_Particle:
        if ALLOW_RUNTIME_CHECK:
            chex_runtime_assert(jnp.allclose(x.temperature, t))
            chex_runtime_assert(x.temperature_idx == t_idx)
        assert jnp.ndim(x.temperature) == jnp.ndim(x.temperature_idx) == 0
        return lax.cond(t_idx == 0,
                        lambda: new_from_ref(key),
                        lambda: hmc_core.move(key, x, params))

    def reset_temperature(x: HMC_PT_Particle) -> HMC_PT_Particle:
        return x._replace(
            temperature=t, temperature_idx=t_idx,
            logpdf_target=geometric_anneal(t, x.logpdf_ref, x.logpdf_final_target),
            gradlogpdf_target=geometric_anneal(t, x.gradlogpdf_ref, x.gradlogpdf_final_target)
        )

    return HMC_PT_MCMC(
        init_extend=hmc_core.init_extend, move=move, adapt_params=hmc_core.adapt_params,
        adapt_nsteps=hmc_core.adapt_nsteps, reset_temperature=reset_temperature
    ) 