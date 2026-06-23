r"""This module contains skeletons for MCMC algorithms. Each algorithm has two
versions: a standalone version and an integrated version for use inside PT.
"""
from __future__ import annotations
from collections import namedtuple
from functools import partial
from typing import Callable, Protocol, Any, Union, NamedTuple
from jax import Array, random, vmap, numpy as jnp, lax
from jax.typing import ArrayLike
from mppt.abstract.pt_concept import PTParticle, PTParticles, PTProblem
from mppt.jax_configs import ALLOW_RUNTIME_CHECK, chex_runtime_assert
from mppt.utils import pytree_length, dummy_use, follow_protocol, WhileLoop, bisection

RdVector = Array
Float = Union[float, ArrayLike]
Key = Any

class MCMCParticle(Protocol):
    r"""Protocol for extended state of an MCMC algorithm. An extended state
    should contain (at least) the position and the (unnormalized) logpdf.
    Concrete implementations may have additional attributes.
    """
    position: RdVector
    logpdf_target: Float

class MCMCParticles(Protocol):
    r"""A pytree of multiple extended states
    """
    position: Array
    logpdf_target: Array

class MCMC(Protocol):
    r"""Describes an MCMC algorithm. An MCMC algorithm is a quintuple of functions
    as described below.

    :param init_extend: extend an R^d vector into an MCMCParticle (e.g. bundling the
    logpdf_target with the position)
    :param extend: extend the inner state of the algorithm into an MCMCParticle. Per
    the specification in `docs/extend.md`, this method should be called `inner_extend`
    but is kept to `extend` for RWMH implementations for historical reasons
    :param move: one step of the MCMC algorithm
    :param adapt_params: adapt MCMC parameters using a set of particles
    :param adapt_nsteps: adapt number of MCMC steps using a set of particles
    and the adapted params. The output is a WhileLoop object which can be executed
    either fully or partially in order to increase performance.

    For a full description of `init_extend` and `extend`, see `docs/extend.md`
    For examples of classes following this protocol, see `impl/mcmc_random_walk.py`
    and `impl/mcmc_hmc.py`.
    """
    init_extend: Callable[[Key, RdVector], Ext]
    extend: Callable[[...], Ext]
    move: Callable[[Key, Ext, MCMCParams], Ext]
    adapt_params: Callable[[Key, Exts], MCMCParams]
    adapt_nsteps: Callable[[Key, Exts, MCMCParams], WhileLoop[Any, int]]

class PTMCMC(Protocol):
    r"""Describes an MCMC algorithm as part of Parallel Tempering. Most attributes
    are the same as the `MCMC` class. However, the particle type is `PTParticle`
    instead of `MCMCParticle`. The `reset_temperature` method allows to reset
    the temperature of a particle to the defined temperature at the creation
    of the `PTMCMC` object. See `PTMCMCEngine`.

    See `docs/extend.md` for the philosophy and `impl/pt_mcmc_random_walk.py`
    and `impl/pt_mcmc_hmc.py` for classes implementing this protocol.
    """
    init_extend: Callable[[Key, RdVector], Ext]
    extend: Callable[[...], Ext]
    move: Callable[[Key, Ext, MCMCParams], Ext]
    adapt_params: Callable[[Key, Exts], MCMCParams]
    adapt_nsteps: Callable[[Key, Exts, MCMCParams], WhileLoop[Any, int]]
    reset_temperature: Callable[[Ext], Ext]

class PTMCMCEngine(Protocol):
    def __call__(self, pt_problem: PTProblem, t: Float, t_idx: int) -> PTMCMC:
        r"""Returns a PTMCMC object for each temperature.
        :param t: temperature (could be a JAX tracer)
        :param t_idx: temperature index (could be a JAX tracer)

        Warning
        -------
        It is always assumed in this codebase that t=0 if t_idx=0; that t takes
        values between 0 and 1 where 0 corresponds to the reference distribution
        and 1 the target.
        """
        raise NotImplementedError

def batch_move(key, x, params, mcmc: Union[MCMC, PTMCMC]):
    r"""Moves multiple particles in parallel using vmap
    """
    N = pytree_length(x)
    subkeys = random.split(key, N)

    @vmap
    def mover(_k, _x):
        return mcmc.move(_k, _x, params)

    return mover(subkeys, x)

def repeated_batch_move(key, x, params, mcmc, nrepeat: int):
    r"""Moves multiple particles in parallel using vmap, then repeats that
    ``nrepeat`` times with lax.for_loops
    """
    body_func = partial(_rbm_body_func, params=params, mcmc=mcmc)
    val = (key, x)
    return lax.fori_loop(0, nrepeat, body_func, val)[1]

def _rbm_body_func(i, val, params, mcmc):
    dummy_use(i)
    key, x = val
    key, subkey = random.split(key)
    x = batch_move(subkey, x, params, mcmc)
    return key, x

def batch_extend(x, mcmc):
    r"""Apply ``mcmc.extend`` to each particle in x via vmap."""
    return vmap(mcmc.extend)(x)

def batch_init_extend(key, x, mcmc):
    r"""Apply ``mcmc.init_extend`` to each particle in x via vmap, splitting
    the key for each particle."""
    subkeys = random.split(key, pytree_length(x))
    return vmap(mcmc.init_extend)(subkeys, x)

def batch_reset_temperature(x, mcmc: PTMCMC):
    r"""Apply ``mcmc.reset_temperature`` to each particle in x via vmap.
    Used after swaps to update temperature metadata to match new Z assignment.
    """
    return vmap(mcmc.reset_temperature)(x)

def generic_stepsize_adapter(
        key, sample: Union[MCMCParticles, Any], stepsize_to_params: Callable[[Float], MCMCParams],
        mcmc: Union[MCMC, PTMCMC], target_acc_range=(0.8, 0.95),
        strategy: str = 'hmc', niter_max: int = 10, debug=False
) -> MCMCParams:
    r"""Adapt the step size in MCMC method by binary search based on a desired
    acceptance rate.
    :param sample: the current sample, following the `MCMCParticles` protocol.
    It should have the `accepted` attribute.
    :param stepsize_to_params: a function sending a value of step size epsilon into an
    MCMCParameters object of the `mcmc` argument
    :param mcmc: an MCMC algorithm following the `MCMC` or `PTMCMC` protocol.
    :param target_acc_range: the desired range of acceptance rates
    We recommend (0.2, 0.35) for random walk, (0.45, 0.6) for MALA and
    (0.8, 0.95) for HMC.
    :param strategy: how to select the initial guess for the bisection algorithm.
    Three strategies are available: `rw` sets ``eps=1.3*2.38/sqrt(d)``;
    `mala` sets ``eps = 1.25*1.65/d**(1/6)``; and `hmc` sets ``eps=1.65/d**(1/6)``.
    :param niter_max: the maximum number of iterations in the binary search
    """
    N, d = sample.position.shape
    if strategy == 'rw':
        eps_upper = 1.3 * 2.38 / d**(1/2)
    elif strategy == 'mala':
        eps_upper = 1.25 * 1.65 / d**(1/6)
    elif strategy == 'hmc':
        eps_upper = 1.65 / d**(1/6)
    else:
        raise ValueError('Unknown strategy')
    acc_default = (target_acc_range[0] + target_acc_range[1])/2
    acc_delta = abs(target_acc_range[1] - target_acc_range[0])/2

    def f(eps):
        params = stepsize_to_params(eps)
        acc = batch_move(key, sample, params, mcmc).accepted  # reuse key for smoothness
        return jnp.mean(acc) - acc_default
    if ALLOW_RUNTIME_CHECK:
        chex_runtime_assert(jnp.allclose(f(0.), 1 - acc_default))

    eps_star = bisection(f, 0., eps_upper, acc_delta, niter_max, verbose=debug,
                         verbose_offset=acc_default)
    return stepsize_to_params(eps_star)

def generic_nsteps_adapter(
        mcmc: M, key, exts: Exts, mcmc_params, conf, max_niter) -> WhileLoop:
    r"""A generic method to adapt the number of MCMC steps, given a population
    of particles. It applies MCMC iterations until the distances between the
    terminal particles and the initial particles are large enough, based on a
    statistical test with confidence interval ``conf``.

    In practice, applying this method on a d-dimensional Gaussian target and an
    optimally-tuned random walk MCMC results in around 1.7 * d MCMC steps and a
    correlation of 0.35 between the logpdf of the initial and the final samples.
    """
    batch_updater = partial(batch_move, params=mcmc_params, mcmc=mcmc)

    cont_cond, body_func = _generic_nsteps_adapter_helper(
        exts.logpdf_target, conf, max_niter, batch_updater)
    init_val = _AdapterState(key, exts, 0)
    # noinspection PyTypeChecker
    return WhileLoop(cont_cond, body_func, init_val, lambda e: e.niter)

class _AdapterState(NamedTuple):
    key: Any
    x: MCMCParticles
    niter: int

def _generic_nsteps_adapter_helper(init_logpdf_target, conf: Float, maxiter: int, batch_updater):
    r"""Build the cond/body functions for adaptive nsteps.

    Stopping criterion: MCMC iterations continue until a sufficient fraction
    of particles have moved far enough from their initial logpdf values. "Far
    enough" is defined by the (1-conf) quantile of pairwise absolute differences
    in the initial logpdf array. The 1.25 multiplier provides a small margin
    above the nominal (1-conf) fraction.
    """
    q = quantile_absolute_difference(init_logpdf_target, 1 - conf)
    def stop_cond(val: _AdapterState):
        failed_idx = jnp.abs(val.x.logpdf_target - init_logpdf_target) <= q
        cond1 = jnp.mean(failed_idx) <= (1.25 * (1 - conf))
        cond2 = val.niter >= maxiter
        return jnp.logical_or(cond1, cond2)

    def cont_cond(val):
        return jnp.logical_not(stop_cond(val))

    def body_func(val: _AdapterState) -> _AdapterState:
        key, x, niter = val
        key, subkey = random.split(key)
        x = batch_updater(subkey, x)
        return _AdapterState(key, x, niter + 1)

    return cont_cond, body_func

def quantile_absolute_difference(x, q):
    r"""Compute the q-th quantile of |x_i - x_{N-1-i}| (reverse-paired
    absolute differences). This gives a scale estimate for the spread of x
    without requiring O(N^2) pairwise comparisons.
    """
    _, = x.shape
    dat = jnp.abs(x - x[::-1])
    return jnp.quantile(dat, q)

@follow_protocol(MCMC)
class MagicMCMC(NamedTuple):
    r"""Magic MCMC kernel, mostly for testing purposes
    """
    direct_rvs: Callable[[Key], RdVector]
    direct_logpdf: Callable[[RdVector], Float]
    _mcmc_particle = namedtuple('_mcmc_particle', ['logpdf_target', 'position'])

    def extend(self, x):
        return self._mcmc_particle(self.direct_logpdf(x), x)

    def init_extend(self, key, x):
        dummy_use(key)
        return self.extend(x)

    def move(self, key, x, params):
        dummy_use(x, params)
        return self.extend(self.direct_rvs(key))

    @staticmethod
    def adapt_params(key, x):
        dummy_use(key, x)
        return None

    @staticmethod
    def adapt_nsteps(key, x, params):
        dummy_use(key, x, params)
        return WhileLoop(lambda e: False, lambda e: e, 1, lambda e: e)