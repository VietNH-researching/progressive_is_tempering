from collections import namedtuple
from typing import NamedTuple, Callable, Protocol
from jax import Array, lax, random
from jax.typing import ArrayLike
from mppt.abstract.mcmc import MCMCParticle, MCMC, MCMCParticles, generic_nsteps_adapter, generic_stepsize_adapter
from mppt.utils import follow_protocol, has_nan, dummy_use, WhileLoop, set_named_tuple_attr
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular

RdVector = Array
Float = ArrayLike

class FixedCovMVNormal(NamedTuple):
    r"""Multivariate Normal distribution with fixed covariance matrix and fixed
    scale. See documentation of function ``fixed_cov_mv_normal``.
    """
    eps: Float
    cov: Array
    chol: Array
    log_det_chol: Float
    chol_failed: bool

    @property
    def d(self):
        d, _d = self.chol.shape
        assert _d == d
        return d

    @property
    def const(self):
        return -1/2 * (self.d*jnp.log(2*jnp.pi) + 2*self.d*jnp.log(self.eps) +
                       2*self.log_det_chol)

    def rvs(self, key, mean: RdVector) -> RdVector:
        return mean + (self.chol @ random.normal(key, (self.d, ))) * self.eps

    def logpdf(self, mean: RdVector, x: RdVector) -> Float:
        v = solve_triangular(self.chol, x - mean, lower=True)
        return -0.5 * jnp.sum(v ** 2) / self.eps ** 2 + self.const

def fixed_cov_mv_normal(eps: Float, cov: Array) -> FixedCovMVNormal:
    r"""Creates a multivariate normal distribution with fixed covariance matrix
    and fixed scale. The returned object has a ``logpdf`` method which takes
    two inputs ``mean`` and ``x`` and which calculates the log of
        N(x | mean, eps ** 2 * cov).
    The similar interpretation holds for the ``rvs`` method.

    Note: if the input covariance matrix is degenerate it will be diagonalized.
    """
    d, _d = cov.shape
    assert _d == d

    chol = jnp.linalg.cholesky(cov)
    chol_failed = has_nan(chol)
    chol = lax.cond(chol_failed, lambda: _chol_diag(cov), lambda: chol)
    # if cholesky failed we'll just take the diag
    log_det_chol = jnp.sum(jnp.log(jnp.diag(chol)))

    return FixedCovMVNormal(eps, cov, chol, log_det_chol, chol_failed)

def _chol_diag(m):
    # Floor diagonal at a small epsilon to avoid log(0) = -inf in log_det_chol
    return jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(m), 1e-30)))

@follow_protocol(MCMCParticle)
class RW_Particle(Protocol):
    r"""Minimal protocol for a random walk particle. Concrete implementations
    might contain additional attributes.
    """
    position: RdVector
    logpdf_target: Float
    accepted: bool

@follow_protocol(MCMCParticles)
class RW_Particles(Protocol):
    r"""Minimal protocol for a set of random walk particles.
    """
    position: Array
    logpdf_target: Array
    accepted: Array

@follow_protocol(MCMC)
class RW_MCMC(NamedTuple):
    r"""Core random walk MCMC algorithm. See docs of the parent protocol ``MCMC``
    for more details.

    :param extend: function to extend an R^d vector into a RW_Particle object
    containing position, logpdf etc... In particular the target distribution is
    already encoded in the `extend` function.
    :param adapt_nsteps_max: the max number of steps when an adaptive number of
    MCMC steps is asked for
    :param use_bisection_for_step_size: in the parameter adaptation routine,
    whether to find the step size by binary search, or to use the default
    recommendation 2.38/sqrt(d).
    """
    extend: Callable[[RdVector], RW_Particle]
    adapt_nsteps_max: int = jnp.iinfo(jnp.int32).max
    use_bisection_for_step_size: bool = False

    def init_extend(self, key, x):
        dummy_use(key)
        assert x.ndim == 1
        return self.extend(x)

    def move(self, key, x: RW_Particle, params: FixedCovMVNormal) -> RW_Particle:
        r"""One RW-MH step: propose from N(x, eps^2 * cov), accept/reject."""
        assert x.position.ndim == 1
        key, subkey = random.split(key)
        x_star = self.extend(params.rvs(subkey, x.position))
        log_acc = x_star.logpdf_target - x.logpdf_target
        key, subkey = random.split(key)
        log_u = jnp.log(random.uniform(subkey))
        # noinspection PyTypeChecker
        return lax.cond(log_u <= log_acc,
                        lambda: set_named_tuple_attr(x_star, True),
                        lambda: set_named_tuple_attr(x, False))

    def adapt_params(self, key, x: RW_Particles, target_acc_rate=(0.2, 0.35),
                     niter_max=10, debug=False) -> FixedCovMVNormal:
        r"""Adapt the parameters of random walk based on a given sample.
        :param x: multiple `RW_Particle` bundled into one
        """
        _check_dims_rw_particles(x)
        _, d = x.position.shape
        emp_cov = jnp.cov(x.position, rowvar=False)
        if not self.use_bisection_for_step_size:
            return fixed_cov_mv_normal(2.38/jnp.sqrt(d), emp_cov)
        else:
            stepsize_to_params = lambda eps: fixed_cov_mv_normal(eps, emp_cov)
            return generic_stepsize_adapter(
                key, x, stepsize_to_params, self, target_acc_range=target_acc_rate, strategy='rw',
                niter_max=niter_max, debug=debug
            )

    def adapt_nsteps(self, key, x: RW_Particles,
                     params: FixedCovMVNormal) -> WhileLoop:
        r"""Adapt the number of MCMC steps between two swaps. This function
        simply fails back to the generic version.
        """
        return generic_nsteps_adapter(self, key, x, params,
                                      0.95, self.adapt_nsteps_max)

def _check_dims_rw_particles(x: RW_Particles):
    N1, d1 = x.position.shape
    N2, = x.logpdf_target.shape
    N3, = x.accepted.shape
    assert N1 == N2 == N3

def rw_mcmc(target_logpdf: Callable[[RdVector], Float], use_bisection=False) -> \
        RW_MCMC:
    r"""Concrete constructor for the RW_MCMC class. See docs of that class
    for details.
    """
    _rw_particle = namedtuple('_rw_particle', ['position', 'logpdf_target', 'accepted'])
    def extend(x):
        assert x.ndim == 1
        return _rw_particle(x, target_logpdf(x), True)
    # noinspection PyTypeChecker
    return RW_MCMC(extend, use_bisection_for_step_size=use_bisection)