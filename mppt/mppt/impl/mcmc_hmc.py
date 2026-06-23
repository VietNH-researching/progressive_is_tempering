from __future__ import annotations
from functools import partial
from typing import NamedTuple, Callable, Protocol, Union
from mppt.impl.mcmc_random_walk import FixedCovMVNormal, fixed_cov_mv_normal
from mppt.utils import follow_protocol
from mppt.abstract.mcmc import MCMC, MCMCParticle, MCMCParticles, generic_stepsize_adapter, generic_nsteps_adapter
from jax import numpy as jnp, Array, random, lax, value_and_grad
from jax.typing import ArrayLike
from collections import namedtuple

Float = ArrayLike
Bool = ArrayLike
RdVector = Array

@follow_protocol(MCMCParticle)
class HMCParticle(Protocol):
    r"""Protocol for a Hamiltonian Monte Carlo particle.
    Implementations may have additional attributes.
    """
    position: RdVector
    logpdf_target: Float
    accepted: Bool
    veloc: RdVector
    gradlogpdf_target: RdVector

@follow_protocol(MCMCParticles)
class HMCParticles(Protocol):
    r"""Protocol for multiple Hamiltonian Monte Carlo particles.
    """
    position: Array
    logpdf_target: Array
    accepted: Array
    veloc: Array
    gradlogpdf_target: Array

class HMCParticleNoVeloc(Protocol):
    r"""Protocol for a Hamiltonian Monte Carlo particle without velocity.
    """
    position: RdVector
    logpdf_target: Float
    accepted: Bool
    gradlogpdf_target: RdVector

class HMCParameters(NamedTuple):
    r"""Parameters of a Hamiltonian Monte Carlo algorithm
    :param scaled_cov: contains info about the covariance matrix and
    the proposal scale `epsilon`
    :param refresh_rate: v_new ~ N(\sqrt{1-r} v_old, r v_new)

    Interpretation:
        * Time before refresh: eps / R
        * Discretization fineness: 1 / R
        * No. of cyckes: Niter * R
        * Acceptance rate is determined by: eps
    """
    scaled_cov: FixedCovMVNormal
    refresh_rate: Float

@follow_protocol(MCMC)
class HMC(NamedTuple):
    r"""Hamiltonian Monte Carlo algorithm.
    The distribution is coupled with an N(0,1) variable and the deterministic dynamic
    is:
        dx/dt = Lv
        dv/dt = L^T \gradlog \pi(x)
    where optimally L should be close to the covariance of pi.

    :param feed_extend: extend an R^d vector into an HMCParticle object, except the
    velocity field.
    :param default_refresh_rate: default refreshment rate. Roughly speaking the velocity is
    completely refreshed after O(1/refresh_rate) steps.
    :param mala_mode: whether to run MALA instead of HMC. Then the refresh rate
    will be set to 1. regardless of user's choice. Additionally the parameter
    adapter will aim for an acceptance rate between 0.45 and 0.6.

    To clearly distinguish between the internal extend and the external extend,
    this class has no extend method but only `internal_extend` and `init_extend`.
    See `docs/extend.md` for full explanation.
    """
    feed_extend: Callable[[RdVector], Union[HMCParticleNoVeloc, namedtuple]]
    HMCParticleType: type[namedtuple]
    adapt_nsteps_max: int = jnp.iinfo(jnp.int32).max
    default_refresh_rate: float = 0.07
    mala_mode: bool = False

    @property
    def default_refresh_rate_(self):
        return 1.0 if self.mala_mode else self.default_refresh_rate

    def init_extend(self, key, x: RdVector):
        assert x.ndim == 1
        return self.HMCParticleType(veloc=random.normal(key, shape=x.shape),
                                    **self.feed_extend(x)._asdict())

    def inner_extend(self, x: RdVector, v: RdVector) -> HMCParticle:
        assert x.ndim == v.ndim == 1
        assert x.shape == v.shape
        return self.HMCParticleType(veloc=v,
                                    **self.feed_extend(x)._asdict())

    @staticmethod
    def _check_shapes(x: HMCParticle):
        d, = x.position.shape
        assert x.veloc.shape == (d, )
        assert x.logpdf_target.shape == ()
        assert x.gradlogpdf_target.shape == (d, )
        assert x.accepted.shape == ()

    @staticmethod
    def _check_shapes_multiple(x: HMCParticles):
        N, d = x.position.shape
        assert x.veloc.shape == (N, d)
        assert x.logpdf_target.shape == (N, )
        assert x.gradlogpdf_target.shape == (N, d)
        assert x.accepted.shape == (N, )

    @staticmethod
    def _update_veloc(x: HMCParticle, v: RdVector) -> HMCParticle:
        # noinspection PyUnresolvedReferences
        return x._replace(veloc=v)

    @staticmethod
    def _update_acc(x: HMCParticle, acc: bool) -> HMCParticle:
        # noinspection PyUnresolvedReferences
        return x._replace(accepted=acc)

    @staticmethod
    def _joint_logpdf(x: HMCParticle):
        return x.logpdf_target - 0.5 * jnp.sum(x.veloc**2)

    def move(self, key, x: HMCParticle, params: HMCParameters) -> HMCParticle:
        r"""One HMC step: leapfrog jump, Metropolis accept/reject, velocity refresh."""
        self._check_shapes(x)
        x_new = self._hmc_jump(x, params)
        log_acc_rate = self._joint_logpdf(x_new) - self._joint_logpdf(x)
        key, subkey = random.split(key)
        log_u = jnp.log(random.uniform(subkey))
        x_ret = lax.cond(log_u < log_acc_rate,
                         lambda: self._update_acc(x_new, True),
                         lambda: self._update_acc(x, False))
        key, subkey = random.split(key)
        return self._hmc_refresh_veloc(subkey, x_ret, params.refresh_rate)

    def _hmc_jump(self, x: HMCParticle, params: HMCParameters) -> HMCParticle:
        r"""Verlet integrator: half step position, full step velocity, then
        half step position again
        """
        x = self._hmc_jump_pos(x, params)
        x = self._hmc_jump_veloc(x, params)
        x = self._hmc_jump_veloc(x, params)
        x = self._hmc_jump_pos(x, params)

        return self._update_veloc(x, -x.veloc)

    def _hmc_jump_pos(self, x: HMCParticle, params: HMCParameters) -> HMCParticle:
        pos_new = x.position + \
                  0.5 * params.scaled_cov.eps * params.scaled_cov.chol @ x.veloc
        return self.inner_extend(pos_new, x.veloc)

    def _hmc_jump_veloc(self, x: HMCParticle, params: HMCParameters) -> HMCParticle:
        return self._update_veloc(
            x, x.veloc + (
                    0.5*params.scaled_cov.eps*params.scaled_cov.chol.T@x.gradlogpdf_target
            )
        )

    def _hmc_refresh_veloc(self, key, x: HMCParticle, refresh_rate) -> HMCParticle:
        r"""Partial velocity refresh: mix current velocity with fresh noise.
        At refresh_rate=1 this is a full refresh (MALA-like).
        """
        z = random.normal(key, x.position.shape)
        v_new = -jnp.sqrt(1-refresh_rate) * x.veloc + jnp.sqrt(refresh_rate) * z
        # Uncommenting the following lines breaks the test.
        # v_new = jnp.sqrt(1 - refresh_rate) * x.veloc + jnp.sqrt(refresh_rate) * z
        # v_new = -jnp.sqrt(1-refresh_rate) * x.veloc + refresh_rate * z
        return self._update_veloc(x, v_new)

    def adapt_params(self, key, x: HMCParticles, target_acc_rate=(0.8, 0.95),
                     niter_max=10, debug=False) -> HMCParameters:
        r"""Adaptation strategy: the covariance matrix is the empirical covariance
        and the step size is adjusted based on the given acc rate.
        """
        self._check_shapes_multiple(x)
        emp_cov = jnp.cov(x.position, rowvar=False)
        emp_cov = fixed_cov_mv_normal(eps=1, cov=emp_cov)

        def stepsize_to_params(eps):
            cov_obj = emp_cov._replace(eps=eps)
            return HMCParameters(cov_obj, self.default_refresh_rate_)

        if not self.mala_mode:
            # noinspection PyTypeChecker
            return generic_stepsize_adapter(
                key, x, stepsize_to_params, self, target_acc_rate, strategy='hmc',
                niter_max=niter_max, debug=debug
            )
        else:
            # noinspection PyTypeChecker
            return generic_stepsize_adapter(
                key, x, stepsize_to_params, self, (0.4, 0.65), strategy='mala',
                niter_max=niter_max, debug=debug
            )

    def adapt_nsteps(self, key, x: HMCParticles, params):
        r"""Adapt the number of MCMC steps between two swaps. This function
        simply fails back to the generic version.
        """
        # noinspection PyTypeChecker
        return generic_nsteps_adapter(self, key, x, params,
                                      0.95, self.adapt_nsteps_max)

    # === USAGE EXAMPLE ===
    # === This is an example of a fully-featured implementation of HMC with a feed_extend ===

    class SimpleHMCState(NamedTuple):
        r"""Minimal HMC particle type for standalone (non-PT) usage.
        See ``HMC.simpleInstance`` for construction.
        """
        position: RdVector
        logpdf_target: Float
        accepted: Bool
        veloc: RdVector
        gradlogpdf_target: RdVector

        SimpleHMCNoVeloc = namedtuple('SimpleHMCNoVeloc',
                                      ['position', 'logpdf_target', 'accepted', 'gradlogpdf_target'])

        @classmethod
        def feed_extend(cls, x: RdVector, logpdf_target) -> 'SimpleHMCNoVeloc':
            val, grad = value_and_grad(logpdf_target)(x)
            return cls.SimpleHMCNoVeloc(x, val, True, grad)

    @classmethod
    def simpleInstance(cls, logpdf_target):
        r"""Create a simple instance of ``HMC`` given the logpdf of a target
        distribution
        """
        feed_extend = partial(cls.SimpleHMCState.feed_extend,
                              logpdf_target=logpdf_target)
        return cls(feed_extend, cls.SimpleHMCState)