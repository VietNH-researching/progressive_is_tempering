from typing import NamedTuple, Optional
import jax
from mppt.abstract.mcmc import PTMCMCEngine, batch_reset_temperature, repeated_batch_move
from mppt.abstract.pt_concept import PTParticles, Float, PTProblem, geometric_anneal
from jax import numpy as jnp, random
from jax.scipy.special import logsumexp, softmax
from mppt.jax_configs import ALLOW_RUNTIME_CHECK, chex_runtime_assert
from mppt.utils import is_all_same, bisection, pytree_index, pytree_length
from mppt.impl import logZ_estimators as logZ_library

_KNOWN_LOGZ_ESTIMATORS = ('SMC_like', 'robust', 'reverse_logistic')

class OneStepSMCResult(NamedTuple):
    r"""Output of one SMC step: the resampled-and-moved particles at the new
    temperature, local log-normalizing-constant estimates, and the number of
    MCMC steps used.
    """
    xnext: PTParticles
    logZ_local: dict[str, Float]  # local log Z estimated by different methods
    niter_mcmc: int

    @property
    def t(self):
        ret = self.xnext.temperature[0]
        assert ret.ndim == 0
        return ret

    @property
    def t_idx(self):
        ret = self.xnext.temperature_idx[0]
        assert ret.ndim == 0
        return ret

def _systematic_resampling(key, w):
    r"""Core systematic resampling: generate N evenly spaced points on [0,1]
    with a single random offset, then map through the CDF of weights w.
    """
    N = w.shape[0]
    key_u, key_perm = jax.random.split(key)

    sorted_us = (jnp.arange(N) + jax.random.uniform(key_u)) / N

    # Uncomment the next line should make test fail
    # sorted_us = jnp.sort(jax.random.uniform(key_u, (N, )))

    # Clip to N-1: floating-point error in cumsum(w) can make the last CDF
    # value slightly < 1.0, causing searchsorted to return N (out of bounds).
    idx = jnp.searchsorted(jnp.cumsum(w), sorted_us)
    return jnp.minimum(idx, N - 1)

def systematic_resampling(key, w):
    r"""Systematic resampling with a random pre-permutation to break
    ordering artifacts. Returns indices into the original array.
    """
    N = w.shape[0]
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, N)
    key, subkey = jax.random.split(key)
    return perm[_systematic_resampling(subkey, w[perm])]

def ess_ratio(lw):
    r"""Effective sample size (ESS) as a fraction of the total number of
    particles. ESS = 1/sum(W_i^2) where W_i are normalized weights.
    Returns a value in (0, 1]; 1 means all weights are equal.
    """
    log_total = logsumexp(lw)
    W = jax.lax.cond(
        jnp.isfinite(log_total),
        lambda: jnp.exp(lw - log_total),
        lambda: jnp.zeros_like(lw),
    )
    denom = lw.shape[0] * jnp.sum(W**2)
    return jax.lax.cond(
        denom > 0,
        lambda: 1 / denom,
        lambda: jnp.array(0., dtype=denom.dtype),
    )

def get_next_temperature(x: PTParticles, new_ess_ratio: Float, tol) -> Float:
    r"""Find the next temperature t_next > t_current such that reweighting
    from t_current to t_next drops the ESS ratio to ``new_ess_ratio``.
    Uses bisection on [t_current, 1]. If ESS stays above threshold even
    at t=1, returns 1.0 directly.
    """
    if ALLOW_RUNTIME_CHECK:
        chex_runtime_assert(is_all_same(x.temperature))
        chex_runtime_assert(is_all_same(x.temperature_idx))
    t_old = x.temperature[0]
    assert jnp.ndim(t_old) == 0
    def obj(t):
        # lw = (t - t_old) * (x.logpdf_final_target - x.logpdf_ref)
        lw = geometric_anneal(t, x.logpdf_ref, x.logpdf_final_target) -\
            geometric_anneal(t_old, x.logpdf_ref, x.logpdf_final_target)
        return ess_ratio(lw) - new_ess_ratio
    def _bisect():
        t = bisection(obj, t_old, 1., tol, 30, False)
        return jax.lax.cond(1.0 - t < 1e-6, lambda: 1.0, lambda: t)
    return jax.lax.cond(obj(1.0) >= 0, lambda: 1.0, _bisect)

def one_step_smc(key, x: PTParticles, problem: PTProblem, mcmc_engine: PTMCMCEngine, niter_mcmc: Optional[int], new_ess_ratio: Float = 0.5, logZ_estimators: tuple[str] = ('SMC_like', 'robust', 'reverse_logistic')) -> OneStepSMCResult:
    r"""One step of SMC annealing:
    1. Find next temperature via ESS-based bisection
    2. Reweight and resample particles at new temperature
    3. Adapt MCMC parameters and run MCMC moves
    4. Estimate local log-normalizing constant

    :param niter_mcmc: fixed number of MCMC steps, or None for adaptive
    :param new_ess_ratio: target ESS ratio for temperature selection (default 0.5)
    :param logZ_estimators: names of estimator functions in ``impl.logZ_estimators``
    """
    # Choose next temperature
    t_next = get_next_temperature(x, new_ess_ratio, 1e-2)

    # Resample
    key, subkey = random.split(key)
    # lw = (t_next - x.temperature[0]) * (x.logpdf_final_target - x.logpdf_ref)
    lw = geometric_anneal(t_next, x.logpdf_ref, x.logpdf_final_target) -\
        geometric_anneal(x.temperature[0], x.logpdf_ref, x.logpdf_final_target)
    idx = systematic_resampling(subkey, softmax(lw))
    x_next = pytree_index(x, idx)

    # Change temperature
    # noinspection PyTypeChecker
    mcmc = mcmc_engine(problem, t_next, x.temperature_idx[0] + 1)
    x_next = batch_reset_temperature(x_next, mcmc)

    # Move with adaptive parameters
    key, subkey = random.split(key)
    mcmc_params = mcmc.adapt_params(subkey, x_next)
    if niter_mcmc is None:
        key, subkey = random.split(key)
        niter_mcmc = mcmc.adapt_nsteps(subkey, x_next, mcmc_params).execute()
    key, subkey = random.split(key)
    x_next = repeated_batch_move(subkey, x_next, mcmc_params, mcmc, niter_mcmc)

    # Get logZ
    key, subkey = random.split(key)
    logZ = _get_logZ(subkey, x, x_next, x.temperature[0], t_next, logZ_estimators)

    return OneStepSMCResult(x_next, logZ, niter_mcmc)

def _get_logZ(key, x: PTParticles, x_next: PTParticles, t, t_next, logZ_estimators):
    r"""Estimate the local log-normalizing constant between temperatures t and
    t_next using particles from both the old and new populations. Constructs
    a two-level (T=2) array and delegates to the standard logZ estimators.
    """
    x_old = x.logpdf_target
    x_new = geometric_anneal(t_next, x.logpdf_ref, x.logpdf_final_target)
    x_next_old = geometric_anneal(t, x_next.logpdf_ref, x_next.logpdf_final_target)
    x_next_new = x_next.logpdf_target

    N = pytree_length(x)
    assert x_old.shape == x_new.shape == x_next_old.shape ==\
           x_next_new.shape == (N, )

    arr_old = jnp.array([x_old, x_next_old])
    arr_new = jnp.array([x_new, x_next_new])

    res = {}
    for est in logZ_estimators:
        key, subkey = random.split(key)
        estimator = _resolve_logZ_estimator(est)
        res[est] = estimator(subkey, arr_old, arr_new, jnp.array([0., 1.]))
    return res

def _resolve_logZ_estimator(name: str):
    r"""Resolve SMC logZ estimator names with a stable error boundary.

    SMC calls this inside a jitted step, so an unknown name should fail while
    tracing with an actionable ``ValueError`` rather than an ``AttributeError``
    from a dynamic module lookup.
    """
    if name not in _KNOWN_LOGZ_ESTIMATORS:
        known = ', '.join(_KNOWN_LOGZ_ESTIMATORS)
        raise ValueError(f"Unknown SMC logZ estimator '{name}'. Known estimators: {known}")
    return getattr(logZ_library, name)
