from mppt.jax_configs import x64, default_gpu_memory_setting; x64(), default_gpu_memory_setting()  # THIS LINE MUST BE ON TOP
from collections import namedtuple
from itertools import count
from typing import NamedTuple, Union
import pytest
from jax.scipy.stats import multivariate_normal as mvn
import numpy as np
from tqdm import trange, tqdm
from scipy.stats import multivariate_normal as smvn
from mppt.tests.distribution_test_helper import np_random_mean_and_cov, multivariate_sample_tester
from mppt.utils import follow_protocol, dummy_use, new_jax_key, assert_no_nan
from mppt.abstract.mcmc import MCMC, generic_nsteps_adapter, batch_move, batch_extend, repeated_batch_move
from jax import numpy as jnp, random, jit
from mppt.impl.mcmc_random_walk import rw_mcmc, fixed_cov_mv_normal
from functools import partial
import chex

@follow_protocol(MCMC)
class OU(NamedTuple):
    d: int
    rho: float
    conf: float
    maxniter: Union[int,float] = jnp.inf
    OUParticle = namedtuple('OUParticle', ['position', 'logpdf_target'])

    def raw_logpdf_target(self, x):
        assert x.shape == (self.d, )
        return mvn.logpdf(x, jnp.zeros(self.d), jnp.identity(self.d))

    def extend(self, x):
        return self.OUParticle(x, self.raw_logpdf_target(x))

    def move(self, key, extended, params):
        dummy_use(params)
        x = self.rho * extended.position + jnp.sqrt(1 - self.rho**2) * \
            random.normal(key, (self.d, ))
        return self.extend(x)

    def adapt_params(self, key, extended):
        dummy_use(self, key, extended)
        return None

    def adapt_nsteps(self, key, extended, params):
        # noinspection PyTypeChecker
        return generic_nsteps_adapter(self, key, extended, params, self.conf, self.maxniter)

def _test_OU(seed_init, d, rho, nparticles, nsteps, verbose):
    mean_init, cov_init = np_random_mean_and_cov(d, seed=seed_init)
    np.random.seed()
    init_particles = np.random.multivariate_normal(mean=mean_init, cov=cov_init,
                                                   size=nparticles)
    target_dist = smvn(mean=np.zeros(d), cov=np.identity(d))
    if verbose:
        multivariate_sample_tester(init_particles, target_dist, True, 1.0, True, 'starting sample')

    obj = OU(d, rho, 0.)
    x = batch_extend(init_particles, obj)
    key = new_jax_key()

    for _ in trange(nsteps, disable=not verbose):
        key, subkey = random.split(key)
        # noinspection PyTypeChecker
        x = batch_move(subkey, x, None, obj)

    multivariate_sample_tester(x.position, target_dist, True, 0.02, verbose, 'terminal sample')

@pytest.mark.slow
def test_OU():
    _test_OU(5, 5, 0.92, 10_000, 70, False)

def _quantile_distance_sn(key, d, alpha, samplesize=100_000):
    r"""The alpha quantile of |logpdf(x) - logpdf(y)| for x and y independently
    drawn from the standard normal distribution in dimension d.
    """
    key, subkey = random.split(key)
    x = random.normal(subkey, shape=(samplesize, d))
    x = jnp.sum(x**2, axis=1) * 0.5

    key, subkey = random.split(key)
    y = random.normal(subkey, shape=(samplesize, d))
    y = jnp.sum(y**2, axis=1) * 0.5

    return jnp.quantile(jnp.abs(x - y), alpha)

def _quantile_test_violation(key, rho, d, alpha, samplesize=100_000):
    r"""Calculates
        Proba(|logpdf(x^rho) - logpdf(x)| <= q_alpha)
    where x^rho is x after an OU step with coefficient rho,
    and q_alpha is the result of the function `_quantile_distance_sn`.
    Note that if x^rho were independent of x, the above probability would
    be exactly alpha; whereas if x^rho were exactly equal to x, it would be
    exactly one.
    As such, this probability can be interpreted as the proportion of particle
    pairs that fail the independence test at level ``alpha``.
    """
    key, subkey = random.split(key)
    q_alpha = _quantile_distance_sn(subkey, d, alpha, samplesize)

    key, subkey = random.split(key)
    x = random.normal(subkey, shape=(samplesize, d))
    x_logpdf = jnp.sum(x**2, axis=1) * 0.5

    key, subkey = random.split(key)
    y = rho * x + jnp.sqrt(1-rho**2) * random.normal(subkey, shape=(samplesize, d))
    y_logpdf = jnp.sum(y**2, axis=1) * 0.5

    return jnp.mean(jnp.abs(x_logpdf - y_logpdf) <= q_alpha)

def _min_nsteps(key, d, alpha, rho, samplesize=100_000, verbose1=False,
                verbose2=False):
    r"""Returns the minimal number of OU steps such that
        Proba(|logpdf(Y) - logpdf(X)| <= q_alpha) <= 1.25 * alpha
    where Y ~ OU(X) for nsteps, and `q_alpha` is the value returned by
     `_quantile_distance_sn`.
    """
    @jit
    def prob(k, nstep):
        return _quantile_test_violation(k, rho**nstep, d, alpha, samplesize)

    for n in tqdm(count(), disable=not verbose1):
        key, subkey = random.split(key)
        current_prob = prob(subkey, n)
        if verbose2: print('n', n, 'rho**n', rho**n, 'prob', current_prob)
        if current_prob < 1.25 * alpha:
            return n

def _test_generic_nsteps_adapter(d, alpha, rho, threshold, maxiter=jnp.inf, samplesize=100_000, verbose=False):
    mcmc = OU(d, rho, 1-alpha, maxiter)
    key = new_jax_key()
    key, subkey = random.split(key)
    sample = random.normal(subkey, shape=(samplesize, d))
    sample = batch_extend(sample, mcmc)

    if verbose: print('Running adaptation...')
    keu, subkey = random.split(key)
    nsteps = mcmc.adapt_nsteps(subkey, sample, None).execute()
    if verbose: print('nsteps', nsteps)

    key, subkey = random.split(key)
    nsteps_ref = _min_nsteps(subkey, d, alpha, rho, samplesize, verbose)
    if verbose: print('nsteps_ref', nsteps_ref)
    nsteps_ref = jnp.minimum(nsteps_ref, maxiter)

    assert jnp.abs(nsteps_ref - nsteps) <= threshold

@pytest.mark.slow
def test_generic_nsteps_adapter():
    _test_generic_nsteps_adapter(4, 0.07, 0.985, 7, samplesize=200_000, verbose=False)
    _test_generic_nsteps_adapter(4, 0.07, 0.985, 7, maxiter=16, samplesize=200_000, verbose=False)
    _test_generic_nsteps_adapter(4, 0.05, 0.99, 7, samplesize=200_000, verbose=False)

def _test_repeated_batch_move(nparticles, nrepeats, verbose):
    mean = jnp.array([0.5, -0.25, 1.0])
    cov = jnp.array([
        [1.2, 0.1, 0.0],
        [0.1, 1.5, 0.2],
        [0.0, 0.2, 1.1],
    ])
    mean, cov = jnp.array(mean), jnp.array(cov)
    mcmc = rw_mcmc(partial(mvn.logpdf, mean=mean, cov=cov))
    params = fixed_cov_mv_normal(2.38/jnp.sqrt(3), cov)
    xstart = random.multivariate_normal(random.key(321), mean, cov, shape=(nparticles,))
    xstart = batch_extend(xstart, mcmc)

    # The oracle is equivalence between the lax.fori_loop implementation and
    # the explicit Python loop under the same key stream, not distribution fit.
    key = random.key(123)
    if verbose: print('JAX.LAX...')
    res1 = repeated_batch_move(key, xstart, params, mcmc, nrepeats)

    if verbose: print('For loop...')
    res2 = xstart
    for _ in range(nrepeats):
        key, subkey = random.split(key)
        res2 = batch_move(subkey, res2, params, mcmc)

    if verbose: print('Acc rate', jnp.mean(res2.accepted))
    chex.assert_trees_all_close(res1, res2)
    assert_no_nan(res1)
    assert_no_nan(res2)

def test_repeated_batch_move():
    _test_repeated_batch_move(50, 5, False)
