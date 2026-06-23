from __future__ import annotations
from collections import namedtuple
from typing import Protocol, Callable

from jax.typing import ArrayLike
import numpy as np
from jax import numpy as jnp, random, Array, vmap, lax
from jax.scipy.stats import multivariate_normal as jmvn, norm as jnorm
from mppt.abstract.pt_concept import PTProblem, geometric_anneal
from mppt.abstract.mcmc import MCMC, batch_move, batch_init_extend
from mppt.tests.distribution_test_helper import np_random_mean_and_cov
from mppt.utils import random_split
from mppt.jax_configs import allow_runtime_check, chexify_no_async

Float = ArrayLike

def create_random_pt_problem(d, seed, logZ=0):
    mean_init, cov_init = np_random_mean_and_cov(d, seed=seed)
    mean_target, cov_target = np_random_mean_and_cov(d, seed=seed+42)
    np.random.seed()

    def rvs_ref(key):
        return random.multivariate_normal(key, mean_init, cov_init)
    def logpdf_ref(x):
        assert x.shape == (d, )
        return jmvn.logpdf(x, mean_init, cov_init)
    def logpdf_final_target(x):
        assert x.shape == (d, )
        return jmvn.logpdf(x, mean_target, cov_target) + logZ
    def get_intermediate_dist(t):
        return dist_pt_intermediate(mean_init, cov_init, mean_target, cov_target, t)

    return PTProblem(rvs_ref, logpdf_ref, logpdf_final_target), get_intermediate_dist

def dist_pt_intermediate(mean_init, cov_init, mean_target, cov_target, t):
    inv0 = jnp.linalg.inv(cov_init)
    inv1 = jnp.linalg.inv(cov_target)
    # inv = (1-t) * inv0 + t * inv1
    inv = geometric_anneal(t, inv0, inv1)
    cov = jnp.linalg.inv(inv)
    # mean = cov @ ((1-t) * (inv0@mean_init) + t * (inv1@mean_target))
    mean = cov @ geometric_anneal(t, inv0@mean_init, inv1@mean_target)
    return mean, cov

def create_pt_problem_1d(mean0, sigma0, mean1, sigma1, logZ=1234):
    def rvs_ref(key):
        return random.normal(key) * sigma0 + mean0
    def logpdf_ref(x):
        assert jnp.ndim(x) == 0
        return jnorm.logpdf(x, loc=mean0, scale=sigma0)
    def logpdf_final_target(x):
        assert jnp.ndim(x) == 0
        return jnorm.logpdf(x, loc=mean1, scale=sigma1) + logZ
    def get_intermediate_dist(t):
        return dist_pt_intermediate_1d(mean0, sigma0, mean1, sigma1, t)
    return PTProblem(rvs_ref, logpdf_ref, logpdf_final_target), get_intermediate_dist

LocScale = namedtuple('LocScale', ['loc', 'scale'])

def dist_pt_intermediate_1d(mean0, sigma0, mean1, sigma1, t):
    inv0 = 1/sigma0**2
    inv1 = 1/sigma1**2
    inv = geometric_anneal(t, inv0, inv1)
    cov = 1/inv
    mean = cov * geometric_anneal(t, inv0 * mean0, inv1 * mean1)
    return LocScale(mean, jnp.sqrt(cov))

class Dist(Protocol):
    rvs: Callable[[Key], Draw]
    logpdf: Callable[[Draw], Float]
EstimateResult = namedtuple('EstimateResult', ['est', 'lo', 'hi'])

def tv_est_symmetric(key, dist1: Dist, dist2: Dist, samplesize: int,
                     conf: float = 0.95) -> EstimateResult:
    r"""Estimates TV distance using samples from both distributions
    """
    key, subkey = random_split(key, samplesize+1)
    mix_rvs = vmap(_generate_mixture_draw, in_axes=(0, None, None))
    sample = mix_rvs(subkey, dist1, dist2)
    sample = exp_ratio(vmap(dist1.logpdf)(sample),
                       vmap(dist2.logpdf)(sample))
    return mean_est(sample, conf)

def _generate_mixture_draw(key, dist1, dist2):
    key, subkey = random.split(key)
    ber = random.bernoulli(subkey)
    key, subkey = random.split(key)
    return lax.cond(ber, lambda: dist1.rvs(subkey), lambda: dist2.rvs(subkey))

def exp_ratio(a, b):
    r"""Calculates |e^a - e^b|/(e^a + e^b) stably and with extended precision
    """
    c = -jnp.abs(a - b)
    return jnp.abs(jnp.expm1(c)) / (1 + jnp.exp(c))

def mean_est(sample: Array, conf: float) -> EstimateResult:
    r"""Estimates mean of a sample and calculates CI using basic CLT
    """
    N, = sample.shape
    est = jnp.mean(sample)
    std = jnp.std(sample)/jnp.sqrt(N)
    coef = jnorm.ppf((1+conf)/2)
    return EstimateResult(est, est - coef * std, est + coef * std)

_normal_product = namedtuple('_normal_product', ['rvs', 'logpdf'])
def normal_product(mean0, sigma0, mean1, sigma1):
    def rvs(key):
        key0, key1 = random.split(key)
        x0 = random.normal(key0) * sigma0 + mean0
        x1 = random.normal(key1) * sigma1 + mean1
        return jnp.array([x0, x1])

    def logpdf(x):
        assert x.shape == (2, )
        x0, x1 = x
        return jnorm.logpdf(x0, loc=mean0, scale=sigma0) +\
            jnorm.logpdf(x1, loc=mean1, scale=sigma1)

    return _normal_product(rvs, logpdf)

def rrate_two_normals(key, mean0, sigma0, mean1, sigma1, N) -> EstimateResult:
    r"""Estimates the rejection rate of the swap step between two normal distributions
    N(mean0, sigma0**2) and N(mean1, sigma1**2) using N samples
    """
    dist_a = normal_product(mean0, sigma0, mean1, sigma1)
    dist_b = normal_product(mean1, sigma1, mean0, sigma0)
    # noinspection PyTypeChecker
    return tv_est_symmetric(key, dist_a, dist_b, N)

def adapt_stepsize_tester(key, mcmc: MCMC, sample, target_range, max_iter, verbose):
    r"""Stepsize adapter tester for random walk and HMC
    :param sample: an (N, d) sample in R^d.
    """
    key, subkey = random.split(key)
    sample = batch_init_extend(subkey, sample, mcmc)
    with allow_runtime_check(True):
        mcmc_adapt_params = mcmc.adapt_params
        if not verbose:
            mcmc_adapt_params = chexify_no_async(mcmc_adapt_params)
        key, subkey = random.split(key)
        # RW_MCMC and HMC have optional arguments to adapt_params in addition to the
        # MCMC protocol.
        # noinspection PyArgumentList
        params = mcmc_adapt_params(
            subkey, sample, target_acc_rate=target_range, niter_max=max_iter, debug=verbose
        )
        # Validate with the same key
        acc = batch_move(subkey, sample, params, mcmc).accepted
        acc = jnp.mean(acc)
        if verbose:
            print('Real acceptance rate', acc)
        assert target_range[0] <= acc <= target_range[1]