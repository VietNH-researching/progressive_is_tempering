from mppt.jax_configs import x64, default_gpu_memory_setting; x64(); default_gpu_memory_setting()  # THIS LINE MUST BE AT TOP
from mppt.tests.distribution_test_helper import np_random_mean_and_cov, multivariate_sample_tester
from mppt.utils import new_jax_key
from mppt.impl.mcmc_random_walk import fixed_cov_mv_normal, rw_mcmc
from jax import random, numpy as jnp, jit
from jax.scipy.stats import multivariate_normal as jmvn
import numpy as np
from tqdm import trange
from mppt.abstract.mcmc import batch_move, batch_extend
from scipy.stats import multivariate_normal as smvn
from functools import partial
from mppt.tests.pt_test_helper import adapt_stepsize_tester

def _test_multivar_normal_(mean, cov, cov2, eps, verbose):
    if verbose:
        print('=== New test ===')
        print('mean', mean)
        print(f'cov\n{cov}')
        print(f'cov2\n{cov2}')
        print('eps', eps)
    cov = eps**2 * cov
    key = new_jax_key()
    x_ref = random.multivariate_normal(key, mean, cov)
    logpdf_ref = jmvn.logpdf(x_ref, mean, cov)
    if verbose:
        print('x_ref', x_ref)
        print('logpdf_ref', logpdf_ref)

    res = fixed_cov_mv_normal(eps, cov2)
    x_res = res.rvs(key, mean)  # intentional key reuse
    logpdf_res = res.logpdf(mean, x_res)
    if verbose:
        print('x_res', x_res)
        print('logpdf_res', logpdf_res)

    assert jnp.allclose(x_ref, x_res)
    assert jnp.allclose(logpdf_ref, logpdf_res)

def _test_multivar_normal(d, eps, verbose):
    mean, cov = np_random_mean_and_cov(d)
    _test_multivar_normal_(mean, cov, cov, eps, verbose)

def _test_multivar_normal_degenerate(d, eps, verbose):
    mean, cov = np_random_mean_and_cov(d)
    cov = jnp.diag(jnp.diag(cov))

    cov2 = np.array(cov)
    for i in range(d):
        for j in range(d):
            if i != j:
                cov2[i, j] = cov2[i, j] - 10_000
    try:
        np.linalg.cholesky(cov2)
    except np.linalg.LinAlgError:
        pass
    else:
        raise AssertionError('Matrix should be degenerate')

    _test_multivar_normal_(mean, cov, cov2, eps, verbose)

def test_multivar_normal():
    _test_multivar_normal(3, 0.45, False)
    _test_multivar_normal(4, 1.35, False)
    _test_multivar_normal_degenerate(5, 1.2, False)

def _test_RW_invariance(dim, seed_init, seed_target, niter, nparticles, acc_rate_range, tv_threshold, verbose, nparams_setter=20_000, diag_target=False, range_std=(1, 2)):
    mean_init, cov_init = np_random_mean_and_cov(dim, seed=seed_init)
    mean_target, cov_target = np_random_mean_and_cov(dim, seed=seed_target,
                                                     range_std=range_std)
    if diag_target:
        cov_target = jnp.diag(jnp.diag(cov_target))
    np.random.seed()
    key = new_jax_key()

    def logpdf_target(p):
        assert p.shape == (dim, )
        return jmvn.logpdf(p, mean_target, cov_target)
    mcmc = rw_mcmc(logpdf_target)
    key, subkey = random.split(key)
    x_init = random.multivariate_normal(subkey, mean_init, cov_init,
                                        shape=(nparticles, ))

    key, subkey = random.split(key)
    _test_RW_invariance_helper(subkey, mean_target, cov_target, x_init, mcmc, niter, acc_rate_range, tv_threshold, verbose, nparams_setter)

def _test_RW_invariance_helper(key, mean_target, cov_target, x_init, mcmc, niter, acc_rate_range, tv_threshold, verbose, nparams_setter):
    x = batch_extend(x_init, mcmc)

    target_scipy = smvn(mean=mean_target, cov=cov_target)
    params_setter = target_scipy.rvs(size=nparams_setter)
    params_setter = batch_extend(params_setter, mcmc)
    key, subkey = random.split(key)
    params = mcmc.adapt_params(subkey, params_setter)
    if verbose and params.chol_failed:
        print('Warning: cholesky failed')
    del params_setter

    mover = jit(partial(batch_move, params=params, mcmc=mcmc))

    for _ in trange(niter, disable=not verbose):
        key, subkey = random.split(key)
        x = mover(subkey, x)

    if verbose:
        multivariate_sample_tester(x_init, target_scipy, True, 1.0, True, 'init sample')

    multivariate_sample_tester(x.position, target_scipy, True, tv_threshold, verbose, 'end sample')
    acc_rate = jnp.mean(x.accepted)
    if verbose: print('acc_rate', acc_rate)
    assert acc_rate_range[0] <= acc_rate <= acc_rate_range[1]

def test_RW_invariance():
    r"""Tests invariance of RWMH and the covariance matrix adaptation (aiming for
    acc. rate of 23%).
    Runtime: ~ 5 mins on CPU
    """
    _test_RW_invariance(3, 34, 45, 1000, 30_000, (0.25, 0.35), 0.02, False)
    _test_RW_invariance(30, 123, 456, 1000, 30_000, (0.19, 0.29), 0.02, False)
    _test_RW_invariance(100, 48, 294, 2000, 10_000, (0.19, 0.29), 0.02, False)
    _test_RW_invariance(100, 17, 89, 1000, 10000, (0.19, 0.29), 0.02, False, 80, True, (1, 10))

def _test_RW_adapt(seed_problem, seed_exec, d, N, target_range, max_niter, verbose):
    np.random.seed(seed_problem)
    mean, cov = np_random_mean_and_cov(d)
    def target_logpdf(x):
        assert x.shape == (d, )
        return jmvn.logpdf(x, mean, cov)
    mcmc = rw_mcmc(target_logpdf, use_bisection=True)
    np.random.seed()

    key = random.key(seed_exec)
    key, subkey = random.split(key)
    sample = random.multivariate_normal(subkey, mean=mean, cov=cov, shape=(N, ))

    key, subkey = random.split(key)
    adapt_stepsize_tester(subkey, mcmc, sample, target_range, max_niter, verbose)

def test_RW_adapt():
    _test_RW_adapt(3, 4, 64, 1000, (0.2, 0.35), 4, False)
    _test_RW_adapt(3, 4, 16, 1000, (0.02, 0.03), 10, False)
    _test_RW_adapt(3, 4, 16, 1000, (0.85, 0.86), 10, False)
