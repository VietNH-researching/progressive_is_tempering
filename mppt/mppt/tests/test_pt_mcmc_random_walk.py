from mppt.jax_configs import x64, default_gpu_memory_setting, no_async; x64(), no_async(), default_gpu_memory_setting()  # LINE MUST BE AT TOP
from mppt.impl.pt_mcmc_random_walk import rw_pt_mcmc
from mppt.tests.distribution_test_helper import np_random_mean_and_cov
from jax import random, numpy as jnp, vmap, jit, block_until_ready
from jax.scipy.stats import multivariate_normal as jmvn
import numpy as np
import pytest
from mppt.tests.test_mcmc_random_walk import _test_RW_invariance_helper
from mppt.utils import new_jax_key, assert_no_nan
import chex
from mppt.abstract.mcmc import batch_extend
from mppt.jax_configs import allow_runtime_check, chexify_no_async
from mppt.impl.mcmc_random_walk import rw_mcmc, fixed_cov_mv_normal
from functools import partial
from mppt.tests.pt_test_helper import create_random_pt_problem

def _test_rw_pt_mcmc_invariance(
        d, seed_pt, seed_test, t, nparticles,
        niter, acc_rate_range, tv_threshold, verbose, nparams_setter=20_000
):
    pt_problem, intermediate_dist_getter = create_random_pt_problem(d, seed_pt)
    mean_target, cov_target = intermediate_dist_getter(t)

    mean_init, cov_init = np_random_mean_and_cov(d, seed=seed_test)
    np.random.seed()
    x_init = np.random.multivariate_normal(mean=mean_init, cov=cov_init,
                                           size=nparticles)

    mcmc = rw_pt_mcmc(pt_problem, t, int(t != 0))

    key = new_jax_key()
    _test_RW_invariance_helper(
        key, mean_target, cov_target, x_init, mcmc,
        niter, acc_rate_range, tv_threshold, verbose, nparams_setter
    )

@pytest.mark.slow
def test_rw_pt_mcmc_invariance():
    _test_rw_pt_mcmc_invariance(20, 30, 40, 0.65, 20_000, 600, (0.19, 0.29), 0.02, False)
    _test_rw_pt_mcmc_invariance(20, 11, 22, 0, 20_000, 1, (0., 1.), 0.02, False)

def _test_rw_pt_temperature_reset(d, seed, good_t, bad_t, nsamples):
    pt_problem, intermediate_dist_getter = create_random_pt_problem(d, seed)
    _ = intermediate_dist_getter(good_t)
    # Temperature reset is a field-recomputation contract; deterministic
    # positions expose wiring bugs without turning this into a sampler test.
    sample = jnp.linspace(-1., 1., nsamples * d).reshape((nsamples, d))
    bad_mcmc = rw_pt_mcmc(pt_problem, bad_t, 1)
    bad_sample = batch_extend(sample, bad_mcmc)
    good_mcmc = rw_pt_mcmc(pt_problem, good_t, 1)
    good_sample = batch_extend(sample, good_mcmc)

    try:
        chex.assert_trees_all_close(good_sample, bad_sample)
    except AssertionError:
        pass
    else:
        raise AssertionError

    with allow_runtime_check(True):
        assert_no_nan(good_sample)
        chex.assert_trees_all_close(
            good_sample,
            chexify_no_async(vmap(good_mcmc.reset_temperature))(bad_sample)
        )

def test_rw_pt_temperature_reset():
    _test_rw_pt_temperature_reset(6, 5, 0.4, 0.3, 132)

def _test_rw_pt_nsteps_adapt(d, seed, t, nsamples, nrepeats, verbose):
    pt_problem, intermediate_dist_getter = create_random_pt_problem(d, seed)
    np.random.seed()
    mean_target, cov_target = intermediate_dist_getter(t)
    wrapped_mcmc = rw_pt_mcmc(pt_problem, t, 1)
    naked_mcmc = rw_mcmc(lambda e: jmvn.logpdf(e, mean_target, cov_target))
    params = fixed_cov_mv_normal(2.38/jnp.sqrt(d), cov_target)

    @partial(jit, static_argnums=(1, ))
    @partial(vmap, in_axes=(0, None))
    def nsteps(k, engine):
        k, sk = random.split(k)
        sample = random.multivariate_normal(sk, mean=mean_target, cov=cov_target,
                                            shape=(nsamples, ))
        sample = batch_extend(sample, engine)
        k, sk = random.split(k)
        return engine.adapt_nsteps(sk, sample, params).execute()

    key = new_jax_key()
    subkeys = random.split(key, nrepeats)
    if verbose: print('Running...')
    wrap_res = block_until_ready(nsteps(subkeys, wrapped_mcmc))
    naked_res = block_until_ready(nsteps(subkeys, naked_mcmc))
    if verbose:
        print('Done...')
        print(wrap_res)

    assert jnp.array_equal(wrap_res, naked_res)

@pytest.mark.slow
def test_rw_pt_nsteps_adapt():
    _test_rw_pt_nsteps_adapt(10, 12, 0.65, 1000, 10, False)
