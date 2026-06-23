from mppt.jax_configs import x64; x64()
from mppt.tests.pt_test_helper import adapt_stepsize_tester
from collections import namedtuple
from functools import partial
from jax import random, numpy as jnp, jit
from jax.scipy.stats import multivariate_normal as jmvn
from scipy.stats import multivariate_normal as smvn
from mppt.impl.mcmc_hmc import HMC, HMCParameters
from mppt.impl.mcmc_random_walk import fixed_cov_mv_normal
from mppt.tests.distribution_test_helper import np_random_mean_and_cov, multivariate_sample_tester
from mppt.abstract.mcmc import batch_init_extend, batch_move
from mppt.utils import new_jax_key
import numpy as np
import chex
from tqdm import trange

DummyHMCProblem = namedtuple('DummyHMCProblem',
                             ['mean', 'cov', 'target_logpdf', 'hmc_obj', 'hmc_params',
                              'scipy_target_dist'])

def dummy_problem(seed, d, R, adapt_cov=True, eps=None):
    np.random.seed(seed)
    mean, cov = np_random_mean_and_cov(d)
    if adapt_cov:
        prop_cov = cov
    else:
        prop_cov = np_random_mean_and_cov(d)[1]
    if eps is None:
        eps = 1.65 / d**(1/6)
    np.random.seed()

    target_logpdf_ = partial(jmvn.logpdf, mean=jnp.array(mean), cov=jnp.array(cov))
    def target_logpdf(x):
        assert x.shape == (d, )
        return target_logpdf_(x)
    hmc_obj = HMC.simpleInstance(target_logpdf)
    cov_obj = fixed_cov_mv_normal(eps, jnp.array(prop_cov))
    params = HMCParameters(cov_obj, R)
    scipy_dist = smvn(mean=mean, cov=cov)
    return DummyHMCProblem(mean, cov, target_logpdf, hmc_obj, params, scipy_dist)

def _test_reversibility(seed, d, eps, verbose=False):
    problem = dummy_problem(seed, d, 0.66, adapt_cov=False, eps=eps)
    hmc_obj = problem.hmc_obj
    x = problem.scipy_target_dist.rvs()
    x = hmc_obj.init_extend(random.key(seed), x)
    if verbose:
        print('init position')
        print(x.position)
        print('init veloc')
        print(x.veloc)
    y = hmc_obj._hmc_jump(x, problem.hmc_params)
    z = hmc_obj._hmc_jump(y, problem.hmc_params)
    if verbose:
        print('terminal position')
        print(y.position)
        print('terminal veloc')
        print(y.veloc)
    chex.assert_trees_all_close(x, z)

def test_reversibility():
    for _ in range(30):
        _test_reversibility(np.random.randint(low=0, high=2**31), 3, np.random.rand())

def _run_dummy_pb(dp: DummyHMCProblem, N: int, niter: int,
                  acc_threshold: tuple, TV_threshold: float, verbose):
    mcmc = dp.hmc_obj
    key = new_jax_key()
    x_init = jnp.array([dp.mean] * N)
    key, subkey = random.split(key)
    x_init = batch_init_extend(subkey, x_init, mcmc)

    x = x_init
    mover = partial(batch_move, params=dp.hmc_params, mcmc=mcmc)
    mover = jit(mover)
    for _ in trange(niter, disable=not verbose):
        key, subkey = random.split(key)
        x = mover(subkey, x)

    multivariate_sample_tester(x.position, dp.scipy_target_dist, True, TV_threshold, verbose, 'target')

    acc_rate = jnp.mean(x.accepted)
    if verbose:
        print('Acc rate', acc_rate)
    assert acc_threshold[0] <= acc_rate <= acc_threshold[1]

def test_recover_langevin():
    dummy_prob = dummy_problem(82, 60, 1)
    _run_dummy_pb(dummy_prob,30_000, 50, (0.53, 0.63), 0.02, False)

def test_invariant():
    dummy_prob = dummy_problem(82, 25, 0.05, eps=0.48)
    _run_dummy_pb(dummy_prob,20_000, 100, (0.9, 1.0), 0.02, False)

    dummy_prob_skeleton = partial(dummy_problem, seed=42, d=5, adapt_cov=False)

    dummy_prob = dummy_prob_skeleton(eps=0.45, R=0.2)
    _run_dummy_pb(dummy_prob, 30_000, 50, (0.8, 0.95), 0.02, False)

    dummy_prob = dummy_prob_skeleton(eps=0.4, R=0.02)
    _run_dummy_pb(dummy_prob, 30_000, 300, (0.8, 0.95), 0.02, False)

def _test_adapt_stepsize(d, N, seed_problem, seed_exec, target_acc, niter_max, verbose):
    dummy_prob = dummy_problem(seed_problem, d, 0.5, False)
    key = random.key(seed_exec)
    key, subkey = random.split(key)
    x = random.multivariate_normal(subkey, dummy_prob.mean, dummy_prob.cov, (N, ))
    key, subkey = random.split(key)
    adapt_stepsize_tester(subkey, dummy_prob.hmc_obj, x, target_acc, niter_max, verbose)

def test_adapt_stepsize():
    _test_adapt_stepsize(20, 1000, 9, 99, (0.8, 0.95), 4, False)
    _test_adapt_stepsize(20, 1000, 9, 99, (0.2, 0.3), 10, False)