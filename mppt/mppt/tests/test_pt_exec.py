from mppt.jax_configs import no_async, numpy_raise_errors, x64, default_gpu_memory_setting; x64(); default_gpu_memory_setting(); no_async(); numpy_raise_errors()
from collections import namedtuple
import chex
from mppt.abstract.pt_concept import PTState
import numpy as np
from mppt.abstract.pt_exec import make_contiguous, reset_temperature, _separate_into_halves, _merge_halves, \
    pt_initialize, pt_explore, pt_explore_nonadaptive
from mppt.utils import inverse_permutation, pytree_index, should_raise, new_jax_key, assert_no_nan, jit_with_info, \
    random_split, tree_stack
from jax import numpy as jnp, vmap, random, Array, jit
from mppt.tests.pt_test_helper import create_random_pt_problem
from mppt.impl.pt_mcmc_random_walk import rw_pt_mcmc
from typing import Union
from mppt.jax_configs import allow_runtime_check, chexify_no_async
from functools import partial
from tqdm import trange
from scipy.stats import multivariate_normal as smvn
from mppt.tests.distribution_test_helper import multivariate_sample_tester
from mppt.impl.mcmc_random_walk import fixed_cov_mv_normal

def _permute_state(state: PTState) -> PTState:
    T, N = state.Z.shape
    perm = jnp.array(np.random.permutation(T * N))
    inv = inverse_permutation(perm)
    return PTState(pytree_index(state.X, inv), perm[state.Z]).validate_shape()

def _test_make_contiguous(T, N, trivialZ: bool):
    cls = namedtuple('cls', ['f1', 'f2'])
    X = cls(jnp.array(np.random.rand(T*N)), jnp.array(np.random.rand(T*N, 2, 3)))
    Z = np.arange(T * N)
    if not trivialZ: Z = np.random.permutation(Z)
    Z = Z.reshape((T, N))

    init_state = PTState(X, jnp.array(Z)).validate_shape()
    permuted_state = _permute_state(init_state)
    recovered_init = make_contiguous(init_state)
    recovered_permuted = make_contiguous(permuted_state)

    chex.assert_trees_all_equal(recovered_init, recovered_permuted)
    if trivialZ:
        chex.assert_trees_all_equal(recovered_init, init_state)

    with should_raise(AssertionError):
        chex.assert_trees_all_equal(permuted_state, init_state)
    if not trivialZ:
        with should_raise(AssertionError):
            chex.assert_trees_all_equal(init_state, recovered_init)
        with should_raise(AssertionError):
            chex.assert_trees_all_equal(permuted_state, recovered_init)

def test_make_contiguous():
    _test_make_contiguous(12, 15, True)
    _test_make_contiguous(12, 15, False)

def _test_reset_temperature(d, T, N, seed, verbose):
    pt_problem, getter = create_random_pt_problem(d, seed)
    temperature_arr = np.random.rand(T)
    temperature_arr[0] = 0
    temperature_arr = jnp.array(temperature_arr)
    x_init = jnp.array(np.random.multivariate_normal(*getter(0), size=T*N))
    x_init_2D = np.array(x_init).reshape((T, N, d))

    @jit
    @vmap
    def aug_rand_t(k, e):
        idx: Union[int, Array] = random.randint(k, shape=(), minval=0, maxval=T)
        return rw_pt_mcmc(pt_problem, pytree_index(temperature_arr, idx), idx).extend(e)

    key = new_jax_key()
    subkeys = random.split(key, T*N)
    x_init_aug = aug_rand_t(subkeys, x_init)
    if verbose: print(x_init_aug.temperature_idx, '\n')
    test_state = PTState(x_init_aug, jnp.arange(T*N).reshape((T, N))).validate_shape()
    test_state = _permute_state(test_state)
    # noinspection PyTypeChecker
    res = reset_temperature(test_state, pt_problem,
                            temperature_arr, rw_pt_mcmc).validate_shape()
    if verbose: print('res\n', res)

    assert jnp.array_equal(res.Z, jnp.arange(T*N).reshape((T, N)))
    for t in range(T):
        for n in range(N):
            coded_idx = t * N + n
            assert jnp.allclose(x_init_2D[t,n], res.X.position[coded_idx])

            true_extended = rw_pt_mcmc(
                pt_problem, pytree_index(temperature_arr, t), t).extend(
                jnp.array(x_init_2D[t,n]))
            chex.assert_trees_all_close(pytree_index(res.X, coded_idx), true_extended)
            assert_no_nan(true_extended)

def test_reset_temperature():
    _test_reset_temperature(2, 10, 30, 12, False)

def _test_separator(T, N, d):
    x = {'one': np.random.rand(T*N, d, d+1), 'two': np.random.rand(T*N)}
    key = new_jax_key()
    y, z, perm = _separate_into_halves(key, x, T, N)
    chex.assert_tree_shape_prefix(y, (T, (N - N % 2)//2))
    chex.assert_tree_shape_prefix(z, (T, (N + N % 2)//2))
    xr = _merge_halves(y, z, perm)
    chex.assert_trees_all_close(xr, x)
    assert_no_nan(xr)

def test_separator():
    _test_separator(32, 18, 4)
    _test_separator(33, 19, 4)

def _test_pt_explore_big(seed, d, N, T, nrepeats, max_niter, TV_threshold, acc_rate_thresholds, niter_thresholds, runtime_check, verbose, print_while_loops):
    with allow_runtime_check(runtime_check):
        pt_problem, int_dist_getter = create_random_pt_problem(d, seed)
        temperatures = jnp.array([0.5 * s/(T-1) for s in range(T-1)] + [1])
        mcmc = partial(rw_pt_mcmc, nsteps_max=max_niter)
        key = new_jax_key()
        key, subkey = random.split(key)
        initializer = pt_initialize
        if runtime_check:
            initializer = chexify_no_async(initializer)
        if verbose: print('Initializing...')
        # noinspection PyTypeChecker
        state = initializer(subkey, pt_problem, temperatures, N, mcmc, True)
        if verbose: print('Done.')
        jitted_explore = jit(partial(
            pt_explore,
            problem=pt_problem, temperatures=temperatures, mcmc_engine=mcmc
        ))
        if runtime_check:
            jitted_explore = chexify_no_async(jitted_explore)
        elif verbose:
            jitted_explore = jit_with_info(jitted_explore, True)
        while_loop = None
        for i in trange(nrepeats, disable=(not verbose) or print_while_loops):
            key, subkey = random.split(key)
            state, while_loop, _ = jitted_explore(subkey, state)
            if print_while_loops:
                print('==============')
                print('iter', i)
                print('first while loop niters+done')
                print(while_loop[0][0])
                print(while_loop[0][1])
                print('second while loop niters+done')
                print(while_loop[1][0])
                print(while_loop[1][1])

        assert niter_thresholds[0] <= jnp.max(while_loop[0][0]) <= niter_thresholds[1]
        assert niter_thresholds[0] <= jnp.max(while_loop[1][0]) <= niter_thresholds[1]
        for i, t in enumerate(temperatures):
            _test_sample_fit(state, int_dist_getter, i, t, TV_threshold,
                             acc_rate_thresholds, verbose)

def _test_sample_fit(state, int_dist_getter, i, t, TV_threshold, acc_rate_thresholds, verbose):
    sample = state.X.position[state.Z[i]]
    mean, cov = int_dist_getter(t)
    true_dist = smvn(mean=mean, cov=cov, allow_singular=False)
    multivariate_sample_tester(sample, true_dist, True, TV_threshold, verbose, f'temperature {i}')

    acc_rate = jnp.mean(state.X.accepted[state.Z[i]])
    if verbose: print(f'acc_rate at temperature {i}', acc_rate)
    if i > 0:
        assert acc_rate_thresholds[0] <= acc_rate <= acc_rate_thresholds[1]

def test_pt_explore_big():
    r"""Tests that the PT Exploration mechanism with a large number of particles
    converges to the right distribution, has the right acceptance rate and the
    right number of MCMC steps.

    This test should be run on the server, not on personal PC
    """
    _test_pt_explore_big(123, 10, 100, 15, 5, 100, 1., (0.0, 1.0), (0, 100), True, False, False)
    chex.block_until_chexify_assertions_complete()
    _test_pt_explore_big(123, 10, 20_000, 20, 10, 40, 0.02, (0.2, 0.3), (21, 25), False, False, False)

def _test_pt_explore_small(seed, d, N, T, nrepeats, niter, TV_threshold, verbose):
    pt_problem, getter = create_random_pt_problem(d, seed)
    temperatures = jnp.array([0.5 * s / (T - 1) for s in range(T - 1)] + [1])
    mcmc = partial(rw_pt_mcmc, nsteps_max=1)

    key = new_jax_key()
    initializer = partial(pt_initialize, problem=pt_problem, temperatures=temperatures,
                          mcmc_engine=mcmc, nparticles=N)
    initializer = vmap(initializer)
    initializer = jit_with_info(initializer) if verbose else jit(initializer)
    key, subkeys = random_split(key, nrepeats+1)
    state = initializer(subkeys)

    runner = partial(pt_explore, problem=pt_problem, temperatures=temperatures,
                     mcmc_engine=mcmc)
    runner = vmap(runner)
    runner = jit_with_info(runner) if verbose else jit(runner)

    for i in range(niter):
        if verbose: print(f'iter {i}')
        key, subkeys = random_split(key, nrepeats+1)
        state, _, __ = runner(subkeys, state)

    assert state.X.accepted.shape == (nrepeats, T*N)
    acc_rates = state.X.accepted.reshape((nrepeats, T, N))
    if verbose: print('acc_rates', jnp.mean(acc_rates, axis=(0, 2)))

    assert state.X.position.shape == (nrepeats, T*N, d)
    all_samples = state.X.position.reshape((nrepeats, T, N, d))
    for i, t in enumerate(temperatures):
        sample = all_samples[:, i, ...].reshape((nrepeats * N, d))
        mean, cov = getter(t)
        dist = smvn(mean=mean, cov=cov, allow_singular=False)
        if verbose: print(f'Temperature {i}')
        multivariate_sample_tester(sample, dist, True, TV_threshold, verbose, title=f'temperature {i}')

def test_pt_explore_small():
    r"""Checks that the adaptation process in ``pt_explore`` indeed leaves
    the target invariant -- in particular, parameters of an MCMC kernel acting
    on a half of particles should only depend on the other half.
    """
    _test_pt_explore_small(999, 4, 8, 6, 2000, 100, 0.02, False)

def _test_pt_explore_nonadaptive(seed, d, N, temperatures, niter, TV_threshold, acc_range, runtime_check, verbose):
    temperatures = jnp.array(temperatures)
    pt_problem, getter = create_random_pt_problem(d, seed)
    covs = [getter(t)[1] for t in temperatures]
    params = [fixed_cov_mv_normal(2.38 / jnp.sqrt(d), cov) for cov in covs]
    params = tree_stack(*params)

    initializer = pt_initialize
    explorer = partial(pt_explore_nonadaptive, problem=pt_problem, temperatures=temperatures,
                       mcmc_params=params, niter=niter, mcmc_engine=rw_pt_mcmc)
    explorer = jit(explorer)
    if runtime_check:
        initializer = chexify_no_async(initializer)
        explorer = chexify_no_async(explorer)
    elif verbose:
        explorer = jit_with_info(explorer, True)

    with allow_runtime_check(runtime_check):
        key = new_jax_key()
        key, subkey = random.split(key)
        # noinspection PyTypeChecker
        state = initializer(subkey, pt_problem, temperatures, N, rw_pt_mcmc)
        if verbose:
            print('Starting...')
            for i, t in enumerate(temperatures):
                _test_sample_fit(state, getter, i, t, 1.0, (0., 2.), True)
        key, subkey = random.split(key)
        # noinspection PyTypeChecker
        new_state = explorer(subkey, state)
        print('Done.')
        for i, t in enumerate(temperatures):
            _test_sample_fit(new_state, getter, i, t, TV_threshold,
                             acc_range, verbose)

def test_pt_explore_nonadaptive():
    _test_pt_explore_nonadaptive(9, 16, 300, [0., 0.25, 0.5, 0.75, 1],
                                 30, 1.1, (-0.2, 3.3), True, False)
    _test_pt_explore_nonadaptive(9, 16, 30_000, [0., 0.25, 0.5, 0.75, 1],
                                 300, 0.02, (0.2, 0.3), False, False)

def test_pt_initialize_rejects_bad_temperatures():
    r"""pt_initialize must reject temperatures that don't start at 0 or end at 1."""
    d, N = 2, 10
    pt_problem, _ = create_random_pt_problem(d, seed=0)

    # Does not start at 0
    with should_raise(ValueError):
        pt_initialize(new_jax_key(), pt_problem, jnp.array([0.1, 0.5, 1.0]), N, rw_pt_mcmc)

    # Does not end at 1
    with should_raise(ValueError):
        pt_initialize(new_jax_key(), pt_problem, jnp.array([0., 0.5, 0.9]), N, rw_pt_mcmc)

    # Valid temperatures should work
    pt_initialize(new_jax_key(), pt_problem, jnp.array([0., 0.5, 1.0]), N, rw_pt_mcmc)

def test_pt_explore_rejects_n1():
    r"""pt_explore must reject N<2 since the two-half split produces an empty
    first half (0 particles), which breaks adaptation.
    """
    d, T, N = 2, 3, 1
    pt_problem, _ = create_random_pt_problem(d, seed=0)
    temperatures = jnp.array([0., 0.5, 1.])
    state = pt_initialize(new_jax_key(), pt_problem, temperatures, N, rw_pt_mcmc)
    with should_raise(ValueError):
        pt_explore(new_jax_key(), state, pt_problem, temperatures, rw_pt_mcmc)