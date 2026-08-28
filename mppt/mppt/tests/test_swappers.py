from mppt.jax_configs import x64, debug_nan, no_async; x64(), debug_nan(), no_async()
from functools import partial
import numpy as np
import pytest
from jax import numpy as jnp, random, jit
from tqdm import trange
from mppt.impl.swappers import _aux_func, basic_swapper
from mppt.jax_configs import allow_runtime_check, chexify_no_async
from scipy.stats import norm
from mppt.tests.distribution_test_helper import TV_continuous_samples, plot_continuous_samples
from mppt.utils import new_jax_key, jit_with_info
from mppt.abstract.pt_concept import repeated_swap, PTState
from mppt.tests.pt_test_helper import create_pt_problem_1d, rrate_two_normals
from mppt.abstract.pt_exec import pt_swap
from mppt.impl.pt_mcmc_random_walk import rw_pt_mcmc
from mppt.abstract.mcmc import batch_extend

def _test__aux_func(x):
    res1 = jnp.abs(1 - jnp.exp(x)) / (1 + jnp.exp(x))
    res2 = _aux_func(x)
    assert jnp.allclose(res1, res2)

def test__aux_func():
    _test__aux_func(0.4)
    _test__aux_func(1.3)
    _test__aux_func(-1.3)
    _test__aux_func(-0.4)

def test_basic_swapper_synthetic_accept_reject():
    log_pi0 = jnp.array([0., 0.])
    perm = jnp.array([0, 1])

    accept = basic_swapper(
        random.key(0), log_pi0, jnp.array([1e6, 0.]), perm,
        debug=True, mix=False
    )
    assert jnp.array_equal(accept.perm, jnp.array([1, 0]))
    assert jnp.array_equal(accept.rejection_rate_naive_A, jnp.array(0.))

    reject = basic_swapper(
        random.key(0), log_pi0, jnp.array([-1e6, 0.]), perm,
        debug=True, mix=False
    )
    assert jnp.array_equal(reject.perm, perm)
    assert jnp.array_equal(reject.rejection_rate_naive_A, jnp.array(1.))

def _test_basic_swapper(mean0, sigma0, mean1, sigma1, N, niter, TV_threshold, TV_threshold_discrete, runtime_check, verbose):
    r"""Generates N particles from N(mean0, sigma0) and N particles from
    N(mean1, sigma1), mixes them randomly, then runs the swapper which after
    ``niter`` iterations is expected to recover a reasonable arrangement.
    """
    dist0 = norm(loc=mean0, scale=sigma0)
    dist1 = norm(loc=mean1, scale=sigma1)
    x = np.concatenate([dist0.rvs(size=N), dist1.rvs(size=N)])
    x = np.random.permutation(x)
    perm = jnp.array(np.random.permutation(2 * N))
    old_perm = perm  # only to test repeated_swap
    if verbose:
        _test_swap_sample_fit(x, perm, dist0, dist1, 1.0, True, 'before')
    log_pi0 = jnp.array(dist0.logpdf(x)) + 10
    log_pi1 = jnp.array(dist1.logpdf(x)) - 8
    obj = None
    swap_worker = partial(basic_swapper, debug=True)
    swap_worker = jit(swap_worker)
    if runtime_check:
        swap_worker = chexify_no_async(swap_worker)
    elif verbose:
        swap_worker = jit_with_info(swap_worker, True)

    with allow_runtime_check(runtime_check):
        key = new_jax_key()
        old_key = key  # only to test repeated_swap
        for _ in trange(niter, disable=not verbose):
            key, subkey = random.split(key)
            # obj = basic_swapper(subkey, log_pi0, log_pi1, perm, True)
            obj = swap_worker(subkey, log_pi0, log_pi1, perm)
            perm = obj.perm

        swap_repeater = repeated_swap
        if runtime_check: swap_repeater = chexify_no_async(swap_repeater)
        alt_obj = swap_repeater(old_key, log_pi0, log_pi1,
                                old_perm, niter, basic_swapper)
        assert jnp.array_equal(alt_obj.perm, obj.perm)
        assert jnp.allclose(alt_obj.rejection_rate, obj.rejection_rate)

    _test_swap_sample_fit(x, perm, dist0, dist1, TV_threshold, verbose, 'after')
    res = [obj.rejection_rate, obj.rejection_rate_naive_A, obj.rejection_rate_naive_B]
    if verbose:
        print('Rejection rates')
        print(res)
    assert (np.max(res) - np.min(res)) <= TV_threshold_discrete

def _test_swap_sample_fit(x, perm, dist0, dist1, TV_threshold, verbose, title):
    N = perm.shape[0] // 2
    x0 = x[perm[0:N]]; x1 = x[perm[N:2 * N]]
    tv0 = TV_continuous_samples(x0, dist0, True)
    tv1 = TV_continuous_samples(x1, dist1, True)
    if verbose:
        print('==========')
        print(title)
        print('tv0', tv0)
        print('tv1', tv1)
        fig = plot_continuous_samples(x0, dist0, True)
        fig.axes[0].set_title(title + ' 0')
        fig.show()
        fig2 = plot_continuous_samples(x1, dist1, True)
        fig2.axes[0].set_title(title + ' 1')
        fig2.show()
    assert tv0 <= TV_threshold
    assert tv1 <= TV_threshold

def test_basic_swapper_runtime_check_rejects_invalid_perm():
    r"""Runtime checks should reject duplicate/lost particle ids.

    Distribution recovery belongs in the slow suite; this unit test targets the
    actual safety contract used by PT swaps.
    """
    log_pi0 = jnp.zeros(4)
    log_pi1 = jnp.zeros(4)
    duplicate_perm = jnp.array([0, 1, 1, 3])

    with allow_runtime_check(True):
        with pytest.raises(AssertionError):
            basic_swapper(
                random.key(0), log_pi0, log_pi1, duplicate_perm,
                debug=True, mix=False
            )

@pytest.mark.slow
def test_basic_swapper_distribution_recovery():
    _test_basic_swapper(3.4, 1.5, 5., 0.75, 20_000, 50, 0.02, 0.01, False, False)


def _validate_shape(pt_state, temperatures=None):
    return PTState.validate_shape(pt_state, temperatures)

def _test_pt_swap(mean0, sigma0, mean1, sigma1, temperatures, N, n_outer_swaps, n_inner_swaps, TV_threshold_discrete, TV_threshold_continuous, runtime_check, verbose):
    r"""Tests the swap step of PT by generating particles from T distributions
    interpolating between N(mean0, sigma0**2) and N(mean1, sigma1**2), then
    scattering them over T levels in a random way. The swaps should be able to
    recover a reasonable arrangement.
    """
    temperatures = jnp.array(temperatures)
    pt_problem, getter = create_pt_problem_1d(mean0, sigma0, mean1, sigma1)
    T = len(temperatures)
    validator = _validate_shape
    if runtime_check: validator = chexify_no_async(validator)

    theo_rej = _theoretical_rej_rates(getter, temperatures)
    if verbose: print('Theoretical rejection rates\n', theo_rej)

    if verbose: print('Generating data...')
    X = []
    for t in temperatures:
        loc_and_scale = getter(t)._asdict()
        X.append(np.random.normal(**loc_and_scale, size=N))
    # noinspection PyTypeChecker
    X = jnp.array(np.random.permutation(np.concatenate(X)))
    # noinspection PyTypeChecker
    dummy_mcmc = rw_pt_mcmc(pt_problem, jnp.array(np.random.rand()), 1)
    X = batch_extend(X, dummy_mcmc)
    Z = np.random.permutation(T * N).reshape((T, N))
    state = validator(PTState(X, jnp.array(Z)))
    if verbose:
        print('Done.')
        _test_state_fit(state, getter, temperatures, 1.1, 'before')

    swap_worker = partial(pt_swap, n_outer_swaps=n_outer_swaps, n_inner_swaps=n_inner_swaps,
                          inner_swapper=basic_swapper)
    swap_worker = jit(swap_worker)
    if runtime_check:
        swap_worker = chexify_no_async(swap_worker)
    elif verbose:
        swap_worker = jit_with_info(swap_worker, True)
    new_state, rej = swap_worker(new_jax_key(), state, temperatures)

    _test_state_fit(new_state, getter, temperatures, TV_threshold_continuous, 'after' if verbose else False)
    if verbose: print('Obtained rejection rates\n', rej)
    assert np.max(np.abs(rej - theo_rej)) <= TV_threshold_discrete

def _test_state_fit(state, getter, temperatures, TV_threshold, verbose):
    if verbose:
        print('Test', verbose)
        print('=================')
    for i, t in enumerate(temperatures):
        loc_and_scale = getter(t)._asdict()
        sample = state.X.position[state.Z[i]]
        assert sample.shape == (state.N, )
        dist = norm(**loc_and_scale)
        tv = TV_continuous_samples(sample, dist, True)
        if verbose:
            print(f'tv {verbose} temp {i}', tv)
            fig = plot_continuous_samples(sample, dist, True)
            fig.axes[0].set_title(f'{verbose} temp {i}')
            fig.show()
        assert tv < TV_threshold

def _theoretical_rej_rates(getter, temperatures):
    res = []
    for i in range(len(temperatures) - 1):
        res.append(
            rrate_two_normals(new_jax_key(),
                              *getter(temperatures[i]), *getter(temperatures[i+1]),
                              10_000).est
        )
    return np.array(res)

@pytest.mark.slow
def test_pt_swap_runtime_check():
    r"""Tests the module ``abstract.pt_exec.pt_swap``
    """
    _test_pt_swap(-0.5, 1, 5.5, 2, [0., 0.5, 1.], 30, 2, 2, 1., 1., True, False)

@pytest.mark.slow
def test_pt_swap_distribution_recovery():
    r"""Statistical recovery tests for ``abstract.pt_exec.pt_swap``."""
    _test_pt_swap(-0.5, 1, 5.5, 2, [0., 0.4, 0.6, 0.8, 0.9, 1.], 20_000, 10, 3, 0.01, 0.02, False, False)
    _test_pt_swap(-0.5, 1, 5.5, 2, [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1], 20_000, 11, 3, 0.01, 0.02, False, False)
