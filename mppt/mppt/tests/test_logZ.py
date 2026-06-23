from mppt.jax_configs import x64, no_async, debug_nan, _set_allow_runtime_check, default_gpu_memory_setting, allow_runtime_check; _set_allow_runtime_check(True), x64(), no_async(), debug_nan(), default_gpu_memory_setting()
import pytest
from mppt.abstract.mcmc import batch_extend
from mppt.abstract.pt_concept import LogZ_Estimator, PTState
from jax import jit, vmap, random, numpy as jnp
from mppt.abstract.pt_exec import default_Z, estimate_logZ
from mppt.impl.logZ_estimators import SMC_like, robust, reverse_logistic
from mppt.impl.pt_mcmc_random_walk import rw_pt_mcmc
from mppt.jax_configs import chexify_no_async
from mppt.tests.pt_test_helper import create_pt_problem_1d
from mppt.utils import new_jax_key
import numpy as np
from typing import Callable

def _test_logZ_helper(mean0, sigma0, mean1, sigma1, temperatures, N, nrepeats, estimator: LogZ_Estimator, iq_threshold, verbose, logZ_true=1234, runtime_check=True):
    temperatures = np.array(temperatures)
    problem, getter = create_pt_problem_1d(mean0, sigma0, mean1, sigma1, logZ_true)
    dists = [getter(t) for t in temperatures]
    m_all = jnp.array([e[0] for e in dists])
    sigma_all = jnp.array([e[1] for e in dists])

    @jit
    @vmap
    def get_logZ(k):
        k, sk = random.split(k)
        X = random.normal(sk, (len(temperatures), N)) * sigma_all[:, None] + m_all[:, None]
        X = jnp.concatenate(jnp.array(X))
        assert X.shape == (len(temperatures) * N, )
        dummy_mcmc = rw_pt_mcmc(problem, 0, 0)
        X = batch_extend(X, dummy_mcmc)
        Z = default_Z(len(temperatures), N)
        state = PTState(X, Z).validate_shape(temperatures)
        k, sk = random.split(k)
        return estimate_logZ(sk, state, temperatures, estimator)

    if runtime_check:
        get_logZ = chexify_no_async(get_logZ)

    if verbose:
        print('========================')
        print('Generating different logZ estimates...')
    key = new_jax_key()
    subkeys = random.split(key, nrepeats)
    logZs = get_logZ(subkeys)
    if verbose: print('Done.')

    q0, q1 = jnp.quantile(logZs, jnp.array([0.25, 0.75]))
    if verbose:
        print('trueZ', logZ_true)
        print('quantiles', q0, q1)
    assert q0 >= logZ_true - iq_threshold
    assert q1 <= logZ_true + iq_threshold

def _test_logZ_standard_set(mean0=3.0, sigma0=2.0, mean1=3.9, sigma1=0.7, est: Callable = SMC_like, threshold=0.02, verbose=False, runtime_check=True):
    _test_logZ_helper(mean0, sigma0, mean1, sigma1, [0., 0.5, 1.], 10000, 50, est, threshold, verbose, runtime_check=runtime_check)
    _test_logZ_helper(mean0, sigma0, mean1, sigma1, np.linspace(0, 1, 200), 100, 50, est, threshold, verbose, runtime_check=runtime_check)
    random_temperatures = sorted([0] + np.random.rand(10).tolist() + [1])
    _test_logZ_helper(mean0, sigma0, mean1, sigma1, random_temperatures, 10000, 50, est, threshold, verbose, runtime_check=runtime_check)
    _test_logZ_helper(mean0, sigma0, mean1, sigma1, [0., 1.], 20000, 50, est, threshold, verbose, runtime_check=runtime_check)

def test_logZ_SMC_like():
    _test_logZ_standard_set()

def test_logZ_robust():
    _test_logZ_standard_set(est=robust)
    _test_logZ_standard_set(3.9, 0.7, 3, 2, est=robust)

def test_robust_vs_smc_like():
    r"""Verify that robust() and SMC_like() agree on the same data, and that
    both recover the known logZ across different distribution parameters.
    Uses a dense temperature grid (T=20) to reduce discretization bias.
    """
    scenarios = [
        # (mean0, sigma0, mean1, sigma1, label)
        (0.0, 1.0, 5.0, 1.0, 'same variance, different means'),
        (0.0, 1.0, 0.0, 0.3, 'same mean, different variance'),
        (0.0, 2.0, 10.0, 0.5, 'different mean and variance'),
        (-3.0, 0.5, 3.0, 0.5, 'symmetric, far apart'),
    ]
    logZ_true = 0.0  # both ref and target are normalized; estimator should return 0
    N = 10000
    nrepeats = 50
    temperatures = jnp.array(np.linspace(0, 1, 20))  # dense grid to reduce bias
    T = len(temperatures)

    for mean0, sigma0, mean1, sigma1, label in scenarios:
        problem, getter = create_pt_problem_1d(mean0, sigma0, mean1, sigma1, logZ_true)
        dists = [getter(float(t)) for t in temperatures]
        m_all = jnp.array([e[0] for e in dists])
        sigma_all = jnp.array([e[1] for e in dists])

        @jit
        @vmap
        def get_both(k):
            k, sk = random.split(k)
            X = random.normal(sk, (T, N)) * sigma_all[:, None] + m_all[:, None]
            X = jnp.concatenate(jnp.array(X))
            dummy_mcmc = rw_pt_mcmc(problem, 0, 0)
            X = batch_extend(X, dummy_mcmc)
            Z = default_Z(T, N)
            state = PTState(X, Z).validate_shape(temperatures)
            k, sk1 = random.split(k)
            k, sk2 = random.split(k)
            logZ_smc = estimate_logZ(sk1, state, temperatures, SMC_like)
            logZ_rob = estimate_logZ(sk2, state, temperatures, robust)
            return logZ_smc, logZ_rob

        get_both_c = chexify_no_async(get_both)
        key = new_jax_key()
        subkeys = random.split(key, nrepeats)
        logZs_smc, logZs_rob = get_both_c(subkeys)

        median_smc = float(jnp.median(logZs_smc))
        median_rob = float(jnp.median(logZs_rob))

        # Both should recover true logZ within tolerance
        assert abs(median_smc - logZ_true) < 0.05, \
            f'SMC_like failed for {label}: median={median_smc:.4f}, true={logZ_true}'
        assert abs(median_rob - logZ_true) < 0.05, \
            f'robust failed for {label}: median={median_rob:.4f}, true={logZ_true}'

        # They should agree with each other
        assert abs(median_smc - median_rob) < 0.05, \
            f'SMC_like vs robust disagree for {label}: {median_smc:.4f} vs {median_rob:.4f}'

def test_ess_ratio_edge_cases():
    r"""ess_ratio must return finite values even with degenerate weights."""
    from mppt.abstract.smc import ess_ratio

    # All-equal weights: ESS ratio should be 1.0
    lw_equal = jnp.zeros(100)
    assert jnp.isclose(ess_ratio(lw_equal), 1.0)

    # One dominant weight: ESS ratio should be near 1/N
    lw_skewed = jnp.array([-1000.0] * 99 + [0.0])
    result = ess_ratio(lw_skewed)
    assert jnp.isfinite(result)
    assert result < 0.05  # close to 1/100

    # All -inf weights: denom=0, must not return inf
    lw_all_neg_inf = jnp.full(100, -jnp.inf)
    result = ess_ratio(lw_all_neg_inf)
    assert jnp.isfinite(result), f'ess_ratio returned {result} for all-inf weights'

@pytest.mark.slow
def test_logZ_reverse_logistics():
    with allow_runtime_check(False):
        _test_logZ_standard_set(est=reverse_logistic, verbose=False, runtime_check=False)
        _test_logZ_standard_set(3.9, 0.7, 3, 2, est=reverse_logistic, verbose=False, runtime_check=False)