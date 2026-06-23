from mppt.jax_configs import x64, default_gpu_memory_setting; x64(); default_gpu_memory_setting()
from typing import NamedTuple
from mppt.impl.pt_mcmc_hmc import hmc_pt_mcmc, HMCParameters
from mppt.jax_configs import chex_runtime_assert
from mppt.tests.pt_test_helper import create_random_pt_problem
from jax import numpy as jnp, random, vmap
import numpy as np
from mppt.utils import new_jax_key
from mppt.ui.ui_main import parallel_tempering, pt_manual_runners, default_collectors
from mppt.impl.mcmc_hmc import fixed_cov_mv_normal
from scipy.stats import multivariate_normal as smvn
from mppt.tests.distribution_test_helper import multivariate_sample_tester

# Note: this file only tests the HMC kernel wrapper in Parallel Tempering.
# Tests of the underlying kernel are found in ``tests/test_mcmc_hmc.py``.

def test_reset_temperature():
    problem, getdist = create_random_pt_problem(10, 10)
    mean_tar, cov_tar = getdist(1.)

    hmc_pt_t0 = hmc_pt_mcmc(problem, 0.2, 2)
    hmc_pt_t1 = hmc_pt_mcmc(problem, 0.3, 2)
    x_raw = np.random.multivariate_normal(mean=mean_tar, cov=cov_tar)
    x_raw = jnp.array(x_raw)

    key = new_jax_key()
    key, subkey = random.split(key)
    x_t0 = hmc_pt_t0.init_extend(subkey, x_raw)
    x_t1 = hmc_pt_t1.init_extend(subkey, x_raw)

    _matches_all_but_veloc(x_t0, hmc_pt_t0.reset_temperature(x_t1))

def _matches_all_but_veloc(x: NamedTuple, y, verbose=False):
    for field in x._fields:
        if verbose: print(field)
        if field != 'veloc':
            chex_runtime_assert(jnp.allclose(
                getattr(x, field), getattr(y, field)
            ))

def _test_integrated_hmc(D, SEED, LOGZ, TEMPERATURES, NITER_PT, NPARTICLES, NITER_MCMC, MANUAL_PARAMS: bool, R=0.07, MALA: bool = False, verbose=False, target_rate=0.95, rate_threshold=0.01, tv_threshold=0.02, logZ_threshold=0.02, n_inner_swaps=None, n_outer_swaps=None):
    collectors = default_collectors.copy()
    del collectors[0]  # no need SMC-like log Z estimate
    del collectors[1]  # no need reverse logistic log Z estimate

    problem, true_dist_getter = create_random_pt_problem(D, SEED, LOGZ)
    if MALA:
        ENGINE = {'name': 'mala'}
    else:
        ENGINE = {'name': 'hmc', 'refresh_rate': R}
    if not MANUAL_PARAMS:
        EXPLORER = {'mode': 'adaptive', 'niter': NITER_MCMC}
    else:
        @vmap
        def get_params(t):
            _, cov = true_dist_getter(t)
            scaled_cov = fixed_cov_mv_normal(1.65/D**(1/6), cov)
            return HMCParameters(scaled_cov, R)
        EXPLORER = {'mode': 'nonadaptive', 'niter': NITER_MCMC,
                    'mcmc_params': get_params(jnp.array(TEMPERATURES))}
    key = new_jax_key()

    state, res, res_last = parallel_tempering(
        key=key, problem=problem, temperatures=TEMPERATURES, n_iter=NITER_PT,
        n_particles=NPARTICLES, explorer=EXPLORER, engine=ENGINE, verbose=verbose,
        n_inner_swaps=n_inner_swaps, n_outer_swaps=n_outer_swaps,
        runtime_check=(not verbose) and MANUAL_PARAMS
    )

    if verbose:
        print('Swap rejection rates')
        print(res_last['swap_rejection_rates'])
    mcmc_acc_rates = res_last['mcmc_acc_rates']
    if verbose:
        print('MCMC acc rates')
        print(mcmc_acc_rates)
    logZ = res['logZ_robust']
    if verbose:
        print('log Z')
        print(logZ)

    assert abs(mcmc_acc_rates[-1] - target_rate) <= rate_threshold
    assert abs(logZ[-1] - LOGZ) <= logZ_threshold

    for i, t_ in enumerate(TEMPERATURES):
        sample = state.X.position[state.Z[i]]
        assert sample.shape == (NPARTICLES, D)
        true_dist = smvn(*true_dist_getter(t_))
        multivariate_sample_tester(sample, true_dist, True, tv_threshold, verbose, f'the {i}-th distribution')

def test_feed_extend_nan_gradient_guard():
    r"""feed_extend should detect NaN gradients when ALLOW_RUNTIME_CHECK is True.
    Without the guard, NaN gradients propagate silently through leapfrog.
    """
    from mppt.jax_configs import allow_runtime_check
    from mppt.abstract.pt_concept import PTProblem
    from mppt.utils import should_raise

    # Target with log(x[0]) produces NaN gradient at x[0]=0
    def bad_logpdf(x):
        return jnp.log(x[0])  # grad is 1/x[0] -> inf at x=0 -> NaN in anneal

    problem = PTProblem(
        rvs_ref=lambda key: random.normal(key, (2,)),
        logpdf_ref=lambda x: -0.5 * jnp.sum(x**2),
        logpdf_final_target=bad_logpdf
    )
    engine = hmc_pt_mcmc(problem, 0.5, 1)
    x_at_zero = jnp.zeros(2)
    key = new_jax_key()

    # With runtime checks ON, should raise due to NaN gradient
    with allow_runtime_check(True):
        with should_raise(AssertionError):
            engine.init_extend(key, x_at_zero)

def test_pt_manual_runners_basic():
    r"""Smoke test for pt_manual_runners: build a runner, run two steps,
    check output structure. First step initializes from scratch via
    cold_init (compiled initializer_nothing), second step runs from state.
    """
    D, SEED, N = 3, 0, 500
    problem, _ = create_random_pt_problem(D, SEED)
    temperatures = [0., 0.5, 1.]

    runner = pt_manual_runners(
        problem, temperatures, n_particles=N,
        engine={'name': 'random_walk', 'maxniter': 10},
        explorer={'mode': 'adaptive', 'niter': 5},
    )

    key = new_jax_key()
    # First step: initialize from prior + one PT iteration
    key, subkey = random.split(key)
    from mppt.abstract.pt_exec import pt_initialize
    from mppt.impl.pt_mcmc_random_walk import rw_pt_mcmc
    init_state = pt_initialize(subkey, problem, jnp.array(temperatures), N, rw_pt_mcmc)

    key, subkey = random.split(key)
    state, diags = runner(subkey, init_state, full_output=True)

    # State should be a valid PTState
    assert state.Z.shape == (3, N)
    # Diagnostics should contain expected keys
    assert 'logZ_robust' in diags
    assert 'mcmc_acc_rates' in diags
    assert 'swap_rejection_rates' in diags

def test_integrated_hmc():
    _test_integrated_hmc(5, 33, 12, [0, 0.25, 0.5, 0.75, 0.875, 1], 30, 30_000, 13, False, verbose=False, target_rate=0.95, rate_threshold=0.05)  # if R were close to zero this would fail

    _test_integrated_hmc(5, 33, 12, [0, 0.25, 0.5, 0.75, 0.875, 1], 800, 30_000, 1, False, verbose=False,target_rate=0.95, rate_threshold=0.05)  # without either n_iter_mcmc or swaps, this would fail

    # This last experiment should only be run on a server
    _test_integrated_hmc(32, 32, 32, [0, 0.03125, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9375, 0.9675, 1], 600, 30_000, 1, True, verbose=False, target_rate=0.58, rate_threshold=0.05, logZ_threshold=0.1)