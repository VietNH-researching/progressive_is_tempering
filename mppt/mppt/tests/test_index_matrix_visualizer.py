from mppt.jax_configs import x64, no_async, debug_nan; x64(), debug_nan(), no_async()
from mppt.impl.mcmc_random_walk import fixed_cov_mv_normal
from mppt.tests.pt_test_helper import create_random_pt_problem
from mppt.ui.collectors import Collector
from mppt.ui.explorers import SimpleExploreResult
from mppt.ui.index_matrix_visualizer import cumcompose, _plot_data, _proportion_init_temperature, _find_positions, \
    _integrize, _discretize_proportion
from mppt.ui.ui_main import default_collectors, parallel_tempering
from mppt.utils import new_jax_key, tree_stack, dummy_use, pytree_index
from jax import numpy as jnp
import numpy as np

def logpdf_ref_collector():
    def func(key, expresult: SimpleExploreResult, pt_problem, temperatures, engine):
        dummy_use(key, pt_problem, temperatures, engine)
        return expresult.new_state.X.logpdf_ref
    return Collector('logpdf_ref', False, True, func)

def _test_cumcompose(d, seed, temperatures, nparticles, niter, verbose):
    T = len(temperatures)
    N = nparticles
    temperatures = jnp.array(temperatures)
    problem, _ = create_random_pt_problem(d, seed)
    key = new_jax_key()
    dummy_params = [fixed_cov_mv_normal(1., jnp.identity(d))
                    for _ in range(len(temperatures))]
    dummy_params = tree_stack(*dummy_params)
    collectors = default_collectors.copy() + [logpdf_ref_collector()]
    res = parallel_tempering(key, problem, temperatures, niter, None, nparticles, explorer={'mode': 'nonadaptive', 'niter': 0, 'mcmc_params': dummy_params}, verbose=verbose, runtime_check=True, collectors=collectors)
    if verbose: print('reject rate\n', res[-1]['swap_rejection_rates'])
    res = res[1]
    composed_Zs = cumcompose(res['index_matrix'])
    init_matrix = res['logpdf_ref'][0]
    for mat, Z in zip(res['logpdf_ref'], composed_Zs):
        mat = mat.reshape((T, N))
        mat_prime = pytree_index(init_matrix, Z.ravel()).reshape((T, N))
        if verbose:
            print('============')
            print('logpdf_arr\n', mat)
            print('logpdf_arr_prime\n', mat_prime)
            print('cumulated Z\n', Z)
        assert jnp.allclose(mat, mat_prime)

def test_cumcompose():
    _test_cumcompose(2, 4, [0., 0.2, 0.5, 1.], 5, 10, False)
    _test_cumcompose(3, 3, jnp.linspace(0, 1, 10), 40, 20, False)

def _test__plot_data(T, N, Nd):
    # tests are included in implementation
    Z = np.random.permutation(T*N).reshape((T, N))
    tracked_particles = np.random.randint(low=0, high=T*N, size=3)
    _plot_data(Z, tracked_particles, Nd, True)

def test_plot_data():
    for _ in range(10):
        _test__plot_data(10, 20, 37)
        _test__plot_data(20, 10, 37)

def _test__proportion_init_temperature(T, N, verbose):
    Z = np.random.permutation(T * N).reshape((T, N))
    if verbose:
        print('Simplified Z')
        print(Z//N)
    R = _proportion_init_temperature(Z, check=True)  # tests are embedded in implementation
    if verbose:
        print('Response')
        print(R)

def test_proportion_init_temperature():
    for _ in range(100):
        _test__proportion_init_temperature(3, 2, False)
        _test__proportion_init_temperature(2, 3, False)
        _test__proportion_init_temperature(10, 20, False)
        _test__proportion_init_temperature(20, 10, False)

def _test__find_positions(T, N, num):
    Z = np.random.permutation(T * N).reshape((T, N))
    tracked = np.random.randint(low=0, high=T*N, size=num)
    _find_positions(tracked, Z, check=True)  # tests are embedded in implementation

def test__find_positions():
    for _ in range(100):
        _test__find_positions(3, 2, 4)
        _test__find_positions(2, 3, 4)
        _test__find_positions(10, 20, 30)
        _test__find_positions(20, 10, 30)

def _test__integrize(p, N, threshold):
    p = jnp.array(p)
    res = _integrize(p, N)
    assert jnp.sum(res) == N
    assert jnp.issubdtype(res.dtype, jnp.integer)
    prop = res/N
    assert 0.5 * jnp.sum(jnp.abs(prop - p)) < threshold

def test_integrize():
    _test__integrize([0.6, 0., 0.4], 10_000, 0.01)
    for _ in range(10):
        p = np.random.rand(6)
        p = p/p.sum()
        _test__integrize(p, 10_000, 0.01)
    for _ in range(10):
        p = np.random.rand(7)
        p = p/p.sum()
        _test__integrize(p, 10_000, 0.01)

def _test__discretize_proportion(p, N, threshold):
    p = jnp.array(p)
    res = _discretize_proportion(p, N)
    assert res.shape == (N, )
    T, = p.shape
    emp_p = jnp.sum(res == jnp.arange(T)[:, jnp.newaxis], axis=1)/N
    assert 0.5 * jnp.sum(jnp.abs(emp_p - p)) < threshold

def test__discretize_proportion():
    _test__discretize_proportion([0.6, 0., 0.4], 10_000, 0.01)
    for _ in range(10):
        p = np.random.rand(6)
        p = p/p.sum()
        _test__discretize_proportion(p, 10_000, 0.01)
    for _ in range(10):
        p = np.random.rand(7)
        p = p/p.sum()
        _test__discretize_proportion(p, 10_000, 0.01)