from mppt.jax_configs import x64, no_async, debug_nan; x64(), debug_nan(), no_async()
from mppt.ui.index_matrix_visualizer import cumcompose, _plot_data, _proportion_init_temperature, _find_positions, \
    _integrize, _discretize_proportion
from mppt.utils import pytree_index
from jax import numpy as jnp
import numpy as np

def _deterministic_Z(T, N, offset=1):
    # Hand-built permutations make visualization/indexing failures reproducible.
    return np.roll(np.arange(T * N), offset).reshape((T, N))

def test_cumcompose():
    Z_arr = jnp.array([
        [[1, 0], [2, 3]],
        [[0, 2], [1, 3]],
    ])

    composed_Zs = cumcompose(Z_arr)

    assert jnp.array_equal(composed_Zs[0], jnp.array([[0, 1], [2, 3]]))
    assert jnp.array_equal(composed_Zs[1], jnp.array([[1, 0], [2, 3]]))
    assert jnp.array_equal(composed_Zs[2], jnp.array([[1, 2], [0, 3]]))

    payload = jnp.array([10, 20, 30, 40])
    assert jnp.array_equal(
        pytree_index(payload, composed_Zs[-1].ravel()).reshape((2, 2)),
        jnp.array([[20, 30], [10, 40]]),
    )

def _test__plot_data(T, N, Nd):
    # tests are included in implementation
    Z = _deterministic_Z(T, N)
    tracked_particles = np.array([0, T * N // 2, T * N - 1])
    _plot_data(Z, tracked_particles, Nd, True)

def test_plot_data():
    for _ in range(2):
        _test__plot_data(10, 20, 37)
        _test__plot_data(20, 10, 37)

def _test__proportion_init_temperature(T, N, verbose):
    Z = _deterministic_Z(T, N)
    if verbose:
        print('Simplified Z')
        print(Z//N)
    R = _proportion_init_temperature(Z, check=True)  # tests are embedded in implementation
    if verbose:
        print('Response')
        print(R)

def test_proportion_init_temperature():
    _test__proportion_init_temperature(3, 2, False)
    _test__proportion_init_temperature(2, 3, False)
    _test__proportion_init_temperature(10, 20, False)
    _test__proportion_init_temperature(20, 10, False)

def _test__find_positions(T, N, num):
    Z = _deterministic_Z(T, N)
    tracked = np.linspace(0, T * N - 1, num=num, dtype=int)
    _find_positions(tracked, Z, check=True)  # tests are embedded in implementation

def test__find_positions():
    _test__find_positions(3, 2, 4)
    _test__find_positions(2, 3, 4)
    _test__find_positions(10, 20, 30)
    _test__find_positions(20, 10, 30)

def _test__integrize(p, N, threshold):
    p = jnp.array(p)
    p = p / jnp.sum(p)
    res = _integrize(p, N)
    assert jnp.sum(res) == N
    assert jnp.issubdtype(res.dtype, jnp.integer)
    prop = res/N
    assert 0.5 * jnp.sum(jnp.abs(prop - p)) < threshold

def test_integrize():
    _test__integrize([0.6, 0., 0.4], 10_000, 0.01)
    # Fixed probability vectors cover zeros, uneven mass, and several lengths
    # without relying on a lucky random draw.
    _test__integrize([1, 2, 4, 8, 16, 32], 10_000, 0.01)
    _test__integrize([0, 3, 1, 0, 5, 9, 2], 10_000, 0.01)

def _test__discretize_proportion(p, N, threshold):
    p = jnp.array(p)
    p = p / jnp.sum(p)
    res = _discretize_proportion(p, N)
    assert res.shape == (N, )
    T, = p.shape
    emp_p = jnp.sum(res == jnp.arange(T)[:, jnp.newaxis], axis=1)/N
    assert 0.5 * jnp.sum(jnp.abs(emp_p - p)) < threshold

def test__discretize_proportion():
    _test__discretize_proportion([0.6, 0., 0.4], 10_000, 0.01)
    _test__discretize_proportion([1, 2, 4, 8, 16, 32], 10_000, 0.01)
    _test__discretize_proportion([0, 3, 1, 0, 5, 9, 2], 10_000, 0.01)
