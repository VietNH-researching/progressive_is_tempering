from functools import partial
from typing import Callable
from matplotlib.collections import QuadMesh, PathCollection
from matplotlib.animation import FuncAnimation
from mppt.abstract.pt_exec import default_Z
from mppt.utils import pytree_index, func_animation_parameters
from jax import numpy as jnp, Array, vmap, jit
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from inspect import signature

def cumcompose(Z_arr):
    r"""Cumulative composition of Z matrices.
    """
    T, N = Z_arr[0].shape
    res = [default_Z(T, N)]
    for Z in Z_arr:
        latest = res[-1].ravel()
        new = pytree_index(latest, Z.ravel()).reshape((T, N))
        res.append(new)
    return res

def _arrange(Z, check=False):
    r"""Arranges each row of the Z-matrix according to the initial temperature,
    then the particle index.
    """
    T, N = Z.shape
    Z_backup = Z

    Z = jnp.sort(Z, axis=1)
    arg = jnp.argsort(Z // N, axis=1)
    Z = jnp.take_along_axis(Z, arg, axis=1)

    if check:
        z1 = jnp.sort(Z_backup, axis=1)
        z2 = jnp.sort(Z, axis=1)
        assert jnp.array_equal(z1, z2)

    return Z

def _proportion_init_temperature(sorted_Z, check=False):
    r"""Calculates, for each row of Z, the proportion of particles with a given
    initial temperature.
    :param sorted_Z: Z-array of size (T, N) outputted by ``_arrange_Z``
    :return array of size (T, T)
    """
    Z = sorted_Z
    T, N = Z.shape
    Z = Z//N
    T_array = jnp.arange(T)
    res = jnp.sum(Z[..., jnp.newaxis] == T_array, axis=1)/N
    assert res.shape == (T, T)
    if check:
        assert jnp.allclose(jnp.sum(res, axis=1), 1)
    return res

def _discretize_proportion(p_vector: Array, N: int) -> Array:
    r"""Creates a vector with ``N`` elements having approximately the same
    proportion of elements as the ``p_vector``.
    :param p_vector: array of shape ``(T, )``, summing to one.
    :param N: array of shape ``(N, )`` having elements in ``[0, T[``
    """
    T, = p_vector.shape
    nrepeats = _integrize(p_vector, N)
    return jnp.repeat(jnp.arange(T), nrepeats, total_repeat_length=N)

discretize_proportion = vmap(_discretize_proportion, in_axes=(0, None))

def _integrize(p_vector: Array, N: int) -> Array:
    r"""Given a proportion vector ``p_vector`` of shape ``(T, )``, creates
    a new integer-valued vector of the same shape summing to ``N`` having
    approximately the same proportion as ``p_vector``.
    """
    x = jnp.round(jnp.cumsum(p_vector) * N)
    x = jnp.array(x, dtype=int)
    x = jnp.r_[0, x]
    return jnp.diff(x)

def _find_positions(tracked_particles, sorted_Z, check=False):
    r"""Finds the positions of tracked particles in a sorted Z-array
    :param tracked_particles: array containing elements in ``[0, T*N[``
    :param sorted_Z: Z-array of size (T, N) outputted by ``_arrange_Z``
    :return: Two arrays A and B such that (A[i], (N-1) * B[i]) is the position
    of tracked_particles[i]
    """
    T, N = sorted_Z.shape
    assert jnp.ndim(tracked_particles) == 1
    S = sorted_Z == tracked_particles[:, jnp.newaxis, jnp.newaxis]
    res_check, res_a, res_b = jnp.nonzero(S) if check\
        else jnp.nonzero(S, size=len(tracked_particles))
    if check:
        assert jnp.array_equal(res_check, jnp.arange(len(tracked_particles)))
        assert len(res_a) == len(res_b) == len(tracked_particles)
        for ra, rb, p in zip(res_a, res_b, tracked_particles):
            assert jnp.ndim(ra) == jnp.ndim(rb) == 0
            assert sorted_Z[ra, rb] == p
    return res_a, res_b/(N-1)

def _plot_data(Z, tracked_particles, N_discretize, check=False):
    r"""Returns a heat map (T x N_discretize array of elements in {0, 1, ...
    T-1}) and positions of tracked particles (two arrays referring to t and
    n/N).
    """
    Z = _arrange(Z, check)
    proportion = _proportion_init_temperature(Z, check)
    heat_map = discretize_proportion(proportion, N_discretize)
    assert heat_map.shape == (Z.shape[0], N_discretize)
    # return (_proportion_init_temperature(Z, check),
    #         _find_positions(tracked_particles, Z, check))
    return heat_map, _find_positions(tracked_particles, Z, check)

def _no_jit_if_check(func, jitter):
    jitted = jitter(func)
    func_signature = signature(func)

    def wrapped(*args, **kwargs):
        parse_result = func_signature.bind(*args, **kwargs)
        parse_result.apply_defaults()
        if parse_result.arguments['check']:
            return func(*args, **kwargs)
        else:
            return jitted(*args, **kwargs)

    return wrapped

_plot_data = _no_jit_if_check(_plot_data, partial(jit, static_argnums=(2, 3)))

def _plotter(ax: plt.Axes, heat_map, tracked_t, tracked_n, tracked_colours, animated=False):
    T, Nd = heat_map.shape
    assert jnp.ndim(tracked_t) == jnp.ndim(tracked_n) == 1
    assert len(tracked_t) == len(tracked_n) == len(tracked_colours)

    mesh = ax.pcolormesh(np.linspace(0, 1, Nd + 1), np.linspace(0, 1, T + 1), (heat_map+1)/T, cmap='gray_r', vmin=0, vmax=1, animated=animated)
    scattered = ax.scatter(tracked_n, (tracked_t+0.5)/T, s=70, c=tracked_colours, animated=animated)

    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('initial temperature')
    ax.set_ylabel('current temperature')
    ax.get_xaxis().set_visible(False)
    ax.set_aspect('equal')

    return mesh, scattered

def visualize_index_matrix(index_matrix_output, tracked_temperatures_idx, tracked_colours, Ndiscretize=300, debug=False) -> Callable:
    r"""Visualize how the particles move between temperatures using the Z-matrices.
    :param index_matrix_output: list of Z matrices after the swap steps. This can
    be obtained from the `index_matrix` key of the second element of the output
    of ``parallel_tempering``. For best results it is recommended to remove a few
    first 'burn-in' elements
    :param tracked_temperatures_idx: the visualization follows the path of certain
    particles. This argument determines their initial temperatures.
    :param tracked_colours: the colours of the tracked particles.
    :param Ndiscretize: number of pixels to discretize each row of the heatmap
    :return: a function which takes an iteration time and outputs a visualization
    of the particle system at that time
    """
    T, N = index_matrix_output[0].shape
    tracked_particles = jnp.array([int(N * (t_idx + 0.5)) for t_idx in tracked_temperatures_idx])
    Z_all = cumcompose(index_matrix_output)

    def plot(i):
        heat_map, tracked_pos = _plot_data(Z_all[i], tracked_particles, Ndiscretize, debug)
        fig, ax = plt.subplots()
        _plotter(ax, heat_map, *tracked_pos, tracked_colours)
        fig.show()

    return plot

def _animation_helper(Nparticles: int, tracked_temperatures_idx, tracked_colours, Ndiscretize=300, debug=False):
    tracked_particles = jnp.array(
        [int(Nparticles * (t_idx + 0.5)) for t_idx in tracked_temperatures_idx])

    def plot_init(ax, Z):
        heat_map, tracked_positions = _plot_data(Z, tracked_particles, Ndiscretize, debug)
        return _plotter(ax, heat_map, *tracked_positions, tracked_colours, animated=True)

    def plot_update(mesh: QuadMesh, scattered: PathCollection, Z):
        heat_map, tracked_positions = _plot_data(Z, tracked_particles, Ndiscretize, debug)
        T, Nd = heat_map.shape
        mesh.set_array((heat_map+1)/T)

        tracked_t, tracked_n = tracked_positions
        new_x = tracked_n; new_y = (tracked_t+0.5)/T
        scattered.set_offsets(jnp.c_[new_x, new_y])

        return mesh, scattered

    return plot_init, plot_update

def visualize_index_matrix_video(index_matrix_output, tracked_temperatures_idx, tracked_colours, video_len_in_sec: float = 10.0, Ndiscretize=300, fps: int = 24, debug=False) -> FuncAnimation:
    r"""Creates a video showing the evolution of the index matrix. In other words,
    a video version of ``visualize_index_matrix``.
    """
    Z_all = cumcompose(index_matrix_output)
    del index_matrix_output

    params = func_animation_parameters(len(Z_all), fps, video_len_in_sec)
    T, Nparticles = Z_all[0].shape
    plot_init, plot_update = _animation_helper(Nparticles, tracked_temperatures_idx,
                                               tracked_colours, Ndiscretize, debug)

    fig, ax = plt.subplots()
    mesh, scattered = plot_init(ax, Z_all[0])

    def animate(i):
        return plot_update(mesh, scattered, Z_all[i])

    return FuncAnimation(fig, animate, tqdm(params.frames_idx), interval=params.interval,
                         blit=True)