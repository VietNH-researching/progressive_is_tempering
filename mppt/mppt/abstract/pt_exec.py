r"""Implements the general Parallel Tempering algorithm.
"""
from __future__ import annotations
from collections import namedtuple
from functools import partial
from typing import NamedTuple, Optional
from mppt.abstract.mcmc import PTMCMCEngine, batch_reset_temperature, repeated_batch_move, batch_init_extend
from mppt.abstract.pt_concept import PTState, PTProblem, TwoLevelSwapper, repeated_swap, geometric_anneal, \
    LogZ_Estimator
from jax import numpy as jnp, Array, vmap, tree as jax_tree, random, lax
from mppt.utils import pytree_index, random_split, while_loops_executor, pytree_length, inverse_permutation, dummy_use
from jax.typing import ArrayLike

Float = ArrayLike


def default_Z(T, N):
    r"""The default Z component of a ``PTState`` object, corresponding to the
    contiguous arrangement.
    """
    return jnp.arange(T * N).reshape((T, N))

def make_contiguous(state: PTState) -> PTState:
    r"""Returns a new Parallel Tempering state with the same information but
    with contiguous arrangement.
    """
    T, N = state.Z.shape
    newZ = default_Z(T, N)
    return PTState(pytree_index(state.X, state.Z.ravel()), newZ).validate_shape()

def reset_temperature(state: PTState, problem: PTProblem, temperatures: Array,
                      mcmc_engine: PTMCMCEngine) -> PTState:
    r"""Resets the temperatures of the particles in `state.X` to match the
    prescribed temperatures in `state.Z`. The reset is necessary after swap
    operations because they may change the temperature of particles. The
    ``mcmc_engine`` argument should match the particle type in ``state``.

    Warning
    -------
    It is always assumed in this codebase that the first element of
    ``teperatures`` is 0.
    """
    T, = temperatures.shape
    assert state.Z.shape[0] == T

    state = make_contiguous(state)
    @vmap
    def reset_row(i):
        mcmc = mcmc_engine(problem, temperatures[i], i)
        return batch_reset_temperature(pytree_index(state.X, state.Z[i]), mcmc)

    X = reset_row(jnp.arange(T))
    return PTState(jax_tree.map(jnp.concatenate, X), state.Z).validate_shape()

def _validate_temperatures(temperatures: Array):
    r"""Check that temperatures start at 0 and end at 1.
    Uses float() to extract concrete values, so this must be called
    outside of JIT-traced code.
    """
    t0, t_last = float(temperatures[0]), float(temperatures[-1])
    if abs(t0) > 1e-6:
        raise ValueError(f'temperatures must start at 0 (got {t0})')
    if abs(t_last - 1.) > 1e-6:
        raise ValueError(f'temperatures must end at 1 (got {t_last})')

def pt_initialize(key, problem: PTProblem, temperatures: Array, nparticles: Optional[int],
                  mcmc_engine: PTMCMCEngine, debug=False, particles=None) -> PTState:
    r"""Initialize a PT state with T*N particles.

    Three initialization modes:

    1. **From prior** (default): ``particles=None, debug=False``.
       Draws T*N iid samples from ``problem.rvs_ref``. Each temperature
       gets independent particles. Standard usage.

    2. **From provided particles**: ``particles=(T, N, d) array``.
       Uses the given array directly (e.g., SMC output reshaped to (T, N, d)).
       ``nparticles`` is ignored and inferred from ``particles.shape[1]``.

    3. **Debug mode**: ``debug=True``.
       Draws N particles from the prior and replicates them identically
       across all T temperatures. Useful for testing swap mechanics in
       isolation (all temperatures start from the same population).

    In all modes, particles are extended via ``mcmc_engine`` and temperatures
    are assigned via ``reset_temperature``.
    """
    T, = temperatures.shape
    rvs_ref = vmap(problem.rvs_ref)
    if debug:
        key, subkeys = random_split(key, nparticles+1)
        x = rvs_ref(subkeys)
        x = jnp.concatenate(jnp.array([x] * T))
    elif particles is None:
        key, subkeys = random_split(key, T * nparticles + 1)
        x = rvs_ref(subkeys)
    else:
        assert particles.shape[0] == T
        nparticles = particles.shape[1]
        x = jnp.concatenate(particles)
    assert x.shape[0] == T * nparticles
    key, subkey = random.split(key)
    x = batch_init_extend(subkey, x, mcmc_engine(problem, 0., 0))
    state = PTState(x, default_Z(T, nparticles)).validate_shape(temperatures)
    return reset_temperature(state, problem, temperatures, mcmc_engine)

def pt_explore(key, state: PTState, problem: PTProblem, temperatures: Array,
               mcmc_engine: PTMCMCEngine, niter=None) -> 'ExploreResult':
    r"""Performs the exploration step of parallel tempering. It divides the set
    of particles at each temperature into two parts, applying MCMC to one part
    using parameters found based on the other part. See documentation of the
    private function `_pt_explore` for details on the interactions between the
    two parts, and documentation of ``ExploreResult`` class for details on the
    output.

    Warning: The function assumes that the ``state`` argument is contiguous.
    See ``make_contiguous``.
    """
    T, N = state.Z.shape
    if N < 2:
        raise ValueError(f'pt_explore requires N >= 2 particles per temperature (got N={N}). '
                         f'The two-half exploration split produces an empty half with N=1.')
    state.assert_contiguous()
    state.validate_shape(temperatures)

    key, subkey = random.split(key)
    first_half, second_half, perm = _separate_into_halves(subkey, state.X, T, N)
    key, subkey = random.split(key)
    new_second_half_res = _pt_explore(subkey, first_half, second_half, problem,
                                      temperatures, mcmc_engine, niter=niter)
    key, subkey = random.split(key)
    new_first_half_res = _pt_explore(subkey, new_second_half_res.new_state, first_half,
                                     problem, temperatures, mcmc_engine, niter=niter)

    new_state_X = _merge_halves(new_first_half_res.new_state, new_second_half_res.new_state,
                                perm)
    new_state = PTState(new_state_X, default_Z(T, N)).validate_shape(temperatures)
    res_wl = tuple([new_second_half_res.while_loop_exec_res, new_first_half_res.while_loop_exec_res])
    res_pr = tuple([new_second_half_res.mcmc_params, new_first_half_res.mcmc_params])
    return ExploreResult(new_state, res_wl, res_pr)

class ExploreResult(NamedTuple):
    r"""Result of an explore step in PT.

    :param new_state: the output state of the exploration step. Should already
    be contiguous.
    :param while_loop_exec_res: a tuple of two elements containing the results
    of the adaptations of the number of MCMC steps on the first half and the second
    half of the particle system. Each element is a tuple of two arrays: the first
    one is the number of steps chosen for each temperature, and the second one
    indicates whether the adaptation process, at each temperature, reached the
    end or was interrupted to save computing time.
    :param mcmc_params: a tuple of two elements containing the MCMC parameters
    found based on the first and the second half of the particle system.
    """
    new_state: PTState
    while_loop_exec_res: tuple[tuple[ArrayInt, ArrayBool], ...]
    mcmc_params: tuple[MCMCParams, ...]

_HalfExploreResult = namedtuple('_HalfExploreResult', ['new_state', 'while_loop_exec_res', 'mcmc_params'])

def _pt_explore(
        key, base_half: PyTree, updated_half: PyTree, problem: PTProblem,
        temperatures: Array, mcmc_engine: PTMCMCEngine, completion_threshold=0.5, niter=None
) -> _HalfExploreResult:
    r"""Update the ``updated_half`` argument by running MCMC with parameters
    determined by the ``base_half`` argument. Each half is a PyTree of length
    T containing particles at T different temperatures. The algorithm proceeds
    as follows:
        * Using ``base_half``, get parameters for the MCMC method (e.g. covariance
        matrix, step size) [step A]
        * Using ``base_half`` and the previously obtained parameters, find the
        optimal number of MCMC iterations at each temperature. This is done by
        triggering ``adapt_nsteps`` at each temperature, producing T different
        while loops. Since the mixing of MCMC can vary between temperatures, we
        avoid deadlock by only running these while loops until 50% of them are
        done and assigning the resulting number of iterations to the variable
        `niter` below. The proportion of 50% can be changed via the ``completion_threshold``
        argument. [step B]
        * Run MCMC on ``updated_half`` using the previously obtained parameters
        for `niter` iterations [step C]
    """
    # the halves are PyTree of shape prefix (T, N//2)
    # In the following functions: k=key, i=index, ti=temperature[i], pr=params (mcmc)
    @vmap
    def get_parameters(k, i, ti, x):  # step A
        return mcmc_engine(problem, ti, i).adapt_params(k, x)
    def get_nadapts_loops(obj):  # step B
        k, i, ti, pr, x = obj
        return mcmc_engine(problem, ti, i).adapt_nsteps(k, x, pr)
    def run_mcmc(k, i, ti, pr, x, n):  # step C
        engine = mcmc_engine(problem, ti, i)
        return repeated_batch_move(k, x, pr, engine, n)

    T, = temperatures.shape

    # Step A
    key, subkeys = random_split(key, T+1)
    mcmc_params = get_parameters(subkeys, jnp.arange(T), temperatures, base_half)

    # Step B
    if niter is None:
        key, subkeys = random_split(key, T+1)
        while_loop_res = while_loops_executor(
            get_nadapts_loops,
            (subkeys, jnp.arange(T), temperatures, mcmc_params, base_half),
            completion_threshold
        )
        assert while_loop_res[0].shape == (T, )
        assert jnp.issubdtype(while_loop_res[0].dtype, jnp.integer)
        niter = jnp.max(while_loop_res[0])
    else:
        while_loop_res = None

    # Step C
    key, subkeys = random_split(key, T+1)
    vrun_mcmc = vmap(partial(run_mcmc, n=niter))
    new_half = vrun_mcmc(subkeys, jnp.arange(T), temperatures, mcmc_params, updated_half)

    return _HalfExploreResult(new_half, while_loop_res, mcmc_params)

def _separate_into_halves(key, X: PyTree, T: int, N: int):
    r"""Split flat particles into two randomly selected halves per temperature.

    Reshapes X from (T*N, ...) to (T, N, ...), applies a random permutation
    along the N axis, then takes the first and second half. The permutation
    ensures unbiased selection and must be passed to ``_merge_halves`` to
    reassemble.

    :return: (first_half, second_half, perm) where each half has shape
    prefix (T, N//2) and perm is the permutation used.
    """
    assert pytree_length(X) == T * N
    perm = random.permutation(key, N)
    # def reshape_to_T_N(arr):
    #     old_shape = list(arr.shape)
    #     assert old_shape[0] == T * N
    #     new_shape = [T, N] + old_shape[1:]
    #     return jnp.reshape(arr, new_shape)
    # reshape_to_T_N = partial(PTState.reshape_to_T_N, T=T, N=N)
    def take_first_half(arr):  # arr has shape (T, N, ...)
        return arr[:, perm[0:N//2], ...]
    def take_second_half(arr):
        return arr[:, perm[N//2:], ...]
    # X = jax_tree.map(reshape_to_T_N, X)
    X = PTState.reshape_to_T_N(X, T, N)
    return jax_tree.map(take_first_half, X), jax_tree.map(take_second_half, X), perm

def _merge_halves(X, Y, perm):
    r"""Inverse of ``_separate_into_halves``: merge two (T, N//2) halves back
    into a flat (T*N, ...) pytree, undoing the permutation so particles return
    to their original ordering.
    """
    def concatenate(x, y):
        return jnp.concatenate([x, y], axis=1)
    def reorder(arr):
        return arr[:, inverse_permutation(perm), ...]
    res = jax_tree.map(concatenate, X, Y)
    res = jax_tree.map(reorder, res)
    return PTState.flatten_first_two_dims(res)

def pt_explore_nonadaptive(
        key, state: PTState, problem: PTProblem, temperatures: Array,
        mcmc_params, niter: int, mcmc_engine: PTMCMCEngine) -> PTState:
    r"""Exploration step for Parallel Tempering without adaptation.
    As such, the function requires the MCMC parameters for all temperatures
    (``mcmc_params``: PyTree) and the number of MCMC steps (``niter``).
    The ``state`` is assumed to be contiguous.
    """
    state.assert_contiguous()
    state.validate_shape(temperatures)
    dZ = default_Z(state.T, state.N)

    @vmap
    def explore_row(k, i, t, params):
        mcmc = mcmc_engine(problem, t, i)
        x = pytree_index(state.X, dZ[i])
        return repeated_batch_move(k, x, params, mcmc, niter)

    subkeys = random.split(key, state.T)
    newX = explore_row(subkeys, jnp.arange(state.T), temperatures, mcmc_params)
    newX = PTState.flatten_first_two_dims(newX)
    return PTState(newX, dZ).validate_shape(temperatures)

def _pt_swap_once(
        key, state: PTState, rej_rates: Array, mode: str, temperatures: Array,
        n_inner_swaps: int, inner_swapper: TwoLevelSwapper) -> tuple[PTState, Array]:
    r"""One pass of pairwise swaps between adjacent temperatures.

    :param mode: ``'even'`` swaps pairs (0,1), (2,3), ...; ``'odd'`` swaps
    pairs (1,2), (3,4), ...
    :param n_inner_swaps: number of Metropolis swap attempts per pair
    :return: (updated PTState with modified Z, updated rejection rates array)

    Note: state.X.temperature cannot be used here because swapping changes Z
    only -- the particle metadata still reflects pre-swap temperatures.
    """
    T, N = state.Z.shape
    @vmap
    def swap_row(k, i):  # swap particles at temperatures i and i+1
        Z_cc = jnp.concatenate(pytree_index(state.Z, jnp.array([i, i+1])))  # cc: concatenated
        logpdf_rvs_cc = pytree_index(state.X.logpdf_ref, Z_cc)
        logpdf_ft_cc = pytree_index(state.X.logpdf_final_target, Z_cc)
        log_pi_i = geometric_anneal(pytree_index(temperatures, i),
                                    logpdf_rvs_cc, logpdf_ft_cc)
        log_pi_ip1 = geometric_anneal(pytree_index(temperatures, i+1),
                                      logpdf_rvs_cc, logpdf_ft_cc)
        perm = jnp.arange(2 * N)
        perm, reject_rate = repeated_swap(k, log_pi_i, log_pi_ip1, perm,
                                          n_inner_swaps, inner_swapper)
        return (jnp.array([i, i+1]), pytree_index(Z_cc, perm).reshape((2, N)),
                i, reject_rate)

    if mode == 'even':
        i_idx = jnp.arange(0, T-1, 2)
    elif mode == 'odd':
        i_idx = jnp.arange(1, T-1, 2)
    else:
        raise ValueError('Unknown mode')
    subkeys = random.split(key, len(i_idx))
    swap_res = swap_row(subkeys, i_idx)

    new_Z = state.Z.at[swap_res[0]].set(swap_res[1])
    new_rej_rates = rej_rates.at[swap_res[2]].set(swap_res[3])

    return PTState(state.X, new_Z).validate_shape(temperatures), new_rej_rates

class SwapResult(NamedTuple):
    new_state: PTState
    rej_rates: Array

def pt_swap(key, state: PTState, temperatures: Array, n_outer_swaps: int,
            n_inner_swaps: int, inner_swapper: TwoLevelSwapper) -> SwapResult:
    r"""Performs the swap step in parallel tempering. This function alternates
    between odd and even swaps between adjacent temperatures. The exact working
    of the swaps between two temperatures is handled by the ``inner_swapper``
    object.
    :return: a tuple whose first component is the new state and second component
    is the rejection rates at all temperatures (ie rej[i] is between temperatures[i]
    and temperatures[i+1]).

    Warning
    -------
    The swap step changes only ``state.Z`` and not ``state.X`` (in  particular
    ``state.X.temperature`` and ``state.X.temperature_idx``). Therefore, the
    ``reset_temperature`` function must be applied before the next exploration
    step.
    """
    init_val = {'key': key, 'state': state, 'rej_rates': jnp.zeros(state.T - 1)}
    body_func = partial(_pt_swap_body_func, temperatures=temperatures,
                        n_inner_swaps=n_inner_swaps, inner_swapper=inner_swapper)
    res = lax.fori_loop(0, n_outer_swaps, body_func, init_val)
    return SwapResult(res['state'], res['rej_rates'])

def _pt_swap_body_func(i, val, temperatures, n_inner_swaps, inner_swapper):
    r"""Body of the fori_loop in ``pt_swap``: one even swap pass followed by
    one odd swap pass.
    """
    dummy_use(i)
    key, subkey = random.split(val['key'])
    state_temp, rej_temp = _pt_swap_once(subkey, val['state'], val['rej_rates'],
                                         'even', temperatures, n_inner_swaps, inner_swapper)
    key, subkey = random.split(key)
    state_new, rej_new = _pt_swap_once(subkey, state_temp, rej_temp,
                                       'odd', temperatures, n_inner_swaps, inner_swapper)
    return {'key': key, 'state': state_new, 'rej_rates': rej_new}

def estimate_logZ(key, state: PTState, temperatures: Array,
                  estimator: LogZ_Estimator) -> Float:
    r"""Estimates logZ. The state is supposed to be contiguous. See
    ``impl/logZ_estimators`` for different logZ estimators that can be
    used with this function.
    """
    state.assert_contiguous()
    # reshaper = partial(PTState.reshape_to_T_N, T=state.T, N=state.N)
    # reshaped_state = jax_tree.map(reshaper, state.X)
    reshaped_state = PTState.reshape_to_T_N(state.X, state.T, state.N)
    return estimator(key, reshaped_state.logpdf_ref, reshaped_state.logpdf_final_target,
                     temperatures)