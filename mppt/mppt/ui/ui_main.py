from __future__ import annotations
from collections import defaultdict
from functools import partial
from numbers import Integral, Real
from typing import Union

import jax
from mppt.abstract.mcmc import batch_init_extend
from mppt.abstract.pt_concept import PTProblem, PTState
from jax import Array, random, lax, numpy as jnp, jit, vmap, block_until_ready
from mppt.abstract.pt_exec import pt_initialize, reset_temperature, pt_swap, _validate_temperatures
from mppt.abstract.smc import one_step_smc
from mppt.jax_configs import chexify_no_async, allow_runtime_check
from mppt.ui.collectors import parse_collectors
from mppt.ui.engine_ui import parse_engine
from mppt.ui.explorers import parse_explorer
from mppt.ui.frozen import FrozenDict, FrozenList
from mppt.ui.swappers_api import parse_swappers
from mppt.utils import verbose_scan, jit_with_info, random_split, run_time

def _validate_int_count(name: str, value, minimum: int, *, allow_none: bool = False):
    r"""Validate Python-side control counts before JAX tracing/compilation.

    Silent negative or missing counts are especially costly here: they can turn
    a ``lax.fori_loop`` into a no-op or fail much later inside compiled code.
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be an integer >= {minimum} (got None)")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum} (got {value!r})")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum} (got {value})")
    return value

def _validate_open_unit_interval(name: str, value):
    r"""Validate probabilities controlling adaptive schedules.

    For SMC, endpoints are invalid: ESS ratio 1 can produce no temperature
    progress, while 0 asks for complete degeneracy.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a float in (0, 1) (got {value!r})")
    value = float(value)
    if not 0. < value < 1.:
        raise ValueError(f"{name} must be a float in (0, 1) (got {value})")
    return value

default_collectors = FrozenList([
    {'name': 'logZ', 'estimator': 'SMC_like'}, {'name': 'logZ', 'estimator': 'robust'},
    {'name': 'logZ', 'estimator': 'reverse_logistic'},
    {'name': 'mcmc_acc_rates'},
    {'name': 'niter'}, {'name': 'mcmc_parameters', 'at_every_iter': False},
    {'name': 'swap_rejection_rates'}  # , {'name': 'index_matrix'} (don't collect to save memory)
])

def _build_pt_step(problem, temperatures, engine, explorer, collectors, *,
                   n_outer_swaps, n_inner_swaps, swapper):
    r"""Parse configs and build the PT step closure.

    Returns (step, mcmc) where step is the one-iteration closure and mcmc
    is the parsed MCMC engine (needed for initialization and temperature reset).
    """
    mcmc = parse_engine(engine)
    explorer = parse_explorer(**explorer)
    collectors_obj = parse_collectors(collectors)
    swapper = parse_swappers(**swapper)
    if n_outer_swaps is None:
        n_outer_swaps = len(temperatures)
    if n_inner_swaps is None:
        n_inner_swaps = 3
    n_outer_swaps = _validate_int_count('n_outer_swaps', n_outer_swaps, 0)
    n_inner_swaps = _validate_int_count('n_inner_swaps', n_inner_swaps, 0)

    def step(carry, _, last_iter: bool = False):
        r"""One PT iteration: explore -> collect post-explore stats -> swap ->
        collect post-swap stats -> reset temperatures. When last_iter=True,
        collectors marked at_every_iter=False also run.
        """
        col_res = {}
        s = carry['state']
        k = carry['key']

        # Explore
        k, sk = random.split(k)
        s = explorer(sk, s, problem, temperatures, mcmc)
        for co in collectors_obj:
            if (co.at_every_iter or last_iter) and (not co.after_swap):
                k, sk = random.split(k)
                col_res[co.unique_name] = co.func(sk, s, problem, temperatures, mcmc)

        # Swap
        k, sk = random.split(k)
        s = pt_swap(sk, s.new_state, temperatures, n_outer_swaps, n_inner_swaps,
                    swapper)
        for co in collectors_obj:
            if (co.at_every_iter or last_iter) and co.after_swap:
                k, sk = random.split(k)
                col_res[co.unique_name] = co.func(sk, s, problem, temperatures, mcmc)

        # Reset
        s = reset_temperature(s.new_state, problem, temperatures, mcmc)
        return {'key': k, 'state': s}, col_res

    return step, mcmc

def parallel_tempering(
        key, problem: PTProblem, temperatures: Union[Array, list], n_iter: int,
        init_state: Union[PTState, Array] = None, n_particles: int = None,
        explorer: Union[dict, FrozenDict] = FrozenDict({'mode': 'adaptive'}),
        engine: Union[dict, FrozenDict] = FrozenDict({'name': 'random_walk', 'maxniter': 100}),
        collectors: Union[list, FrozenList] = default_collectors,
        n_inner_swaps: int = None, n_outer_swaps: int = None,
        verbose: bool = True, runtime_check: bool = False,
        swapper: Union[dict, FrozenDict] = FrozenDict({'mix': True}),
) -> tuple[PTState, CollectResult, CollectResultLast]:
    r"""Run parallel tempering with a fixed temperature schedule.

    See ``docs/api.md`` for a full description of the algorithm and all options.

    :param key: JAX random key.
    :param problem: a ``PTProblem`` (reference sampler, reference log-density,
        target log-density).
    :param temperatures: schedule from 0 to 1, e.g. ``[0, 0.1, 0.3, 0.6, 1.0]``.
    :param n_iter: number of PT iterations (each = explore + swap).
    :param init_state: starting particles. Either a ``PTState`` from a previous
        run, a ``(T, N, d)`` array (e.g. from SMC), or ``None`` to sample from
        the reference.
    :param n_particles: particles per temperature. Ignored if ``init_state``
        is provided.
    :param engine: MCMC kernel. ``{'name': 'hmc', 'refresh_rate': 0.07}``
        (default: random walk). See ``docs/api.md#engines``.
    :param explorer: exploration mode. ``{'mode': 'adaptive', 'niter': 50}``
        adapts step size with fixed MCMC steps; ``{'mode': 'adaptive'}``
        adapts both. See ``docs/api.md#explorers``.
    :param collectors: statistics to collect at each iteration. Default:
        logZ (3 estimators), acceptance rates, swap rejection rates, MCMC
        parameters (last iteration only). See ``docs/api.md#collectors``.
    :param n_inner_swaps: how many times to re-pair and attempt swaps within
        each adjacent temperature pair. Default: 3.
    :param n_outer_swaps: how many even/odd sweep passes over all temperature
        pairs. Default: ``len(temperatures)``.
    :param verbose: print progress bar and timing.
    :param runtime_check: enable chex debug assertions (slow).
    :param swapper: ``{'mix': True}`` (recommended) randomly permutes particles
        before pairing for swaps; ``{'mix': False}`` pairs by index order.
    :return: ``(final_state, intermediate_stats, final_stats)`` where
        ``intermediate_stats`` is a dict of arrays stacked over iterations
        0..n_iter-2, and ``final_stats`` includes all collectors.
    """
    temperatures = jnp.array(temperatures)
    _validate_temperatures(temperatures)
    n_iter = _validate_int_count('n_iter', n_iter, 1)
    if init_state is None:
        n_particles = _validate_int_count('n_particles', n_particles, 1)
    elif n_particles is not None:
        n_particles = _validate_int_count('n_particles', n_particles, 1)
    step, mcmc = _build_pt_step(
        problem, temperatures, engine, explorer, collectors,
        n_outer_swaps=n_outer_swaps, n_inner_swaps=n_inner_swaps, swapper=swapper
    )

    subkey = key
    if not isinstance(init_state, PTState):
        key, subkey = random.split(key)
        init_state = pt_initialize(subkey, problem, temperatures, n_particles, mcmc,
                                   particles=init_state)
    else:
        init_state = reset_temperature(init_state, problem, temperatures, mcmc)
    init_carry = {'key': subkey, 'state': init_state}

    if verbose:
        res0 = verbose_scan(step, init_carry, None, n_iter-1, runtime_check,
                            compile_options={'donate_argnums': (0, 1)})
    else:
        with allow_runtime_check(runtime_check):
            temp_func = lambda: lax.scan(step, init_carry, length=n_iter-1)
            if runtime_check: temp_func = chexify_no_async(temp_func)
            res0 = temp_func()

    if verbose: print('Running last iteration...')
    with allow_runtime_check(False):
        step_last = partial(step, last_iter=True)
        step_last = jit_with_info(step_last, compile_options={'donate_argnums': (0, 1)})\
            if verbose else jit(step_last, donate_argnums=(0, 1))
        res1 = step_last(res0[0], None)
    return res1[0]['state'], res0[1], res1[1]

def pt_manual_runners(
        problem: PTProblem, temperatures: Union[Array, list], n_particles: int,
        explorer: Union[dict, FrozenDict] = FrozenDict({'mode': 'adaptive'}),
        engine: Union[dict, FrozenDict] = FrozenDict({'name': 'random_walk', 'maxniter': 100}),
        collectors: Union[list, FrozenList] = default_collectors,
        n_inner_swaps: int = None, n_outer_swaps: int = None,
        swapper: Union[dict, FrozenDict] = FrozenDict({'mix': True}),
) -> callable:
    r"""Pre-compile a one-step PT function for manual iteration.

    Useful for periodic checkpointing or custom stopping criteria. All
    internal functions are JIT-compiled eagerly, so the first call to this
    function takes 10--60s, but subsequent calls to the returned ``one_step``
    have zero compilation overhead.

    See ``docs/api.md`` for a full description of the algorithm and all options.

    :param problem: a ``PTProblem`` (reference sampler, reference log-density,
        target log-density).
    :param temperatures: schedule from 0 to 1.
    :param n_particles: particles per temperature. Must match the actual
        particle count passed at runtime (a mismatch raises ``ValueError``).
    :param engine: MCMC kernel. ``{'name': 'hmc', 'refresh_rate': 0.07}``
        (default: random walk). See ``docs/api.md#engines``.
    :param explorer: exploration mode. ``{'mode': 'adaptive', 'niter': 50}``
        adapts step size with fixed MCMC steps; ``{'mode': 'adaptive'}``
        adapts both. See ``docs/api.md#explorers``.
    :param collectors: statistics to collect. See ``docs/api.md#collectors``.
    :param n_inner_swaps: how many times to re-pair and attempt swaps within
        each adjacent temperature pair. Default: 3.
    :param n_outer_swaps: how many even/odd sweep passes over all temperature
        pairs. Default: ``len(temperatures)``.
    :param swapper: ``{'mix': True}`` (recommended) randomly permutes particles
        before pairing for swaps; ``{'mix': False}`` pairs by index order.
    :return: a callable ``one_step(key, prev, full_output)`` where ``prev``
        is a ``PTState`` or ``(T, N, d)`` array, and ``full_output=False``
        collects only per-iteration stats while ``True`` includes all
        collectors. Returns ``(next_state, diagnostics_dict)``.
    """
    temperatures = jnp.array(temperatures)
    _validate_temperatures(temperatures)
    n_particles = _validate_int_count('n_particles', n_particles, 1)
    step, mcmc = _build_pt_step(
        problem, temperatures, engine, explorer, collectors,
        n_outer_swaps=n_outer_swaps, n_inner_swaps=n_inner_swaps, swapper=swapper
    )

    def one_step_ordinary_compact(key, prev):
        carry_in = {'key': key, 'state': prev}
        carry_out, diag = step(carry_in, None, False)
        return carry_out['state'], diag

    def one_step_ordinary_full(key, prev):
        carry_in = {'key': key, 'state': prev}
        carry_out, diag = step(carry_in, None, True)
        return carry_out['state'], diag

    def initializer_smc(key, prev):
        return pt_initialize(key, problem, temperatures, None, mcmc,
                             particles=prev)

    def initializer_nothing(key):
        return pt_initialize(key, problem, temperatures, n_particles, mcmc)

    one_step_ordinary_compact, one_step_ordinary_full, \
        initializer_smc, initializer_nothing = _compile_all(
            one_step_ordinary_compact, one_step_ordinary_full,
            initializer_smc, initializer_nothing
        )

    def one_step(key, prev, full_output):
        if isinstance(prev, Array):
            if prev.ndim == 3 and prev.shape[1] != n_particles:
                raise ValueError(
                    f"pt_manual_runners was configured with n_particles={n_particles}, "
                    f"but received array with shape {prev.shape} "
                    f"({prev.shape[1]} particles). These must match."
                )
            key, subkey = random.split(key)
            prev = initializer_smc(subkey, prev)
        elif isinstance(prev, PTState) and prev.N != n_particles:
            raise ValueError(
                f"pt_manual_runners was configured with n_particles={n_particles}, "
                f"but received PTState with N={prev.N}. These must match."
            )
        if full_output:
            return one_step_ordinary_full(key, prev)
        else:
            return one_step_ordinary_compact(key, prev)

    return one_step

def _compile_all(compact_step, full_step, smc_init, cold_init):
    r"""Eagerly compile all four JIT-compiled functions used by
    ``pt_manual_runners``:
        compact_step: one-step with at_every_iter collectors only
        full_step: one-step with all collectors
        smc_init: initialize from SMC output array
        cold_init: initialize from scratch
    Compilation happens once at runner creation time so that subsequent
    calls have zero compilation overhead.
    """
    import warnings
    key = random.key(0)
    print('Compiling PT runner:', flush=True)
    with run_time('  initializer'):
        cold_init = jit(cold_init).lower(key).compile()

    state: PTState = cold_init(key)
    arr = PTState._reshape_to_T_N(state.X.position, state.T, state.N)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Some donated buffers were not usable')
        with run_time('  SMC initializer'):
            smc_init = jit(smc_init, donate_argnums=(1, )).lower(key, arr).compile()
        with run_time('  one-step (compact)'):
            compact_step = jit(compact_step, donate_argnums=(1, )).lower(key, state).compile()
        with run_time('  one-step (full)'):
            full_step = jit(full_step, donate_argnums=(1, )).lower(key, state).compile()
    print('Compilation done.')
    return compact_step, full_step, smc_init, cold_init

def smc_sampler(key, problem: PTProblem, n_particles, engine, n_mcmc_steps, local_ess_ratio, verbose, runtime_check, keep_all_states=False, keep_mcmc_params=False):
    r"""SMC sampler that anneals from t=0 to t=1 with adaptive temperature
    schedule. At each step: find next temperature via ESS-based bisection,
    resample, run MCMC, estimate local logZ. Temperatures are chosen so that
    each resampling step drops the ESS ratio to ``local_ess_ratio``.

    :param keep_all_states: if True, return all intermediate particle positions
    as a stacked array (shape: n_steps+1, n_particles, d); otherwise return
    only the final particles.
    :param keep_mcmc_params: if True, additionally return the per-rung adapted
    MCMC parameters (e.g. ``HMCParameters``, stacked over a leading axis) as a
    4th return value. Entry ``k-1`` of the stack is the parameters adapted and
    used for the move into rung ``k`` (``k = 1, ..., len(temperatures_list)-1``);
    rung 0 has no associated MCMC move. Off by default to keep the return shape
    backward compatible.
    :return: (particles, logZ_dict, temperatures_list) or, if
    ``keep_mcmc_params=True``, (particles, logZ_dict, temperatures_list, mcmc_params)

    When ``verbose=True``, each iteration prints:

      * ``dlogZ``  = log(Z_t / Z_{t_prev}), the step's incremental log-ratio
      * ``logZ(t)`` = log(Z_t / Z_0),        cumulative so far

    Values are reported for the three logZ estimators; their order is printed
    in the header and matches the order of ``logZ_dict`` in the return value.
    At t = 1, ``logZ(t)`` equals the final log(Z_target / Z_ref).
    """
    n_particles = _validate_int_count('n_particles', n_particles, 1)
    n_mcmc_steps = _validate_int_count(
        'n_mcmc_steps', n_mcmc_steps, 0, allow_none=True
    )
    local_ess_ratio = _validate_open_unit_interval(
        'local_ess_ratio', local_ess_ratio
    )
    mcmc = parse_engine(engine)

    # Initialize and compile
    if verbose:
        print('Initializing and compiling...', flush=True)
    key, subkeys = random_split(key, n_particles+1)
    x = vmap(problem.rvs_ref)(subkeys)
    key, subkey = random.split(key)
    x = jit(batch_init_extend, static_argnums=(2, ))(subkey, x,
                                                     mcmc(problem, 0., 0))
    x = block_until_ready(x)

    mover = partial(one_step_smc, problem=problem, mcmc_engine=mcmc, niter_mcmc=n_mcmc_steps, new_ess_ratio=local_ess_ratio)
    if runtime_check:
        mover = chexify_no_async(jit(mover))
    else:
        mover = jit(mover)

    # Run
    logZ_ret = defaultdict(float)   # cumulative log(Z_t / Z_0) per estimator
    all_temperatures = [0.]
    all_states = [x.position]
    all_mcmc_params = []
    header_printed = False
    while not jnp.allclose(x.temperature, 1):
        key, subkey = random.split(key)
        x, logZ_local, niter_mcmc, mcmc_params = mover(subkey, x)
        for _k, _v in logZ_local.items():
            logZ_ret[_k] += _v
        if verbose:
            if not header_printed:
                est_order = ', '.join(logZ_local.keys())
                print(f'--- SMC anneal | estimators: ({est_order}) ---')
                print('    dlogZ = log(Z_t / Z_prev)   |   logZ(t) = log(Z_t / Z_0)')
                header_printed = True
            dlogZ_str = '(' + ', '.join(f'{float(v):>9.3f}' for v in logZ_local.values()) + ')'
            cumZ_str  = '(' + ', '.join(f'{float(logZ_ret[k]):>9.3f}' for k in logZ_local.keys()) + ')'
            t_val = float(x.temperature[0])
            t_str = f'{t_val:.2e}' if t_val < 0.001 else f'{t_val:.6f}'
            print(f'iter {int(x.temperature_idx[0]):>3d} | t={t_str} '
                  f'| acc={float(jnp.mean(x.accepted)):.2f} | niter={niter_mcmc:>3d} '
                  f'| dlogZ={dlogZ_str} | logZ(t)={cumZ_str}')
        all_temperatures.append(x.temperature[0])
        if keep_all_states:
            all_states.append(x.position)
        if keep_mcmc_params:
            all_mcmc_params.append(mcmc_params)

    if verbose:
        est_order = ', '.join(logZ_ret.keys())
        total_str = '(' + ', '.join(f'{float(v):+.4f}' for v in logZ_ret.values()) + ')'
        print(f'--- SMC done | logZ = log(Z_1 / Z_0) = {total_str}   [{est_order}] ---')

    if keep_all_states:
        x = jnp.array(all_states)
    if keep_mcmc_params:
        mcmc_params_stacked = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves), *all_mcmc_params
        )
        return x, logZ_ret, all_temperatures, mcmc_params_stacked
    return x, logZ_ret, all_temperatures
