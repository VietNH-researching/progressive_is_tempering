from functools import partial
from typing import NamedTuple, Any, Protocol
from jax import Array, numpy as jnp
from jax.typing import ArrayLike
from mppt.abstract.mcmc import PTMCMCEngine
from mppt.abstract.pt_concept import PTState, PTProblem
from mppt.abstract.pt_exec import pt_explore, pt_explore_nonadaptive


class SimpleExploreResult(NamedTuple):
    r"""Simplified output of an exploration step, used by the UI layer.
    Wraps the richer ``ExploreResult`` from ``pt_exec`` with just the
    fields needed by collectors and the main loop.
    """
    new_state: PTState
    niter: int         # number of MCMC steps used (max across temperatures)
    mcmc_params: Any   # MCMC parameters from the second half's adaptation

class UIExplorer(Protocol):
    r"""Callable signature for exploration strategies."""
    def __call__(self, key, state: PTState, problem: PTProblem, temperatures: Array,
                 engine: PTMCMCEngine) -> SimpleExploreResult:
        ...

def parse_explorer(mode: str, niter=None, mcmc_params=None) -> UIExplorer:
    r"""Select an exploration strategy from a config dict.

    Modes:
        ``'adaptive'``: adapt MCMC params and (optionally) nsteps from data.
        If ``niter`` is given, use that fixed step count with adaptive params.
        ``'nonadaptive'``: use fixed ``mcmc_params`` and ``niter`` (both required).
    """
    if mode == 'adaptive':
        if niter is None:
            return _adaptive
        else:
            return partial(_adaptive_with_niter, niter=niter)
    elif mode == 'nonadaptive':
        if niter is None:
            raise ValueError("nonadaptive explorer requires 'niter'")
        if mcmc_params is None:
            raise ValueError("nonadaptive explorer requires 'mcmc_params'")
        return partial(_nonadaptive, mcmc_params=mcmc_params, niter=niter)
    else:
        raise ValueError(f"Unknown explorer mode '{mode}'")

def _adaptive(key, state: PTState, problem: PTProblem, temperatures: Array,
              engine: PTMCMCEngine) -> SimpleExploreResult:
    r"""Fully adaptive exploration: adapt both MCMC parameters and iteration
    count from the data. Returns the max niter across temperatures.
    """
    res = pt_explore(key, state, problem, temperatures, engine)
    # while_loop_exec_res is (half1_result, half2_result); each is (niters, done_flags)
    niters = res.while_loop_exec_res[1][0]
    assert jnp.ndim(niters) == 1 and jnp.issubdtype(niters.dtype, jnp.integer)
    niter: ArrayLike = jnp.max(niters)
    return SimpleExploreResult(res.new_state, niter, res.mcmc_params[1])

def _adaptive_with_niter(
        key, state: PTState, problem: PTProblem, temperatures: Array,
        engine: PTMCMCEngine, niter: int
) -> SimpleExploreResult:
    r"""Adaptive MCMC parameters but fixed iteration count."""
    res = pt_explore(key, state, problem, temperatures, engine, niter=niter)
    return SimpleExploreResult(res.new_state, niter, res.mcmc_params[1])

def _nonadaptive(key, state: PTState, problem: PTProblem, temperatures: Array,
                 engine: PTMCMCEngine, mcmc_params, niter: int) -> SimpleExploreResult:
    r"""Fixed MCMC parameters and fixed iteration count. Requires both
    ``mcmc_params`` and ``niter`` to be provided at construction time.
    """
    new_state = pt_explore_nonadaptive(key, state, problem, temperatures,
                                       mcmc_params, niter, engine)
    return SimpleExploreResult(new_state, niter, mcmc_params)
