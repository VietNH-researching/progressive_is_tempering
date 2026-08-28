from __future__ import annotations
from collections import namedtuple
from typing import Protocol, Callable, Union, Any
from jax.typing import ArrayLike
from jax import Array, numpy as jnp
from mppt.abstract.mcmc import PTMCMCEngine
from mppt.abstract.pt_concept import PTProblem
from mppt.abstract.pt_exec import SwapResult, estimate_logZ
from mppt.ui.explorers import SimpleExploreResult
from mppt.utils import dummy_use
from mppt.impl import logZ_estimators

Float = ArrayLike
Int = ArrayLike

class CollectorProtocol(Protocol):
    r"""Template for a collector object. It calculates a certain statistics at
    each PT iteration.

    :param unique_name: a unique name identifying the collector and its parameters
    :param after_swap: whether to collect after the swap step or after the exploration
    :param at_every_iter: whether to collect at every iteration or only at the end
    :param func: calculates the statistics to be collected
    """
    unique_name: str
    after_swap: bool
    at_every_iter: bool
    func: Callable[[Key, StepOutput, PTProblem, Temperatures, Engine], Any]

def is_collector(obj):
    r"""Duck-type check: does obj have the four Collector fields?"""
    return hasattr(obj, 'unique_name') and hasattr(obj, 'after_swap') and\
        hasattr(obj, 'at_every_iter') and hasattr(obj, 'func')

Collector = namedtuple('Collector', ['unique_name', 'after_swap', 'at_every_iter', 'func'])
all_collectors = {}

def add_collector(f):
    r"""Decorator that registers a collector factory by its function name.
    Registered collectors can be selected via ``parse_collectors``.
    """
    all_collectors[f.__name__] = f
    return f

def parse_collectors(collector_list: list[Union[dict, CollectorProtocol]]) ->\
        list[CollectorProtocol]:
    r"""``collector_list`` is a list of dictionaries of the form
        {'name': logZ, 'estimator': 'SMC_like'}. It can also contain
        custom objects satisfying CollectorProtocol
    """
    res = []
    for element in collector_list:
        if is_collector(element):
            res.append(element)
        else:
            element = element.copy()
            try:
                name = element.pop('name')
            except KeyError as exc:
                raise ValueError("collector config must include a 'name' field") from exc
            try:
                collector_factory = all_collectors[name]
            except KeyError as exc:
                known = ', '.join(sorted(all_collectors))
                raise ValueError(f"Unknown collector '{name}'. Known collectors: {known}") from exc
            try:
                res.append(collector_factory(**element))
            except TypeError as exc:
                raise ValueError(f"Invalid config for collector '{name}': {exc}") from exc
    return res

@add_collector
def logZ(estimator: str) -> Collector:
    if not hasattr(logZ_estimators, estimator):
        raise ValueError(f"Unknown logZ estimator '{estimator}'")

    def func(key, out: SimpleExploreResult, problem: PTProblem,
             temperatures: Array, engine: PTMCMCEngine) -> Float:
        dummy_use(problem, engine)
        return estimate_logZ(key, out.new_state, temperatures,
                             getattr(logZ_estimators, estimator))
    return Collector(unique_name='logZ_' + estimator, after_swap=False,
                     at_every_iter=True, func=func)

@add_collector
def mcmc_acc_rates() -> Collector:
    r"""Collect mean MCMC acceptance rate at each temperature. Shape: (T,)."""
    def func(key, out: SimpleExploreResult, problem, temperatures, engine) -> Array:
        dummy_use(key, problem, temperatures, engine)
        state = out.new_state
        state.assert_contiguous()
        arr = state.reshape_to_T_N(state.X.accepted, state.T, state.N)
        res = jnp.mean(arr, axis=1)
        assert res.shape == (state.T, )
        return res
    return Collector(unique_name='mcmc_acc_rates', after_swap=False,
                     at_every_iter=True, func=func)

@add_collector
def niter() -> Collector:
    r"""Collect the number of MCMC steps used in the exploration step."""
    def func(key, out: SimpleExploreResult, problem, temperatures, engine) -> Int:
        dummy_use(key, problem, temperatures, engine)
        return out.niter
    return Collector(unique_name='mcmc_niter', after_swap=False,
                     at_every_iter=True, func=func)

@add_collector
def mcmc_parameters(at_every_iter: bool) -> Collector:
    r"""Collect adapted MCMC parameters (covariance, step size, etc.).
    Set ``at_every_iter=False`` to collect only at the last iteration
    (saves memory for large parameter objects).
    """
    def func(key, out: SimpleExploreResult, problem, temperatures, engine):
        dummy_use(key, problem, temperatures, engine)
        return out.mcmc_params
    return Collector(unique_name='mcmc_parameters', after_swap=False,
                     at_every_iter=at_every_iter, func=func)

@add_collector
def swap_rejection_rates() -> Collector:
    r"""Collect swap rejection rates between adjacent temperatures. Shape: (T-1,).
    Collected after the swap step.
    """
    def func(key, out: SwapResult, problem, temperatures, engine) -> Array:
        dummy_use(key, problem, temperatures, engine)
        return out.rej_rates
    return Collector(unique_name='swap_rejection_rates', after_swap=True,
                     at_every_iter=True, func=func)

@add_collector
def index_matrix() -> Collector:
    r"""Collect the full Z index matrix after swaps. Shape: (T, N).
    Disabled by default in ``default_collectors`` due to high memory cost.
    Collected after the swap step.
    """
    def func(key, out: SwapResult, problem, temperatures, engine) -> Array:
        dummy_use(key, problem, temperatures, engine)
        return out.new_state.Z
    return Collector(unique_name='index_matrix', after_swap=True,
                     at_every_iter=True, func=func)
