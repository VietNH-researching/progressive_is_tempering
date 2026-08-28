from functools import partial
from mppt.abstract.mcmc import PTMCMCEngine
from mppt.impl.pt_mcmc_hmc import hmc_pt_mcmc
from mppt.impl.pt_mcmc_random_walk import rw_pt_mcmc

all_engines = {}

def register_engine(e):
    r"""Decorator that registers an engine factory by its function name.
    Registered engines can be selected via ``parse_engine({'name': ...})``.
    """
    all_engines[e.__name__] = e
    return e

def parse_engine(engine: dict) -> PTMCMCEngine:
    r"""Look up and instantiate an MCMC engine from a config dict.
    The ``'name'`` key selects the engine; remaining keys are passed as kwargs.
    """
    engine = engine.copy()
    try:
        name = engine.pop('name')
    except KeyError as exc:
        raise ValueError("engine config must include a 'name' field") from exc
    try:
        engine_factory = all_engines[name]
    except KeyError as exc:
        known = ', '.join(sorted(all_engines))
        raise ValueError(f"Unknown engine '{name}'. Known engines: {known}") from exc
    try:
        return engine_factory(**engine)
    except TypeError as exc:
        raise ValueError(f"Invalid config for engine '{name}': {exc}") from exc

@register_engine
def random_walk(maxniter: int = 100, identity_kernel=False,
                use_bisection=False) -> PTMCMCEngine:
    r"""Random walk Metropolis-Hastings engine.
    :param maxniter: max MCMC steps per exploration when using adaptive nsteps
    :param identity_kernel: if True, skip MCMC moves entirely (particles stay
    fixed). Debugging tool for testing swap mechanics in isolation.
    :param use_bisection: if True, tune step size by bisection to target 0.2-0.35
    acceptance; if False, use the 2.38/sqrt(d) rule
    """
    # noinspection PyTypeChecker
    return partial(rw_pt_mcmc, nsteps_max=maxniter, identity_kernel=identity_kernel,
                   use_bisection=use_bisection)

@register_engine
def hmc(refresh_rate: float = None):
    r"""HMC engine
    :param refresh_rate: velocity refreshment rate. Velocity is roughly completely
    refreshed after O(1/refresh_rate) steps. Default to 0.07 if not specified.
    """
    return partial(hmc_pt_mcmc, default_refresh_rate=refresh_rate)

@register_engine
def mala():
    r"""MALA (Metropolis-Adjusted Langevin Algorithm) engine. Equivalent to
    HMC with refresh_rate=1 (full velocity refresh every step) and acceptance rate
    tuned to the 0.4--0.65 range instead of HMC's 0.8--0.95.
    """
    return partial(hmc_pt_mcmc, mala=True)
