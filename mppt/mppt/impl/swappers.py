r"""Implements local swappers between two distributions
"""
from collections import namedtuple
from jax import Array, vmap, numpy as jnp, random, lax
from mppt.abstract.pt_concept import TwoLevelSwapper, TwoLevelSwapperResult
from mppt.utils import follow_protocol
from typing import Union

_DebugSwapperType = namedtuple('_DebugSwapperType', ['perm', 'rejection_rate', 'rejection_rate_naive_A', 'rejection_rate_naive_B'])

@follow_protocol(TwoLevelSwapper)
def basic_swapper(key, log_pi0: Array, log_pi1: Array, perm: Array,
                  debug: bool = False, mix: bool = True) \
        -> Union[TwoLevelSwapperResult, _DebugSwapperType]:
    r"""Swaps particles between two PT levels. See documentation of
    ``TwoLevelSwapper`` for more details.
    """
    TwoLevelSwapper.shape_check(log_pi0, log_pi1, perm)
    N = log_pi0.shape[0] // 2

    key, subkey = random.split(key)
    perm0 = perm[0:N]
    if mix:
        perm1 = random.permutation(subkey, perm[N:])
    else:
        perm1 = perm[N:]
    log_pi0_vals = jnp.array([log_pi0[perm0], log_pi0[perm1]]).T
    log_pi1_vals = jnp.array([log_pi1[perm0], log_pi1[perm1]]).T

    res = _basic_swapper(random.split(key, N), log_pi0_vals, log_pi1_vals, perm0, perm1)
    new_perm = jnp.concatenate([res.perm[0], res.perm[1]])
    rejection_rate = jnp.mean(res.rejection_rate)
    rejection_rate_naive_A = jnp.mean(res.rejection_rate_naive_A)
    rejection_rate_naive_B = jnp.mean(res.rejection_rate_naive_B)
    TwoLevelSwapper.shape_check(log_pi0, log_pi1, new_perm)

    if debug:
        return _DebugSwapperType(new_perm, rejection_rate, rejection_rate_naive_A,
                                 rejection_rate_naive_B)
    else:
        return TwoLevelSwapperResult(new_perm, rejection_rate)

@vmap
def _basic_swapper(key, log_pi0_vals, log_pi1_vals, i0, i1):
    r"""Metropolis swap for one pair of particles. Accepts swap if
    log(u) <= log(pi1[0]/pi0[0]) - log(pi1[1]/pi0[1]).
    Returns three rejection rate estimates: theoretical, empirical, and
    probability-based (via expm1).
    """
    assert log_pi0_vals.shape == log_pi1_vals.shape == (2, )
    assert i0.shape == i1.shape == ()
    assert jnp.issubdtype(i0, jnp.integer) and jnp.issubdtype(i1, jnp.integer)
    diff = log_pi1_vals - log_pi0_vals
    log_p = diff[0] - diff[1]
    log_u = jnp.log(random.uniform(key))
    acc = log_u <= log_p
    log_prob_acc = jnp.minimum(0, log_p)
    rejection_rate = 0.5 * (_aux_func(log_p) + _aux_func(-log_p))
    rrates = rejection_rate, 1 - acc, -jnp.expm1(log_prob_acc)
    return lax.cond(acc,
                    lambda: _DebugSwapperType((i1, i0), *rrates),
                    lambda: _DebugSwapperType((i0, i1), *rrates))

def _aux_func(x):
    r"""Calculates |1 - e^x|/(1 + e^x)
    """
    assert jnp.ndim(x) == 0
    x = -jnp.abs(x)
    return -jnp.expm1(x)/(1 + jnp.exp(x))