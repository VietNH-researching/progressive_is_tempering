r"""Swapper configuration parsing.

Translates the user-facing ``swapper`` dict (e.g. ``{'mix': True}``) into
a configured swapper callable for use in ``pt_swap``.
"""
from functools import partial
from mppt.impl.swappers import basic_swapper


def parse_swappers(*, mix: bool):
    r"""Return a swapper function with the ``mix`` parameter pre-applied.

    :param mix: if True, randomly permute particles before pairing for swaps.
    This breaks correlation between particle indices and swap partners,
    improving mixing when N is large. If False, particles are paired in
    their current index order.
    """
    return partial(basic_swapper, mix=mix)
