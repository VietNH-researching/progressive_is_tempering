r"""Immutable wrappers for default configuration values.

Used for default arguments in ``parallel_tempering`` and ``pt_manual_runners``.
Python's mutable default argument gotcha means a bare ``def f(x={})`` shares
one dict across all calls. These wrappers prevent accidental mutation while
allowing callers to get a mutable copy via ``.copy()`` when they need to
modify the config.
"""

class FrozenDict:
    r"""Read-only dict wrapper. Access values with ``[]``, get a mutable
    copy with ``.copy()``.
    """
    def __init__(self, d: dict):
        self._dict = d

    def __getitem__(self, item):
        return self._dict[item]

    def copy(self) -> dict:
        r"""Return a mutable shallow copy of the underlying dict."""
        return self._dict.copy()

    def keys(self):
        return self._dict.keys()

class FrozenList:
    r"""Read-only list wrapper. Access elements with ``[]``, get a mutable
    copy with ``.copy()``.
    """
    def __init__(self, li: list):
        self._list = li

    def __getitem__(self, item):
        return self._list[item]

    def copy(self) -> list:
        r"""Return a mutable shallow copy of the underlying list."""
        return self._list.copy()
