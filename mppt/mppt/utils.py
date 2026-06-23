from __future__ import annotations
import time
from collections import namedtuple
from functools import partial
from jax import numpy as jnp, random, jit, block_until_ready, lax, Array, vmap
from secrets import randbits
import jax
import chex
from typing import Any, Callable, List, Union, NamedTuple, Tuple
import numpy as np
from tqdm import tqdm
from itertools import count
from mppt.jax_configs import chexify_no_async, allow_runtime_check, ALLOW_RUNTIME_CHECK, chex_runtime_assert


def follow_protocol(*args, **kwargs):
    r"""Decorates that a class or function follows a protocol.
    Has no real effect (returns the identity function).
    """
    # assert isinstance(p, type)
    dummy_use(args, kwargs)
    def id_func(obj): return obj
    return id_func

def has_nan(x):
    return jnp.sum(jnp.isnan(x)) >= 1

def new_jax_key():
    return random.key(randbits(63))

# noinspection PyUnusedLocal
def dummy_use(*args, **kwargs):
    r"""Prevents type checker warning about unused variables
    """
    return

def pytree_length(tree):
    r"""Returns the ``length`` of a pytree, after asserting that all leaves have
    the same length
    """
    leaves = jax.tree.leaves(tree)
    chex.assert_equal_shape_prefix(leaves, 1)
    return len(leaves[0])

def is_diagonal(mat):
    return jnp.allclose(mat - jnp.diag(jnp.diag(mat)), 0)

def random_split(key, N):
    keys = jax.random.split(key, N)
    return keys[0], keys[1:]

def set_named_tuple_attr(x, val, attr='accepted'):
    r"""Returns a namedtuple of the same content as x, but with the ``accepted``
    attribute having the specified value
    """
    assert hasattr(x, attr)
    y = dict(x._asdict())
    y[attr] = val
    return type(x)(**y)

def jit_with_info(func, jitted=False, compile_options: dict = None):
    r"""Returns a jitted function which displays compilation and evaluation steps.
    """
    if compile_options is None:
        compile_options = {}
    func = jit(func, **compile_options) if not jitted else func
    compiled: List[Union[Callable, None]] = [None]  # list is modifiable

    def returned_func(*a, **k):
        try:
            return block_until_ready(compiled[0](*a, **k))
        except TypeError:
            with run_time(f'Compiling {func.__name__}'):
                compiled[0] = func.lower(*a, **k).compile()
                block_until_ready(compiled[0])
            with run_time(f'Running {func.__name__}'):
                return block_until_ready(compiled[0](*a, **k))

    return returned_func

class run_time:
    r"""Context manager to print run time.
    Usage:
        with run_time('Running my function...'):
            my_function()
    """
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        self.start = time.time()
        print(self.text, end='', flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if exc_type is None:
            print(f' ({elapsed:.1f}s)')
        else:
            print(f' (crashed after {elapsed:.1f}s)')

class add_run_time_to:
    r"""Context manager that adds elapsed time to a numpy array in-place.
    Useful for accumulating timing across multiple calls.
    """
    def __init__(self, x):
        if not isinstance(x, np.ndarray):
            raise ValueError('x must be a numpy array')
        self.x = x

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.x += self.end - self.start

class WhileLoop(NamedTuple):
    r"""Represents components of a while loop that is parsed to ``jax.lax.while_loop``.
    This class is used to represent an adaptor that has unknown execution time
    (for instance, adapting the number of MCMC steps). The execution can be triggered
    by the ``execute`` method, but objects of this class can also be gathered to
    form a custom loop (for example, given 100 different ``WhileLoop`` objects,
    run them until half of them are completed).
    """
    cond_func: Callable[[T], bool]
    body_func: Callable[[T], T]
    init_val: T
    extractor: Callable[[T], R]

    def execute(self) -> R:
        return self.extractor(
            lax.while_loop(self.cond_func, self.body_func, self.init_val))

    def verbose_execute(self, print_mode=False, print_func=lambda e: e) -> R:
        cond_func = jit(self.cond_func)
        body_func = jit(self.body_func)
        if ALLOW_RUNTIME_CHECK:
            cond_func = chexify_no_async(cond_func)
            body_func = chexify_no_async(body_func)
        iterator = iter(tqdm(count(), disable=print_mode))

        val = self.init_val
        if print_mode: print(print_func(val))
        while cond_func(val):
            val = body_func(val)
            if print_mode: print(print_func(val))
            next(iterator)
        return self.extractor(val)

def while_loops_executor(
        f: Callable[[pytree], WhileLoop[T, R]], x: pytree, proportion: float
) -> Tuple[BatchedR, Array]:
    r"""Executes multiple ``WhileLoop`` objects until a proportion of them has
    finished.
    :param f: function that sends a pytree to a WhileLoop object
    :param x: a batched pytree that will be sent to a vectorized version of f
    :param proportion: float between 0 and 1
    :return: a tuple whose first element is the batched output and second element
    is an array of booleans indicating whether the individual loops have finished.
    """
    OuterVal = namedtuple('OuterVal', ['inner_val', 'cond', 'extracted'])

    @vmap
    def init_val(e):
        wl = f(e)
        inner_init = wl.init_val
        return OuterVal(inner_init, wl.cond_func(inner_init),
                        wl.extractor(inner_init))

    @vmap
    def body_func(e, ov: OuterVal):
        def if_true():
            wl = f(e)
            next_iv = wl.body_func(ov.inner_val)
            return OuterVal(next_iv, wl.cond_func(next_iv),
                            wl.extractor(next_iv))
        def if_false():
            return ov
        return lax.cond(ov.cond, if_true, if_false)

    def cond_func(ov):
        return jnp.mean(1 - ov.cond) < proportion

    # noinspection PyTypeChecker
    raw_res = lax.while_loop(cond_func, partial(body_func, x), init_val(x))
    # noinspection PyUnresolvedReferences
    return raw_res.extracted, jnp.logical_not(raw_res.cond)

class temporary_numpy_seed:
    """
    Context handler to temporary set numpy seed to some value, then revert back to original setting once done.

    Usage:
        with temporary_numpy_seed(42):
            ...  # do something
    """
    # tested
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.old_state = np.random.get_state()
        np.random.seed(self.seed)
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        # noinspection PyTypeChecker
        np.random.set_state(self.old_state)

def pytree_index(tree, idx):
    r"""Apply leave[idx] for all leaves of a pytree
    """

    def get_idx(obj):
        fill_value = jnp.nan
        if jnp.issubdtype(obj.dtype, jnp.integer):
            fill_value = jnp.iinfo(jnp.int32).max
        return obj.at[idx].get(mode='fill', fill_value=fill_value)

    return jax.tree.map(get_idx, tree)

def inverse_permutation(p: jax.Array) -> jax.Array:
    """Calculates the inverse of a permutation
    :param p: an array of length N and taking values in [0,N-1]. Should be thought of as a map.
    :return q: an array such that p[q] = q[p] = arange(N)
    """
    res = jnp.zeros(len(p), dtype=int)
    res = res.at[p].set(jnp.arange(len(p)))
    return res

class should_raise:
    r"""Context manager to assert whether a code block raises an error.
    Usage:

    >>> from mppt.utils import should_raise
    >>> with should_raise(ZeroDivisionError):  # should pass
    ...     1/0
    ...
    >>> with should_raise(ZeroDivisionError):  # should not pass
    ...     1+0
    ...
    Traceback (most recent call last):
      ...
    AssertionError
    >>> with should_raise(ZeroDivisionError):  # should not pass
    ...     raise RuntimeError
    ...
    Traceback (most recent call last):
      ...
    AssertionError
    """
    def __init__(self, expected_exc_type, reraise_type=AssertionError):
        if expected_exc_type is None:
            raise ValueError('Expected execption type must not be None')
        self.expected_exc_type = expected_exc_type
        self.reraise_type = reraise_type

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (exc_type is None) or (not issubclass(exc_type, self.expected_exc_type)):
            raise self.reraise_type
        return True

def assert_no_nan(pytree):
    def local_assert(arr):
        if jnp.sum(jnp.isnan(arr)) >= 1:
            raise AssertionError
    jax.tree.map(local_assert, pytree)

def _stack(*args):
    return jnp.stack(args)

def tree_stack(*trees):
    r"""Warning: Compare with ``tree_concatenate``
    """
    if len(trees) == 1:
        print('Warning: make sure you call tree_stack with *')
    return jax.tree.map(_stack, *trees)

def verbose_scan(f, init, xs, length=None, runtime_check=False, compile_options: dict = None):
    r"""Python-loop replacement for ``lax.scan`` with a tqdm progress bar.

    Trade-off: 2-10x slower than ``lax.scan`` because each iteration is a
    separate GPU dispatch (no cross-iteration fusion, no donate_argnums).
    Use ``verbose=False`` in ``parallel_tempering`` for production runs,
    which falls back to ``lax.scan``.
    """
    if compile_options is None:
        compile_options = {}
    with allow_runtime_check(runtime_check):
        if runtime_check:
            f = chexify_no_async(jit(f, **compile_options))
        else:
            f = jit_with_info(f, compile_options=compile_options)
        if xs is None:
            xs = [None] * length
        carry = init
        ys = []
        for x in tqdm(xs):
            carry, y = f(carry, x)
            ys.append(y)
        return carry, tree_stack(*ys) if ys else None

FuncAnimationParameters = namedtuple('FuncAnimationParameters', ['interval', 'frames_idx'])

def func_animation_parameters(niter: int, n_fps: int, video_length: float) -> FuncAnimationParameters:
    r"""Finds the appropriate parameters to pass into ``FuncAnimation``.
    :param niter: number of algorithm iterations
    :param n_fps: desired number of frames per second
    :param video_length: desired length of the video in seconds
    :return: a tuple containing the ``interval`` parameter and the indices of
    the frames in the video (between 0 inclusive and ``niter`` exclusive).
    """
    nframes = int(n_fps * video_length)
    frame_idx = jnp.round(jnp.linspace(0, niter-1, nframes))
    frame_idx = jnp.array(frame_idx, dtype=int)
    return FuncAnimationParameters(1000/n_fps, frame_idx)

_bisection_type = namedtuple('_bisection_type',
                             ['xleft', 'xright', 'fleft', 'fright', 'niter'])

def bisection(f, xleft_, xright_, ftol, max_iter, verbose=False, verbose_offset=0):
    r"""Bisection method to find root of a monotonic function.
    f(xleft) and f(xright) does not need to have opposite signs

    :param verbose_offset: offset the printed value of the function f. Can be useful
    to interpret results when it is more natural to state the problem as finding
    f(x^*) = alpha for some alpha rather than f(x^*) = 0.
    """
    def create(xleft, xright, fleft, fright, niter):
        if ALLOW_RUNTIME_CHECK:
            chex_runtime_assert(xleft <= xright)
            chex_runtime_assert(jnp.allclose(f(xleft), fleft))
            chex_runtime_assert(jnp.allclose(f(xright), fright))
        return _bisection_type(xleft, xright, fleft, fright, niter)

    def loop(s: _bisection_type) -> _bisection_type:
        xleft, xright, fleft, fright, niter = s
        return lax.cond(
            jnp.sign(fleft) * jnp.sign(fright) >= 0,
            lambda: same_sign(s), lambda: different_sign(s)
        )

    def same_sign(s: _bisection_type) -> _bisection_type:
        r"""Action when f_left * f_right > 0
        """
        xleft, xright, fleft, fright, niter = s
        return lax.cond(
            jnp.abs(fleft) >= jnp.abs(fright),
            lambda: create(xright, 2*xright-xleft, fright, f(2*xright-xleft), niter+1),
            lambda: create(2*xleft-xright, xleft, f(2*xleft-xright), fleft, niter+1)
        )

    def different_sign(s: _bisection_type) -> _bisection_type:
        r"""Action when f_left * f_right <= 0
        """
        xleft, xright, fleft, fright, niter = s
        xmid = 0.5 * xleft + 0.5 * xright
        fmid = f(xmid)
        return lax.cond(
            jnp.sign(fleft) * jnp.sign(fmid) <= 0,
            lambda: create(xleft, xmid, fleft, fmid, niter+1),
            lambda: create(xmid, xright, fmid, fright, niter+1)
        )

    def cond(s: _bisection_type) -> Union[bool, Array]:
        cond_1 = jnp.abs(s.fleft) > ftol
        cond_2 = s.niter < max_iter
        return jnp.logical_and(cond_1, cond_2)

    def print_func(s: _bisection_type) -> _bisection_type:
        assert isinstance(s, _bisection_type)
        return s._replace(fleft=s.fleft + verbose_offset,
                          fright=s.fright + verbose_offset)

    init_val = create(xleft_, xright_, f(xleft_), f(xright_), 0)
    while_loop = WhileLoop(cond, loop, init_val, lambda e: e)
    if verbose:
        return while_loop.verbose_execute(print_mode=True,
                                          print_func=print_func).xleft
    else:
        return while_loop.execute().xleft

def is_all_same(x: Array):
    y = x.ravel()
    return jnp.allclose(y - y[0], 0.)