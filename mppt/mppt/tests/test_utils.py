from mppt.jax_configs import default_gpu_memory_setting; default_gpu_memory_setting()
from functools import partial
from jax import jit
from mppt.utils import WhileLoop, while_loops_executor, bisection
import numpy as np
from mppt.jax_configs import allow_runtime_check, chexify_no_async

def _create_while_loop(start, delta, step, injax=False):
    def cond(x):
        return x != (start + step * delta)
    def body(x):
        return x + delta
    def extractor(x):
        return x ** 2
    res = WhileLoop(cond, body, start, extractor)
    if not injax:
        assert res.execute() == (start + step * delta) ** 2
        print([_prettyprint(start + delta * s) for s in range(step+1)])
    return res

def _prettyprint(x):
    if x < 10:
        return str('0{}'.format(x))
    else:
        return str(x)

def _WL_manual_answer(start, delta, step, query_step):
    if query_step < step:
        return (start + delta * query_step) ** 2, False
    else:
        return (start + delta * step) ** 2, True

def _f_internal(x):
    return _create_while_loop(x[0], x[1], x[2], True)

_wle = jit(partial(while_loops_executor, _f_internal))

def _test_while_loops_executor(starts, deltas, steps, proportion, verbose):
    starts = np.array(starts, dtype=int)
    deltas = np.array(deltas, dtype=int)
    steps = np.array(steps, dtype=int)
    nthreads = len(starts)
    if verbose:
        print('')
        for sta, delt, ste in zip(starts, deltas, steps):
            _create_while_loop(sta, delt, ste, False)

    # res = while_loops_executor(_f_internal, (starts, deltas, steps), proportion)
    res = _wle((starts, deltas, steps), proportion)

    ntop = int(np.ceil(proportion * nthreads))
    finished_at = steps[np.argsort(steps)[ntop-1]] if nthreads > 0 else -1
    manual_ans = [_WL_manual_answer(sta, delt, ste, finished_at)
                  for sta, delt, ste in zip(starts, deltas, steps)]
    expected_res1 = [e[0] for e in manual_ans]
    expected_res2 = [e[1] for e in manual_ans]

    if verbose:
        print('Expected res1\n', expected_res1)
        print('Obtained res1\n', res[0])
        print('Expected res2\n', expected_res2)
        print('Obtained res2\n', res[1])

    assert np.array_equal(res[0], expected_res1)
    assert np.array_equal(res[1], expected_res2)

def test_while_loops_executor():
    # Fixed cases cover empty batches, zero-step loops, tied stopping times, and
    # mixed long/short loops. The oracle is the manually computed stop frontier.
    _test_while_loops_executor([], [], [], 0.5, False)
    _test_while_loops_executor([0], [3], [0], 1.0, False)
    _test_while_loops_executor([0, 2, 5, 7], [1, 2, 3, 4], [0, 1, 1, 3], 0.5, False)
    _test_while_loops_executor([9, 1, 4, 0, 6], [2, 5, 1, 3, 4], [8, 2, 5, 2, 0], 0.6, False)
    _test_while_loops_executor(
        [0, 1, 2, 3, 4, 5, 6],
        [1, 1, 2, 2, 3, 3, 4],
        [12, 3, 25, 1, 0, 7, 19],
        0.75,
        False,
    )

def _test_bisection(xleft, xright, tol, K, increasing, verbose):
    if increasing:
        func = lambda x: x - K
    else:
        func = lambda x: K - x
    if not verbose:
        bisection_c = chexify_no_async(bisection)
    else:
        bisection_c = bisection
    with allow_runtime_check(True):
        res = bisection_c(func, xleft, xright, tol, 100, verbose=verbose)
    assert abs(res - K) <= 2 * tol

def test_bisection():
    dt = np.sqrt(2)

    _test_bisection(92., 93+dt, 0.01, 100, True, False)
    _test_bisection(92., 108+dt, 0.01, 100, True, False)
    _test_bisection(108., 110., 0.01, 100, True, False)

    _test_bisection(92., 93+dt, 0.01, 100, False, False)
    _test_bisection(92., 108+dt, 0.01, 100, False, False)
    _test_bisection(108., 110., 0.01, 100, False, False)
