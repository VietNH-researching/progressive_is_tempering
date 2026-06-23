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

def _test_while_loops_executor(nthreads, stepmax, proportion, verbose):
    starts = np.random.randint(low=0, high=10, size=nthreads)
    deltas = np.random.randint(low=1, high=10, size=nthreads)
    steps = np.random.randint(low=0, high=stepmax, size=nthreads)
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
    for _ in range(2000):
        _test_while_loops_executor(
            nthreads=np.random.randint(low=0, high=40),
            stepmax=np.random.randint(low=1, high=5),
            proportion=np.random.rand(), verbose=False
        )
        _test_while_loops_executor(
            nthreads=np.random.randint(low=0, high=40),
            stepmax=np.random.randint(low=1, high=10),
            proportion=np.random.rand(), verbose=False
        )
        _test_while_loops_executor(
            nthreads=np.random.randint(low=0, high=40),
            stepmax=np.random.randint(low=1, high=100),
            proportion=np.random.rand(), verbose=False
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
        res = bisection_c(func, xleft, xright, tol, 10_000, verbose=verbose)
    assert abs(res - K) <= 2 * tol

def test_bisection():
    dt = np.sqrt(2)

    _test_bisection(92., 93+dt, 0.01, 100, True, False)
    _test_bisection(92., 93., 0.01, 100, True, False)
    _test_bisection(92., 108+dt, 0.01, 100, True, False)
    _test_bisection(92., 108., 0.01, 100, True, False)
    _test_bisection(108., 110+dt, 0.01, 100, True, False)
    _test_bisection(108., 110., 0.01, 100, True, False)

    _test_bisection(92., 93+dt, 0.01, 100, False, False)
    _test_bisection(92., 93., 0.01, 100, False, False)
    _test_bisection(92., 108+dt, 0.01, 100, False, False)
    _test_bisection(92., 108., 0.01, 100, False, False)
    _test_bisection(108., 110+dt, 0.01, 100, False, False)
    _test_bisection(108., 110., 0.01, 100, False, False)