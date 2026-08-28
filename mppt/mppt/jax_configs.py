r"""JAX environment configuration utilities.

These functions set environment variables and JAX config flags. They must be
called **before** any JAX import or operation to take effect (JAX reads env
vars at import time). Test files call these on their first line for this reason.
"""
import os

def x64():
    r"""Enable 64-bit floating point precision. Without this, JAX defaults to
    32-bit which can cause numerical issues in MCMC acceptance ratios and
    log-density calculations.
    """
    os.environ['JAX_ENABLE_X64'] = 'True'
    import jax
    jax.config.update('jax_enable_x64', True)

def debug_nan():
    r"""Enable NaN detection. When active, JAX raises an error immediately
    when a NaN is produced, rather than silently propagating it. Useful for
    debugging numerical instabilities in MCMC kernels.
    """
    os.environ['JAX_DEBUG_NANS'] = 'True'
    import jax
    jax.config.update('jax_debug_nans', True)

def _gpu_memory_saver():
    r"""Disable GPU memory pre-allocation. By default JAX pre-allocates most
    GPU memory at startup. This disables that behavior, letting memory grow
    on demand (slower but avoids OOM when running multiple processes).
    """
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def no_async():
    r"""Disable asynchronous dispatch on CPU. Makes execution synchronous,
    which simplifies debugging (errors appear at the call site rather than
    later) but reduces throughput.
    """
    os.environ['JAX_CPU_ENABLE_ASYNC_DISPATCH'] = 'False'
    import jax
    jax.config.update('jax_cpu_enable_async_dispatch', False)

def no_frame_filter():
    r"""Show full JAX stack traces instead of filtered ones. Useful for
    debugging JIT compilation errors where the filtered trace hides the
    actual source location.
    """
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

def chexify_no_async(*args, **kwargs):
    r"""Wrap a function with chex's runtime assertion checking in synchronous
    mode. The wrapped function will check index bounds and user assertions
    after each call. Significant performance cost; use for debugging only.
    """
    import chex
    from jax.experimental.checkify import index_checks, user_checks
    return chex.chexify(*args, **kwargs, async_check=False,
                        errors=index_checks | user_checks)

def chex_runtime_assert(x):
    r"""Assert that x is True at runtime inside JIT-traced code. Requires
    the calling function to be wrapped with ``chexify_no_async`` or to be
    called outside of JIT. Guard calls with ``if ALLOW_RUNTIME_CHECK:``.
    """
    import chex
    from jax import numpy as jnp
    return chex.assert_trees_all_equal(jnp.array(x), jnp.array(True))

class _AllowRuntimeCheck:
    r"""Global flag controlling whether chex runtime assertions are active.
    Default is False (off). When True, ``chex_runtime_assert`` calls scattered
    throughout the codebase become active, which severely degrades performance
    but catches shape/value errors inside JIT.

    Usage::

        if ALLOW_RUNTIME_CHECK:
            chex_runtime_assert(...)
    """
    def __init__(self, allow=False):
        self._allow = allow

    def set_allow(self, allow):
        self._allow = allow

    def __bool__(self):
        return self._allow

ALLOW_RUNTIME_CHECK = _AllowRuntimeCheck()

def _set_allow_runtime_check(allow: bool):
    r"""Set the global ``ALLOW_RUNTIME_CHECK`` flag."""
    ALLOW_RUNTIME_CHECK.set_allow(allow)

class allow_runtime_check:
    r"""Context manager to temporarily enable or disable runtime checks.

    Usage::

        with allow_runtime_check(True):
            # chex assertions are active here
            ...
        # reverts to previous state
    """
    def __init__(self, allow: bool):
        self._allow = allow
        self._current_state = bool(ALLOW_RUNTIME_CHECK)

    def __enter__(self):
        _set_allow_runtime_check(self._allow)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_allow_runtime_check(self._current_state)

def numpy_raise_errors():
    r"""Configure numpy to raise exceptions on divide-by-zero, overflow, and
    invalid operations instead of silently producing inf/nan.
    """
    import numpy as np
    np.seterr(divide='raise', over='raise', invalid='raise')

def _get_gpu_memory():
    r"""Query free GPU memory in MB for each device via nvidia-smi.
    Returns a list of integers, one per GPU, ordered by device ID.
    """
    import subprocess
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv'])
    lines = out.split(b'\n')[1:-1]
    return [int(line.split()[0]) for line in lines]

def _choose_best_default_device():
    r"""Set JAX's default device to the GPU with the most free memory.
    Falls back silently if nvidia-smi is not available (e.g., CPU-only).
    """
    try:
        mem = _get_gpu_memory()
    except FileNotFoundError:
        return
    idx = mem.index(max(mem))
    import jax
    devices = jax.devices()
    for dev in devices:
        if dev.id == idx:
            jax.config.update('jax_default_device', dev)

def default_gpu_memory_setting():
    r"""Standard GPU setup: disable memory pre-allocation and select the
    GPU with the most free memory. Called at the top of test files.
    """
    _gpu_memory_saver()
    _choose_best_default_device()
