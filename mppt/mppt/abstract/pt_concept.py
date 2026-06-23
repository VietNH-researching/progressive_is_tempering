r"""Defines the parallel tempering problem and the state of the algorithm.
"""
from functools import partial
from typing import NamedTuple, Any, Callable, Protocol, Union
from jax import Array, numpy as jnp, lax, random, tree as jax_tree
from jax.typing import ArrayLike
from mppt.jax_configs import ALLOW_RUNTIME_CHECK, chex_runtime_assert
from mppt.utils import pytree_length, dummy_use

PyTree = Any
Key = Any
RdVector = Array
Float = Union[float, ArrayLike]

class PTState(NamedTuple):
    r"""Describes the current state of the parallel tempering algorithm.
    Parallel tempering maintains K sets of particles at K different temperatures.
    Each set has N particles.

    :param X: an object of length K * N, representing the K * N particles. For
    example, in a problem of dimension d, this could be a JAX array of size
    (K * N, d). More frequently though, X is a NamedTuple of multiple attributes,
    each of which has shape (K * N, ...). See class `PTParticles`
    :param Z: a JAX array of shape (K, N) where the k-th row contains indices
    of elements in X which are at the k-th temperature. The idea is that swap
    operations do not change X, but only change Z.
    """
    X: PyTree
    Z: Array

    def validate_shape(self, temperatures: Array = None):
        r"""Assert that X has length T*N and Z has shape (T, N). When
        runtime checks are on, also verify Z is a valid permutation of
        0..T*N-1. Returns self for chaining.
        """
        T, N = self.Z.shape
        assert pytree_length(self.X) == T * N
        if temperatures is not None:
            t, = temperatures.shape
            assert T == t
        if ALLOW_RUNTIME_CHECK:
            Z_sorted = jnp.sort(self.Z.ravel())
            chex_runtime_assert(jnp.array_equal(Z_sorted, jnp.arange(T * N)))
        return self

    @property
    def N(self):
        return self.Z.shape[1]

    @property
    def T(self):
        return self.Z.shape[0]

    def assert_contiguous(self):
        r"""Asserts that the Z attribute is contiguous
        """
        if ALLOW_RUNTIME_CHECK:
            chex_runtime_assert(jnp.array_equal(
                self.Z.ravel(), jnp.arange(self.T * self.N)
            ))

    @staticmethod
    def _reshape_to_T_N(arr, T, N):
        r"""Reshape an array of size (T * N, ...) into (T, N, ...)
        """
        old_shape = list(arr.shape)
        assert old_shape[0] == T * N
        new_shape = [T, N] + old_shape[1:]
        return jnp.reshape(arr, new_shape)

    @classmethod
    def reshape_to_T_N(cls, tree, T, N):
        reshaper = partial(cls._reshape_to_T_N, T=T, N=N)
        return jax_tree.map(reshaper, tree)

    @staticmethod
    def _flatten_first_two_dims(arr):
        r"""Reshape an array of size (T, N, ...) into (T * N, ...)
        """
        old_shape = list(arr.shape)
        new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
        return jnp.reshape(arr, new_shape)

    @classmethod
    def flatten_first_two_dims(cls, tree):
        return jax_tree.map(cls._flatten_first_two_dims, tree)

class PTProblem(NamedTuple):
    r"""Define a parallel tempering problem.

    The algorithm anneals between a tractable reference distribution (at
    temperature t=0) and the target distribution (at t=1) via geometric
    interpolation: log pi_t(x) = (1-t)*logpdf_ref(x) + t*logpdf_final_target(x).

    :param rvs_ref: draw one sample from the reference distribution.
        Signature: ``key -> x`` where ``x`` has shape ``(d,)``.
    :param logpdf_ref: log-density of the reference distribution. Must be
        normalized (i.e. integrates to 1). Signature: ``x -> scalar``.
    :param logpdf_final_target: log-density of the target distribution.
        Can be unnormalized. Signature: ``x -> scalar``.

    All three functions must operate on vectors of the same dimension d
    and be compatible with JAX transformations (jit, grad, vmap).

    Example::

        problem = PTProblem(
            rvs_ref=lambda key: random.normal(key, shape=(d,)),
            logpdf_ref=lambda x: -0.5 * jnp.sum(x**2) - 0.5 * d * jnp.log(2*jnp.pi),
            logpdf_final_target=my_target_logpdf,
        )
    """
    rvs_ref: Callable[[Key], RdVector]
    logpdf_ref: Callable[[RdVector], Float]
    logpdf_final_target: Callable[[RdVector], Float]

class PTParticle(Protocol):
    r"""Describes a particle in the parallel tempering algorithm. Below is the
    minimum attributes, specific implementations might add others.

    Warning: `logpdf_final_target` is self-explanatory, whereas `logpdf_target`
    refers to the **current** temperature. This implementation assumes that the
    temperature lies between 0 and 1, with 0 corresponding to the reference
    distribution and 1 the target.
    """
    position: RdVector
    temperature: Float
    temperature_idx: int
    logpdf_ref: Float
    logpdf_final_target: Float
    logpdf_target: Float  # normally equals to logpdf_ref * (1-temperature) +
    # + logpdf_final_target * temperature

class PTParticles(Protocol):
    r"""Describes a pytree bundling multiple PTParticle
    """
    position: Array
    temperature: Array
    temperature_idx: Array
    logpdf_ref: Array
    logpdf_final_target: Array
    logpdf_target: Array

def geometric_anneal(t, logpdf_0, logpdf_1):
    r"""Geometric annealing: ``(1-t) * logpdf_0 + t * logpdf_1``.

    Wrapped in a named function so that all uses of this specific annealing
    schedule can be found and replaced if a different schedule is needed.
    """
    return (1-t) * logpdf_0 + t * logpdf_1

class TwoLevelSwapper(Protocol):
    r"""Template for a swapper between two levels. Concrete implementations are
    given in ``impl/swappers.py``
    """
    def __call__(self, key, log_pi0: Array, log_pi1: Array,
                 perm: Array) -> 'TwoLevelSwapperResult':
        r"""Swaps particles between two distributions ``pi_0`` and ``pi_1``
        :param log_pi0: array of size `(2N, )` containing logpdf of pi_0 evaluated
        on the 2N particles
        :param log_pi1: array of size `(2N, )` containing logpdf of pi_1 evaluated
        on the 2N particles
        :param perm: a permutation of {1, 2, ..., 2N} such that the first N elements
        give indices of particles following pi_0 and the last N elements give indices
        of particles following pi_1.
        :return: A ``TwoLevelSwapperResult`` object containing the following
        attributes::
            * perm: a new permutation obtained by applying MCMC on the passed ``perm``
            * rejection_rate: an estimate of the swap rejection rate
        """
        ...

    @staticmethod
    def shape_check(log_pi0: Array, log_pi1: Array, perm: Array) -> None:
        r"""Checks shapes of the arguments
        """
        assert log_pi0.shape == log_pi1.shape == perm.shape
        two_N, = log_pi0.shape
        assert two_N % 2 == 0
        if ALLOW_RUNTIME_CHECK:
            chex_runtime_assert(jnp.array_equal(
                jnp.sort(perm), jnp.arange(two_N)
            ))

class TwoLevelSwapperResult(NamedTuple):
    perm: Array
    rejection_rate: Float

def repeated_swap(key, log_pi0, log_pi1, perm, nrepeats, swapper: TwoLevelSwapper) -> TwoLevelSwapperResult:
    r"""Applies a swapper repeatedly on a permutation
    """
    init_val = {'key': key, 'res': TwoLevelSwapperResult(perm, 1.0)}
    body_func = partial(_repeated_swap_body_func, log_pi0=log_pi0, log_pi1=log_pi1,
                        swapper=swapper)
    return lax.fori_loop(0, nrepeats, body_func, init_val)['res']

def _repeated_swap_body_func(i, val, log_pi0, log_pi1, swapper):
    dummy_use(i)
    key, subkey = random.split(val['key'])
    res = swapper(subkey, log_pi0, log_pi1, val['res'].perm)
    return {'key': key, 'res': res}

class LogZ_Estimator(Protocol):
    r"""Template for an estimator of log Z. Concrete implementations are
    found in ``impl/logZ_estimators.py``
    """
    def __call__(self, key, logpdf_ref: Array, logpdf_final_target: Array,
                 temperatures: Array) -> Float:
        r"""Estimates the logZ
        :param logpdf_ref: an array of shape (T, N) containing the logpdf of the
        reference evaluated at all particles
        :param logpdf_final_target: an array of shape (T, N) containing the logpdf
        of the final target evaluated at all particles
        :param temperatures: the array of temperatures, starting at 0 and ending
        at 1
        :return: the estimate of logZ.
        """
        ...

    @staticmethod
    def shape_check(logpdf_ref, logpdf_final_target, temperatures) -> tuple[int, int]:
        T, N = logpdf_ref.shape
        assert logpdf_final_target.shape == (T, N)
        assert temperatures.shape == (T, )
        if ALLOW_RUNTIME_CHECK:
            chex_runtime_assert(jnp.allclose(temperatures[0], 0))
            chex_runtime_assert(jnp.allclose(temperatures[-1], 1))
        return T, N