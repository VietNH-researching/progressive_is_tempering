"""
Unit tests for progressive_is_tempering.net.dit (PITA's DIT3D, Flax port).

Tests cover:
  - Parameter count of the ALDP configuration (3,179,267, as PITA's).
  - Output shape and dtype; exactly zero output at initialization (zero-init
    final layer and adaLN modulations).
  - Dropout: deterministic calls are reproducible, training calls need and use
    a dropout key.
  - Conditioning: after a parameter perturbation, the output depends on the
    noise level and on beta.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progressive_is_tempering.net.dit import DIT3D

B, N = 3, 22


@pytest.fixture(scope="module")
def net():
    return DIT3D()


@pytest.fixture(scope="module")
def params(net):
    return net.init(jax.random.key(0), jnp.zeros(1), jnp.zeros((1, 3 * N)), jnp.zeros(1))


def _inputs(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return (jax.random.normal(k1, (B,)), jax.random.normal(k2, (B, 3 * N)),
            jax.random.uniform(k3, (B,)))


def test_parameter_count(params):
    assert sum(x.size for x in jax.tree_util.tree_leaves(params)) == 3_179_267


def test_shape_and_zero_init(net, params):
    t, x, b = _inputs(jax.random.key(1))
    out = net.apply(params, t, x, b)
    assert out.shape == (B, 3 * N) and out.dtype == jnp.float32
    assert np.all(np.asarray(out) == 0.0)


def _perturb(params, key):
    leaves, tree = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(
        tree, [l + 0.05 * jax.random.normal(k, l.shape, l.dtype) for l, k in zip(leaves, keys)])


def test_dropout_and_conditioning(net, params):
    p = _perturb(params, jax.random.key(2))
    t, x, b = _inputs(jax.random.key(3))
    out = net.apply(p, t, x, b)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(net.apply(p, t, x, b)))
    assert not np.allclose(out, net.apply(p, t + 1.0, x, b))
    assert not np.allclose(out, net.apply(p, t, x, b + 0.5))
    with pytest.raises(Exception):
        net.apply(p, t, x, b, deterministic=False)            # needs a dropout key
    d1 = net.apply(p, t, x, b, deterministic=False, rngs={"dropout": jax.random.key(4)})
    d2 = net.apply(p, t, x, b, deterministic=False, rngs={"dropout": jax.random.key(5)})
    assert not np.allclose(d1, d2)


def test_jit(net, params):
    t, x, b = _inputs(jax.random.key(6))
    np.testing.assert_array_equal(np.asarray(jax.jit(net.apply)(params, t, x, b)),
                                  np.asarray(net.apply(params, t, x, b)))
