"""
Unit tests for progressive_is_tempering.targets.

Tests cover:
  - GMMTarget log_prob / energy consistency
  - grad_energy correctness (finite-difference check)
  - log_prob sums correctly over dimensions
  - PTProblem interface functions are JAX-traceable
  - make_gmm_target factory defaults
"""

import jax
import jax.numpy as jnp
import pytest

from progressive_is_tempering.targets import GMMTarget, make_gmm_target


@pytest.fixture
def gmm():
    return make_gmm_target()


class TestGMMTarget:
    def test_log_prob_shape(self, gmm):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (gmm.dim,))
        lp = gmm.log_prob(x)
        assert lp.shape == ()

    def test_energy_is_neg_log_prob(self, gmm):
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (gmm.dim,))
        assert jnp.allclose(gmm.energy(x), -gmm.log_prob(x), atol=1e-5)

    def test_grad_energy_finite_difference(self, gmm):
        """∇E(x) from jax.grad should match central finite differences."""
        key = jax.random.PRNGKey(2)
        x = jax.random.normal(key, (gmm.dim,))
        grad = gmm.grad_energy(x)

        eps = 1e-4
        fd_grad = jnp.zeros_like(x)
        for i in range(gmm.dim):
            e_i = jnp.zeros(gmm.dim).at[i].set(1.0)
            fd_grad = fd_grad.at[i].set(
                (gmm.energy(x + eps * e_i) - gmm.energy(x - eps * e_i)) / (2 * eps)
            )
        # float32 FD with eps=1e-4 has ~1e-2 truncation + rounding error
        assert jnp.allclose(grad, fd_grad, atol=0.05), f"max diff: {jnp.max(jnp.abs(grad - fd_grad))}"

    def test_log_prob_factorizes(self, gmm):
        """log π(x) = Σ_i log π_1d(x_i)."""
        key = jax.random.PRNGKey(3)
        x = jax.random.normal(key, (gmm.dim,))
        # compute per-dimension contributions manually
        per_dim = jnp.array([
            jax.nn.logsumexp(
                gmm.log_weights
                - 0.5 * ((x[i] - gmm.means) / gmm.std) ** 2
                - jnp.log(gmm.std)
                - 0.5 * jnp.log(2.0 * jnp.pi)
            )
            for i in range(gmm.dim)
        ])
        assert jnp.allclose(gmm.log_prob(x), jnp.sum(per_dim), atol=1e-5)

    def test_log_prob_finite(self, gmm):
        key = jax.random.PRNGKey(4)
        xs = jax.random.normal(key, (10, gmm.dim))
        lps = jax.vmap(gmm.log_prob)(xs)
        assert jnp.all(jnp.isfinite(lps))

    def test_rvs_ref_shape(self, gmm):
        key = jax.random.PRNGKey(5)
        x = gmm.rvs_ref(key)
        assert x.shape == (gmm.dim,)

    def test_logpdf_ref_matches_gaussian(self, gmm):
        key = jax.random.PRNGKey(6)
        x = jax.random.normal(key, (gmm.dim,))
        import jax.scipy.stats as jss
        expected = jnp.sum(jss.norm.logpdf(x))
        assert jnp.allclose(gmm.logpdf_ref(x), expected, atol=1e-5)

    def test_make_gmm_target_defaults(self):
        gmm = make_gmm_target()
        assert gmm.dim == 16
        assert gmm.std == 0.5
        assert gmm.means.shape == (2,)

    def test_jit_compatible(self, gmm):
        key = jax.random.PRNGKey(7)
        x = jax.random.normal(key, (gmm.dim,))
        jit_lp = jax.jit(gmm.log_prob)(x)
        assert jnp.isfinite(jit_lp)

    def test_vmap_compatible(self, gmm):
        key = jax.random.PRNGKey(8)
        xs = jax.random.normal(key, (5, gmm.dim))
        lps = jax.vmap(gmm.log_prob)(xs)
        assert lps.shape == (5,)