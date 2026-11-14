"""
Unit tests for progressive_is_tempering.targets.

Tests cover:
  - GMMTarget log_prob / energy consistency
  - grad_energy correctness (finite-difference check)
  - log_prob sums correctly over dimensions
  - PTProblem interface functions are JAX-traceable
  - make_gmm_target factory defaults
  - DW4 / LennardJones pair energies against hand-computed values, traps,
    translation invariance, batching and input validation
"""

import jax
import jax.numpy as jnp
import pytest

from progressive_is_tempering.targets import DW4, GMMTarget, LennardJones, make_gmm_target


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

def _two_particle_config(n_particles, spatial_dim, r, far=1e3):
    """Particle 0 at the origin, particle 1 at distance r, the rest far away.

    `far` makes every other pair term negligible for decaying potentials.
    """
    pos = jnp.zeros((n_particles, spatial_dim))
    pos = pos.at[1, 0].set(r)
    for i in range(2, n_particles):
        pos = pos.at[i, 0].set(far * i)
    return pos


class TestDW4:
    def test_dim(self):
        assert DW4().dim == 8

    def test_pair_potential_single_distance(self):
        """All six pairs at the same distance r: U = 6 [a (r-r0)^4 + b (r-r0)^2 + c]."""
        target = DW4(a=0.9, b=-4.0, c=0.5, r0=4.0, trap_scale=0.0)
        # square of side s: 4 edges at s, 2 diagonals at s·√2
        s = 3.0
        pos = jnp.array([[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]])
        pair = lambda r: 0.9 * (r - 4.0) ** 4 - 4.0 * (r - 4.0) ** 2 + 0.5
        expected = -(4 * pair(s) + 2 * pair(s * jnp.sqrt(2.0)))
        assert jnp.allclose(target(pos), expected, rtol=1e-5)

    def test_trap_individual(self):
        """Trap adds (trap_scale/2) Σ_i ||x_i||² to the energy."""
        pos = jax.random.normal(jax.random.PRNGKey(0), (4, 2))
        no_trap = DW4(trap_scale=0.0)(pos)
        trapped = DW4(trap_scale=2.0, trap_mode="individual")(pos)
        assert jnp.allclose(no_trap - trapped, jnp.sum(pos ** 2), rtol=1e-5)

    def test_trap_com(self):
        """'com' trap adds (trap_scale/2) N ||CoM||² to the energy."""
        pos = jax.random.normal(jax.random.PRNGKey(1), (4, 2))
        no_trap = DW4(trap_scale=0.0)(pos)
        trapped = DW4(trap_scale=2.0, trap_mode="com")(pos)
        expected = 4 * jnp.sum(pos.mean(axis=0) ** 2)
        assert jnp.allclose(no_trap - trapped, expected, rtol=1e-5)

    def test_translation_invariant_without_trap(self):
        target = DW4(trap_scale=0.0)
        pos = jax.random.normal(jax.random.PRNGKey(2), (4, 2))
        shift = jnp.array([3.0, -1.5])
        assert jnp.allclose(target(pos), target(pos + shift), rtol=1e-5)

    def test_beta_scales_log_density(self):
        pos = jax.random.normal(jax.random.PRNGKey(3), (4, 2))
        assert jnp.allclose(DW4(beta=0.5)(pos), 0.5 * DW4(beta=1.0)(pos), rtol=1e-5)

    def test_batched_and_jit(self):
        target = DW4()
        xs = jax.random.normal(jax.random.PRNGKey(4), (7, 4, 2))
        batched = target(xs)
        assert batched.shape == (7,)
        assert jnp.allclose(batched, jax.vmap(target)(xs), rtol=1e-6)
        assert jnp.allclose(batched, jax.jit(target)(xs), rtol=1e-6)

    def test_grad_finite_at_coincident_particles(self):
        """The 1e-8 floor keeps ∇ finite when particles coincide."""
        grad = jax.grad(DW4(trap_scale=0.0))(jnp.zeros((4, 2)))
        assert jnp.all(jnp.isfinite(grad))

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            DW4()(jnp.zeros((8,)))

    @pytest.mark.parametrize("kwargs", [{"beta": 0.0}, {"trap_mode": "bad"}])
    def test_rejects_bad_params(self, kwargs):
        with pytest.raises(ValueError):
            DW4(**kwargs)


class TestLennardJones:
    def test_dim(self):
        assert LennardJones().dim == 39
        assert LennardJones(n_particles=55).dim == 165

    def test_pair_minimum_is_minus_epsilon(self):
        """A single pair at r = rm contributes exactly -ε."""
        target = LennardJones(n_particles=2, spatial_dim=3, epsilon=2.0, rm=1.3, trap_scale=0.0)
        pos = _two_particle_config(2, 3, 1.3)
        assert jnp.allclose(target(pos), 2.0, rtol=1e-5)

    def test_pair_potential_value(self):
        target = LennardJones(n_particles=2, spatial_dim=3, epsilon=2.0, rm=1.0, trap_scale=0.0)
        r = 1.7
        expected = -2.0 * ((1.0 / r) ** 12 - 2.0 * (1.0 / r) ** 6)
        assert jnp.allclose(target(_two_particle_config(2, 3, r)), expected, rtol=1e-5)

    def test_distant_pairs_negligible(self):
        """With the other particles far away, only the close pair contributes."""
        target = LennardJones(n_particles=13, spatial_dim=3, epsilon=2.0, rm=1.0, trap_scale=0.0)
        assert jnp.allclose(target(_two_particle_config(13, 3, 1.0)), 2.0, rtol=1e-4)

    def test_trap_individual(self):
        pos = 2.0 * jax.random.normal(jax.random.PRNGKey(0), (13, 3))
        no_trap = LennardJones(trap_scale=0.0)(pos)
        trapped = LennardJones(trap_scale=1.0)(pos)
        assert jnp.allclose(no_trap - trapped, 0.5 * jnp.sum(pos ** 2), rtol=1e-4)

    def test_translation_invariant_without_trap(self):
        target = LennardJones(trap_scale=0.0)
        pos = 2.0 * jax.random.normal(jax.random.PRNGKey(1), (13, 3))
        shift = jnp.array([1.0, -2.0, 0.5])
        assert jnp.allclose(target(pos), target(pos + shift), rtol=1e-4)

    def test_batched_and_jit(self):
        target = LennardJones(epsilon=2.0)
        xs = 2.0 * jax.random.normal(jax.random.PRNGKey(2), (5, 13, 3))
        batched = target(xs)
        assert batched.shape == (5,)
        # float32 on GPU: jit may sum the pair terms in a different order (~1e-6 rel).
        assert jnp.allclose(batched, jax.vmap(target)(xs), rtol=1e-5)
        assert jnp.allclose(batched, jax.jit(target)(xs), rtol=1e-5)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            LennardJones()(jnp.zeros((12, 3)))

    @pytest.mark.parametrize(
        "kwargs",
        [{"n_particles": 1}, {"spatial_dim": 0}, {"epsilon": 0.0}, {"rm": 0.0},
         {"beta": 0.0}, {"trap_mode": "bad"}],
    )
    def test_rejects_bad_params(self, kwargs):
        with pytest.raises(ValueError):
            LennardJones(**kwargs)
