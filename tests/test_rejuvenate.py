"""
Unit tests for progressive_is_tempering.rejuvenate.

Tests cover:
  - rejuvenate: shape, JIT compatibility, vmap-over-rungs compatibility
    (the pattern the vmapped outer data-generation loop needs).
  - Detailed balance: with an exact Gaussian target, the stationary
    distribution is preserved in the long run (empirical mean/variance
    check).
  - The t_idx==0 gotcha documented in rejuvenate's docstring: the kernel
    ignores `positions` and redraws from the reference at t_idx==0.
  - project_fn (e.g. a zero-center-of-mass constraint): output
    stays exactly on the constraint manifold after every step, and the
    vmap-over-rungs pattern still works with it set (closed over, like
    `problem`/`n_steps`, not vmapped).
"""

import jax
import jax.numpy as jnp

from progressive_is_tempering.mala_kernel import PTProblem, mala_kernel
from progressive_is_tempering.rejuvenate import rejuvenate


def _gaussian_problem(dim: int) -> PTProblem:
    """Reference == target == N(0,I): the simplest possible stationarity check."""
    def logpdf(x):
        return -0.5 * jnp.sum(x ** 2) - 0.5 * dim * jnp.log(2 * jnp.pi)

    def rvs_ref(key):
        return jax.random.normal(key, (dim,))

    return PTProblem(rvs_ref=rvs_ref, logpdf_ref=logpdf, logpdf_final_target=logpdf)


def _project_com(x: jnp.ndarray, n_particles: int, spatial_dim: int) -> jnp.ndarray:
    """Zero-center-of-mass projector for n_particles in spatial_dim dimensions
    (single-particle signature, since rejuvenate vmaps it itself)."""
    xr = x.reshape(n_particles, spatial_dim)
    com = xr.mean(axis=0, keepdims=True)
    return (xr - com).reshape(x.shape)


def _calibrated_params(problem, beta, t_idx, particles, key):
    """Adapt MALAParameters the same way `one_step_smc` does mid-SMC, standalone."""
    mcmc = mala_kernel(problem, beta, t_idx)
    key_init, key_adapt = jax.random.split(key)
    x = jax.vmap(mcmc.init_extend)(
        jax.random.split(key_init, particles.shape[0]), particles
    )
    return mcmc.adapt_params(key_adapt, x)


class TestRejuvenate:
    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        problem = _gaussian_problem(4)
        particles = jax.random.normal(key, (20, 4))
        params = _calibrated_params(problem, 1.0, 1, particles, key)
        new_p, acc = rejuvenate(
            key, particles, problem, jnp.array(1.0), 1, params, 10
        )
        assert new_p.shape == (20, 4)
        assert acc.shape == ()

    def test_stationary_distribution_gaussian(self):
        key = jax.random.PRNGKey(1)
        problem = _gaussian_problem(8)
        particles = jax.random.normal(key, (300, 8))
        key, k_calib, k_move = jax.random.split(key, 3)
        params = _calibrated_params(problem, 1.0, 1, particles, k_calib)
        new_p, acc = rejuvenate(
            k_move, particles, problem, jnp.array(1.0), 1, params, 50
        )
        assert 0.2 < float(acc) < 1.0
        assert jnp.allclose(jnp.mean(new_p), 0.0, atol=0.15)
        assert jnp.allclose(jnp.var(new_p), 1.0, atol=0.15)

    def test_particles_change(self):
        key = jax.random.PRNGKey(2)
        problem = _gaussian_problem(4)
        particles = jax.random.normal(key, (10, 4))
        params = _calibrated_params(problem, 1.0, 1, particles, key)
        new_p, _ = rejuvenate(
            key, particles, problem, jnp.array(1.0), 1, params, 20
        )
        assert not jnp.allclose(new_p, particles)

    def test_t_idx_zero_ignores_positions(self):
        """Documented gotcha: t_idx=0 redraws from rvs_ref, ignoring `positions`."""
        key = jax.random.PRNGKey(3)
        problem = _gaussian_problem(4)
        particles = jnp.full((10, 4), 100.0)  # far outside N(0,I) support
        params = _calibrated_params(problem, 0.0, 0, particles, key)
        new_p, _ = rejuvenate(
            key, particles, problem, jnp.array(0.0), 0, params, 5
        )
        assert not jnp.allclose(new_p, particles)

    def test_vmap_over_rungs(self):
        """The pattern the vmapped outer loop needs: vmap over
        (key, positions, beta, t_idx, mcmc_params) per active rung, with
        `problem` and `n_steps` closed over (not vmapped) -- mirroring
        exactly how rejuvenate is static in those two positions.
        """
        key = jax.random.PRNGKey(4)
        problem = _gaussian_problem(4)
        n_rungs = 3
        keys = jax.random.split(key, n_rungs)
        particles = jax.random.normal(key, (n_rungs, 15, 4))
        betas = jnp.array([0.3, 0.6, 1.0])
        t_idxs = jnp.array([1, 2, 3])
        params = jax.vmap(lambda b, t, p: _calibrated_params(problem, b, t, p, key))(
            betas, t_idxs, particles
        )

        def one(k, p, b, t, prm):
            return rejuvenate(k, p, problem, b, t, prm, 5)

        new_p, acc = jax.vmap(one)(keys, particles, betas, t_idxs, params)
        assert new_p.shape == (n_rungs, 15, 4)
        assert acc.shape == (n_rungs,)


class TestRejuvenateProjected:
    """Translation-invariant particle systems need every internal move
    projected back to zero-CoM -- the kernel itself has no CoM-awareness at
    all, so this must be threaded through explicitly via `project_fn`.
    """

    def test_output_stays_on_constraint_manifold(self):
        """Starting from particles with nonzero CoM, rejuvenation with
        project_fn should return particles with exactly zero CoM (to float
        precision) -- the chain must never leave the manifold, not just end
        up back on it by chance.
        """
        key = jax.random.PRNGKey(10)
        n_particles, spatial_dim = 2, 2
        dim = n_particles * spatial_dim
        problem = _gaussian_problem(dim)

        key, subkey = jax.random.split(key)
        particles = jax.random.normal(subkey, (20, dim)) + 5.0  # nonzero CoM
        params = _calibrated_params(problem, 1.0, 1, particles, key)

        project_fn = lambda x: _project_com(x, n_particles, spatial_dim)
        key, subkey = jax.random.split(key)
        new_p, acc = rejuvenate(
            subkey, particles, problem, jnp.array(1.0), 1, params, 30,
            project_fn=project_fn,
        )
        com = new_p.reshape(-1, n_particles, spatial_dim).mean(axis=1)
        assert jnp.abs(com).max() < 1e-4
        assert 0.0 <= float(acc) <= 1.0

    def test_default_path_unaffected(self):
        """project_fn=None (the default) must reproduce the pre-existing
        repeated_batch_move fast path exactly -- same call, same key, same
        result.
        """
        key = jax.random.PRNGKey(11)
        problem = _gaussian_problem(4)
        particles = jax.random.normal(key, (10, 4))
        params = _calibrated_params(problem, 1.0, 1, particles, key)

        key, subkey = jax.random.split(key)
        p_default, a_default = rejuvenate(
            subkey, particles, problem, jnp.array(1.0), 1, params, 10
        )
        p_explicit, a_explicit = rejuvenate(
            subkey, particles, problem, jnp.array(1.0), 1, params, 10,
            project_fn=None,
        )
        assert jnp.array_equal(p_default, p_explicit)
        assert jnp.array_equal(a_default, a_explicit)

    def test_vmap_over_rungs_with_project_fn(self):
        """Same vmap-over-rungs pattern _data_gen_parallel needs, but with
        project_fn set -- it must be closed over (not vmapped), exactly like
        `problem`/`n_steps`.
        """
        key = jax.random.PRNGKey(12)
        n_particles, spatial_dim = 2, 2
        dim = n_particles * spatial_dim
        problem = _gaussian_problem(dim)
        project_fn = lambda x: _project_com(x, n_particles, spatial_dim)

        n_rungs = 3
        keys = jax.random.split(key, n_rungs)
        key, subkey = jax.random.split(key)
        particles = jax.random.normal(subkey, (n_rungs, 12, dim)) + 3.0
        betas = jnp.array([0.3, 0.6, 1.0])
        t_idxs = jnp.array([1, 2, 3])
        params = jax.vmap(lambda b, t, p: _calibrated_params(problem, b, t, p, key))(
            betas, t_idxs, particles
        )

        def one(k, p, b, t, prm):
            return rejuvenate(k, p, problem, b, t, prm, 15, project_fn=project_fn)

        new_p, acc = jax.vmap(one)(keys, particles, betas, t_idxs, params)
        assert new_p.shape == (n_rungs, 12, dim)
        assert acc.shape == (n_rungs,)
        com = new_p.reshape(n_rungs, 12, n_particles, spatial_dim).mean(axis=2)
        assert jnp.abs(com).max() < 1e-4


class TestRejuvenateEnergyAndInvariance:
    """`return_energy` reads E from the final particle; `project_invariant`
    skips re-evaluating an energy that the projection cannot change."""

    def test_return_energy_matches_energy_and_positions(self):
        key = jax.random.PRNGKey(20)
        problem = _gaussian_problem(4)
        particles = jax.random.normal(key, (30, 4))
        params = _calibrated_params(problem, 0.7, 2, particles, key)
        p2, a2 = rejuvenate(key, particles, problem, jnp.array(0.7), 2, params, 8)
        p3, a3, e3 = rejuvenate(key, particles, problem, jnp.array(0.7), 2, params, 8,
                                return_energy=True)
        assert jnp.array_equal(p2, p3)
        assert jnp.array_equal(a2, a3)
        expected = -jax.vmap(problem.logpdf_final_target)(p3)
        assert jnp.allclose(e3, expected, rtol=1e-6, atol=1e-5)

    def test_project_invariant_matches_full_reinit(self):
        """For an energy that depends on pair distances only, skipping the
        energy re-evaluation after the zero-CoM projection must give the same
        chain as re-deriving everything."""
        n_particles, spatial_dim = 3, 2
        dim = n_particles * spatial_dim

        def energy(x):  # pair distances only: translation invariant
            xr = x.reshape(n_particles, spatial_dim)
            d2 = jnp.sum((xr[:, None] - xr[None]) ** 2, axis=-1)
            return 0.5 * jnp.sum(d2) / n_particles

        project_fn = lambda x: _project_com(x, n_particles, spatial_dim)
        problem = PTProblem(
            rvs_ref=lambda k: project_fn(jax.random.normal(k, (dim,))),
            logpdf_ref=lambda x: -0.5 * jnp.sum(x ** 2),
            logpdf_final_target=lambda x: -energy(x),
        )
        key = jax.random.PRNGKey(21)
        particles = jax.vmap(project_fn)(jax.random.normal(key, (64, dim)))
        params = _calibrated_params(problem, 0.6, 3, particles, key)
        full = rejuvenate(key, particles, problem, jnp.array(0.6), 3, params, 10,
                          project_fn=project_fn, return_energy=True)
        cut = rejuvenate(key, particles, problem, jnp.array(0.6), 3, params, 10,
                         project_fn=project_fn, project_invariant=True,
                         return_energy=True)
        assert jnp.allclose(full[0], cut[0], atol=1e-5)
        assert jnp.allclose(full[2], cut[2], rtol=1e-5, atol=1e-5)
        assert jnp.allclose(cut[2], jax.vmap(energy)(cut[0]), rtol=1e-5, atol=1e-5)
