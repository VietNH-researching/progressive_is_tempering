"""
Unit tests for progressive_is_tempering.utils.smc_diagnostics.

Tests cover:
  - exchangeable_degeneracy: i.i.d. input recovers ~N independent particles,
    ancestrally degenerate input reports far fewer, threshold defaulting,
    reference control, input validation.
  - reference_anchored_window / honest_density_hist: window contains the
    reference, off-window fraction is reported honestly.
  - smc_seed_quality_figure: assembles without a display and returns stats.
"""

import numpy as np
import pytest

from progressive_is_tempering.utils.smc_diagnostics import (
    exchangeable_degeneracy,
    honest_density_hist,
    reference_anchored_window,
    smc_seed_quality_figure,
)

N_PARTICLES = 4_000
N_FEATURES = 16


def _iid_features(seed=0, n=N_PARTICLES, f=N_FEATURES):
    """Genuinely i.i.d. exchangeable features."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, f))


def _degenerate_features(seed=0, n=N_PARTICLES, f=N_FEATURES, n_ancestors=20):
    """Ancestrally degenerate: n particles descended from n_ancestors lineages.

    Each particle inherits its ancestor's per-feature sign (the "basin"), then
    gets independent within-basin jitter — mimicking local MCMC that erases
    literal duplicates while leaving the basin assignment frozen.
    """
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_ancestors, f))
    parent = rng.integers(0, n_ancestors, size=n)
    return signs[parent] * 2.0 + rng.normal(scale=0.3, size=(n, f))


class TestExchangeableDegeneracy:
    def test_iid_recovers_particle_count(self):
        """i.i.d. features should imply ~N independent particles."""
        st = exchangeable_degeneracy(_iid_features(seed=1))
        # Estimator is itself noisy with only F=16 features; an order of
        # magnitude either way is the meaningful resolution here.
        assert 0.1 * N_PARTICLES < st["n_eff_test"] < 10 * N_PARTICLES

    def test_iid_spread_matches_iid_standard_error(self):
        st = exchangeable_degeneracy(_iid_features(seed=2))
        assert 0.3 < st["inflation"] < 3.0

    def test_degenerate_reports_far_fewer(self):
        """Degenerate population must report far below its particle count."""
        st = exchangeable_degeneracy(_degenerate_features(seed=3, n_ancestors=20))
        assert st["n_eff_test"] < 0.05 * N_PARTICLES
        assert st["inflation"] > 4.0

    def test_degenerate_rejected_by_chi_square(self):
        st = exchangeable_degeneracy(_degenerate_features(seed=4, n_ancestors=20))
        assert st["p_value"] < 1e-6

    def test_iid_not_rejected_by_chi_square(self):
        st = exchangeable_degeneracy(_iid_features(seed=5))
        assert st["p_value"] > 1e-3

    def test_more_ancestors_means_more_effective_particles(self):
        few = exchangeable_degeneracy(_degenerate_features(seed=6, n_ancestors=10))
        many = exchangeable_degeneracy(_degenerate_features(seed=6, n_ancestors=200))
        assert many["n_eff_test"] > few["n_eff_test"]

    def test_reference_control_is_calibrated(self):
        """The i.i.d. reference, passed as control, recovers ~its own size."""
        st = exchangeable_degeneracy(
            _degenerate_features(seed=7), _iid_features(seed=8)
        )
        assert 0.1 * N_PARTICLES < st["n_eff_ref"] < 10 * N_PARTICLES
        assert st["n_eff_test"] < st["n_eff_ref"]

    def test_threshold_defaults_to_reference_median(self):
        ref = _iid_features(seed=9) + 5.0
        st = exchangeable_degeneracy(_iid_features(seed=10), ref)
        assert st["threshold"] == pytest.approx(float(np.median(ref)), abs=1e-9)

    def test_threshold_defaults_to_test_median_without_reference(self):
        ft = _iid_features(seed=11)
        st = exchangeable_degeneracy(ft)
        assert st["threshold"] == pytest.approx(float(np.median(ft)), abs=1e-9)

    def test_explicit_threshold_is_used(self):
        st = exchangeable_degeneracy(_iid_features(seed=12), threshold=0.25)
        assert st["threshold"] == pytest.approx(0.25)

    def test_reported_shapes_and_pooled_probability(self):
        ft = _iid_features(seed=13)
        st = exchangeable_degeneracy(ft)
        assert st["n_test"] == N_PARTICLES
        assert st["n_features"] == N_FEATURES
        assert st["per_feature_test"].shape == (N_FEATURES,)
        assert st["p_test"] == pytest.approx(0.5, abs=0.05)

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            exchangeable_degeneracy(np.zeros(10))

    def test_rejects_single_feature(self):
        with pytest.raises(ValueError, match="at least 2"):
            exchangeable_degeneracy(np.zeros((10, 1)))

    def test_rejects_mismatched_reference(self):
        with pytest.raises(ValueError, match="features_ref"):
            exchangeable_degeneracy(np.zeros((10, 4)), np.zeros((10, 5)))


class TestPlotHelpers:
    def test_window_contains_reference(self):
        ref = np.linspace(-1.0, 1.0, 100)
        lo, hi = reference_anchored_window(np.array([0.0, 0.5]), ref)
        assert lo <= ref.min() and hi >= ref.max()

    def test_window_caps_extreme_test_outliers(self):
        """A single r^-12 blow-up must not flatten the reference to a pixel."""
        ref = np.linspace(0.0, 1.0, 1000)
        test = np.concatenate([ref, [1e15]])
        lo, hi = reference_anchored_window(test, ref, max_spans=4.0)
        assert hi < 10.0

    def test_window_without_reference(self):
        lo, hi = reference_anchored_window(np.linspace(0, 1, 100), None)
        assert lo < hi

    def test_honest_hist_reports_off_window_fraction(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x = np.concatenate([np.zeros(90), np.full(10, 100.0)])
        frac = honest_density_hist(ax, x, -1.0, 1.0, 10, "t", "#000000")
        plt.close(fig)
        assert frac == pytest.approx(0.1)


class TestSeedQualityFigure:
    def test_figure_and_stats(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ft = _degenerate_features(seed=14)
        fr = _iid_features(seed=15)
        fig, st = smc_seed_quality_figure(
            features_test=ft,
            features_ref=fr,
            energy_test=ft.sum(axis=1),
            energy_ref=fr.sum(axis=1),
            feature_label="feature",
            title="test",
        )
        plt.close(fig)
        assert st["n_features"] == N_FEATURES
        assert "frac_out_energy" in st
        assert "frac_out_feature" in st
