"""
SMC seed-quality diagnostics.

The top rung of the SMC ladder targets :math:`\\pi_{\\beta_M} = \\pi` — the same
distribution a ground-truth reference set is drawn from — so the two should
agree.  This is the one check worth running *before* committing to a training
run: it puts a ceiling on what the seeded replay buffers can teach the network.

Two failure modes are covered, and they are not the same thing:

1. **Distributional mismatch.**  The pooled summary statistic (coordinates, or
   inter-particle distances) and the energy simply disagree with the reference.
   Caught by :func:`plot_value_comparison`.

2. **Ancestral degeneracy.**  The pooled histogram matches the reference
   beautifully, yet the particle population carries far less independent
   information than its particle count suggests.  This happens when SMC
   resampling collapses the ancestral lineages: local MCMC then re-equilibrates
   each particle *within* its basin, erasing literal duplicates, while the
   basin assignment itself stays frozen.  A distinct-row count will report
   100% unique and miss it entirely.  Caught by
   :func:`exchangeable_degeneracy`.

The degeneracy test needs a set of **exchangeable scalar features** — features
that are statistically identical to one another under the target:

===================  ==================================================
Target               Exchangeable features
===================  ==================================================
factorised           the ``d`` coordinates (i.i.d. across coordinates)
n-particle system    the ``C(n,2)`` pairwise distances (particle-
                     permutation symmetry)
===================  ==================================================

Each feature's exceedance probability :math:`p_f = P(\\text{feature}_f > \\tau)`
must be the same for every ``f``.  Estimated from ``N`` particles, each
:math:`\\hat p_f` has standard error :math:`\\sqrt{p(1-p)/N}` *if the particles
are independent*.  Comparing the observed spread of :math:`\\{\\hat p_f\\}`
against that standard error gives an implied independent-particle count.

The reference set is passed through the identical computation as a control: it
is genuinely i.i.d., so its implied count should come back close to its true
size.  When it does not, the features are not exchangeable for this target and
the test is uninformative — read ``n_eff_ref`` before trusting ``n_eff_test``.

Only NumPy is needed for the statistics.  matplotlib and scipy are imported
lazily inside the plotting/statistics helpers so that importing this module
never requires them.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "C_TEST",
    "C_REF",
    "REF_FILL_ALPHA",
    "C_INK",
    "C_INK_2",
    "style_axis",
    "reference_anchored_window",
    "honest_density_hist",
    "plot_value_comparison",
    "exchangeable_degeneracy",
    "smc_seed_quality_figure",
    "print_seed_quality_report",
]

# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------
# Colour follows the entity, never the panel: C_TEST is always whatever is being
# evaluated (the SMC seed, or the model), C_REF is always ground truth — so the
# eye carries meaning across every figure in every notebook.  The reference is
# additionally drawn as a filled histogram (the test as a step outline), so
# identity never rests on colour alone.
C_TEST = "#eb6834"   # orange
C_REF = "#2a78d6"    # blue
REF_FILL_ALPHA = 0.35
C_INK = "#0b0b0b"    # primary text
C_INK_2 = "#52514e"  # secondary text


def style_axis(ax) -> None:
    """Recessive grid and axes: the data is the ink, the frame is not."""
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8c7c2")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=C_INK_2, labelsize=9, length=3, width=0.8)


# ---------------------------------------------------------------------------
# Distribution comparison
# ---------------------------------------------------------------------------
def reference_anchored_window(x_test, x_ref, max_spans: float = 4.0):
    """Plot window that always shows the reference in full.

    Widens to include the test bulk (1st-99th pct) when the two are comparable,
    then caps at ``max_spans`` reference-spans either side.  Without the cap, a
    single close contact — LJ energy is unbounded above, going like r^-12 —
    drags the axis out by 15 orders of magnitude and flattens the reference to
    one pixel.
    """
    x_test = np.asarray(x_test).ravel()
    if x_ref is None or np.asarray(x_ref).size == 0:
        lo, hi = float(np.percentile(x_test, 1)), float(np.percentile(x_test, 99))
    else:
        x_ref = np.asarray(x_ref).ravel()
        ref_lo, ref_hi = float(x_ref.min()), float(x_ref.max())
        span = max(ref_hi - ref_lo, 1e-6)
        lo = max(min(ref_lo, float(np.percentile(x_test, 1))), ref_lo - max_spans * span)
        hi = min(max(ref_hi, float(np.percentile(x_test, 99))), ref_hi + max_spans * span)
    pad = 0.05 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def honest_density_hist(ax, x, lo, hi, bins, label, color, filled=False) -> float:
    """Step histogram whose area equals the IN-WINDOW fraction of the sample.

    ``filled=True`` draws a translucent filled histogram behind the step
    outlines — the style reserved for the ground-truth reference.

    Not ``density=True``.  Matplotlib's density normalizes over the values that
    land inside ``range``, so a sample with 97% of its mass off-window would be
    drawn as though the surviving 3% were its entire distribution — the panel
    would show a confident-looking curve for a diverged model.  Weighting by
    ``1/(N_total * bin_width)`` instead keeps every curve on the same absolute
    scale: a sample that mostly misses the window is drawn small, which is the
    truth.  Two curves that overlay are genuinely the same distribution.

    Returns the fraction of the sample that fell outside the window.
    """
    x = np.asarray(x).ravel()
    n_total = max(x.size, 1)
    bin_width = (hi - lo) / bins
    w = np.full(n_total, 1.0 / (n_total * bin_width))
    if filled:
        ax.hist(x, bins=bins, range=(lo, hi), weights=w, histtype="stepfilled",
                color=color, alpha=REF_FILL_ALPHA, label=label, zorder=1)
    else:
        ax.hist(x, bins=bins, range=(lo, hi), weights=w, histtype="step",
                linewidth=1.5, color=color, label=label, zorder=2)
    return float(np.sum((x < lo) | (x > hi))) / n_total


def plot_value_comparison(ax, v_test, v_ref, xlabel, label_test="SMC seed",
                          bins=150, clip_at_zero=False) -> float:
    """Density of a scalar statistic: test vs reference, on shared bin edges."""
    lo, hi = reference_anchored_window(v_test, v_ref)
    if clip_at_zero:
        lo = max(lo, 0.0)
    if v_ref is not None:
        honest_density_hist(ax, v_ref, lo, hi, bins, "reference", C_REF, filled=True)
    frac_out = honest_density_hist(ax, v_test, lo, hi, bins, label_test, C_TEST)
    if frac_out > 0.0005:
        ax.text(0.98, 0.94, f"{frac_out:.1%} of {label_test} off-window",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=C_INK_2)
    ax.set_xlabel(xlabel, color=C_INK_2, fontsize=10)
    ax.set_ylabel("density", color=C_INK_2, fontsize=10)
    ax.legend(fontsize=9, frameon=False, labelcolor=C_INK)
    style_axis(ax)
    return frac_out


# ---------------------------------------------------------------------------
# Ancestral degeneracy
# ---------------------------------------------------------------------------
def exchangeable_degeneracy(features_test, features_ref=None, threshold=None):
    """Quantify ancestral degeneracy from a set of exchangeable features.

    Parameters
    ----------
    features_test : array, shape (N, F)
        Exchangeable scalar features of the particles under test.
    features_ref : array, shape (N_ref, F), optional
        The same features for a ground-truth reference set.  Passed through the
        identical computation as a control: being i.i.d., its implied
        independent-particle count should come back close to ``N_ref``.
    threshold : float, optional
        Exceedance threshold ``tau``.  Defaults to the median of the pooled
        reference features (or of the test features when no reference is
        given), which puts ``p`` near 0.5 where the estimator is best
        conditioned.  Pass an explicit value to probe a specific structure —
        e.g. the mode boundary of a mixture.

    Returns
    -------
    dict
        ``threshold``, ``n_test``, ``n_features``, ``p_test``/``p_ref``
        (pooled exceedance probability), ``per_feature_test``/
        ``per_feature_ref`` (the F per-feature estimates), ``se_iid`` (standard
        error of one estimate under independence), ``s_test``/``s_ref``
        (observed spread across features), ``n_eff_test``/``n_eff_ref``
        (implied independent particles), ``inflation`` (``s_test/se_iid``),
        ``chi2`` and ``p_value`` (goodness of fit of the F counts against a
        common ``p``).

    Notes
    -----
    Features need not be independent *of each other* within a particle — the 78
    distances of a 13-particle cluster are heavily constrained.  Cross-feature
    correlation deflates the observed spread, which makes the test conservative:
    it under-reports degeneracy rather than inventing it.  ``n_eff_ref`` is the
    guard against the correlation being severe enough to make the test useless.
    """
    ft = np.asarray(features_test)
    if ft.ndim != 2:
        raise ValueError(f"features_test must be 2-D (N, F), got shape {ft.shape}")
    n_test, n_features = ft.shape
    if n_features < 2:
        raise ValueError("need at least 2 exchangeable features to measure spread")

    fr = None if features_ref is None else np.asarray(features_ref)
    if fr is not None:
        if fr.ndim != 2 or fr.shape[1] != n_features:
            raise ValueError(
                f"features_ref must be (N_ref, {n_features}), got shape {fr.shape}"
            )

    if threshold is None:
        pool = ft if fr is None else fr
        threshold = float(np.median(pool))
    threshold = float(threshold)

    per_test = (ft > threshold).mean(axis=0)          # (F,)
    p_test = float(per_test.mean())
    s_test = float(per_test.std(ddof=1))

    # Reference exceedance probability anchors the null; fall back to the test's
    # own pooled value when no reference is available.
    if fr is not None:
        per_ref = (fr > threshold).mean(axis=0)
        p_ref = float(per_ref.mean())
        s_ref = float(per_ref.std(ddof=1))
        n_ref = fr.shape[0]
        var_ref = p_ref * (1.0 - p_ref)
        n_eff_ref = var_ref / s_ref**2 if s_ref > 0 else float("inf")
    else:
        per_ref, p_ref, s_ref, n_ref, n_eff_ref = None, p_test, None, None, None

    var = p_ref * (1.0 - p_ref)
    se_iid = float(np.sqrt(var / n_test)) if var > 0 else 0.0
    n_eff_test = var / s_test**2 if s_test > 0 else float("inf")

    # Goodness of fit of the F per-feature counts against a common p.
    if var > 0:
        chi2 = float(((per_test * n_test - p_ref * n_test) ** 2 / (n_test * var)).sum())
        from scipy import stats as _stats
        p_value = float(_stats.chi2.sf(chi2, df=n_features))
    else:
        chi2, p_value = float("nan"), float("nan")

    return {
        "threshold": threshold,
        "n_test": n_test,
        "n_features": n_features,
        "n_ref": n_ref,
        "p_test": p_test,
        "p_ref": p_ref,
        "per_feature_test": per_test,
        "per_feature_ref": per_ref,
        "se_iid": se_iid,
        "s_test": s_test,
        "s_ref": s_ref,
        "n_eff_test": float(n_eff_test),
        "n_eff_ref": None if n_eff_ref is None else float(n_eff_ref),
        "inflation": float(s_test / se_iid) if se_iid > 0 else float("inf"),
        "chi2": chi2,
        "p_value": p_value,
    }


def _plot_per_feature(ax, stats, label_test="SMC seed"):
    """Per-feature exceedance probability against the i.i.d. band."""
    per_test = stats["per_feature_test"]
    idx = np.arange(stats["n_features"])
    p_ref, se = stats["p_ref"], stats["se_iid"]

    ax.axhspan(p_ref - 2 * se, p_ref + 2 * se, color=C_REF, alpha=0.22,
               label=f"±2 s.e. if {stats['n_test']:,} i.i.d.", zorder=1)
    ax.axhline(p_ref, color=C_REF, linestyle="--", linewidth=1.5,
               label="reference", zorder=2)
    if stats["per_feature_ref"] is not None:
        ax.plot(idx, stats["per_feature_ref"], "o", ms=3.5, color=C_REF,
                alpha=0.55, zorder=3, label="reference per-feature")
    ax.plot(idx, per_test, "o", ms=4.5, color=C_TEST, zorder=4, label=label_test)

    ax.set_xlabel("exchangeable feature index", color=C_INK_2, fontsize=10)
    ax.set_ylabel(r"$P(\mathrm{feature} > \tau)$", color=C_INK_2, fontsize=10)
    ax.legend(fontsize=8, frameon=False, labelcolor=C_INK)
    style_axis(ax)

    # Boxed: the per-feature cloud can reach the top of the axes (a fully frozen
    # population pins features near 0 and 1), and unboxed text lands on top of it.
    n_eff = stats["n_eff_test"]
    ax.text(0.02, 0.96,
            f"spread {stats['inflation']:.0f}× the i.i.d. s.e.\n"
            f"implied independent particles: {n_eff:,.0f} of {stats['n_test']:,}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            color=C_INK_2, zorder=5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#c8c7c2", linewidth=0.6, alpha=0.88))


def _plot_occupancy(ax, features_test, features_ref, threshold,
                    label_test="SMC seed"):
    """Per-particle count of features above threshold: test vs reference."""
    ft = np.asarray(features_test)
    n_features = ft.shape[1]
    cnt_test = (ft > threshold).sum(axis=1)
    bins = np.arange(-0.5, n_features + 1.5)

    if features_ref is not None:
        cnt_ref = (np.asarray(features_ref) > threshold).sum(axis=1)
        ax.hist(cnt_ref, bins=bins, density=True, histtype="stepfilled",
                color=C_REF, alpha=REF_FILL_ALPHA, label="reference", zorder=1)
    ax.hist(cnt_test, bins=bins, density=True, histtype="step", linewidth=1.5,
            color=C_TEST, label=label_test, zorder=2)
    ax.set_xlabel(r"# features above $\tau$ (per particle)",
                  color=C_INK_2, fontsize=10)
    ax.set_ylabel("probability", color=C_INK_2, fontsize=10)
    ax.legend(fontsize=9, frameon=False, labelcolor=C_INK)
    style_axis(ax)


def smc_seed_quality_figure(
    features_test,
    features_ref,
    energy_test,
    energy_ref,
    feature_label,
    title,
    label_test="SMC seed",
    threshold=None,
    bins=150,
    features_nonneg=False,
):
    """Assemble the standard four-panel SMC seed-quality figure.

    Panels: pooled feature density, energy density, per-feature exceedance
    against the i.i.d. band, and the per-particle occupancy distribution.
    The first two catch distributional mismatch; the last two catch ancestral
    degeneracy that the first two cannot see.

    Returns
    -------
    (fig, stats)
        ``stats`` is the :func:`exchangeable_degeneracy` dict, extended with
        ``frac_out_feature`` and ``frac_out_energy``.
    """
    import matplotlib.pyplot as plt

    stats = exchangeable_degeneracy(features_test, features_ref, threshold)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    frac_feat = plot_value_comparison(
        axes[0, 0], np.asarray(features_test).ravel(),
        None if features_ref is None else np.asarray(features_ref).ravel(),
        feature_label, label_test, bins=bins, clip_at_zero=features_nonneg,
    )
    axes[0, 0].set_title("Pooled summary statistic", color=C_INK, fontsize=11)

    frac_e = plot_value_comparison(
        axes[0, 1], energy_test, energy_ref, "energy $E(x)$", label_test, bins=bins,
    )
    axes[0, 1].set_title("Energy", color=C_INK, fontsize=11)

    _plot_per_feature(axes[1, 0], stats, label_test)
    axes[1, 0].set_title(
        rf"Per-feature exceedance ($\tau$={stats['threshold']:.3g})",
        color=C_INK, fontsize=11,
    )

    _plot_occupancy(axes[1, 1], features_test, features_ref,
                    stats["threshold"], label_test)
    axes[1, 1].set_title("Joint occupancy distribution", color=C_INK, fontsize=11)

    fig.suptitle(title, fontsize=12, color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    stats["frac_out_feature"] = frac_feat
    stats["frac_out_energy"] = frac_e
    return fig, stats


def print_seed_quality_report(stats, label_test="SMC seed") -> None:
    """Print the numbers behind :func:`smc_seed_quality_figure`."""
    print(f"  {label_test}: {stats['n_test']:,} particles, "
          f"{stats['n_features']} exchangeable features")
    print(f"  threshold tau                 : {stats['threshold']:.4f}")
    print(f"  pooled P(feature > tau)       : {stats['p_test']:.4f}   "
          f"reference {stats['p_ref']:.4f}")
    print(f"  s.e. if {stats['n_test']:,} i.i.d. particles : {stats['se_iid']:.5f}")
    print(f"  observed spread across features: {stats['s_test']:.5f}  "
          f"({stats['inflation']:.0f}x the i.i.d. s.e.)")
    print(f"  implied independent particles : {stats['n_eff_test']:,.0f} "
          f"of {stats['n_test']:,}")
    if stats["n_eff_ref"] is not None:
        print(f"    control — same statistic on the reference: "
              f"{stats['n_eff_ref']:,.0f} of {stats['n_ref']:,}")
    print(f"  chi2({stats['n_features']}) = {stats['chi2']:,.0f}, "
          f"p = {stats['p_value']:.2e}")
    if "frac_out_energy" in stats:
        print(f"  off-window fraction: feature {stats['frac_out_feature']:.2%}, "
              f"energy {stats['frac_out_energy']:.2%}")
