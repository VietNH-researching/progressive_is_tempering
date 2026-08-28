from mppt.jax_configs import x64; x64()

import io

import numpy as np
import pytest
from jax import numpy as jnp

from mppt import summarize_diagnostics
from mppt.ui.diagnostics import (
    format_diagnostics_summary,
    print_diagnostics_summary,
)


def test_summarize_diagnostics_extracts_plain_python_fields():
    r"""A normal diagnostics dict should become stable Python scalars/lists.

    Plausible bug: returning device arrays makes checkpoint manifests and logs
    harder to serialize or compare outside JAX.
    """
    diagnostics = {
        'logZ_SMC_like': jnp.array(10.0),
        'logZ_robust': np.array(10.2),
        'logZ_reverse_logistic': 9.8,
        'mcmc_acc_rates': jnp.array([1.0, 0.84, 0.91]),
        'swap_rejection_rates': jnp.array([0.2, 0.35]),
        'mcmc_niter': jnp.array(12),
    }

    summary = summarize_diagnostics(diagnostics)

    assert summary.logZ == {
        'SMC_like': 10.0,
        'robust': 10.2,
        'reverse_logistic': 9.8,
    }
    assert summary.logZ_range == pytest.approx(0.4)
    assert summary.mcmc_acc_rates == [1.0, 0.84, 0.91]
    assert summary.mcmc_acc_min_nonref == pytest.approx(0.84)
    assert summary.mcmc_acc_max_nonref == pytest.approx(0.91)
    assert summary.swap_rejection_rates == [0.2, 0.35]
    assert summary.swap_rej_min == pytest.approx(0.2)
    assert summary.swap_rej_max == pytest.approx(0.35)
    assert summary.mcmc_niter == 12
    assert summary.warnings == ()


def test_summarize_diagnostics_warns_on_obvious_pathologies():
    r"""Warnings should cover high-signal Monte Carlo failure modes.

    Plausible bug: a run with divergent logZ estimates, stuck MCMC moves, or
    nearly impossible swaps looks normal because the raw diagnostics are arrays.
    """
    diagnostics = {
        'logZ_SMC_like': 0.0,
        'logZ_robust': 2.5,
        'mcmc_acc_rates': jnp.array([1.0, 0.01, 1.0]),
        'swap_rejection_rates': jnp.array([0.25, 0.97]),
    }

    warnings = summarize_diagnostics(diagnostics).warnings

    assert any('logZ estimators disagree' in warning for warning in warnings)
    assert any('MCMC acceptance below' in warning for warning in warnings)
    assert any('MCMC acceptance above' in warning for warning in warnings)
    assert any('swap rejection above' in warning for warning in warnings)


def test_summarize_diagnostics_warns_on_nonfinite_values():
    diagnostics = {
        # Use NumPy non-finites here because the suite enables JAX NaN
        # debugging in some modules; constructing a JAX NaN can raise before
        # this helper gets to inspect the diagnostics.
        'logZ_SMC_like': np.array(np.nan),
        'mcmc_acc_rates': np.array([1.0, np.inf]),
        'mcmc_niter': np.array(np.inf),
    }

    warnings = summarize_diagnostics(diagnostics).warnings

    assert any("logZ_SMC_like" in warning for warning in warnings)
    assert any("mcmc_acc_rates" in warning for warning in warnings)
    assert any("mcmc_niter" in warning for warning in warnings)


def test_summarize_diagnostics_rejects_nonscalar_logZ():
    with pytest.raises(ValueError, match="must be scalar"):
        summarize_diagnostics({'logZ_robust': jnp.array([1.0, 2.0])})


def test_format_and_print_diagnostics_summary_are_stable():
    diagnostics = {
        'logZ_robust': 1.25,
        'mcmc_acc_rates': jnp.array([1.0, 0.8]),
        'swap_rejection_rates': jnp.array([0.2]),
    }
    summary = summarize_diagnostics(diagnostics)
    text = format_diagnostics_summary(summary)

    assert 'MPPT diagnostics summary' in text
    assert 'robust=1.25' in text
    assert 'warnings: none' in text

    stream = io.StringIO()
    returned = print_diagnostics_summary(diagnostics, file=stream)
    assert returned == summary
    assert stream.getvalue().startswith('MPPT diagnostics summary')
