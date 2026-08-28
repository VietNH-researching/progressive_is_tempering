from __future__ import annotations

import sys
from typing import Any, NamedTuple, TextIO

import numpy as np


_LOGZ_PREFIX = 'logZ_'
_PREFERRED_LOGZ_KEYS = ('SMC_like', 'robust', 'reverse_logistic')


class DiagnosticsSummary(NamedTuple):
    r"""Compact, plain-Python summary of one MPPT diagnostics dictionary.

    ``parallel_tempering`` and ``pt_manual_runners`` intentionally return raw
    collector outputs so advanced users can keep full arrays and custom fields.
    This summary is a lightweight UI helper for logs, checkpoints, and quick
    sanity checks. It does not certify convergence; it only flags obvious
    numerical or tuning pathologies.
    """
    logZ: dict[str, float]
    logZ_range: float | None
    mcmc_acc_rates: list[float] | None
    mcmc_acc_min_nonref: float | None
    mcmc_acc_max_nonref: float | None
    swap_rejection_rates: list[float] | None
    swap_rej_min: float | None
    swap_rej_max: float | None
    mcmc_niter: int | float | list[float] | None
    warnings: tuple[str, ...]


def summarize_diagnostics(
        diagnostics: dict[str, Any], *,
        logZ_warning_threshold: float = 1.0,
        low_mcmc_acc_threshold: float = 0.05,
        high_mcmc_acc_threshold: float = 0.99,
        high_swap_rej_threshold: float = 0.90,
) -> DiagnosticsSummary:
    r"""Summarize one MPPT diagnostics dictionary and flag obvious issues.

    Parameters are deliberately soft thresholds, not statistical guarantees.
    Monte Carlo diagnostics are problem dependent; the helper only catches
    clear failure modes that are easy to miss in raw collector arrays:

    * non-finite diagnostic values,
    * large disagreement between logZ estimators,
    * near-zero or near-one MCMC acceptance away from the reference level,
    * near-total swap rejection across a temperature gap.

    The returned ``DiagnosticsSummary`` contains only Python scalars/lists, so
    it can be printed, logged, or serialized alongside checkpoints.
    """
    warnings: list[str] = []

    logZ = _extract_logZ(diagnostics, warnings)
    logZ_range = _range_or_none(list(logZ.values()))
    if logZ_range is not None and logZ_range > logZ_warning_threshold:
        warnings.append(
            f'logZ estimators disagree by {logZ_range:.3g}, '
            f'above threshold {logZ_warning_threshold:.3g}'
        )

    mcmc_acc_rates = _optional_float_list(
        diagnostics, 'mcmc_acc_rates', warnings
    )
    mcmc_acc_min_nonref = None
    mcmc_acc_max_nonref = None
    if mcmc_acc_rates is not None:
        nonref = mcmc_acc_rates[1:] if len(mcmc_acc_rates) > 1 else mcmc_acc_rates
        if nonref:
            mcmc_acc_min_nonref = min(nonref)
            mcmc_acc_max_nonref = max(nonref)
            if mcmc_acc_min_nonref < low_mcmc_acc_threshold:
                warnings.append(
                    f'MCMC acceptance below {low_mcmc_acc_threshold:.3g} '
                    f'away from reference: min={mcmc_acc_min_nonref:.3g}'
                )
            if mcmc_acc_max_nonref > high_mcmc_acc_threshold:
                warnings.append(
                    f'MCMC acceptance above {high_mcmc_acc_threshold:.3g} '
                    f'away from reference: max={mcmc_acc_max_nonref:.3g}'
                )

    swap_rejection_rates = _optional_float_list(
        diagnostics, 'swap_rejection_rates', warnings
    )
    swap_rej_min = None
    swap_rej_max = None
    if swap_rejection_rates is not None and swap_rejection_rates:
        swap_rej_min = min(swap_rejection_rates)
        swap_rej_max = max(swap_rejection_rates)
        if swap_rej_max > high_swap_rej_threshold:
            warnings.append(
                f'swap rejection above {high_swap_rej_threshold:.3g}: '
                f'max={swap_rej_max:.3g}'
            )

    mcmc_niter = _optional_scalar_or_list(diagnostics, 'mcmc_niter', warnings)

    return DiagnosticsSummary(
        logZ=logZ,
        logZ_range=logZ_range,
        mcmc_acc_rates=mcmc_acc_rates,
        mcmc_acc_min_nonref=mcmc_acc_min_nonref,
        mcmc_acc_max_nonref=mcmc_acc_max_nonref,
        swap_rejection_rates=swap_rejection_rates,
        swap_rej_min=swap_rej_min,
        swap_rej_max=swap_rej_max,
        mcmc_niter=mcmc_niter,
        warnings=tuple(warnings),
    )


def format_diagnostics_summary(summary: DiagnosticsSummary) -> str:
    r"""Return a stable multi-line text representation of a summary."""
    lines = ['MPPT diagnostics summary']
    if summary.logZ:
        ordered = _ordered_logZ_items(summary.logZ)
        logZ_text = ', '.join(f'{name}={value:.6g}' for name, value in ordered)
        if summary.logZ_range is not None:
            logZ_text += f' (range={summary.logZ_range:.3g})'
        lines.append(f'  logZ: {logZ_text}')
    else:
        lines.append('  logZ: not collected')

    if summary.mcmc_acc_rates is not None:
        lines.append(
            '  MCMC acceptance: '
            f'min_nonref={_fmt_optional(summary.mcmc_acc_min_nonref)}, '
            f'max_nonref={_fmt_optional(summary.mcmc_acc_max_nonref)}'
        )
    else:
        lines.append('  MCMC acceptance: not collected')

    if summary.swap_rejection_rates is not None:
        lines.append(
            '  swap rejection: '
            f'min={_fmt_optional(summary.swap_rej_min)}, '
            f'max={_fmt_optional(summary.swap_rej_max)}'
        )
    else:
        lines.append('  swap rejection: not collected')

    if summary.mcmc_niter is not None:
        lines.append(f'  MCMC niter: {summary.mcmc_niter}')
    else:
        lines.append('  MCMC niter: not collected')

    if summary.warnings:
        lines.append('  warnings:')
        lines.extend(f'    - {warning}' for warning in summary.warnings)
    else:
        lines.append('  warnings: none')
    return '\n'.join(lines)


def print_diagnostics_summary(
        diagnostics: dict[str, Any], *,
        file: TextIO | None = None,
        **summary_kwargs,
) -> DiagnosticsSummary:
    r"""Print and return ``summarize_diagnostics(diagnostics, **kwargs)``.

    Returning the summary keeps this helper useful in scripts where the same
    object should be logged and later saved to a checkpoint manifest.
    """
    if file is None:
        file = sys.stdout
    summary = summarize_diagnostics(diagnostics, **summary_kwargs)
    print(format_diagnostics_summary(summary), file=file)
    return summary


def _extract_logZ(diagnostics: dict[str, Any], warnings: list[str]) -> dict[str, float]:
    raw = {}
    for key, value in diagnostics.items():
        if key.startswith(_LOGZ_PREFIX):
            raw[key[len(_LOGZ_PREFIX):]] = _scalar_float(key, value, warnings)
    return dict(_ordered_logZ_items(raw))


def _ordered_logZ_items(logZ: dict[str, float]):
    preferred = [(name, logZ[name]) for name in _PREFERRED_LOGZ_KEYS if name in logZ]
    extras = sorted(
        (name, value) for name, value in logZ.items()
        if name not in _PREFERRED_LOGZ_KEYS
    )
    return preferred + extras


def _optional_float_list(
        diagnostics: dict[str, Any], key: str, warnings: list[str]
) -> list[float] | None:
    if key not in diagnostics:
        return None
    values = _float_list(key, diagnostics[key], warnings)
    return values


def _optional_scalar_or_list(
        diagnostics: dict[str, Any], key: str, warnings: list[str]
) -> int | float | list[float] | None:
    if key not in diagnostics:
        return None
    arr = _as_numpy(key, diagnostics[key], warnings)
    values = [float(v) for v in arr.ravel()]
    if not np.all(np.isfinite(values)):
        warnings.append(f"diagnostic '{key}' contains non-finite values")
    if arr.shape == ():
        value = values[0]
        return int(value) if value.is_integer() else value
    return values


def _scalar_float(key: str, value, warnings: list[str]) -> float:
    arr = _as_numpy(key, value, warnings)
    if arr.shape != ():
        raise ValueError(f"diagnostic '{key}' must be scalar (got shape {arr.shape})")
    ret = float(arr)
    if not np.isfinite(ret):
        warnings.append(f"diagnostic '{key}' is not finite")
    return ret


def _float_list(key: str, value, warnings: list[str]) -> list[float]:
    arr = _as_numpy(key, value, warnings)
    ret = [float(v) for v in arr.ravel()]
    if not np.all(np.isfinite(ret)):
        warnings.append(f"diagnostic '{key}' contains non-finite values")
    return ret


def _as_numpy(key: str, value, warnings: list[str]) -> np.ndarray:
    try:
        arr = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"diagnostic '{key}' cannot be converted to an array") from exc
    if arr.dtype == object:
        warnings.append(f"diagnostic '{key}' has object dtype")
    return arr


def _range_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values) - min(values)


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return 'n/a'
    return f'{value:.3g}'
