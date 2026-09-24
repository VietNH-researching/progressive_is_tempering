"""Shared estimation and diagnostic utilities.

Never imports from the core package (one-directional layering).
"""

from progressive_is_tempering.utils.smc_diagnostics import (
    C_INK,
    C_INK_2,
    C_REF,
    C_TEST,
    REF_FILL_ALPHA,
    exchangeable_degeneracy,
    honest_density_hist,
    plot_value_comparison,
    print_seed_quality_report,
    reference_anchored_window,
    smc_seed_quality_figure,
    style_axis,
)
from progressive_is_tempering.utils.metrics import (
    distance_w2,
    energy_w2,
    geometric_w2,
)
from progressive_is_tempering.utils.report import write_record

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
    "distance_w2",
    "energy_w2",
    "geometric_w2",
    "write_record",
]
