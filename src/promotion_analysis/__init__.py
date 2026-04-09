"""
Promotion sensitivity analysis and scenario plotting.
"""

from .promotion_sensitivity import (
    estimate_family_promotion_sensitivity,
    estimate_panel_promotion_sensitivity,
    compute_revenue_proxy_curve,
)
from .promotion_plots import (
    plot_family_promotion_sensitivity,
    plot_revenue_proxy_curve,
    plot_scenario_comparison,
    plot_simulation_output,
)

__all__ = [
    "estimate_family_promotion_sensitivity",
    "estimate_panel_promotion_sensitivity",
    "compute_revenue_proxy_curve",
    "plot_family_promotion_sensitivity",
    "plot_revenue_proxy_curve",
    "plot_scenario_comparison",
    "plot_simulation_output",
]