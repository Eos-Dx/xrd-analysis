"""
SHAP Analysis Package for Cancer Detection using XRD-SKana Features

This package provides comprehensive SHAP-based interpretability tools for
tree-based machine learning models predicting cancer from XRD spectroscopy data.

Modules:
    core: Main SHAP calculation functions (main effects and interactions)
    visualization: Plotting functions for SHAP analysis
    interpretation: Diagnostic rule extraction and natural language explanations
    validation: Model validation and SHAP stability checks
"""

from shap_analysis.core import (
    calculate_shap_interactions,
    calculate_shap_values_with_original,
)
from shap_analysis.higher_order import (
    analyze_conditional_interactions,
    calculate_shapley_taylor_interactions,
    calculate_treeshap_iq,
    visualize_higher_order_interactions,
)
from shap_analysis.interpretation import (
    extract_shap_decision_tree_rules,
    generate_diagnostic_rules,
    generate_sample_explanation,
)
from shap_analysis.validation import validate_shap_additivity
from shap_analysis.visualization import (
    plot_shap_interaction_heatmap,
    plot_shap_summary,
    plot_top_interactions,
)

__version__ = "0.1.0"

__all__ = [
    "calculate_shap_values_with_original",
    "calculate_shap_interactions",
    "plot_shap_interaction_heatmap",
    "plot_shap_summary",
    "plot_top_interactions",
    "generate_diagnostic_rules",
    "generate_sample_explanation",
    "extract_shap_decision_tree_rules",
    "validate_shap_additivity",
    # Higher-order interactions
    "calculate_shapley_taylor_interactions",
    "calculate_treeshap_iq",
    "analyze_conditional_interactions",
    "visualize_higher_order_interactions",
]
