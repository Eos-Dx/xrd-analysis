"""
Visualization functions for SHAP analysis results.

This module provides plotting functions for SHAP interaction heatmaps,
top interaction rankings, and other SHAP-related visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_shap_interaction_heatmap(shap_interaction_results, figsize=(10, 8)):
    """
    Visualize SHAP interaction strength matrix as heatmap.

    Parameters
    ----------
    shap_interaction_results : dict
        Output from calculate_shap_interactions() containing 'interaction_matrix'
        and 'feature_names' keys
    figsize : tuple, default=(10, 8)
        Figure size (width, height) in inches

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the heatmap

    Notes
    -----
    - Diagonal elements represent main effects
    - Off-diagonal elements represent pairwise interactions
    - Symmetric matrix (interaction_ij = interaction_ji)
    """
    matrix = shap_interaction_results["interaction_matrix"]
    feature_names = shap_interaction_results["feature_names"]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".3f",
        square=True,
        cbar_kws={"label": "Mean |SHAP Interaction|"},
        ax=ax,
    )

    ax.set_title(
        "SHAP Interaction Strength Matrix\n(Pairwise SKana Component Interactions)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


def plot_top_interactions(shap_interaction_results, top_k=10, figsize=(10, 6)):
    """
    Bar plot of strongest pairwise SHAP interactions.

    Parameters
    ----------
    shap_interaction_results : dict
        Output from calculate_shap_interactions() containing 'interaction_importance'
    top_k : int, default=10
        Number of top interactions to display
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the bar plot

    Notes
    -----
    - Interactions are ranked by mean absolute SHAP interaction value
    - Higher values indicate stronger synergy or antagonism between features
    """
    df = shap_interaction_results["interaction_importance"].head(top_k)

    fig, ax = plt.subplots(figsize=figsize)

    # Create labels for pairs
    labels = [f"{row['feature_i']} × {row['feature_j']}" for _, row in df.iterrows()]

    ax.barh(labels, df["mean_abs_interaction"], color="steelblue", alpha=0.7)
    ax.set_xlabel("Mean |SHAP Interaction|", fontsize=12)
    ax.set_title(
        f"Top {top_k} SKana Component Interactions for Cancer Prediction",
        fontsize=13,
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    return fig


def plot_shap_summary(shap_results, max_display=10, figsize=(10, 6)):
    """
    Create SHAP summary plot (beeswarm) using shap library.

    Parameters
    ----------
    shap_results : dict
        Output from calculate_shap_values_with_original()
    max_display : int, default=10
        Maximum number of features to display
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object

    Notes
    -----
    This uses the shap library's built-in summary_plot function.
    """
    import shap

    fig, ax = plt.subplots(figsize=figsize)

    shap.summary_plot(
        shap_results["shap_values"],
        shap_results["X_test_scaled"],
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()

    return fig


def plot_shap_force_plot_sample(
    shap_results, sample_idx=0, matplotlib=True, figsize=(20, 3)
):
    """
    Create force plot for individual sample explanation.

    Parameters
    ----------
    shap_results : dict
        Output from calculate_shap_values_with_original()
    sample_idx : int, default=0
        Index of sample to explain
    matplotlib : bool, default=True
        If True, use matplotlib backend; if False, use JavaScript (interactive)
    figsize : tuple, default=(20, 3)
        Figure size for matplotlib backend

    Returns
    -------
    matplotlib.figure.Figure or shap.Explanation
        Depending on matplotlib parameter
    """
    import shap

    if matplotlib:
        shap.initjs()
        fig = shap.force_plot(
            shap_results["expected_value"],
            shap_results["shap_values"][sample_idx, :],
            shap_results["X_test_scaled"].iloc[sample_idx, :],
            matplotlib=True,
            figsize=figsize,
            show=False,
        )
        return fig
    else:
        return shap.force_plot(
            shap_results["expected_value"],
            shap_results["shap_values"][sample_idx, :],
            shap_results["X_test_scaled"].iloc[sample_idx, :],
        )


def plot_shap_dependence(
    shap_results, feature, interaction_feature=None, figsize=(8, 6)
):
    """
    Create dependence plot showing SHAP value vs feature value.

    Parameters
    ----------
    shap_results : dict
        Output from calculate_shap_values_with_original()
    feature : str
        Feature name to plot
    interaction_feature : str, optional
        Feature to color points by (for interaction detection)
    figsize : tuple, default=(8, 6)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import shap

    fig, ax = plt.subplots(figsize=figsize)

    shap.dependence_plot(
        feature,
        shap_results["shap_values"],
        shap_results["X_test_scaled"],
        interaction_index=interaction_feature,
        show=False,
        ax=ax,
    )

    plt.tight_layout()

    return fig
