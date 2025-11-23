"""
Higher-order SHAP interaction analysis.

This module provides tools for analyzing 3-way, 4-way, and higher-order
feature interactions using TreeSHAP-IQ and approximation methods.
"""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd


def calculate_shapley_taylor_interactions(
    model,
    X,
    order=3,
    max_features=None,
    n_samples=100,
    feature_names=None,
    top_k=20,
):
    """
    Compute higher-order Shapley-Taylor interactions (approximation).

    This method approximates higher-order interactions by computing products
    of SHAP values. It's fast but approximate - exact computation requires
    TreeSHAP-IQ (see calculate_treeshap_iq).

    Parameters
    ----------
    model : trained model
        Tree-based model (LightGBM, RandomForest, XGBoost)
    X : pd.DataFrame or np.ndarray
        Feature matrix (preprocessed/scaled)
    order : int, default=3
        Interaction order (3 for S1×S2×S3, 4 for S1×S2×S3×S4)
    max_features : int, optional
        Limit to first N features to avoid combinatorial explosion.
        If None, uses all features (careful with order > 3!)
    n_samples : int, default=100
        Number of samples to analyze
    feature_names : list of str, optional
        Feature names. If None, uses X.columns or generic names
    top_k : int, default=20
        Return only top K strongest interactions

    Returns
    -------
    pd.DataFrame
        Ranked interactions with columns:
        - 'features': tuple of feature names
        - 'interaction_strength': mean interaction value
        - 'abs_strength': absolute value (for ranking)

    Notes
    -----
    Computational complexity: O(C(n, k) × m) where n=features, k=order, m=samples
    For n=10, order=3: 120 combinations
    For n=10, order=4: 210 combinations
    For n=20, order=3: 1140 combinations (getting expensive!)

    Example
    -------
    >>> interactions_3way = calculate_shapley_taylor_interactions(
    ...     model=trained_model,
    ...     X=X_test,
    ...     order=3,
    ...     max_features=5,
    ...     n_samples=100
    ... )
    >>> print(interactions_3way.head())
    """
    import shap

    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)

    # Subset data
    X_subset = X.iloc[:n_samples]
    n_features = X_subset.shape[1]
    n_samples_actual = X_subset.shape[
        0
    ]  # Use actual size (may be < n_samples if data smaller)

    # Limit features if specified
    if max_features is not None and max_features < n_features:
        X_subset = X_subset.iloc[:, :max_features]
        n_features = max_features

    # Validate order
    if order > n_features:
        raise ValueError(f"Order {order} > number of features {n_features}")

    n_combinations = len(list(combinations(range(n_features), order)))
    if n_combinations > 1000:
        warnings.warn(
            f"Computing {n_combinations} combinations of order {order}. "
            f"This may be slow. Consider reducing max_features or order."
        )

    # Get base SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_subset)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class

    print(f"Computing {order}-way interactions...")
    print(f"  Features: {n_features}")
    print(f"  Samples: {n_samples_actual} (requested: {n_samples})")
    print(f"  Combinations: {n_combinations}")

    # Compute higher-order interactions
    interactions = []

    feature_indices = list(range(n_features))

    for feature_tuple in combinations(feature_indices, order):
        feature_names_tuple = tuple(X_subset.columns[i] for i in feature_tuple)

        # Approximate interaction: mean product of SHAP values
        # This estimates joint contribution beyond individual effects
        interaction_values = np.array(
            [
                np.prod(
                    [shap_values[sample_idx, feat_idx] for feat_idx in feature_tuple]
                )
                for sample_idx in range(n_samples_actual)  # Use actual subset size
            ]
        )

        interaction_strength = interaction_values.mean()

        interactions.append(
            {
                "features": feature_names_tuple,
                "interaction_strength": interaction_strength,
                "abs_strength": abs(interaction_strength),
                "std": interaction_values.std(),
            }
        )

    # Convert to DataFrame and sort
    df_interactions = pd.DataFrame(interactions).sort_values(
        "abs_strength", ascending=False
    )

    print(f"✓ Computed {len(df_interactions)} interactions")

    return df_interactions.head(top_k).reset_index(drop=True)


def calculate_treeshap_iq(
    model,
    X,
    max_order=3,
    n_samples=50,
    interaction_type="shapley_taylor",
    feature_names=None,
):
    """
    Compute exact higher-order SHAP interactions using TreeSHAP-IQ.

    TreeSHAP-IQ provides **exact** computation of higher-order interactions
    for tree-based models, unlike the approximation in Shapley-Taylor.

    Parameters
    ----------
    model : trained model
        Tree-based model (LightGBM, RandomForest, XGBoost)
    X : pd.DataFrame or np.ndarray
        Feature matrix (preprocessed/scaled)
    max_order : int, default=3
        Maximum interaction order to compute
        Warning: 4+ is very expensive!
    n_samples : int, default=50
        Number of samples (kept small - this is expensive!)
    interaction_type : str, default='shapley_taylor'
        Type of interaction index: 'shapley_taylor', 'banzhaf', 'faith_shap'
    feature_names : list of str, optional
        Feature names

    Returns
    -------
    dict with keys:
        - 'explainer': TreeSHAPIQ explainer object
        - 'interactions': Computed interaction values
        - 'top_interactions_per_order': Dict mapping order -> top interactions

    Notes
    -----
    Requires: pip install shapiq

    TreeSHAP-IQ is computationally expensive:
    - Order 2: Fast
    - Order 3: Moderate (seconds per sample)
    - Order 4+: Slow (minutes per sample)

    For production use, run on small subset first!

    Example
    -------
    >>> result = calculate_treeshap_iq(
    ...     model=trained_model,
    ...     X=X_test,
    ...     max_order=3,
    ...     n_samples=20
    ... )
    >>> print(result['top_interactions_per_order'][3])
    """
    try:
        from shapiq import TreeSHAPIQ
    except ImportError:
        raise ImportError("TreeSHAP-IQ not installed. Install with: pip install shapiq")

    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)

    # Subset data
    X_subset = X.iloc[:n_samples]
    n_samples_actual = X_subset.shape[0]  # Use actual size

    print(f"TreeSHAP-IQ: Exact {max_order}-order interactions")
    print(f"  Samples: {n_samples_actual} (requested: {n_samples})")
    print(f"  Features: {X_subset.shape[1]}")
    print(f"  Interaction type: {interaction_type}")
    print("  ⚠ This may take several minutes...")

    # Create TreeSHAPIQ explainer
    explainer_iq = TreeSHAPIQ(
        model=model, max_order=max_order, interaction_type=interaction_type
    )

    # Compute interaction values
    interactions_iq = explainer_iq.explain(X_subset.values)

    print("✓ TreeSHAP-IQ computation complete")

    # Extract top interactions per order
    top_interactions_per_order = {}

    for order in range(1, max_order + 1):
        try:
            order_values = interactions_iq.get_n_order_values(order)

            # Convert to list of tuples
            interactions_list = []
            for features, value in order_values:
                feature_names_tuple = tuple(X_subset.columns[i] for i in features)
                interactions_list.append((feature_names_tuple, value))

            # Sort by absolute value
            interactions_list.sort(key=lambda x: abs(x[1]), reverse=True)

            top_interactions_per_order[order] = interactions_list[:20]  # Top 20

            print(f"\nOrder {order}: Found {len(order_values)} interactions")
            print(f"  Top 3:")
            for features, value in interactions_list[:3]:
                features_str = " × ".join(features)
                print(f"    {features_str}: {value:+.4f}")

        except Exception as e:
            warnings.warn(f"Could not extract order {order} interactions: {e}")

    return {
        "explainer": explainer_iq,
        "interactions": interactions_iq,
        "top_interactions_per_order": top_interactions_per_order,
        "X_subset": X_subset,
    }


def analyze_conditional_interactions(
    df_analysis, shap_results, feature_names, min_samples=10
):
    """
    Analyze 3-way interactions through conditional analysis.

    Examines how one feature's SHAP effect changes when two other features
    are jointly high/low. This provides intuitive interpretation of
    higher-order interactions.

    Parameters
    ----------
    df_analysis : pd.DataFrame
        Output from analyze_cancer_decision()
    shap_results : dict
        Output from calculate_shap_values_with_original()
    feature_names : list of str
        Names of features to analyze
    min_samples : int, default=10
        Minimum samples required for each condition

    Returns
    -------
    pd.DataFrame
        Conditional interaction analysis with columns:
        - 'feature': Primary feature
        - 'condition_1', 'condition_2': Conditioning features
        - 'interaction_strength': Effect change under condition
        - 'effect_conditional': SHAP effect when conditions met
        - 'effect_overall': Overall SHAP effect
        - 'n_samples': Number of samples meeting condition

    Example
    -------
    >>> cond_int = analyze_conditional_interactions(
    ...     df_analysis, shap_results, ['S_1', 'S_2', 'S_3']
    ... )
    >>> print(cond_int.head())
    """
    print("=" * 80)
    print("CONDITIONAL 3-WAY INTERACTION ANALYSIS")
    print("=" * 80)

    results = []

    for i, feat_i in enumerate(feature_names):
        for j, feat_j in enumerate(feature_names):
            if i >= j:
                continue
            for k, feat_k in enumerate(feature_names):
                if k in [i, j]:
                    continue

                # Condition: feat_j AND feat_k both high (above median)
                median_j = df_analysis[f"original_{feat_j}"].median()
                median_k = df_analysis[f"original_{feat_k}"].median()

                cond_j = df_analysis[f"original_{feat_j}"] > median_j
                cond_k = df_analysis[f"original_{feat_k}"] > median_k
                condition = cond_j & cond_k

                n_samples_cond = condition.sum()
                if n_samples_cond < min_samples:
                    continue

                # Effect of feat_i under condition vs overall
                effect_conditional = df_analysis[condition][f"shap_{feat_i}"].mean()
                effect_overall = df_analysis[f"shap_{feat_i}"].mean()
                interaction_strength = abs(effect_conditional - effect_overall)

                results.append(
                    {
                        "feature": feat_i,
                        "condition_1": feat_j,
                        "condition_2": feat_k,
                        "interaction_strength": interaction_strength,
                        "effect_conditional": effect_conditional,
                        "effect_overall": effect_overall,
                        "effect_change": effect_conditional - effect_overall,
                        "n_samples": n_samples_cond,
                    }
                )

    df_conditional = pd.DataFrame(results).sort_values(
        "interaction_strength", ascending=False
    )

    print(f"\n✓ Analyzed {len(df_conditional)} conditional interactions")
    print(f"\nTop 10 strongest conditional 3-way interactions:")
    print(df_conditional.head(10).to_string(index=False))

    return df_conditional


def visualize_higher_order_interactions(interactions_df, order=3, top_k=15):
    """
    Visualize higher-order interactions as a bar plot.

    Parameters
    ----------
    interactions_df : pd.DataFrame
        Output from calculate_shapley_taylor_interactions()
    order : int
        Interaction order (for title)
    top_k : int, default=15
        Number of top interactions to display

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    df_plot = interactions_df.head(top_k).copy()

    # Create labels
    df_plot["feature_label"] = df_plot["features"].apply(lambda x: " × ".join(x))

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["red" if x > 0 else "blue" for x in df_plot["interaction_strength"]]

    ax.barh(
        df_plot["feature_label"],
        df_plot["interaction_strength"],
        color=colors,
        alpha=0.7,
    )
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel(f"{order}-Way Interaction Strength", fontsize=12)
    ax.set_title(
        f"Top {top_k} {order}-Way Feature Interactions\n(Shapley-Taylor Approximation)",
        fontsize=13,
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    return fig
