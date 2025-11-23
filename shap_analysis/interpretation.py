"""
Diagnostic rule extraction and interpretation functions.

This module provides tools to translate SHAP analysis into human-readable
diagnostic rules and natural language explanations.
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.tree import DecisionTreeClassifier, export_text


def generate_diagnostic_rules(
    shap_results, interaction_results, df_analysis, threshold_percentile=75
):
    """
    Generate human-readable diagnostic rules from SHAP analysis.

    This function analyzes SHAP contributions to extract interpretable rules
    such as "Cancer diagnosed when S_1 is LOW and S_3 × S_5 interaction is STRONG."

    Parameters
    ----------
    shap_results : dict
        Output from calculate_shap_values_with_original()
    interaction_results : dict
        Output from calculate_shap_interactions()
    df_analysis : pd.DataFrame
        Output from analyze_cancer_decision()
    threshold_percentile : int, default=75
        Percentile threshold for defining "significant" features/interactions

    Returns
    -------
    pd.DataFrame
        DataFrame containing diagnostic rules with columns:
        - feature: Feature name
        - cancer_mean_weight: Mean weight in cancer samples
        - noncancer_mean_weight: Mean weight in non-cancer samples
        - direction: HIGH/LOW/NORMAL relative to non-cancer
        - mean_shap_cancer: Mean SHAP contribution for cancer samples
        - p_value: Mann-Whitney U test p-value
        - interpretation: Human-readable rule string

    Notes
    -----
    This function performs:
    1. Mann-Whitney U test to identify discriminative features
    2. Pairwise interaction significance testing
    3. Per-sample diagnostic generation
    """
    # 1. Main effect rules: Which features discriminate cancer vs non-cancer?
    feature_names = shap_results["X_test_scaled"].columns
    cancer_mask = df_analysis["true_label"] == True
    noncancer_mask = df_analysis["true_label"] == False

    rules = []

    print("=" * 80)
    print("DIAGNOSTIC RULE EXTRACTION: SHAP → Clinical Interpretation")
    print("=" * 80)

    # For each feature, compare cancer vs non-cancer distributions
    for feat in feature_names:
        # Original values (physical weights)
        cancer_vals = df_analysis[cancer_mask][f"original_{feat}"]
        noncancer_vals = df_analysis[noncancer_mask][f"original_{feat}"]

        # SHAP contributions
        cancer_shap = df_analysis[cancer_mask][f"shap_{feat}"]
        noncancer_shap = df_analysis[noncancer_mask][f"shap_{feat}"]

        # Statistics
        cancer_mean = cancer_vals.mean()
        noncancer_mean = noncancer_vals.mean()
        shap_mean_cancer = cancer_shap.mean()

        # Discriminative power: How much does this feature separate classes?
        stat, pval = mannwhitneyu(cancer_vals, noncancer_vals, alternative="two-sided")

        # Rule generation logic
        diff_ratio = cancer_mean / (noncancer_mean + 1e-10)

        if pval < 0.05 and abs(shap_mean_cancer) > 0.1:  # Significant and important
            direction = (
                "HIGH" if diff_ratio > 1.2 else "LOW" if diff_ratio < 0.8 else "NORMAL"
            )

            rule = {
                "feature": feat,
                "cancer_mean_weight": cancer_mean,
                "noncancer_mean_weight": noncancer_mean,
                "direction": direction,
                "mean_shap_cancer": shap_mean_cancer,
                "p_value": pval,
                "interpretation": f"{feat} is {direction} in cancer cases",
            }
            rules.append(rule)

            print(
                f"\n[RULE] {feat} {'INCREASES' if diff_ratio > 1 else 'DECREASES'} "
                f"cancer probability:"
            )
            print(f"  - Cancer mean weight: {cancer_mean:.3f}")
            print(f"  - Non-cancer mean weight: {noncancer_mean:.3f}")
            print(f"  - Mean SHAP contribution (cancer): {shap_mean_cancer:+.3f}")
            print(f"  - Mann-Whitney p-value: {pval:.4f}")

    # 2. Interaction rules: Which pairs matter?
    print("\n" + "=" * 80)
    print("PAIRWISE INTERACTION RULES")
    print("=" * 80)

    top_interactions = interaction_results["interaction_importance"].head(5)

    for _, row in top_interactions.iterrows():
        feat_i = row["feature_i"]
        feat_j = row["feature_j"]
        interaction_strength = row["mean_abs_interaction"]
        interaction_sign = row["mean_interaction"]

        if interaction_strength > 0.05:  # Threshold for meaningful interaction
            effect = "SYNERGY" if interaction_sign > 0 else "ANTAGONISM"

            print(f"\n[INTERACTION] {feat_i} × {feat_j}: {effect}")
            print(f"  - Mean interaction SHAP: {interaction_sign:+.3f}")
            print(
                f"  - Interpretation: {feat_i} and {feat_j} "
                f"{'amplify' if effect=='SYNERGY' else 'counteract'} "
                f"each other's cancer signal"
            )

    # 3. Per-sample diagnostic generator
    def diagnose_sample(sample_idx):
        """Generate natural language diagnosis for specific sample."""
        row = df_analysis.iloc[sample_idx]

        diagnosis = []
        diagnosis.append(
            f"Sample {sample_idx}: Prediction = {row['prediction_prob']:.3f}, "
            f"True label = {'Cancer' if row['true_label'] else 'Non-cancer'}"
        )
        diagnosis.append("\nKey factors:")

        # Extract SHAP contributions
        shap_contribs = [(feat, row[f"shap_{feat}"]) for feat in feature_names]
        shap_contribs.sort(key=lambda x: abs(x[1]), reverse=True)

        for feat, shap_val in shap_contribs[:5]:  # Top 5
            original_val = row[f"original_{feat}"]
            direction = "↑" if shap_val > 0 else "↓"
            diagnosis.append(
                f"  {direction} {feat} = {original_val:.3f} (SHAP: {shap_val:+.3f})"
            )

        return "\n".join(diagnosis)

    print("\n" + "=" * 80)
    print("EXAMPLE SAMPLE-LEVEL DIAGNOSES")
    print("=" * 80)

    # Show 2 cancer and 2 non-cancer examples
    cancer_samples = df_analysis[cancer_mask].index[:2]
    noncancer_samples = df_analysis[noncancer_mask].index[:2]

    for idx in list(cancer_samples) + list(noncancer_samples):
        sample_pos = df_analysis.index.get_loc(idx)
        print(f"\n{diagnose_sample(sample_pos)}")

    return pd.DataFrame(rules)


def extract_shap_decision_tree_rules(shap_results, df_analysis, max_depth=3):
    """
    Fit a shallow decision tree to SHAP values to extract interpretable rules.

    This approach fits a decision tree on SHAP contributions rather than
    raw features, providing a high-level summary of the decision logic.

    Parameters
    ----------
    shap_results : dict
        Output from calculate_shap_values_with_original()
    df_analysis : pd.DataFrame
        Output from analyze_cancer_decision()
    max_depth : int, default=3
        Maximum depth of decision tree (controls rule complexity)

    Returns
    -------
    tuple
        (tree : DecisionTreeClassifier, rules : str)
        - tree: Fitted decision tree model
        - rules: Text representation of decision rules

    Notes
    -----
    - Tree is fit on SHAP values, not raw features
    - Useful for discovering high-level decision patterns
    - Shallow trees (depth 2-4) provide most interpretable rules
    """
    # Features: SHAP contributions per component
    feature_names = shap_results["X_test_scaled"].columns
    X_shap = np.column_stack(
        [df_analysis[f"shap_{feat}"].values for feat in feature_names]
    )
    y = df_analysis["true_label"].values

    # Fit decision tree
    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=10, random_state=42
    )
    tree.fit(X_shap, y)

    # Extract text rules
    rules = export_text(tree, feature_names=[f"SHAP({feat})" for feat in feature_names])

    print("=" * 80)
    print("DECISION TREE RULES FROM SHAP VALUES")
    print("=" * 80)
    print(rules)

    return tree, rules


def generate_sample_explanation(
    shap_results, interaction_results, df_analysis, sample_idx, top_k=5
):
    """
    Generate detailed natural language explanation for a single sample.

    Parameters
    ----------
    shap_results : dict
        Output from calculate_shap_values_with_original()
    interaction_results : dict
        Output from calculate_shap_interactions()
    df_analysis : pd.DataFrame
        Output from analyze_cancer_decision()
    sample_idx : int
        Index of sample to explain
    top_k : int, default=5
        Number of top contributing features to include

    Returns
    -------
    str
        Multi-paragraph natural language explanation

    Example
    -------
    "Sample 42 is predicted as Cancer with 78% confidence.

    Main contributing factors:
    - S_3 (hydroxyapatite) is HIGH (weight=0.65, +0.42 SHAP)
    - S_1 (collagen) is LOW (weight=0.12, -0.38 SHAP)
    - S_5 (disorder marker) is ELEVATED (weight=0.28, +0.31 SHAP)

    Key interactions:
    - S_3 × S_5 show strong synergy (+0.15 SHAP interaction)

    Overall: The combination of low collagen and high mineralization
    with structural disorder strongly indicates cancer."
    """
    row = df_analysis.iloc[sample_idx]
    feature_names = shap_results["X_test_scaled"].columns

    # Header
    prediction_class = "Cancer" if row["predicted_label"] else "Non-cancer"
    true_class = "Cancer" if row["true_label"] else "Non-cancer"
    correct = "✓" if row["correct"] else "✗"

    explanation = []
    explanation.append(f"=== Sample {sample_idx} Explanation {correct} ===\n")
    explanation.append(
        f"Prediction: {prediction_class} ({row['prediction_prob']:.1%} confidence)"
    )
    explanation.append(f"Ground Truth: {true_class}\n")

    # Main contributors
    explanation.append("Main Contributing Factors:")
    shap_contribs = [(feat, row[f"shap_{feat}"]) for feat in feature_names]
    shap_contribs.sort(key=lambda x: abs(x[1]), reverse=True)

    for feat, shap_val in shap_contribs[:top_k]:
        original_val = row[f"original_{feat}"]
        direction = "positive" if shap_val > 0 else "negative"
        explanation.append(
            f"  • {feat}: weight={original_val:.3f}, SHAP={shap_val:+.3f} ({direction})"
        )

    # Top interactions for this sample
    explanation.append("\nKey Pairwise Interactions:")
    shap_int_vals = interaction_results["shap_interaction_values"][sample_idx]
    n_features = len(feature_names)

    interactions_sample = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions_sample.append(
                (feature_names[i], feature_names[j], shap_int_vals[i, j])
            )

    interactions_sample.sort(key=lambda x: abs(x[2]), reverse=True)

    for feat_i, feat_j, int_val in interactions_sample[:3]:
        effect = "synergy" if int_val > 0 else "antagonism"
        explanation.append(f"  • {feat_i} × {feat_j}: {int_val:+.3f} ({effect})")

    return "\n".join(explanation)
