"""
Validation utilities for SHAP analysis.

This module provides functions to validate SHAP computations and assess
the stability of SHAP-based interpretations.
"""

import warnings

import numpy as np
import pandas as pd


def validate_shap_additivity(shap_results, tolerance=0.01, verbose=True):
    """
    Validate that SHAP values satisfy the additivity property.

    The additivity property states that:
    prediction(x) = base_value + sum(shap_values)

    Or in logit space for binary classification:
    logit(prediction) ≈ base_value + sum(shap_values)

    Parameters
    ----------
    shap_results : dict
        Output from calculate_shap_values_with_original() or calculate_shap_interactions()
    tolerance : float, default=0.01
        Maximum acceptable error for additivity check
    verbose : bool, default=True
        If True, print detailed validation results

    Returns
    -------
    dict
        Validation results with keys:
        - 'passed': bool, whether all samples pass validation
        - 'max_error': float, maximum additivity error across samples
        - 'mean_error': float, mean absolute error
        - 'failed_samples': list of int, indices of samples that failed
        - 'errors': np.ndarray, per-sample errors

    Notes
    -----
    For tree-based models, SHAP values should exactly satisfy additivity
    (up to numerical precision). Large errors indicate potential issues
    with the explainer or model.
    """
    shap_values = shap_results["shap_values"]
    predictions = shap_results["predictions"]
    expected_value = shap_results["expected_value"]

    n_samples = len(predictions)
    errors = np.zeros(n_samples)

    for i in range(n_samples):
        # Convert prediction probability to logit
        pred_prob = predictions[i]
        pred_logit = np.log(pred_prob / (1 - pred_prob + 1e-10))

        # Compute expected logit from SHAP
        shap_logit = expected_value + shap_values[i].sum()

        # Error
        errors[i] = abs(pred_logit - shap_logit)

    max_error = errors.max()
    mean_error = errors.mean()
    failed_samples = np.where(errors > tolerance)[0].tolist()
    passed = len(failed_samples) == 0

    if verbose:
        print("=" * 80)
        print("SHAP ADDITIVITY VALIDATION")
        print("=" * 80)
        print(f"Samples validated: {n_samples}")
        print(f"Tolerance: {tolerance}")
        print(f"Max error: {max_error:.6f}")
        print(f"Mean error: {mean_error:.6f}")
        print(f"Failed samples: {len(failed_samples)} / {n_samples}")

        if not passed:
            warnings.warn(
                f"{len(failed_samples)} samples failed additivity check "
                f"(max error: {max_error:.6f}). This may indicate numerical "
                f"issues with the SHAP explainer."
            )
            print(f"\nFailed sample indices: {failed_samples[:10]}...")  # Show first 10
        else:
            print("\n✓ All samples passed additivity validation")

    return {
        "passed": passed,
        "max_error": max_error,
        "mean_error": mean_error,
        "failed_samples": failed_samples,
        "errors": errors,
    }


def evaluate_shap_stability_cv(
    pipeline, df, y_column="isCancerDiagnosed", n_folds=5, **shap_kwargs
):
    """
    Evaluate SHAP value stability across cross-validation folds.

    This function computes SHAP values across multiple CV folds and assesses
    consistency of feature importance rankings and interaction strengths.

    Parameters
    ----------
    pipeline : MLPipeline
        Trained pipeline
    df : pd.DataFrame
        Full dataset
    y_column : str, default='isCancerDiagnosed'
        Target column name
    n_folds : int, default=5
        Number of cross-validation folds
    **shap_kwargs : dict
        Additional arguments passed to calculate_shap_values_with_original()

    Returns
    -------
    dict
        Stability metrics with keys:
        - 'importance_per_fold': list of pd.DataFrame, feature importance per fold
        - 'importance_mean': pd.DataFrame, mean importance across folds
        - 'importance_std': pd.DataFrame, standard deviation of importance
        - 'rank_correlation': float, mean Spearman correlation of rankings across folds
        - 'cv_coefficient': float, coefficient of variation (std/mean) averaged across features

    Notes
    -----
    High stability (CV < 0.3, rank correlation > 0.8) indicates robust interpretations.
    Low stability suggests the model or SHAP values are sensitive to data splits.
    """
    from sklearn.model_selection import StratifiedKFold

    from shap_analysis.core import calculate_shap_values_with_original

    print("=" * 80)
    print(f"SHAP STABILITY ANALYSIS ({n_folds}-Fold Cross-Validation)")
    print("=" * 80)

    importance_per_fold = []
    y = df[y_column]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, y)):
        print(f"\nProcessing fold {fold_idx + 1}/{n_folds}...")

        # Train pipeline on this fold
        df_fold = df.iloc[train_idx]

        # Retrain (assuming pipeline has train method)
        try:
            pipeline.train(
                df_fold,
                y_column=y_column,
                split=False,  # Already split
                print_flag=False,
                show_flag=False,
            )
        except Exception as e:
            warnings.warn(f"Could not retrain pipeline on fold {fold_idx}: {e}")
            continue

        # Calculate SHAP on test fold
        df_test = df.iloc[test_idx]

        shap_results = calculate_shap_values_with_original(
            pipeline, df_test, y_column=y_column, **shap_kwargs
        )

        importance_per_fold.append(shap_results["importance_df"])

    # Compute stability metrics
    if len(importance_per_fold) == 0:
        warnings.warn("No folds successfully computed SHAP values")
        return None

    # Align feature names across folds
    feature_names = importance_per_fold[0]["feature"].tolist()

    # Build importance matrix (n_folds x n_features)
    importance_matrix = np.zeros((len(importance_per_fold), len(feature_names)))

    for fold_idx, imp_df in enumerate(importance_per_fold):
        for feat_idx, feat in enumerate(feature_names):
            importance_matrix[fold_idx, feat_idx] = imp_df[imp_df["feature"] == feat][
                "mean_abs_shap"
            ].values[0]

    # Mean and std across folds
    importance_mean = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap_mean": importance_matrix.mean(axis=0),
            "mean_abs_shap_std": importance_matrix.std(axis=0, ddof=1),
        }
    ).sort_values("mean_abs_shap_mean", ascending=False)

    # Coefficient of variation
    cv_per_feature = importance_mean["mean_abs_shap_std"] / (
        importance_mean["mean_abs_shap_mean"] + 1e-10
    )
    cv_coefficient = cv_per_feature.mean()

    # Rank correlation (Spearman) across folds
    from scipy.stats import spearmanr

    rank_correlations = []
    for i in range(len(importance_per_fold)):
        for j in range(i + 1, len(importance_per_fold)):
            ranks_i = importance_matrix[i, :].argsort().argsort()
            ranks_j = importance_matrix[j, :].argsort().argsort()
            corr, _ = spearmanr(ranks_i, ranks_j)
            rank_correlations.append(corr)

    mean_rank_correlation = np.mean(rank_correlations)

    print("\n" + "=" * 80)
    print("STABILITY RESULTS")
    print("=" * 80)
    print(f"Mean rank correlation: {mean_rank_correlation:.3f}")
    print(f"Coefficient of variation: {cv_coefficient:.3f}")
    print("\nTop features (mean ± std):")
    for _, row in importance_mean.head(5).iterrows():
        print(
            f"  {row['feature']}: "
            f"{row['mean_abs_shap_mean']:.4f} ± {row['mean_abs_shap_std']:.4f}"
        )

    if mean_rank_correlation > 0.8 and cv_coefficient < 0.3:
        print("\n✓ SHAP values show high stability across folds")
    elif mean_rank_correlation > 0.6:
        print("\n⚠ SHAP values show moderate stability")
    else:
        print("\n✗ SHAP values show low stability - interpret with caution")

    return {
        "importance_per_fold": importance_per_fold,
        "importance_mean": importance_mean,
        "rank_correlation": mean_rank_correlation,
        "cv_coefficient": cv_coefficient,
        "importance_matrix": importance_matrix,
    }
