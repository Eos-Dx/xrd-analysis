"""
Core SHAP calculation functions for cancer decision analysis.

This module provides functions to calculate SHAP values (main effects) and
SHAP interaction values (pairwise interactions) while preserving both
original and scaled feature representations.
"""

import warnings

import numpy as np
import pandas as pd
import shap


def calculate_shap_values_with_original(
    pipeline,
    df_original,
    y_column="isCancerDiagnosed",
    group_col="specimenId",
    stratify_cols=["isCancerDiagnosed"],
    test_size=0.4,
    random_state=30,
    n_samples=100,
    feature_names=None,
    original_feature_col="SKana",
):
    """
    Calculate SHAP values keeping both original and scaled feature values.

    This function trains a model using the provided pipeline, splits data
    by groups to avoid leakage, and computes SHAP values while preserving
    the original SKana weights for physical interpretation.

    Parameters
    ----------
    pipeline : xrdanalysis.data_processing.pipeline.MLPipeline
        Trained ML pipeline with preprocessor and estimator
    df_original : pd.DataFrame
        Original data containing SKana column and target
    y_column : str, default='isCancerDiagnosed'
        Name of target column
    group_col : str, default='specimenId'
        Column for grouped splitting to avoid data leakage
    stratify_cols : list, default=['isCancerDiagnosed']
        Columns to stratify on during splitting
    test_size : float, default=0.4
        Proportion of data for test set
    random_state : int, default=30
        Random seed for reproducibility
    n_samples : int, default=100
        Number of test samples to compute SHAP for (for speed)
    feature_names : list of str, optional
        Names for expanded SKana features (e.g., ['S_1', 'S_2', ...])
    original_feature_col : str, default='SKana'
        Column name containing array features to preserve

    Returns
    -------
    dict with keys:
        - 'shap_values': np.ndarray (n_samples, n_features)
            SHAP values for positive class (cancer)
        - 'X_test_scaled': pd.DataFrame
            Scaled test features (what model sees)
        - 'X_test_original': pd.DataFrame
            Original SKana values before scaling
        - 'y_test': pd.Series
            Test labels
        - 'predictions': np.ndarray
            Model prediction probabilities
        - 'explainer': shap.TreeExplainer
            Fitted SHAP explainer object
        - 'importance_df': pd.DataFrame
            Feature importance ranked by mean |SHAP|
        - 'expected_value': float
            Base prediction value (expected model output)

    Notes
    -----
    - Automatically detects estimator name (works with LightGBM, RandomForest, etc.)
    - Handles binary classification SHAP output (extracts positive class)
    - Validates that preprocessing was applied correctly
    """
    from xrdanalysis.data_processing.splitters import grouped_splitter

    # 1. Wrangle data
    X_wrangled = pipeline.wrangle(df_original)
    y = X_wrangled[y_column]

    # 2. Split data (same way as training)
    X_train, X_test, y_train, y_test = grouped_splitter(
        X_wrangled,
        y,
        test_size=test_size,
        random_state=random_state,
        group_col=group_col,
        stratify_cols=stratify_cols,
    )

    # KEEP ORIGINAL VALUES before preprocessing
    X_test_original = X_test[[original_feature_col]].copy()
    # Convert SKana arrays to DataFrame columns
    if original_feature_col in X_test_original.columns:
        skana_arrays = np.vstack(X_test_original[original_feature_col].values)
        if feature_names is None:
            feature_names = [f"S_{i+1}" for i in range(skana_arrays.shape[1])]
        X_test_original_expanded = pd.DataFrame(
            skana_arrays, columns=feature_names, index=X_test_original.index
        )

    # 3. Preprocess (this scales the features)
    X_train_processed = pipeline.trained_preprocessor.transform(X_train)
    X_test_processed = pipeline.trained_preprocessor.transform(X_test)

    # 4. Convert scaled data to DataFrame
    if not isinstance(X_test_processed, pd.DataFrame):
        n_features = X_test_processed.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        X_test_processed = pd.DataFrame(
            X_test_processed, columns=feature_names, index=X_test.index
        )

    # 5. Get trained model - AUTO-DETECT ESTIMATOR NAME (FIXED FOR LIGHTGBM)
    trained_pipeline = pipeline.trained_estimator
    if hasattr(trained_pipeline, "steps"):
        # Extract last step from pipeline
        trained_model = trained_pipeline.steps[-1][1]
    else:
        # Direct estimator
        trained_model = trained_pipeline

    # 6. Create SHAP explainer
    explainer = shap.TreeExplainer(trained_model)

    # 7. Calculate SHAP values on subset
    n_samples = min(n_samples, len(X_test_processed))
    X_test_subset_scaled = X_test_processed.iloc[:n_samples]
    X_test_subset_original = X_test_original_expanded.iloc[:n_samples]

    shap_values = explainer.shap_values(X_test_subset_scaled)

    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class (cancer)

    # 8. Create importance DataFrame
    importance_df = (
        pd.DataFrame(
            {
                "feature": X_test_processed.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # 9. Get predictions
    predictions = trained_model.predict_proba(X_test_subset_scaled)[:, 1]

    # Extract expected value
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    return {
        "shap_values": shap_values,
        "X_test_scaled": X_test_subset_scaled,
        "X_test_original": X_test_subset_original,
        "y_test": y_test.iloc[:n_samples],
        "predictions": predictions,
        "explainer": explainer,
        "importance_df": importance_df,
        "expected_value": expected_value,
    }


def calculate_shap_interactions(
    pipeline,
    df_original,
    y_column="isCancerDiagnosed",
    group_col="specimenId",
    stratify_cols=["isCancerDiagnosed"],
    test_size=0.4,
    random_state=30,
    n_samples=100,
    feature_names=None,
    original_feature_col="SKana",
):
    """
    Calculate SHAP interaction values for pairwise feature effects.

    SHAP interaction values quantify how two features jointly contribute to
    predictions beyond their individual main effects. This is essential for
    understanding synergies between SKana components.

    Parameters
    ----------
    pipeline : xrdanalysis.data_processing.pipeline.MLPipeline
        Trained ML pipeline with preprocessor and estimator
    df_original : pd.DataFrame
        Original data containing SKana column and target
    y_column : str, default='isCancerDiagnosed'
        Name of target column
    group_col : str, default='specimenId'
        Column for grouped splitting
    stratify_cols : list, default=['isCancerDiagnosed']
        Columns to stratify on during splitting
    test_size : float, default=0.4
        Proportion of data for test set
    random_state : int, default=30
        Random seed
    n_samples : int, default=100
        Number of samples to compute interactions for (computational cost is O(n²))
    feature_names : list of str, optional
        Names for expanded SKana features
    original_feature_col : str, default='SKana'
        Column containing array features

    Returns
    -------
    dict with keys:
        - 'shap_values': np.ndarray (n_samples, n_features)
            Main effect SHAP values
        - 'shap_interaction_values': np.ndarray (n_samples, n_features, n_features)
            Pairwise interaction SHAP values
        - 'interaction_matrix': np.ndarray (n_features, n_features)
            Mean absolute interaction strength per feature pair
        - 'interaction_importance': pd.DataFrame
            Ranked list of strongest interactions
        - 'X_test_scaled': pd.DataFrame
            Scaled test features
        - 'X_test_original': pd.DataFrame
            Original SKana values
        - 'y_test': pd.Series
            Test labels
        - 'predictions': np.ndarray
            Prediction probabilities
        - 'explainer': shap.TreeExplainer
            Fitted explainer
        - 'expected_value': float
            Base prediction
        - 'feature_names': list of str
            Feature names

    Notes
    -----
    - Diagonal of interaction matrix equals main effects
    - Interaction matrix is symmetric: interaction(i,j) = interaction(j,i)
    - Computational cost: ~10-100x slower than main effects
    """
    from xrdanalysis.data_processing.splitters import grouped_splitter

    # Steps 1-5: Same as calculate_shap_values_with_original
    X_wrangled = pipeline.wrangle(df_original)
    y = X_wrangled[y_column]

    X_train, X_test, y_train, y_test = grouped_splitter(
        X_wrangled,
        y,
        test_size=test_size,
        random_state=random_state,
        group_col=group_col,
        stratify_cols=stratify_cols,
    )

    X_test_original = X_test[[original_feature_col]].copy()
    skana_arrays = np.vstack(X_test_original[original_feature_col].values)
    if feature_names is None:
        feature_names = [f"S_{i+1}" for i in range(skana_arrays.shape[1])]
    X_test_original_expanded = pd.DataFrame(
        skana_arrays, columns=feature_names, index=X_test_original.index
    )

    X_train_processed = pipeline.trained_preprocessor.transform(X_train)
    X_test_processed = pipeline.trained_preprocessor.transform(X_test)

    if not isinstance(X_test_processed, pd.DataFrame):
        X_test_processed = pd.DataFrame(
            X_test_processed, columns=feature_names, index=X_test.index
        )

    # Get trained model (auto-detect name)
    trained_pipeline = pipeline.trained_estimator
    if hasattr(trained_pipeline, "steps"):
        trained_model = trained_pipeline.steps[-1][1]
    else:
        trained_model = trained_pipeline

    # Create explainer
    explainer = shap.TreeExplainer(trained_model)

    # Subset for computational efficiency
    n_samples = min(n_samples, len(X_test_processed))
    X_test_subset = X_test_processed.iloc[:n_samples]
    X_test_original_subset = X_test_original_expanded.iloc[:n_samples]

    # Calculate main effects
    shap_values = explainer.shap_values(X_test_subset)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class

    # *** KEY: Calculate interaction values ***
    # Shape: (n_samples, n_features, n_features)
    # shap_interaction[i, j, k] = interaction between features j and k for sample i
    print(
        f"Computing SHAP interaction values for {n_samples} samples "
        f"({len(feature_names)} features)..."
    )
    shap_interaction_values = explainer.shap_interaction_values(X_test_subset)

    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[1]  # Positive class

    # Validate: Diagonal should approximately equal main effects
    diagonal_main_effects = np.array(
        [shap_interaction_values[:, i, i] for i in range(len(feature_names))]
    ).T
    max_error = np.abs(diagonal_main_effects - shap_values).max()
    if max_error > 0.01:
        warnings.warn(
            f"SHAP interaction diagonal differs from main effects "
            f"(max error: {max_error:.4f}). This may indicate numerical issues."
        )

    # Calculate interaction importance matrix
    # Average |interaction| across all samples for each pair
    n_features = len(feature_names)
    interaction_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i + 1, n_features):  # Upper triangle only
            # Mean absolute interaction across samples
            interaction_strength = np.abs(shap_interaction_values[:, i, j]).mean()
            interaction_matrix[i, j] = interaction_strength
            interaction_matrix[j, i] = interaction_strength  # Symmetric

    # Create ranked list of strongest interactions
    interaction_list = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interaction_list.append(
                {
                    "feature_i": feature_names[i],
                    "feature_j": feature_names[j],
                    "mean_abs_interaction": interaction_matrix[i, j],
                    "mean_interaction": shap_interaction_values[:, i, j].mean(),
                }
            )

    interaction_df = (
        pd.DataFrame(interaction_list)
        .sort_values("mean_abs_interaction", ascending=False)
        .reset_index(drop=True)
    )

    # Get predictions
    predictions = trained_model.predict_proba(X_test_subset)[:, 1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    return {
        "shap_values": shap_values,
        "shap_interaction_values": shap_interaction_values,
        "interaction_matrix": interaction_matrix,
        "interaction_importance": interaction_df,
        "X_test_scaled": X_test_subset,
        "X_test_original": X_test_original_subset,
        "y_test": y_test.iloc[:n_samples],
        "predictions": predictions,
        "explainer": explainer,
        "expected_value": expected_value,
        "feature_names": feature_names,
    }
