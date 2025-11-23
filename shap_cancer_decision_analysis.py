"""
SHAP Analysis for Cancer Detection Model
Shows how the model decides between cancer/non-cancer samples with spectra visualization
"""

from pathlib import Path

import marimo as mo
import matplotlib.pyplot as plt
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

    Returns
    -------
    dict with keys:
        - 'shap_values': np.ndarray of SHAP values
        - 'X_test_scaled': pd.DataFrame of scaled test features (what model sees)
        - 'X_test_original': pd.DataFrame of original SKana values
        - 'y_test': pd.Series of test labels
        - 'explainer': shap.TreeExplainer object
        - 'importance_df': pd.DataFrame of feature importance
        - 'expected_value': float, base prediction value
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

    # 5. Get trained model
    trained_model = pipeline.trained_estimator.named_steps["rf"]

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

    return {
        "shap_values": shap_values,
        "X_test_scaled": X_test_subset_scaled,
        "X_test_original": X_test_subset_original,
        "y_test": y_test.iloc[:n_samples],
        "predictions": predictions,
        "explainer": explainer,
        "importance_df": importance_df,
        "expected_value": (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, list)
            else explainer.expected_value
        ),
    }


def analyze_cancer_decision(shap_results, threshold=0.5):
    """
    Analyze how the model decides between cancer/non-cancer.

    Returns DataFrame with decision analysis for each sample.
    """
    df_analysis = pd.DataFrame(
        {
            "sample_idx": range(len(shap_results["predictions"])),
            "true_label": shap_results["y_test"].values,
            "prediction_prob": shap_results["predictions"],
            "predicted_label": shap_results["predictions"] > threshold,
            "base_value": shap_results["expected_value"],
        }
    )

    # Add SHAP contributions
    for i, feature in enumerate(shap_results["X_test_scaled"].columns):
        df_analysis[f"shap_{feature}"] = shap_results["shap_values"][:, i]
        df_analysis[f"original_{feature}"] = shap_results["X_test_original"][
            feature
        ].values
        df_analysis[f"scaled_{feature}"] = shap_results["X_test_scaled"][feature].values

    # Calculate total SHAP contribution
    df_analysis["shap_sum"] = shap_results["shap_values"].sum(axis=1)

    # Classification correctness
    df_analysis["correct"] = df_analysis["true_label"] == df_analysis["predicted_label"]

    return df_analysis


def plot_decision_boundary_analysis(shap_results, df_analysis, df_spectra):
    """
    Comprehensive visualization of cancer vs non-cancer decision making.
    """
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.3, height_ratios=[1, 1.2, 1, 2])

    feature_names = shap_results["X_test_scaled"].columns

    # 1. Prediction distribution by true label
    ax1 = fig.add_subplot(gs[0, 0])
    cancer_preds = df_analysis[df_analysis["true_label"] == True]["prediction_prob"]
    noncancer_preds = df_analysis[df_analysis["true_label"] == False]["prediction_prob"]

    ax1.hist(
        cancer_preds,
        bins=30,
        alpha=0.6,
        label="Cancer (true)",
        color="red",
        edgecolor="black",
    )
    ax1.hist(
        noncancer_preds,
        bins=30,
        alpha=0.6,
        label="Non-cancer (true)",
        color="blue",
        edgecolor="black",
    )
    ax1.axvline(
        0.5, color="black", linestyle="--", linewidth=2, label="Decision threshold"
    )
    ax1.set_xlabel("Prediction Probability", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title(
        "Prediction Distribution by True Label", fontsize=12, fontweight="bold"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. SHAP value distributions by label
    ax2 = fig.add_subplot(gs[0, 1])
    cancer_shap_sum = df_analysis[df_analysis["true_label"] == True]["shap_sum"]
    noncancer_shap_sum = df_analysis[df_analysis["true_label"] == False]["shap_sum"]

    ax2.hist(
        cancer_shap_sum,
        bins=30,
        alpha=0.6,
        label="Cancer",
        color="red",
        edgecolor="black",
    )
    ax2.hist(
        noncancer_shap_sum,
        bins=30,
        alpha=0.6,
        label="Non-cancer",
        color="blue",
        edgecolor="black",
    )
    ax2.axvline(0, color="black", linestyle="--", linewidth=2)
    ax2.set_xlabel("Total SHAP Contribution", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("SHAP Value Distribution by Label", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Feature importance comparison
    ax3 = fig.add_subplot(gs[0, 2])
    importance = shap_results["importance_df"]
    ax3.barh(
        importance["feature"], importance["mean_abs_shap"], color="steelblue", alpha=0.7
    )
    ax3.set_xlabel("Mean |SHAP|", fontsize=11)
    ax3.set_title("Feature Importance", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="x")

    # 4. Average spectra: Cancer vs Non-cancer (ORIGINAL weights)
    ax4 = fig.add_subplot(gs[1, :])
    wavl = df_spectra["wavl"].values

    # Get average original weights for each group
    cancer_mask = df_analysis["true_label"] == True
    noncancer_mask = df_analysis["true_label"] == False

    cancer_weights_avg = np.array(
        [df_analysis[cancer_mask][f"original_{feat}"].mean() for feat in feature_names]
    )
    noncancer_weights_avg = np.array(
        [
            df_analysis[noncancer_mask][f"original_{feat}"].mean()
            for feat in feature_names
        ]
    )

    # Calculate average spectra
    cancer_spectrum = sum(
        df_spectra[f"S_{i+1}"].values * cancer_weights_avg[i]
        for i in range(len(feature_names))
    )
    noncancer_spectrum = sum(
        df_spectra[f"S_{i+1}"].values * noncancer_weights_avg[i]
        for i in range(len(feature_names))
    )

    ax4.plot(
        wavl, cancer_spectrum, "r-", linewidth=2.5, label="Average Cancer", alpha=0.8
    )
    ax4.plot(
        wavl,
        noncancer_spectrum,
        "b-",
        linewidth=2.5,
        label="Average Non-cancer",
        alpha=0.8,
    )
    ax4.fill_between(wavl, cancer_spectrum, alpha=0.2, color="red")
    ax4.fill_between(wavl, noncancer_spectrum, alpha=0.2, color="blue")
    ax4.set_xlabel("q (wavl)", fontsize=11)
    ax4.set_ylabel("Intensity", fontsize=11)
    ax4.set_title(
        "Average Spectra: Cancer vs Non-Cancer (Original Weights)",
        fontsize=12,
        fontweight="bold",
    )
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. Component weights comparison (Original)
    ax5 = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(len(feature_names))
    width = 0.35

    ax5.bar(
        x_pos - width / 2,
        cancer_weights_avg,
        width,
        label="Cancer",
        color="red",
        alpha=0.7,
    )
    ax5.bar(
        x_pos + width / 2,
        noncancer_weights_avg,
        width,
        label="Non-cancer",
        color="blue",
        alpha=0.7,
    )
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(feature_names)
    ax5.set_ylabel("Average Weight (Original)", fontsize=11)
    ax5.set_title(
        "Component Weights: Cancer vs Non-Cancer", fontsize=12, fontweight="bold"
    )
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")

    # 6. SHAP contributions comparison
    ax6 = fig.add_subplot(gs[2, 1])
    cancer_shap_avg = np.array(
        [df_analysis[cancer_mask][f"shap_{feat}"].mean() for feat in feature_names]
    )
    noncancer_shap_avg = np.array(
        [df_analysis[noncancer_mask][f"shap_{feat}"].mean() for feat in feature_names]
    )

    ax6.bar(
        x_pos - width / 2,
        cancer_shap_avg,
        width,
        label="Cancer",
        color="red",
        alpha=0.7,
    )
    ax6.bar(
        x_pos + width / 2,
        noncancer_shap_avg,
        width,
        label="Non-cancer",
        color="blue",
        alpha=0.7,
    )
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(feature_names)
    ax6.set_ylabel("Average SHAP Value", fontsize=11)
    ax6.set_title(
        "SHAP Contributions: Cancer vs Non-Cancer", fontsize=12, fontweight="bold"
    )
    ax6.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis="y")

    # 7. Confusion matrix
    ax7 = fig.add_subplot(gs[2, 2])
    cm = pd.crosstab(
        df_analysis["true_label"],
        df_analysis["predicted_label"],
        rownames=["True"],
        colnames=["Predicted"],
    )
    im = ax7.imshow(cm, cmap="Blues", aspect="auto")
    ax7.set_xticks([0, 1])
    ax7.set_yticks([0, 1])
    ax7.set_xticklabels(["Non-cancer", "Cancer"])
    ax7.set_yticklabels(["Non-cancer", "Cancer"])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax7.text(
                j,
                i,
                cm.iloc[i, j],
                ha="center",
                va="center",
                color="white" if cm.iloc[i, j] > cm.values.max() / 2 else "black",
                fontsize=14,
                fontweight="bold",
            )

    ax7.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax7)

    # 8. Individual components for cancer vs non-cancer
    ax8 = fig.add_subplot(gs[3, :])
    for i, feat in enumerate(feature_names):
        # Plot component weighted by average weights
        cancer_comp = df_spectra[feat].values * cancer_weights_avg[i]
        noncancer_comp = df_spectra[feat].values * noncancer_weights_avg[i]

        ax8.plot(
            wavl,
            cancer_comp,
            "--",
            alpha=0.6,
            linewidth=1.5,
            label=f"{feat} Cancer (w={cancer_weights_avg[i]:.3f})",
        )
        ax8.plot(
            wavl,
            noncancer_comp,
            ":",
            alpha=0.6,
            linewidth=1.5,
            label=f"{feat} Non-cancer (w={noncancer_weights_avg[i]:.3f})",
        )

    ax8.set_xlabel("q (wavl)", fontsize=11)
    ax8.set_ylabel("Weighted Intensity", fontsize=11)
    ax8.set_title(
        "Component Contributions: Cancer vs Non-Cancer", fontsize=12, fontweight="bold"
    )
    ax8.legend(fontsize=8, ncol=2)
    ax8.grid(True, alpha=0.3)

    plt.suptitle(
        "Cancer Detection Decision Analysis", fontsize=16, fontweight="bold", y=0.998
    )

    return fig


def create_decision_summary_table(df_analysis):
    """
    Create summary statistics showing key differences between cancer and non-cancer.
    """
    summary = []

    for label in [True, False]:
        label_name = "Cancer" if label else "Non-cancer"
        subset = df_analysis[df_analysis["true_label"] == label]

        summary.append(
            {
                "Group": label_name,
                "N": len(subset),
                "Avg Prediction": f"{subset['prediction_prob'].mean():.3f}",
                "Avg SHAP Sum": f"{subset['shap_sum'].mean():.3f}",
                "Accuracy": f"{subset['correct'].mean()*100:.1f}%",
            }
        )

    return pd.DataFrame(summary)


# ==================== MARIMO CELLS EXAMPLE ====================
# The code below shows example usage in marimo cells
# Copy these into separate cells in your marimo notebook

if __name__ == "__main__":
    # Example usage
    pass
