#!/usr/bin/env python3
"""
Example: Using MLPipelineMulti with weighted soft-label samples for multiclass classification

This example shows how to:
1. Generate soft labels from specimen_status + biopsy
2. Expand to weighted duplicates
3. Train multiclass models with sample_weight using MLPipelineMulti
4. Get ROC curves and metrics for each class
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from xrdanalysis.data_processing.pipeline import MLPipelineMulti
from xrdanalysis.data_processing.splitters import grouped_splitter
from xrdanalysis.data_processing.transformers import (
    SoftLabelToWeightedSamples,
    SpecimenStatusToSoftLabels,
)


def create_example_data(n_patients=50, measurements_per_patient=5):
    """Create synthetic data for demonstration."""
    np.random.seed(42)

    rows = []
    for patient_id in range(n_patients):
        # Patient-level characteristics
        biopsy = np.random.choice([True, False], p=[0.3, 0.7])
        specimen_status = np.random.choice(
            ["CANCER", "BENIGN", "NORMAL"], p=[0.2, 0.3, 0.5]
        )

        for measurement in range(measurements_per_patient):
            # Synthetic radial profile (256 points)
            profile = np.random.normal(0, 1, 256)
            if specimen_status == "CANCER":
                profile += 0.5 * np.sin(np.linspace(0, 4 * np.pi, 256))
            elif specimen_status == "BENIGN":
                profile += 0.3 * np.cos(np.linspace(0, 2 * np.pi, 256))

            rows.append(
                {
                    "patientId": f"patient_{patient_id:03d}",
                    "measurementId": f"{patient_id:03d}_{measurement:02d}",
                    "specimen_status": specimen_status,
                    "biopsy": biopsy,
                    "radial_profile_data": profile,
                    # Add some other features
                    "age": np.random.randint(30, 80),
                    "feature_1": np.random.normal(0, 1),
                    "feature_2": np.random.exponential(2),
                }
            )

    return pd.DataFrame(rows)


def main():
    # 1. Create example data
    print("1. Creating example data...")
    df = create_example_data(n_patients=50, measurements_per_patient=5)
    print(f"   Original data: {len(df)} rows, {df['patientId'].nunique()} patients")
    print(f"   Status distribution: {dict(df['specimen_status'].value_counts())}")
    print(f"   Biopsy distribution: {dict(df['biopsy'].value_counts())}")

    # 2. Generate soft labels
    print("\n2. Generating soft labels...")
    soft_maker = SpecimenStatusToSoftLabels(
        status_col="specimen_status",
        rule_cols=["biopsy"],
        output_col="cancer_status_soft",
        class_order=["CANCER", "BENIGN", "NORMAL"],
        normalize=True,
        strict=False,
    )
    df_with_soft = soft_maker.transform(df)
    print(
        f"   Added soft labels. Example: {df_with_soft['cancer_status_soft'].iloc[0]}"
    )

    # 3. Expand to weighted duplicates
    print("\n3. Expanding to weighted duplicates...")
    expander = SoftLabelToWeightedSamples(
        soft_col="cancer_status_soft",
        label_col="cancer_status",
        weight_col="cancer_status_weighted",
        class_names=["CANCER", "BENIGN", "NORMAL"],
        min_weight=0.01,  # Filter out very small weights
        normalize=True,
    )
    df_expanded = expander.transform(df_with_soft)
    print(f"   Expanded data: {len(df_expanded)} rows")
    print(f"   Class distribution: {dict(df_expanded['cancer_status'].value_counts())}")
    print(
        f"   Weight stats: mean={df_expanded['cancer_status_weighted'].mean():.3f}, "
        f"std={df_expanded['cancer_status_weighted'].std():.3f}"
    )

    # 4. Prepare features (simple approach - flatten radial profile + other features)
    print("\n4. Preparing features...")
    feature_cols = ["age", "feature_1", "feature_2"]

    # For this example, let's use just the other features (not the full radial profile)
    # In practice, you'd use your preprocessing transformers
    X_features = df_expanded[feature_cols + ["cancer_status_weighted"]].copy()

    # Add some summary stats from radial profile as features
    X_features["profile_mean"] = df_expanded["radial_profile_data"].apply(np.mean)
    X_features["profile_std"] = df_expanded["radial_profile_data"].apply(np.std)
    X_features["profile_max"] = df_expanded["radial_profile_data"].apply(np.max)

    # 5. Set up pipeline
    print("\n5. Setting up MLPipelineMulti...")
    pipeline = MLPipelineMulti()

    # Use LightGBM classifier
    pipeline.set_estimator(
        (
            "lgbm",
            LGBMClassifier(
                objective="multiclass",
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1,  # Suppress LightGBM output
            ),
        )
    )

    # Set group-aware splitter (split by patient to avoid leakage)
    pipeline.set_splitter(grouped_splitter)

    # 6. Train with sample weights
    print("\n6. Training model with sample weights...")
    results = pipeline.train(
        X_features,
        y_column="cancer_status",
        test_size=0.25,
        random_state=42,
        group_col="patientId",
        stratify_cols=["cancer_status"],
        sample_weight_col="cancer_status_weighted",  # Use the weights!
        show_flag=True,  # Show ROC curves
        print_flag=True,
        print_split_summary=True,
        metrics=["accuracy", "roc_auc_macro"],
    )

    print(f"\n7. Final Results:")
    print(f"   Accuracy: {results.get('accuracy', 'N/A'):.3f}")
    print(f"   ROC AUC (macro): {results.get('roc_auc_macro', 'N/A'):.1f}%")
    if "per_class_auc" in results:
        print("   Per-class AUC:")
        for cls, auc_val in results["per_class_auc"].items():
            print(f"     {cls}: {auc_val}%")


if __name__ == "__main__":
    main()
