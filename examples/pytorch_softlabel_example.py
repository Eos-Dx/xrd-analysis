import pandas as pd

from xrdanalysis.data_processing.estimators_torch import TorchSoftLabelClassifier
from xrdanalysis.data_processing.pipeline import MLPipeline
from xrdanalysis.data_processing.transformers import (
    AzimuthalIntegration,
    ColumnNormalizer,
    SlopeRemoval,
)

"""
Example: Training a PyTorch soft-label classifier within the existing MLPipeline.

Assumptions
- Your DataFrame `df` contains raw measurement data in column 'measurement_data' and calibration
  columns required by AzimuthalIntegration.
- After wrangling, a feature column 'radial_profile_data' is produced per row (1D array).
- A column 'cancer_status_soft' exists with soft labels per sample (list/array of length C), e.g.:
    [0.1, 0.7, 0.2] for 3-class targets (the values should sum to ~1.0).
- A hard label column 'cancer_diagnosis' exists for evaluation (binary or multiclass class id).

Key idea
- We keep preprocess=False so the estimator receives the DataFrame with the array feature column and
  the soft-label column available. The estimator extracts/uses them internally.
- Wrangling performs azimuthal integration and optional normalizations on the curve.

NOTE: Install PyTorch first, e.g. CPU-only on Windows:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""


def build_pipeline(n_classes: int = 3) -> MLPipeline:
    wrangling = [
        (
            "azimuthal",
            AzimuthalIntegration(
                integration_mode="1D",
                transformation_mode="dataframe",
                npt=256,
                thickness_adjustment=False,
                calc_cake_stats=False,
            ),
        ),
        ("slope", SlopeRemoval(columns=["radial_profile_data"], mode="")),
        ("norm", ColumnNormalizer(column="radial_profile_data", norm="l2", mode="1D")),
    ]

    estimator = [
        (
            "torch_soft",
            TorchSoftLabelClassifier(
                feature_column="radial_profile_data",
                soft_label_column="cancer_status_soft",  # <-- change if your soft label column is named differently
                n_classes=n_classes,
                hidden_dims=(512, 256),
                dropout=0.2,
                lr=1e-3,
                weight_decay=1e-5,
                epochs=40,
                batch_size=128,
                device="cpu",  # set to 'cuda' if available and desired
                verbose=True,
            ),
        )
    ]

    return MLPipeline(
        data_wrangling_steps=wrangling,
        preprocessing_steps=[],  # keep empty; estimator handles feature extraction
        estimator=estimator,
    )


def train_example(df: pd.DataFrame, n_classes: int = 3):
    pipe = build_pipeline(n_classes=n_classes)

    # Use wrangle=True, preprocess=False. Keep y_column pointing to hard labels for correct evaluation,
    # while the estimator uses 'cancer_status_soft' internally for training.
    results = pipe.train(
        X=df,
        y_column="cancer_diagnosis",
        wrangle=True,
        split=True,
        preprocess=False,
        print_flag=True,
        show_flag=False,
        test_size=0.2,
        random_state=42,
    )

    print("Training/validation results:", results)

    # Export full pipeline (wrangling + estimator)
    model = pipe.export_pipeline(wrangle=True, preprocess=False)

    # Predict probabilities on new data
    # proba = pipe.predict_proba(df_new, wrangle=True, preprocess=False)

    return pipe, results


if __name__ == "__main__":
    # Placeholder: load your data here
    # df = pd.read_parquet("path/to/your/data.parquet")
    # pipe, results = train_example(df, n_classes=3)
    print("This is a usage example. Import train_example() and pass your DataFrame.")
