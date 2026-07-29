"""Regression contracts for diagnostics shared by both pipeline variants."""

from __future__ import annotations

import re

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from xrdanalysis.data_processing.pipeline import MLPipeline, MLPipelineMulti


def _deterministic_splitter(X, y, **_split_args):
    test_mask = np.arange(len(X)) % 4 == 1
    return (
        X.iloc[~test_mask],
        X.iloc[test_mask],
        y.iloc[~test_mask],
        y.iloc[test_mask],
    )


@pytest.fixture
def binary_data():
    X = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
            "feature_b": [1.0, 1.1, 0.9, 1.2, 2.0, 2.1, 1.9, 2.2],
            "group": [10, 10, 20, 20, 30, 30, 40, 40],
            "stratum": [0, 1, 0, 1, 0, 1, 0, 1],
            "sample_weight": [1.0, 1.1, 0.9, 1.2, 1.0, 1.3, 0.8, 1.4],
        }
    )
    y = pd.Series([0, 0, 1, 1, 0, 1, 1, 0], index=X.index)
    return X, y


def _binary_pipeline():
    return MLPipeline(
        splitter=_deterministic_splitter,
        estimator=(
            "classifier",
            LogisticRegression(solver="liblinear", random_state=0),
        ),
    )


def _train_binary(pipeline, X, y, **kwargs):
    return pipeline.train(
        X,
        y_column=None,
        y_data=y,
        wrangle=False,
        preprocess=False,
        print_flag=False,
        show_flag=False,
        sample_weight_col="sample_weight",
        **kwargs,
    )


def test_binary_split_summary_has_frozen_diagnostic_sections(
    binary_data, capsys
):  # noqa: E501
    X, y = binary_data

    _train_binary(
        _binary_pipeline(),
        X,
        y,
        print_split_summary=True,
        test_size=0.25,
        group_col="group",
        stratify_cols=["stratum"],
    )

    output = capsys.readouterr().out
    normalized_output = re.sub(r"np\.int64\((\d+)\)", r"\1", output)
    assert "Split summary:" in output
    assert (
        "Rows: total=8, target_test=2, test=2, train=6, "
        "test_ratio=0.250" in output  # noqa: E501
    )
    assert "Groups: total=4, test=2, train=4" in output
    assert "Test bin counts: {(1,): 2}" in normalized_output
    assert "Train bin counts: {(0,): 4, (1,): 2}" in normalized_output
    assert "Label distribution (train):" in output
    assert "Label distribution (test): " in output
    assert "Label proportion (train):" in output
    assert "Label proportion (test): " in output
    assert "Weight stats by label (train): sum/mean/std" in output
    assert "Weight stats by label (test): sum/mean/std" in output


def test_split_summary_can_be_disabled_without_training_output(
    binary_data, capsys
):  # noqa: E501
    X, y = binary_data

    _train_binary(_binary_pipeline(), X, y, print_split_summary=False)

    assert capsys.readouterr().out == ""


def test_diagnostic_failure_is_non_fatal_and_weight_is_not_a_feature(
    binary_data,
):
    X, y = binary_data
    pipeline = _binary_pipeline()

    results = _train_binary(
        pipeline,
        X,
        y,
        print_split_summary=True,
        stratify_cols=1,
    )

    assert "split_summary" in results
    classifier = pipeline.trained_estimator.steps[-1][1]
    assert "sample_weight" not in classifier.feature_names_in_


def test_multiclass_train_emits_diagnostics_and_excludes_weight(capsys):
    X = pd.DataFrame(
        {
            "feature_a": np.tile([0.0, 1.0, 2.0], 4),
            "feature_b": np.repeat([0.0, 1.0, 2.0, 3.0], 3),
            "group": np.repeat([10, 20, 30, 40, 50, 60], 2),
            "stratum": np.tile([0, 1, 2], 4),
            "sample_weight": np.linspace(0.8, 1.4, 12),
        }
    )
    y = pd.Series(np.tile([0, 1, 2], 4), index=X.index)
    pipeline = MLPipelineMulti(
        splitter=_deterministic_splitter,
        estimator=(
            "classifier",
            LogisticRegression(max_iter=300, random_state=0),
        ),
    )

    pipeline.train(
        X,
        y_column=None,
        y_data=y,
        wrangle=False,
        preprocess=False,
        print_flag=False,
        show_flag=False,
        metrics=["accuracy"],
        print_split_summary=True,
        test_size=0.25,
        group_col="group",
        stratify_cols=["stratum"],
        sample_weight_col="sample_weight",
    )

    output = capsys.readouterr().out
    assert "Split summary:" in output
    assert (
        "Rows: total=12, target_test=3, test=3, train=9, "
        "test_ratio=0.250" in output  # noqa: E501
    )
    assert "Groups: total=6, test=3, train=6" in output
    assert "Test bin counts:" in output
    assert "Train bin counts:" in output
    assert "Weight stats by label (train): sum/mean/std" in output
    classifier = pipeline.trained_estimator.steps[-1][1]
    assert "sample_weight" not in classifier.feature_names_in_


def test_binary_export_joblib_prediction_parity_and_threshold(
    binary_data, tmp_path
):  # noqa: E501
    X, y = binary_data
    pipeline = _binary_pipeline()
    _train_binary(pipeline, X, y, print_split_summary=False)
    export_path = tmp_path / "binary_pipeline.joblib"

    exported = pipeline.export_pipeline(
        wrangle=False, preprocess=False, save_path=export_path
    )
    restored = joblib.load(export_path)
    features = X.drop(columns="sample_weight")

    np.testing.assert_array_equal(
        pipeline.predict(features, wrangle=False, preprocess=False),
        restored.predict(features),
    )
    np.testing.assert_allclose(
        pipeline.predict_proba(features, wrangle=False, preprocess=False),
        restored.predict_proba(features),
    )
    assert hasattr(exported, "optimal_threshold")
    expected_threshold = pytest.approx(pipeline.optimal_threshold)
    assert restored.optimal_threshold == expected_threshold
