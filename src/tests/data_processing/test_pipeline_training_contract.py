"""Regression contracts for pipeline target and threshold ownership."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from xrdanalysis.data_processing.pipeline import MLPipeline, MLPipelineMulti


class _ColumnRecordingClassifier(ClassifierMixin, BaseEstimator):
    """Minimal classifier that records the columns given to prediction."""

    def fit(self, x, y):
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = np.asarray(x.columns, dtype=object)
        self.proba_columns_ = []
        self.predict_columns_ = []
        return self

    @staticmethod
    def _scores(x):
        return np.asarray(x["signal"], dtype=float)

    def predict_proba(self, x):
        self.proba_columns_.append(tuple(x.columns))
        score = self._scores(x)
        return np.column_stack([1.0 - score, score])

    def predict(self, x):
        self.predict_columns_.append(tuple(x.columns))
        return self._scores(x) > 0.5


class _ScoreClassifier(_ColumnRecordingClassifier):
    """Classifier with fixed score output for threshold export checks."""


def _inferred_target_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": [0.05, 0.15, 0.25, 0.75, 0.85, 0.95],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )


def _train_inferred_target(pipeline, frame):
    kwargs = {
        "X": frame,
        "y_column": "target",
        "wrangle": False,
        "split": False,
        "preprocess": False,
        "print_flag": False,
        "show_flag": False,
    }
    if isinstance(pipeline, MLPipelineMulti):
        kwargs["metrics"] = ["accuracy"]
    return pipeline.train(**kwargs)


@pytest.mark.parametrize("pipeline_type", [MLPipeline, MLPipelineMulti])
def test_inferred_target_is_excluded_from_features_and_input_is_unchanged(
    pipeline_type,
):
    frame = _inferred_target_frame()
    original = frame.copy(deep=True)
    pipeline = pipeline_type(
        estimator=(
            "classifier",
            LogisticRegression(solver="liblinear", random_state=0),
        )
    )

    _train_inferred_target(pipeline, frame)

    fitted = pipeline.trained_estimator.steps[-1][1]
    assert list(fitted.feature_names_in_) == ["signal"]
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("pipeline_type", [MLPipeline, MLPipelineMulti])
def test_feature_only_prediction_works_after_inferred_target_training(
    pipeline_type,
):
    frame = _inferred_target_frame()
    pipeline = pipeline_type(
        estimator=(
            "classifier",
            LogisticRegression(solver="liblinear", random_state=0),
        )
    )

    _train_inferred_target(pipeline, frame)
    predictions = pipeline.predict(
        frame.drop(columns="target"), wrangle=False, preprocess=False
    )

    assert predictions.shape == (len(frame),)


@pytest.mark.parametrize("pipeline_type", [MLPipeline, MLPipelineMulti])
def test_validate_dataset_uses_target_for_y_but_not_prediction_features(
    pipeline_type,
):
    frame = _inferred_target_frame()
    estimator = ("classifier", _ColumnRecordingClassifier())
    pipeline = pipeline_type(estimator=estimator)

    _train_inferred_target(pipeline, frame)
    pipeline.validate_dataset(
        frame,
        y_column="target",
        wrangle=False,
        preprocess=False,
        metrics=["accuracy"],
    )

    fitted = pipeline.trained_estimator.steps[-1][1]
    assert fitted.proba_columns_[-1] == ("signal",)


def test_constrained_binary_threshold_is_persisted_and_exported_consistently(
    tmp_path,
):
    features = pd.DataFrame({"signal": [0.1, 0.4, 0.35, 0.8]})
    y = pd.Series([0, 0, 1, 1], index=features.index)
    pipeline = MLPipeline(estimator=("classifier", _ScoreClassifier()))

    result = pipeline.train(
        features,
        y_column=None,
        y_data=y,
        wrangle=False,
        split=False,
        preprocess=False,
        print_flag=False,
        min_sensitivity=1.0,
    )
    pipeline_path = tmp_path / "constrained-binary.joblib"
    exported = pipeline.export_pipeline(
        wrangle=False,
        preprocess=False,
        save_path=pipeline_path,
    )
    restored = joblib.load(pipeline_path)
    prediction_path = tmp_path / "constrained-predictions.csv"
    pipeline.export_predictions(
        features,
        prediction_path,
        wrangle=False,
        preprocess=False,
    )
    exported_predictions = pd.read_csv(prediction_path, index_col=0)

    assert result["threshold"] == pytest.approx(0.35)
    assert pipeline.optimal_threshold == pytest.approx(result["threshold"])
    assert exported.optimal_threshold == pytest.approx(result["threshold"])
    assert restored.optimal_threshold == pytest.approx(result["threshold"])
    np.testing.assert_array_equal(
        exported_predictions["cancer_diagnosis"].to_numpy(dtype=bool),
        features["signal"].to_numpy() > result["threshold"],
    )
