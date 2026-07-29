"""Serialization and export contracts for the public pipeline wrappers."""

from __future__ import annotations

import inspect

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from xrdanalysis.data_processing.pipeline import MLPipeline, MLPipelineMulti

_EMPTY = inspect.Parameter.empty
_POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
_KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
_VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD


def _parameter(name, default=_EMPTY, kind=_POSITIONAL):
    return (name, kind, default)


_BASE_METHODS = {
    "__init__": (
        _parameter("self"),
        _parameter("data_wrangling_steps", None),
        _parameter("splitter", train_test_split),
        _parameter("preprocessing_steps", None),
        _parameter("estimator", None),
    ),
    "add_data_wrangling_step": (
        _parameter("self"),
        _parameter("name"),
        _parameter("transformer"),
        _parameter("position", None),
    ),
    "add_preprocessing_step": (
        _parameter("self"),
        _parameter("name"),
        _parameter("transformer"),
        _parameter("position", None),
    ),
    "set_estimator": (_parameter("self"), _parameter("estimator")),
    "set_splitter": (_parameter("self"), _parameter("splitter")),
    "wrangle": (_parameter("self"), _parameter("data"), _parameter("stat", False)),
    "preprocess": (_parameter("self"), _parameter("data")),
    "train_preprocess": (_parameter("self"), _parameter("data")),
    "transform": (_parameter("self"), _parameter("data"), _parameter("stat", False)),
    "wrangle_preprocess_transform": (
        _parameter("self"),
        _parameter("data"),
        _parameter("train", True),
    ),
    "infer_y": (
        _parameter("self"),
        _parameter("X"),
        _parameter("y_column"),
        _parameter("y_value", None),
    ),
    "train_preprocessor": (_parameter("self"), _parameter("data")),
    "train_estimator": (
        _parameter("self"),
        _parameter("X"),
        _parameter("y"),
        _parameter("sample_weight", None),
    ),
    "predict": (
        _parameter("self"),
        _parameter("X"),
        _parameter("wrangle", False),
        _parameter("preprocess", True),
    ),
    "predict_proba": (
        _parameter("self"),
        _parameter("X"),
        _parameter("wrangle", False),
        _parameter("preprocess", True),
    ),
    "validate": (
        _parameter("self"),
        _parameter("y_true"),
        _parameter("y_score"),
        _parameter("metrics", ["accuracy", "roc_auc"]),
        _parameter("show_flag", False),
        _parameter("print_flag", False),
        _parameter("min_sensitivity", None),
        _parameter("min_specificity", None),
    ),
    "train": (
        _parameter("self"),
        _parameter("X"),
        _parameter("y_column"),
        _parameter("y_value", None),
        _parameter("y_data", None),
        _parameter("wrangle", True),
        _parameter("split", True),
        _parameter("preprocess", True),
        _parameter("print_flag", True),
        _parameter("show_flag", False),
        _parameter("min_sensitivity", None),
        _parameter("min_specificity", None),
        _parameter("print_split_summary", False),
        _parameter("sample_weight_col", None),
        _parameter("split_args", kind=_VAR_KEYWORD),
    ),
    "export_pipeline": (
        _parameter("self"),
        _parameter("wrangle", False),
        _parameter("preprocess", True),
        _parameter("save_path", None),
    ),
    "export_predictions": (
        _parameter("self"),
        _parameter("data"),
        _parameter("save_path"),
        _parameter("wrangle", False),
        _parameter("preprocess", True),
    ),
    "validate_dataset": (
        _parameter("self"),
        _parameter("data"),
        _parameter("y_column", None),
        _parameter("y_value", None),
        _parameter("y_data", None),
        _parameter("wrangle", False),
        _parameter("preprocess", False),
        _parameter("metrics", ["accuracy", "roc_auc"]),
        _parameter("show_flag", False),
        _parameter("print_flag", False),
        _parameter("min_sensitivity", None),
        _parameter("min_specificity", None),
    ),
    "tune_estimator_optuna": (
        _parameter("self"),
        _parameter("X"),
        _parameter("y_column"),
        _parameter("y_value", None),
        _parameter("y_data", None),
        _parameter("wrangle", True, _KEYWORD_ONLY),
        _parameter("split", True, _KEYWORD_ONLY),
        _parameter("preprocess", True, _KEYWORD_ONLY),
        _parameter("n_trials", 30, _KEYWORD_ONLY),
        _parameter("timeout", None, _KEYWORD_ONLY),
        _parameter("direction", "minimize", _KEYWORD_ONLY),
        _parameter("param_space", None, _KEYWORD_ONLY),
        _parameter("study_name", None, _KEYWORD_ONLY),
        _parameter("sampler", None, _KEYWORD_ONLY),
        _parameter("pruner", None, _KEYWORD_ONLY),
        _parameter("show_flag", False, _KEYWORD_ONLY),
        _parameter("print_flag", False, _KEYWORD_ONLY),
        _parameter("threshold_range", None, _KEYWORD_ONLY),
        _parameter("penalty_weight", 0.0, _KEYWORD_ONLY),
        _parameter("variability_n_runs", 0, _KEYWORD_ONLY),
        _parameter("variability_seed", 0, _KEYWORD_ONLY),
        _parameter("variability_vary_split", True, _KEYWORD_ONLY),
        _parameter("variability_vary_estimator", True, _KEYWORD_ONLY),
        _parameter("split_args", kind=_VAR_KEYWORD),
    ),
    "evaluate_variability": (
        _parameter("self"),
        _parameter("X"),
        _parameter("y_column"),
        _parameter("y_value", None),
        _parameter("y_data", None),
        _parameter("wrangle", True, _KEYWORD_ONLY),
        _parameter("split", True, _KEYWORD_ONLY),
        _parameter("preprocess", True, _KEYWORD_ONLY),
        _parameter("n_runs", 20, _KEYWORD_ONLY),
        _parameter("seed", 0, _KEYWORD_ONLY),
        _parameter("vary_split", True, _KEYWORD_ONLY),
        _parameter("vary_estimator", True, _KEYWORD_ONLY),
        _parameter("show_flag", False, _KEYWORD_ONLY),
        _parameter("print_flag", False, _KEYWORD_ONLY),
        _parameter("split_args", kind=_VAR_KEYWORD),
    ),
}

_MULTI_OVERRIDES = {
    "validate_multiclass": (
        _parameter("self"),
        _parameter("y_true"),
        _parameter("y_proba"),
        _parameter("y_pred"),
        _parameter("metrics", None),
        _parameter("multi_class", "ovr"),
        _parameter("average", "macro"),
        _parameter("show_flag", False),
        _parameter("print_flag", False),
    ),
    "train": (
        _parameter("self"),
        _parameter("X"),
        _parameter("y_column"),
        _parameter("y_value", None),
        _parameter("y_data", None),
        _parameter("wrangle", True),
        _parameter("split", True),
        _parameter("preprocess", True),
        _parameter("print_flag", True),
        _parameter("show_flag", False),
        _parameter("metrics", None),
        _parameter("multi_class", "ovr"),
        _parameter("average", "macro"),
        _parameter("print_split_summary", False),
        _parameter("sample_weight_col", None),
        _parameter("split_args", kind=_VAR_KEYWORD),
    ),
    "export_pipeline": _BASE_METHODS["export_pipeline"],
    "export_predictions": _BASE_METHODS["export_predictions"],
    "validate_dataset": (
        _parameter("self"),
        _parameter("data"),
        _parameter("y_column", None),
        _parameter("y_value", None),
        _parameter("y_data", None),
        _parameter("wrangle", False),
        _parameter("preprocess", False),
        _parameter("metrics", None),
        _parameter("show_flag", False),
        _parameter("print_flag", False),
        _parameter("multi_class", "ovr"),
        _parameter("average", "macro"),
    ),
}


def _assert_method_contract(cls, name, expected):
    method = getattr(cls, name)
    actual = tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(method).parameters.values()
    )
    assert actual == expected
    assert method.__module__ == "xrdanalysis.data_processing.pipeline"
    assert method.__qualname__ == f"{cls.__name__}.{name}"


def test_pipeline_public_method_signatures_defaults_and_identity_are_exact():
    assert MLPipeline.__module__ == "xrdanalysis.data_processing.pipeline"
    assert MLPipeline.__qualname__ == "MLPipeline"
    for name, expected in _BASE_METHODS.items():
        _assert_method_contract(MLPipeline, name, expected)


def test_multiclass_public_method_signatures_defaults_and_identity_are_exact():
    assert MLPipelineMulti.__module__ == "xrdanalysis.data_processing.pipeline"
    assert MLPipelineMulti.__qualname__ == "MLPipelineMulti"
    for name, expected in _MULTI_OVERRIDES.items():
        _assert_method_contract(MLPipelineMulti, name, expected)

    for name in set(_BASE_METHODS) - set(_MULTI_OVERRIDES):
        assert getattr(MLPipelineMulti, name) is getattr(MLPipeline, name)
        _assert_method_contract(MLPipeline, name, _BASE_METHODS[name])


class AddOffset(BaseEstimator, TransformerMixin):
    """DataFrame-preserving wrangling step used by export-order checks."""

    def __init__(self, offset: float = 0.25):
        self.offset = offset

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        out = x.copy()
        out["x0"] = out["x0"] + self.offset
        return out


class CenterFrame(BaseEstimator, TransformerMixin):
    """Fitted DataFrame-preserving preprocessing step for stable exports."""

    def fit(self, x, y=None):
        self.columns_ = list(x.columns)
        self.mean_ = x.loc[:, self.columns_].mean()
        return self

    def transform(self, x):
        return x.loc[:, self.columns_] - self.mean_


def _binary_data() -> tuple[pd.DataFrame, pd.Series]:
    x0 = np.linspace(-3.0, 3.0, 24)
    x1 = np.cos(x0)
    frame = pd.DataFrame({"x0": x0, "x1": x1}, index=np.arange(100, 124))
    labels = pd.Series((x0 > 0.0).astype(int), index=frame.index)
    return frame, labels


def _multiclass_data() -> tuple[pd.DataFrame, pd.Series]:
    labels = np.repeat(["low", "mid", "high"], 10)
    x0 = np.concatenate(
        [
            np.linspace(-3.0, -1.0, 10),
            np.linspace(-0.2, 0.2, 10),
            np.linspace(1.0, 3.0, 10),
        ]
    )
    frame = pd.DataFrame({"x0": x0, "x1": x0**2}, index=np.arange(200, 230))
    return frame, pd.Series(labels, index=frame.index)


def _base_pipeline() -> MLPipeline:
    return MLPipeline(
        data_wrangling_steps=[("offset", AddOffset())],
        preprocessing_steps=[("center", CenterFrame())],
        estimator=("lr", LogisticRegression(max_iter=500, random_state=7)),
    )


def _multiclass_pipeline() -> MLPipelineMulti:
    return MLPipelineMulti(
        data_wrangling_steps=[("offset", AddOffset())],
        preprocessing_steps=[("center", CenterFrame())],
        estimator=("lr", LogisticRegression(max_iter=500, random_state=8)),
    )


@pytest.mark.parametrize(
    ("factory", "data_factory", "qualified_name"),
    [
        (
            _base_pipeline,
            _binary_data,
            "xrdanalysis.data_processing.pipeline.MLPipeline",
        ),
        (
            _multiclass_pipeline,
            _multiclass_data,
            "xrdanalysis.data_processing.pipeline.MLPipelineMulti",
        ),
    ],
)
def test_fitted_wrappers_round_trip_with_canonical_paths_and_predictions(
    tmp_path, factory, data_factory, qualified_name
):
    frame, labels = data_factory()
    pipeline = factory()
    pipeline.train(
        frame,
        y_column="ignored",
        y_data=labels,
        wrangle=True,
        split=False,
        preprocess=True,
        print_flag=False,
        show_flag=False,
    )
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)

    assert type(restored) is type(pipeline)
    assert (
        f"{type(restored).__module__}.{type(restored).__qualname__}" == qualified_name
    )
    assert restored.trained_preprocessor is not None
    assert restored.trained_estimator is not None
    assert restored.feature_names_ == pipeline.feature_names_
    if isinstance(pipeline, MLPipelineMulti):
        assert not hasattr(restored, "optimal_threshold")
    else:
        assert restored.optimal_threshold == pipeline.optimal_threshold
    np.testing.assert_array_equal(
        restored.predict(frame, wrangle=True, preprocess=True),
        pipeline.predict(frame, wrangle=True, preprocess=True),
    )
    np.testing.assert_allclose(
        restored.predict_proba(frame, wrangle=True, preprocess=True),
        pipeline.predict_proba(frame, wrangle=True, preprocess=True),
    )


@pytest.mark.parametrize(
    ("wrangle", "preprocess", "expected_steps"),
    [
        (False, False, ["lr"]),
        (True, False, ["offset", "lr"]),
        (False, True, ["center", "lr"]),
        (True, True, ["offset", "center", "lr"]),
    ],
)
def test_export_pipeline_step_order_and_wrapper_prediction_parity(
    wrangle, preprocess, expected_steps
):
    frame, labels = _binary_data()
    pipeline = _base_pipeline()
    pipeline.train(
        frame,
        y_column="ignored",
        y_data=labels,
        wrangle=True,
        split=False,
        preprocess=True,
        print_flag=False,
    )

    exported = pipeline.export_pipeline(wrangle=wrangle, preprocess=preprocess)
    assert list(exported.named_steps) == expected_steps
    assert exported.optimal_threshold == pipeline.optimal_threshold
    np.testing.assert_array_equal(
        exported.predict(frame),
        pipeline.predict(frame, wrangle=wrangle, preprocess=preprocess),
    )
    np.testing.assert_allclose(
        exported.predict_proba(frame),
        pipeline.predict_proba(frame, wrangle=wrangle, preprocess=preprocess),
    )


def test_binary_and_multiclass_csv_exports_preserve_schema_index_and_scores(tmp_path):
    binary_frame, binary_labels = _binary_data()
    binary = _base_pipeline()
    binary.train(
        binary_frame,
        y_column="ignored",
        y_data=binary_labels,
        wrangle=True,
        split=False,
        preprocess=True,
        print_flag=False,
    )
    binary_path = tmp_path / "binary.csv"
    binary.export_predictions(binary_frame, binary_path, wrangle=True, preprocess=True)
    binary_csv = pd.read_csv(binary_path, index_col=0)
    binary_scores = binary.predict_proba(binary_frame, wrangle=True, preprocess=True)[
        :, 1
    ]
    expected_binary = (
        binary_scores
        >= binary.export_pipeline(wrangle=True, preprocess=True).optimal_threshold
    )
    assert list(binary_csv.columns) == ["cancer_diagnosis"]
    np.testing.assert_array_equal(
        binary_csv.index.to_numpy(dtype=int), binary_frame.index
    )
    np.testing.assert_array_equal(
        binary_csv["cancer_diagnosis"].to_numpy(dtype=bool), expected_binary
    )

    multi_frame, multi_labels = _multiclass_data()
    multi = _multiclass_pipeline()
    multi.train(
        multi_frame,
        y_column="ignored",
        y_data=multi_labels,
        wrangle=True,
        split=False,
        preprocess=True,
        print_flag=False,
    )
    multi_path = tmp_path / "multi.csv"
    multi.export_predictions(multi_frame, multi_path, wrangle=True, preprocess=True)
    multi_csv = pd.read_csv(multi_path, index_col=0)
    multi_proba = multi.predict_proba(multi_frame, wrangle=True, preprocess=True)
    multi_pred = multi.predict(multi_frame, wrangle=True, preprocess=True)
    classes = multi.trained_estimator.steps[-1][1].classes_
    probability_columns = [f"proba_{label}" for label in classes]
    assert list(multi_csv.columns) == ["prediction", *probability_columns]
    np.testing.assert_array_equal(
        multi_csv.index.to_numpy(dtype=int), multi_frame.index
    )
    np.testing.assert_array_equal(multi_csv["prediction"].to_numpy(), multi_pred)
    np.testing.assert_allclose(multi_csv[probability_columns].to_numpy(), multi_proba)


@pytest.mark.parametrize("factory", [_base_pipeline, _multiclass_pipeline])
@pytest.mark.parametrize(
    "method_name", ["predict", "predict_proba", "export_pipeline", "export_predictions"]
)
def test_pre_fit_public_methods_keep_estimator_error(factory, method_name, tmp_path):
    pipeline = factory()
    if method_name == "export_pipeline":
        call = lambda: pipeline.export_pipeline()
    elif method_name == "export_predictions":
        call = lambda: pipeline.export_predictions(
            _binary_data()[0], tmp_path / "predictions.csv"
        )
    elif method_name == "predict":
        call = lambda: pipeline.predict(_binary_data()[0])
    else:
        call = lambda: pipeline.predict_proba(_binary_data()[0])

    with pytest.raises(RuntimeError, match="Estimator has not been fitted yet."):
        call()


@pytest.mark.parametrize("factory", [_base_pipeline, _multiclass_pipeline])
def test_missing_sample_weight_column_keeps_error(factory):
    frame, labels = _binary_data()
    with pytest.raises(
        KeyError, match="Sample weight column 'weight' not found in training data"
    ):
        factory().train(
            frame,
            y_column="ignored",
            y_data=labels,
            wrangle=False,
            split=False,
            preprocess=False,
            print_flag=False,
            sample_weight_col="weight",
        )
