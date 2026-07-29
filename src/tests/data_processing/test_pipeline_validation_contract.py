"""Regression contracts for base-pipeline validation behavior."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.metrics import RocCurveDisplay

import xrdanalysis.data_processing.pipeline as pipeline_module
from xrdanalysis.data_processing.pipeline import MLPipeline

_BINARY_LABELS = np.array([0, 0, 1, 1])
_BINARY_SCORES = np.array([0.1, 0.2, 0.8, 0.9])


@pytest.mark.parametrize(
    "name",
    [
        "LabelEncoder",
        "RocCurveDisplay",
        "accuracy_score",
        "auc",
        "np",
        "plt",
        "roc_auc_score",
        "roc_curve",
    ],
)
def test_legacy_validation_dependency_aliases_remain_importable(name):
    assert hasattr(pipeline_module, name)


def _multiclass_validation_fixture():
    classes = np.array(["alpha", "beta", "gamma"])
    labels = np.array(["alpha", "beta", "gamma", "alpha", "beta", "gamma"])
    scores = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.4, 0.5],
            [0.4, 0.5, 0.1],
            [0.2, 0.4, 0.4],
            [0.4, 0.2, 0.4],
        ]
    )
    pipeline = MLPipeline()
    pipeline.trained_estimator = SimpleNamespace(
        steps=[("classifier", SimpleNamespace(classes_=classes))]
    )
    return pipeline, classes, labels, scores


def test_binary_validation_is_identical_for_1d_and_two_column_scores():
    one_dimensional = MLPipeline()
    two_dimensional = MLPipeline()

    one_dimensional_result = one_dimensional.validate(
        _BINARY_LABELS,
        _BINARY_SCORES,
    )
    two_dimensional_result = two_dimensional.validate(
        _BINARY_LABELS,
        np.column_stack([1.0 - _BINARY_SCORES, _BINARY_SCORES]),
    )

    assert one_dimensional_result == pytest.approx(two_dimensional_result)
    assert one_dimensional.optimal_threshold == pytest.approx(0.8)
    assert two_dimensional.optimal_threshold == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("metrics", "expected_keys"),
    [
        ([], set()),
        (["accuracy"], {"accuracy"}),
        (
            ["roc_auc"],
            {
                "roc_auc",
                "sensitivity",
                "specificity",
                "precision",
                "ba_accuracy",
                "threshold",
            },
        ),
    ],
)
def test_binary_validation_filters_metrics_but_always_persists_threshold(
    metrics,
    expected_keys,
):
    pipeline = MLPipeline()

    result = pipeline.validate(_BINARY_LABELS, _BINARY_SCORES, metrics=metrics)

    assert set(result) == expected_keys
    assert pipeline.optimal_threshold == pytest.approx(0.8)
    if metrics == ["accuracy"]:
        assert result["accuracy"] == 1.0


def test_base_multiclass_validation_preserves_class_order_and_result_schema():
    pipeline, classes, labels, scores = _multiclass_validation_fixture()

    result = pipeline.validate(labels, scores)

    assert set(result) == {
        "accuracy",
        "roc_auc_macro",
        "roc_auc_micro",
        "per_class_auc",
    }
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["roc_auc_macro"] == pytest.approx(89.6)
    assert result["roc_auc_micro"] == pytest.approx(88.9)
    assert list(result["per_class_auc"]) == list(classes)
    assert result["per_class_auc"] == {
        "alpha": pytest.approx(93.8),
        "beta": pytest.approx(81.2),
        "gamma": pytest.approx(93.8),
    }


def test_binary_validation_prints_threshold_then_filtered_result(capsys):
    result = MLPipeline().validate(
        _BINARY_LABELS,
        _BINARY_SCORES,
        metrics=["accuracy"],
        print_flag=True,
    )

    assert result == {"accuracy": 1.0}
    expected_output = "Optimal threshold: 0.8\n{'accuracy': 1.0}\n"
    assert capsys.readouterr().out == expected_output


def test_base_multiclass_validation_emits_per_class_roc_plots(monkeypatch):
    pipeline, classes, labels, scores = _multiclass_validation_fixture()
    roc_calls = []
    titles = []
    shown = []

    def fake_from_predictions(y_true, y_score, name):
        roc_calls.append((np.asarray(y_true), np.asarray(y_score), name))

    monkeypatch.setattr(
        RocCurveDisplay,
        "from_predictions",
        fake_from_predictions,
    )
    monkeypatch.setattr(plt, "title", lambda value: titles.append(value))
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    result = pipeline.validate(
        labels,
        scores,
        metrics=["roc_auc"],
        show_flag=True,
    )

    assert set(result) == {"roc_auc_macro", "roc_auc_micro", "per_class_auc"}
    assert [name for _, _, name in roc_calls] == [
        f"{class_name} vs rest" for class_name in classes
    ]
    for index, (y_true, y_score, _) in enumerate(roc_calls):
        np.testing.assert_array_equal(y_true, labels == classes[index])
        np.testing.assert_array_equal(y_score, scores[:, index])
    assert titles == ["Multiclass ROC (one-vs-rest)"]
    assert shown == [True]
