"""Contracts for multiclass validation metrics and presentation behavior."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xrdanalysis.data_processing.pipeline import MLPipelineMulti  # noqa: E402

_CLASSES = np.asarray(["zeta", "alpha", "mu"], dtype=object)
_Y_TRUE = np.asarray(["zeta", "alpha", "mu", "zeta", "alpha", "mu"])
_Y_PROBA = np.asarray(
    [
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
        [0.80, 0.10, 0.10],
        [0.10, 0.80, 0.10],
        [0.10, 0.10, 0.80],
    ]
)


def _pipeline_with_reordered_string_classes():
    """Return a fitted-shaped pipeline with estimator-defined class ordering."""
    pipeline = MLPipelineMulti()
    pipeline.trained_estimator = SimpleNamespace(
        steps=[("classifier", SimpleNamespace(classes_=_CLASSES))]
    )
    return pipeline


def _predictions():
    return _CLASSES[np.argmax(_Y_PROBA, axis=1)]


def test_default_multiclass_metrics_follow_estimator_class_order():
    pipeline = _pipeline_with_reordered_string_classes()

    result = pipeline.validate_multiclass(_Y_TRUE, _Y_PROBA, _predictions())

    assert set(result) == {"accuracy", "roc_auc_macro"}
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["roc_auc_macro"] == pytest.approx(100.0)


def test_requested_macro_and_weighted_auc_are_correct_for_string_classes():
    pipeline = _pipeline_with_reordered_string_classes()

    result = pipeline.validate_multiclass(
        _Y_TRUE,
        _Y_PROBA,
        _predictions(),
        metrics=["accuracy", "roc_auc_macro", "roc_auc_weighted"],
    )

    assert set(result) == {"accuracy", "roc_auc_macro", "roc_auc_weighted"}
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["roc_auc_macro"] == pytest.approx(100.0)
    assert result["roc_auc_weighted"] == pytest.approx(100.0)


def test_requested_auc_reports_none_when_validation_lacks_a_class():
    pipeline = _pipeline_with_reordered_string_classes()
    subset = np.asarray([0, 1, 3, 4])

    result = pipeline.validate_multiclass(
        _Y_TRUE[subset],
        _Y_PROBA[subset],
        _predictions()[subset],
        metrics=["roc_auc_macro", "roc_auc_weighted"],
    )

    assert result == {"roc_auc_macro": None, "roc_auc_weighted": None}


def test_unseen_multiclass_label_preserves_value_error_contract():
    pipeline = _pipeline_with_reordered_string_classes()
    labels = _Y_TRUE.copy()
    labels[0] = "unseen"

    with pytest.raises(ValueError, match="previously unseen labels"):
        pipeline.validate_multiclass(labels, _Y_PROBA, _predictions())


def test_multiclass_validation_without_show_does_not_attach_a_figure():
    pipeline = _pipeline_with_reordered_string_classes()

    result = pipeline.validate_multiclass(
        _Y_TRUE,
        _Y_PROBA,
        _predictions(),
        metrics=["accuracy"],
        show_flag=False,
    )

    assert result == {"accuracy": 1.0}
    assert "roc_fig" not in result


def test_multiclass_validation_show_attaches_labeled_roc_figure(monkeypatch):
    pipeline = _pipeline_with_reordered_string_classes()
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    result = pipeline.validate_multiclass(
        _Y_TRUE,
        _Y_PROBA,
        _predictions(),
        metrics=["accuracy"],
        show_flag=True,
    )

    figure = result["roc_fig"]
    axis = figure.axes[0]
    assert shown == [True]
    assert axis.get_title() == "One-vs-Rest ROC Curves"
    assert axis.get_xlabel() == "False Positive Rate"
    assert axis.get_ylabel() == "True Positive Rate"
    assert len(axis.lines) == len(_CLASSES) + 1
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "zeta (AUC=1.00)",
        "alpha (AUC=1.00)",
        "mu (AUC=1.00)",
    ]
    plt.close(figure)


def test_multiclass_validation_prints_requested_results(capsys):
    pipeline = _pipeline_with_reordered_string_classes()

    result = pipeline.validate_multiclass(
        _Y_TRUE,
        _Y_PROBA,
        _predictions(),
        metrics=["accuracy"],
        print_flag=True,
    )

    assert result == {"accuracy": 1.0}
    assert capsys.readouterr().out == "{'accuracy': 1.0}\n"
