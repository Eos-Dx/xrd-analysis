"""Regression contracts for utility evaluation and ROC helpers.

These tests freeze behavior before the private evaluation-helper extraction while
keeping ``utility_functions`` as the stable public import surface.
"""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import UndefinedMetricWarning

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xrdanalysis.data_processing.pipeline import MLPipeline  # noqa: E402
from xrdanalysis.data_processing.utility_functions import (  # noqa: E402
    calculate_optimal_threshold,
    custom_splitter_balanced,
    generate_roc_based_metrics,
    metrics,
    viz_roc,
    viz_roc_balanced,
)

Y_TRUE = np.array([0, 0, 1, 1])
Y_SCORE = np.array([0.1, 0.4, 0.35, 0.8])


def test_legacy_star_import_retains_sklearn_metric_names():
    namespace = {}

    exec(
        "from xrdanalysis.data_processing.utility_functions import *",
        namespace,
    )

    for name in (
        "RocCurveDisplay",
        "auc",
        "f1_score",
        "precision_score",
        "roc_curve",
    ):
        assert name in namespace


def test_calculate_optimal_threshold_has_frozen_unconstrained_and_constrained_results():
    """Keep the historical Youden-J selection and constraint precedence."""
    tpr, fpr, index, threshold = calculate_optimal_threshold(Y_TRUE, Y_SCORE)
    np.testing.assert_array_equal(tpr, [0.0, 0.5, 0.5, 1.0, 1.0])
    np.testing.assert_array_equal(fpr, [0.0, 0.0, 0.5, 0.5, 1.0])
    assert index == 1
    assert threshold == pytest.approx(0.8)

    tpr, fpr, index, threshold = calculate_optimal_threshold(
        Y_TRUE, Y_SCORE, min_sensitivity=1.0
    )
    np.testing.assert_array_equal(tpr, [1.0, 1.0])
    np.testing.assert_array_equal(fpr, [0.5, 1.0])
    assert index == 0
    assert threshold == pytest.approx(0.35)

    tpr, fpr, index, threshold = calculate_optimal_threshold(
        Y_TRUE, Y_SCORE, min_specificity=1.0
    )
    np.testing.assert_array_equal(tpr, [0.0, 0.5])
    np.testing.assert_array_equal(fpr, [0.0, 0.0])
    assert index == 1
    assert threshold == pytest.approx(0.8)


def test_calculate_optimal_threshold_preserves_empty_constraint_error():
    """Do not silently change the established failure for impossible constraints."""
    with pytest.raises(ValueError, match="attempt to get argmax of an empty sequence"):
        calculate_optimal_threshold(Y_TRUE, Y_SCORE, min_sensitivity=1.1)


def test_calculate_optimal_threshold_prints_legacy_threshold(capsys):
    calculate_optimal_threshold(Y_TRUE, Y_SCORE, print_flag=True)

    assert capsys.readouterr().out == "Optimal threshold: 0.8\n"


def test_generate_roc_based_metrics_and_text_are_deterministic_without_plotting():
    """Freeze scalar metrics and text formatting used by evaluation notebooks."""
    with pytest.warns(UndefinedMetricWarning):
        values = generate_roc_based_metrics(Y_TRUE, Y_SCORE, show_flag=False)

    assert values == pytest.approx((50.0, 100.0, 0.0, 75.0, 0.8))
    with pytest.warns(UndefinedMetricWarning):
        text = metrics(
            np.array([0.0, 0.5, 0.5, 1.0, 1.0]),
            np.array([0.0, 0.0, 0.5, 0.5, 1.0]),
            np.array([np.inf, 0.8, 0.4, 0.35, 0.1]),
            Y_SCORE,
            Y_TRUE,
            roc_auc=0.75,
        )
    assert text == (
        "\n"
        "           ROC surface : 75.0%\n"
        "           Optimal threshold: 80.0%\n"
        "           Sensitivity: 50.0%\n"
        "           Specificity: 100.0%\n"
        "           PV: 0.0%\n"
        "           F1-score: 0.0%\n"
        "           "
    )


def test_generate_roc_based_metrics_configures_legacy_plot(monkeypatch):
    utility = importlib.import_module("xrdanalysis.data_processing.utility_functions")
    shown = []
    monkeypatch.setattr(utility.plt, "show", lambda: shown.append(True))

    with pytest.warns(UndefinedMetricWarning):
        values = generate_roc_based_metrics(Y_TRUE, Y_SCORE, show_flag=True)

    figure = plt.gcf()
    assert values == pytest.approx((50.0, 100.0, 0.0, 75.0, 0.8))
    assert shown == [True]
    np.testing.assert_allclose(figure.get_size_inches(), [4.0, 4.0])
    assert figure.dpi == 150
    assert figure.get_facecolor() == pytest.approx((1.0, 1.0, 1.0, 1.0))
    plt.close(figure)


def test_custom_splitter_balanced_is_seeded_patient_complete_and_partitioned():
    """Protect patient-level partition semantics while retaining global RNG behavior."""
    frame = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 3, 3, 4, 5, 6],
            "cancer_diagnosis": [True, True, True, True, True, False, False, False],
        }
    )

    np.random.seed(2026)
    train_indices, test_indices = custom_splitter_balanced(frame, split=0.4)
    np.random.seed(2026)
    repeat_train_indices, repeat_test_indices = custom_splitter_balanced(
        frame, split=0.4
    )

    assert train_indices == repeat_train_indices
    assert test_indices == repeat_test_indices
    assert set(train_indices).isdisjoint(test_indices)
    assert set(train_indices) | set(test_indices) == set(range(len(frame)))
    for patient_id in frame["patient_id"].unique():
        patient_indices = set(frame.index[frame["patient_id"] == patient_id])
        assert patient_indices <= set(train_indices) or patient_indices <= set(
            test_indices
        )


class _PredictProbaModel:
    def predict_proba(self, x_test):
        positive = np.asarray(x_test, dtype=float)
        return np.column_stack([1.0 - positive, positive])


def _cluster_fixture():
    return SimpleNamespace(
        X_test=np.array([0.1, 0.4, 0.35, 0.8, 0.9]),
        y_test=np.array([0, 0, 1, 1, 1]),
        model=_PredictProbaModel(),
        df=pd.DataFrame(
            {
                "patient_id": [1, 2, 3, 4, 5],
                "cancer_diagnosis": [False, False, True, True, True],
            }
        ),
    )


def test_roc_visualization_helpers_smoke_without_interactive_backend(monkeypatch):
    """Exercise both plotting paths without opening figures or writing files."""
    utility = importlib.import_module("xrdanalysis.data_processing.utility_functions")
    monkeypatch.setattr(utility.plt, "show", lambda: None)
    cluster = _cluster_fixture()
    predictor = SimpleNamespace(ML_saxs=cluster, ML_waxs=cluster)
    estimator = SimpleNamespace(ML_saxs=cluster, ML_waxs=cluster)
    figure, axes = plt.subplots(1, 2)

    viz_roc(figure, axes, "regression", predictor, text_on=False, legend_on=False)
    viz_roc_balanced(figure, axes, "regression", [estimator])

    plt.close(figure)


def test_public_utility_imports_signatures_and_pipeline_bindings_are_stable():
    """Keep old imports and pipeline globals valid through a private extraction."""
    utility = importlib.import_module("xrdanalysis.data_processing.utility_functions")
    pipeline = importlib.import_module("xrdanalysis.data_processing.pipeline")
    expected = {
        "custom_splitter_balanced": custom_splitter_balanced,
        "viz_roc": viz_roc,
        "metrics": metrics,
        "viz_roc_balanced": viz_roc_balanced,
        "generate_roc_based_metrics": generate_roc_based_metrics,
        "calculate_optimal_threshold": calculate_optimal_threshold,
    }

    for name, function in expected.items():
        assert getattr(utility, name) is function
        assert len(inspect.signature(function).parameters) >= 2
    assert pipeline.calculate_optimal_threshold is calculate_optimal_threshold
    assert pipeline.generate_roc_based_metrics is generate_roc_based_metrics


def test_pipeline_with_custom_splitter_survives_joblib_round_trip(tmp_path):
    """Persisted pipelines may store the splitter function by module path."""
    pipeline = MLPipeline(splitter=custom_splitter_balanced)
    path = tmp_path / "pipeline_with_custom_splitter.joblib"

    joblib.dump(pipeline, path)
    restored = joblib.load(path)

    assert restored.splitter is custom_splitter_balanced
    assert (
        restored.splitter.__module__ == "xrdanalysis.data_processing.utility_functions"
    )


@pytest.mark.parametrize(
    "function",
    [
        custom_splitter_balanced,
        viz_roc,
        metrics,
        viz_roc_balanced,
        generate_roc_based_metrics,
        calculate_optimal_threshold,
    ],
)
def test_public_evaluation_wrappers_keep_legacy_pickle_identity(tmp_path, function):
    path = tmp_path / f"{function.__name__}.joblib"

    joblib.dump(function, path)
    restored = joblib.load(path)

    assert restored is function
    assert restored.__module__ == "xrdanalysis.data_processing.utility_functions"
    assert restored.__name__ == function.__name__
