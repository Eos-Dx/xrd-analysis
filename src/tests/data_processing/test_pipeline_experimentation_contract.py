"""Public contracts for pipeline tuning and repeated-run experimentation."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from xrdanalysis.data_processing.pipeline import MLPipeline


class _RecordingEstimator:
    """Minimal estimator exposing only state used by experimentation methods."""

    def __init__(self, random_state=100):
        self.params = {"random_state": random_state, "class_weight": "balanced"}
        self.set_param_calls = []

    def get_params(self, deep=False):
        assert deep is False
        return dict(self.params)

    def set_params(self, **params):
        self.set_param_calls.append(dict(params))
        self.params.update(params)
        return self


def _pipeline_with_recording_estimator():
    estimator = _RecordingEstimator()
    return MLPipeline(estimator=("classifier", estimator)), estimator


def test_evaluate_variability_progresses_seeds_skips_failures_and_keeps_last_state(
    monkeypatch,
):
    pipeline, estimator = _pipeline_with_recording_estimator()
    calls = []

    def fake_train(_x, _y_column, **kwargs):
        split_seed = kwargs["random_state"]
        calls.append((split_seed, estimator.params["random_state"], kwargs))
        if split_seed == 12:
            raise RuntimeError("synthetic failed split")
        return {
            "roc_auc": float(split_seed),
            "sensitivity": float(split_seed + 10),
            "specificity": float(split_seed + 20),
            "threshold": split_seed / 100,
        }

    monkeypatch.setattr(pipeline, "train", fake_train)
    result = pipeline.evaluate_variability(
        pd.DataFrame({"x": [1, 2]}),
        "target",
        n_runs=4,
        seed=7,
        random_state=10,
        vary_split=True,
        vary_estimator=True,
    )

    assert [
        (split_seed, estimator_seed) for split_seed, estimator_seed, _ in calls
    ] == [
        (10, 100),
        (11, 101),
        (12, 102),
        (13, 103),
    ]
    assert result["runs"] == 3
    assert result["roc_auc_values"] == [10.0, 11.0, 13.0]
    assert result["sensitivity_values"] == [20.0, 21.0, 23.0]
    assert result["specificity_values"] == [30.0, 31.0, 33.0]
    assert result["threshold_values"] == pytest.approx([0.1, 0.11, 0.13])
    assert result["roc_auc"] == pytest.approx({"mean": 34 / 3, "std": 1.527525})
    assert result["threshold"]["mean"] == pytest.approx(0.34 / 3)
    assert estimator.params["random_state"] == 103
    assert all(call[2]["print_split_summary"] is False for call in calls)


def test_tune_estimator_optuna_reports_missing_optional_dependency(monkeypatch):
    pipeline, _ = _pipeline_with_recording_estimator()
    monkeypatch.setitem(sys.modules, "optuna", None)

    with pytest.raises(ImportError, match="Optuna is required for tuning"):
        pipeline.tune_estimator_optuna(pd.DataFrame({"x": [1]}), "target")


def test_tune_estimator_optuna_uses_fake_study_and_refits_best_params(monkeypatch):
    pipeline, estimator = _pipeline_with_recording_estimator()
    train_calls = []
    variability_calls = []

    def fake_train(_x, _y_column, **kwargs):
        train_calls.append(dict(kwargs))
        pipeline.optimal_threshold = 0.3
        return {"roc_auc": 80.0, "sensitivity": 70.0, "specificity": 60.0}

    def fake_variability(*args, **kwargs):
        variability_calls.append((args, kwargs))
        return {"runs": 2}

    class FakeStudy:
        def __init__(self):
            self.best_trial = SimpleNamespace(
                params={
                    "candidate": 3,
                    "random_state_split": 17,
                    "random_state_est": 23,
                }
            )
            self.best_value = None

        def optimize(self, objective, n_trials, timeout):
            assert n_trials == 1
            assert timeout == 12
            self.best_value = objective(SimpleNamespace())

    study = FakeStudy()
    fake_optuna = ModuleType("optuna")
    fake_optuna.create_study = lambda **kwargs: study

    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    monkeypatch.setattr(pipeline, "train", fake_train)
    monkeypatch.setattr(pipeline, "evaluate_variability", fake_variability)

    result = pipeline.tune_estimator_optuna(
        pd.DataFrame({"x": [1, 2]}),
        "target",
        n_trials=1,
        timeout=12,
        param_space=lambda _trial: {
            "candidate": 3,
            "random_state_split": 17,
            "random_state_est": 23,
        },
        threshold_range=(0.4, 0.7),
        penalty_weight=2.0,
        variability_n_runs=2,
        variability_seed=9,
        variability_vary_split=False,
        variability_vary_estimator=False,
    )

    assert study.best_value == pytest.approx(1.1)
    assert result["best_value"] == pytest.approx(1.1)
    assert result["best_results"]["roc_auc"] == 80.0
    assert result["best_params"] == {
        "candidate": 3,
        "random_state_split": 17,
        "random_state_est": 23,
        "random_state": 100,
        "class_weight": "balanced",
    }
    assert len(train_calls) == 2
    assert train_calls[0]["random_state"] == 17
    assert train_calls[0]["print_flag"] is False
    assert train_calls[1]["print_flag"] is False
    assert estimator.set_param_calls[0] == {
        "candidate": 3,
        "random_state": 23,
        "class_weight": "balanced",
    }
    assert variability_calls[0][1]["n_runs"] == 2
    assert variability_calls[0][1]["seed"] == 9
    assert result["variability"] == {"runs": 2}
