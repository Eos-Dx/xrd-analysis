"""Private serialization and CSV-export support for pipeline wrappers."""

from __future__ import annotations

import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline


def _require_trained_estimator(pipeline):
    if not pipeline.trained_estimator:
        raise RuntimeError("Estimator has not been fitted yet.")


def _build_export_pipeline(pipeline, wrangle, preprocess):
    if wrangle and preprocess:
        return Pipeline(
            steps=[
                *pipeline.data_wrangling_steps,
                *pipeline.trained_preprocessor.steps,
                *pipeline.trained_estimator.steps,
            ]
        )
    if wrangle:
        return Pipeline(
            steps=[
                *pipeline.data_wrangling_steps,
                *pipeline.trained_estimator.steps,
            ]
        )
    if preprocess:
        return Pipeline(
            steps=[
                *pipeline.trained_preprocessor.steps,
                *pipeline.trained_estimator.steps,
            ]
        )
    return pipeline.trained_estimator


def export_binary_pipeline(pipeline, wrangle=False, preprocess=True, save_path=None):
    _require_trained_estimator(pipeline)
    full_pipeline = _build_export_pipeline(pipeline, wrangle, preprocess)
    if hasattr(pipeline, "optimal_threshold"):
        full_pipeline.optimal_threshold = pipeline.optimal_threshold
    if save_path:
        dump(full_pipeline, save_path)
    return full_pipeline


def export_multiclass_pipeline(
    pipeline, wrangle=False, preprocess=True, save_path=None
):
    _require_trained_estimator(pipeline)
    full_pipeline = _build_export_pipeline(pipeline, wrangle, preprocess)
    if save_path:
        dump(full_pipeline, save_path)
    return full_pipeline


def export_binary_predictions(
    pipeline, data, save_path, wrangle=False, preprocess=True
):
    _require_trained_estimator(pipeline)
    model = pipeline.export_pipeline(wrangle, preprocess)
    clf = model.steps[-1][1]
    classes = getattr(clf, "classes_", None)
    proba = pipeline.predict_proba(data, wrangle, preprocess)
    multiclass = proba.ndim == 2 and proba.shape[1] > 2

    if multiclass:
        df_pred = pd.DataFrame(
            proba, index=data.index, columns=[f"p_{label}" for label in classes]
        )
        df_pred["cancer_status_soft"] = df_pred[
            [f"p_{label}" for label in classes]
        ].values.tolist()
    else:
        y_score = proba[:, 1] if proba.ndim == 2 else proba
        threshold = getattr(model, "optimal_threshold", 0.5)
        y_pred = y_score >= threshold
        df_pred = pd.DataFrame(
            data=y_pred, index=data.index, columns=["cancer_diagnosis"]
        )

    df_pred.to_csv(save_path)


def export_multiclass_predictions(
    pipeline, data, save_path, wrangle=False, preprocess=True
):
    _require_trained_estimator(pipeline)
    model = pipeline.export_pipeline(wrangle, preprocess)
    y_pred = pipeline.predict(data, wrangle=wrangle, preprocess=preprocess)
    try:
        y_proba = pipeline.predict_proba(data, wrangle=wrangle, preprocess=preprocess)
        df_out = pd.DataFrame(index=data.index)
        df_out["prediction"] = y_pred
        try:
            final_step = (
                model.steps[-1][1]
                if isinstance(model, Pipeline)
                else pipeline.trained_estimator.steps[-1][1]
            )
            classes = getattr(final_step, "classes_", None)
            if classes is not None and y_proba.ndim == 2:
                for index, label in enumerate(classes):
                    df_out[f"proba_{label}"] = y_proba[:, index]
            else:
                df_out["proba"] = y_proba
        except Exception:
            df_out["proba"] = y_proba
    except Exception:
        df_out = pd.DataFrame({"prediction": y_pred}, index=data.index)

    df_out.to_csv(save_path)
