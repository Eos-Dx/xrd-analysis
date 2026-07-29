"""Private validation support for the public pipeline wrappers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder

__all__ = [
    "LabelEncoder",
    "RocCurveDisplay",
    "accuracy_score",
    "auc",
    "np",
    "plt",
    "roc_auc_score",
    "roc_curve",
]


def validate(
    pipeline,
    y_true,
    y_score,
    metrics,
    show_flag,
    print_flag,
    min_sensitivity,
    min_specificity,
    *,
    calculate_optimal_threshold,
    generate_roc_based_metrics,
):
    """Implement the canonical binary and multiclass validation wrapper."""
    results = {}
    y_score = np.asarray(y_score)
    multiclass = y_score.ndim == 2 and y_score.shape[1] > 2

    if not multiclass:
        y_score_bin = (
            y_score[:, 1] if y_score.ndim == 2 and y_score.shape[1] == 2 else y_score
        )
        _, _, _, pipeline.optimal_threshold = calculate_optimal_threshold(
            y_true,
            y_score_bin,
            min_sensitivity=min_sensitivity,
            min_specificity=min_specificity,
            print_flag=print_flag,
        )
        y_pred = y_score_bin >= pipeline.optimal_threshold

        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(y_true, y_pred)

        if "roc_auc" in metrics:
            sensitivity, specificity, precision, ba_accuracy, threshold = (
                generate_roc_based_metrics(
                    y_true,
                    y_score_bin,
                    show_flag,
                    min_sensitivity=min_sensitivity,
                    min_specificity=min_specificity,
                )
            )
            results["roc_auc"] = round(roc_auc_score(y_true, y_score_bin) * 100, 1)
            results["sensitivity"] = sensitivity
            results["specificity"] = specificity
            results["precision"] = precision
            results["ba_accuracy"] = ba_accuracy
            results["threshold"] = threshold

        if print_flag:
            print(results)
        return results

    clf = pipeline.trained_estimator.steps[-1][1]
    classes = np.asarray(getattr(clf, "classes_", np.unique(y_true)))
    if "accuracy" in metrics:
        y_pred_idx = np.argmax(y_score, axis=1)
        results["accuracy"] = accuracy_score(y_true, classes[y_pred_idx])

    if "roc_auc" in metrics:
        results["roc_auc_macro"] = round(
            roc_auc_score(
                y_true, y_score, multi_class="ovr", average="macro", labels=classes
            )
            * 100,
            1,
        )
        results["roc_auc_micro"] = round(
            roc_auc_score(
                y_true, y_score, multi_class="ovr", average="micro", labels=classes
            )
            * 100,
            1,
        )
        per_class_auc = {}
        for index, label in enumerate(classes):
            y_true_bin = (np.asarray(y_true) == label).astype(int)
            per_class_auc[str(label)] = round(
                roc_auc_score(y_true_bin, y_score[:, index]) * 100, 1
            )
            if show_flag:
                RocCurveDisplay.from_predictions(
                    y_true_bin, y_score[:, index], name=f"{label} vs rest"
                )
        if show_flag:
            plt.title("Multiclass ROC (one-vs-rest)")
            plt.show()
        results["per_class_auc"] = per_class_auc

    if print_flag:
        print(results)
    return results


def validate_dataset(
    pipeline,
    data,
    y_column,
    y_value,
    y_data,
    wrangle,
    preprocess,
    metrics,
    show_flag,
    print_flag,
    min_sensitivity,
    min_specificity,
):
    """Validate raw data through the pipeline's public prediction seams."""
    y_true = pipeline.infer_y(data, y_column, y_value) if y_data is None else y_data
    y_score = pipeline.predict_proba(data, wrangle, preprocess)
    return pipeline.validate(
        y_true,
        y_score,
        metrics,
        show_flag,
        print_flag,
        min_sensitivity=min_sensitivity,
        min_specificity=min_specificity,
    )


def validate_multiclass(
    pipeline,
    y_true,
    y_proba,
    y_pred,
    metrics,
    multi_class,
    average,
    show_flag,
    print_flag,
):
    """Compute multiclass metrics in estimator probability-column order."""
    if metrics is None:
        metrics = ["accuracy", "roc_auc_macro"]
    results = {}
    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(y_true, y_pred)

    try:
        clf = pipeline.trained_estimator.steps[-1][1]
        classes = getattr(clf, "classes_", None)
    except Exception:
        classes = None

    if classes is not None:
        class_to_index = {label: index for index, label in enumerate(classes)}
        try:
            y_true_enc = np.asarray([class_to_index[label] for label in y_true])
        except KeyError as error:
            raise ValueError(
                f"y contains previously unseen labels: {error.args[0]!r}"
            ) from error
    else:
        le = LabelEncoder()
        le.fit(pd.unique(y_true))
        classes = list(le.classes_)
        y_true_enc = le.transform(y_true)

    try:
        auc_macro = roc_auc_score(
            y_true_enc, y_proba, multi_class=multi_class, average="macro"
        )
        if "roc_auc_macro" in metrics:
            results["roc_auc_macro"] = round(auc_macro * 100, 1)
    except Exception:
        if "roc_auc_macro" in metrics:
            results["roc_auc_macro"] = None

    try:
        auc_weighted = roc_auc_score(
            y_true_enc, y_proba, multi_class=multi_class, average="weighted"
        )
        if "roc_auc_weighted" in metrics:
            results["roc_auc_weighted"] = round(auc_weighted * 100, 1)
    except Exception:
        if "roc_auc_weighted" in metrics:
            results["roc_auc_weighted"] = None

    if show_flag and classes is not None and y_proba.ndim == 2:
        try:
            fig, ax = plt.subplots(figsize=(7, 6))
            y_true_series = pd.Series(y_true)
            for index, label in enumerate(classes):
                y_bin = (y_true_series == label).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, y_proba[:, index])
                cls_auc = auc(fpr, tpr)
                ax.plot(
                    fpr,
                    tpr,
                    label=f"{label} (AUC={cls_auc:.2f})",
                    linewidth=2,
                )
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("One-vs-Rest ROC Curves")
            ax.legend(loc="lower right", fontsize="small")
            ax.grid(True, alpha=0.3)
            results["roc_fig"] = fig
            try:
                plt.show()
            except Exception:
                pass
        except Exception:
            pass

    if print_flag:
        print(results)
    return results


def validate_multiclass_dataset(
    pipeline,
    data,
    y_column,
    y_value,
    y_data,
    wrangle,
    preprocess,
    metrics,
    show_flag,
    print_flag,
    multi_class,
    average,
):
    """Validate raw multiclass data through canonical public methods."""
    y_true = pipeline.infer_y(data, y_column, y_value) if y_data is None else y_data
    y_proba = pipeline.predict_proba(data, wrangle=wrangle, preprocess=preprocess)
    y_pred = pipeline.predict(data, wrangle=wrangle, preprocess=preprocess)
    return pipeline.validate_multiclass(
        y_true=y_true,
        y_proba=y_proba,
        y_pred=y_pred,
        metrics=metrics,
        multi_class=multi_class,
        average=average,
        show_flag=show_flag,
        print_flag=print_flag,
    )
