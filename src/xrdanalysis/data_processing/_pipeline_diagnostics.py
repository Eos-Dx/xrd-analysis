"""Private, non-fatal diagnostics for pipeline train/test splits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _bin_counts(
    frame: pd.DataFrame, columns: list[str | bytes]
) -> dict[tuple[Any, ...], int]:
    return (
        frame[columns]
        .apply(lambda row: tuple(row[column] for column in columns), axis=1)
        .value_counts()
        .to_dict()
    )


def _print_weight_stats(
    frame: pd.DataFrame,
    labels: Any,
    sample_weight_col: str | None,
    split_name: str,
) -> None:
    if sample_weight_col is None or sample_weight_col not in frame.columns:
        return

    weights = frame[[sample_weight_col]].copy()
    weights["__y__"] = pd.Series(labels).values
    groups = weights.groupby("__y__")[sample_weight_col]
    print(f"Weight stats by label ({split_name}): sum/mean/std")
    summary = groups.agg(["sum", "mean", "std"]).round(5).rename_axis("label")
    print(summary.to_string())


def emit_split_summary(
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Any,
    y_test: Any,
    split_args: Mapping[str, Any],
    sample_weight_col: str | None,
) -> None:
    """Print legacy split diagnostics.

    This mirrors the historical non-fatal train-time diagnostic path.
    """
    try:
        n_total = len(X)
        n_test = len(X_test)
        n_train = len(X_train)
        test_ratio = (n_test / n_total) if n_total else 0.0
        test_size = split_args.get("test_size", None)
        target_test = (
            int(round(float(test_size) * n_total))
            if isinstance(test_size, (int, float))
            else None
        )
        print("Split summary:")
        if target_test is not None:
            print(
                f"Rows: total={n_total}, target_test={target_test}, "
                f"test={n_test}, train={n_train}, test_ratio={test_ratio:.3f}"
            )
        else:
            print(
                f"Rows: total={n_total}, test={n_test}, train={n_train}, "
                f"test_ratio={test_ratio:.3f}"
            )

        group_col = split_args.get("group_col", None)
        if group_col is not None and group_col in X.columns:
            g_total = X[group_col].nunique()
            g_test = X_test[group_col].nunique()
            g_train = X_train[group_col].nunique()
            print(f"Groups: total={g_total}, test={g_test}, train={g_train}")

        stratify_cols = split_args.get("stratify_cols", None)
        if stratify_cols is not None:
            if isinstance(stratify_cols, (str, bytes)):
                strat_cols = [stratify_cols]
            else:
                strat_cols = list(stratify_cols)
            if all(column in X.columns for column in strat_cols):
                print(f"Test bin counts: {_bin_counts(X_test, strat_cols)}")
                print(f"Train bin counts: {_bin_counts(X_train, strat_cols)}")

        try:
            print(
                "Label distribution (train):",
                pd.Series(y_train).value_counts().to_dict(),
            )
            print(
                "Label distribution (test): ",
                pd.Series(y_test).value_counts().to_dict(),
            )
            train_labels = pd.Series(y_train)
            test_labels = pd.Series(y_test)
            train_proportions = train_labels.value_counts(normalize=True)
            test_proportions = test_labels.value_counts(normalize=True)
            train_proportions = train_proportions.round(3).to_dict()
            test_proportions = test_proportions.round(3).to_dict()
            print("Label proportion (train):", train_proportions)
            print("Label proportion (test): ", test_proportions)
            _print_weight_stats(X_train, y_train, sample_weight_col, "train")
            _print_weight_stats(X_test, y_test, sample_weight_col, "test")
        except Exception:
            pass
    except Exception:
        pass
