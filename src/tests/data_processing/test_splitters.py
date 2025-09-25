import numpy as np
import pandas as pd
import pytest

from xrdanalysis.data_processing.splitters import grouped_splitter


def _make_dataset(n_groups=60, rows_per_group=10, rs=123):
    rng = np.random.default_rng(rs)
    sample_ids = np.repeat(np.arange(n_groups), rows_per_group)
    n = len(sample_ids)

    # Stratify columns
    cancer_status = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    biopsy = rng.choice([True, False], size=n, p=[0.7, 0.3])

    df = pd.DataFrame(
        {
            "sampleId": sample_ids,
            "feature": rng.normal(size=n),
            "cancer_status": cancer_status,
            "biopsy": biopsy,
        }
    )
    y = df["cancer_status"]
    return df, y


def _bin_series(df, stratify_cols):
    if stratify_cols is None:
        return pd.Series(["__ALL__"] * len(df), index=df.index)
    if isinstance(stratify_cols, (str, bytes)):
        stratify_cols = [stratify_cols]
    return df[stratify_cols].apply(lambda r: tuple(r[c] for c in stratify_cols), axis=1)


def test_grouped_splitter_size_group_exclusivity_and_stratification():
    df, y = _make_dataset(n_groups=60, rows_per_group=10, rs=123)

    test_size = 0.25
    expected_test_rows = int(round(test_size * len(df)))
    max_group_size = df.groupby("sampleId").size().max()

    # Run the splitter with two stratification columns
    X_tr, X_te, y_tr, y_te = grouped_splitter(
        df,
        y,
        test_size=test_size,
        random_state=42,
        group_col="sampleId",
        stratify_cols=["cancer_status", "biopsy"],
        print_debug=False,
    )

    # 1) Group exclusivity
    train_groups = set(X_tr["sampleId"].unique())
    test_groups = set(X_te["sampleId"].unique())
    assert train_groups.isdisjoint(
        test_groups
    ), "Groups should not overlap between train and test"

    # 2) Test size close to target by entries (within one group size)
    assert (
        abs(len(X_te) - expected_test_rows) <= max_group_size
    ), f"Test rows {len(X_te)} should be close to target {expected_test_rows} within {max_group_size}"

    # 3) Approximate stratification preservation on combined bins
    overall_bins = _bin_series(df, ["cancer_status", "biopsy"]).value_counts(
        normalize=True
    )
    test_bins = _bin_series(X_te, ["cancer_status", "biopsy"]).value_counts(
        normalize=True
    )

    # Ensure all overall bins appear in test distribution (missing -> 0.0)
    test_bins = test_bins.reindex(overall_bins.index, fill_value=0.0)

    # Tolerance: 0.15 absolute proportion per bin (heuristic; groups may prevent perfect match)
    tol = 0.15
    diffs = (overall_bins - test_bins).abs()
    assert (
        diffs <= tol
    ).all(), f"Per-bin proportion diffs too large: {diffs.to_dict()}"

    # 4) Determinism with same random_state
    X_tr2, X_te2, y_tr2, y_te2 = grouped_splitter(
        df,
        y,
        test_size=test_size,
        random_state=42,
        group_col="sampleId",
        stratify_cols=["cancer_status", "biopsy"],
        print_debug=False,
    )
    # Compare test groups equality
    assert set(X_te["sampleId"].unique()) == set(
        X_te2["sampleId"].unique()
    ), "Splitter should be deterministic for same random_state"


def test_grouped_splitter_single_stratify_column():
    df, y = _make_dataset(n_groups=40, rows_per_group=10, rs=321)

    test_size = 0.3
    expected_test_rows = int(round(test_size * len(df)))
    max_group_size = df.groupby("sampleId").size().max()

    X_tr, X_te, y_tr, y_te = grouped_splitter(
        df,
        y,
        test_size=test_size,
        random_state=7,
        group_col="sampleId",
        stratify_cols="cancer_status",
        print_debug=False,
    )

    # Group exclusivity
    assert set(X_tr["sampleId"].unique()).isdisjoint(set(X_te["sampleId"].unique()))

    # Test size close to target by entries (within one group size)
    assert abs(len(X_te) - expected_test_rows) <= max_group_size

    # Stratification check for single column
    overall_bins = df["cancer_status"].value_counts(normalize=True)
    test_bins = X_te["cancer_status"].value_counts(normalize=True)
    test_bins = test_bins.reindex(overall_bins.index, fill_value=0.0)

    tol = 0.15
    diffs = (overall_bins - test_bins).abs()
    assert (diffs <= tol).all(), f"Single-col diffs too large: {diffs.to_dict()}"
