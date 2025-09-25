import numpy as np
import pandas as pd

from xrdanalysis.data_processing.splitters import grouped_splitter


def _make_synthetic_specimen_dataset(n_specimen=50, min_rows=3, max_rows=4, rs=123):
    rng = np.random.default_rng(rs)
    ids = [f"specimen_{i:03d}" for i in range(n_specimen)]
    rows = []
    for sid in ids:
        k = int(rng.integers(min_rows, max_rows + 1))
        for _ in range(k):
            rows.append(
                {
                    "specimenId": sid,
                    "feature": float(rng.normal()),
                    "biopsy": bool(rng.integers(0, 2)),
                    "cancer_status": int(rng.integers(0, 2)),
                }
            )
    df = pd.DataFrame(rows)
    y = df["cancer_status"]
    return df, y


def test_grouped_splitter_specimen_synthetic():
    df, y = _make_synthetic_specimen_dataset(
        n_specimen=50, min_rows=3, max_rows=4, rs=42
    )

    test_size = 0.25
    expected_test_rows = int(round(test_size * len(df)))
    max_group_size = df.groupby("specimenId").size().max()

    X_tr, X_te, y_tr, y_te = grouped_splitter(
        df,
        y,
        test_size=test_size,
        random_state=32,
        group_col="specimenId",
        stratify_cols=["biopsy"],
        print_debug=False,
    )

    # Diagnostics (kept as prints for demonstration; -s will show them)
    print(
        "Rows: total={}, target_test={}, test={}, train={}, test_ratio={:.3f}".format(
            len(df), expected_test_rows, len(X_te), len(X_tr), len(X_te) / len(df)
        )
    )
    print(
        "Groups: total={}, test={}, train={}".format(
            df["specimenId"].nunique(),
            X_te["specimenId"].nunique(),
            X_tr["specimenId"].nunique(),
        )
    )

    overall_bins = df["biopsy"].value_counts(normalize=True)
    test_bins = (
        X_te["biopsy"]
        .value_counts(normalize=True)
        .reindex(overall_bins.index, fill_value=0.0)
    )
    print("Overall biopsy proportions:", overall_bins.to_dict())
    print("Test biopsy proportions:", test_bins.to_dict())
    print("Abs diffs:", (overall_bins - test_bins).abs().to_dict())

    # 1) Group exclusivity
    train_groups = set(X_tr["specimenId"].unique())
    test_groups = set(X_te["specimenId"].unique())
    assert train_groups.isdisjoint(
        test_groups
    ), "Groups should not overlap between train and test"

    # 2) Test size close to target by entries (within one group size)
    assert (
        abs(len(X_te) - expected_test_rows) <= max_group_size
    ), f"Test rows {len(X_te)} should be close to target {expected_test_rows} within {max_group_size}"

    # 3) Approximate stratification preservation on 'biopsy'
    tol = 0.2
    diffs = (overall_bins - test_bins).abs()
    assert (diffs <= tol).all(), f"Biopsy proportion diffs too large: {diffs.to_dict()}"
