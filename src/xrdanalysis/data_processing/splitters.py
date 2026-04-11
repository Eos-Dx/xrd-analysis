#!/usr/bin/env python3
"""
Group-aware splitter that:
- Randomly splits entries with respect to a group column (group exclusivity)
- Targets test_size by number of entries (not by number of groups)
- Optionally stratifies on one or more columns to approximately preserve
  per-bin distributions in the test set

Usage with MLPipeline:
    from xrdanalysis.data_processing.splitters import grouped_splitter
    pipeline.set_splitter(grouped_splitter)
    result = pipeline.train(
        df,
        'cancer_status',
        split=True,
        test_size=0.25,
        random_state=32,
        group_col='sampleId',
        stratify_cols=['cancer_status', 'biopsy'],
    )

Notes:
- Exact satisfaction of both per-bin stratification and exact test_size
  by entries is NP-hard with group constraints; this splitter uses a
  heuristic to achieve near-exact test size and reasonable per-bin matching.
- The total test size is prioritized; per-bin targets are matched greedily
  and then filled by the least-overshooting groups.
"""
from __future__ import annotations

from typing import Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def _combine_strata_row(row: pd.Series, cols: Sequence[str]) -> Tuple:
    """Combine multiple stratify columns into a single bin key (tuple)."""
    return tuple(row[c] for c in cols)


def grouped_splitter(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.25,
    random_state: Optional[int] = None,
    group_col: str = "specimenId",
    stratify_cols: Optional[Iterable[str]] = None,
    print_debug: bool = False,
    **kwargs,
):
    """
    Split X,y into train/test such that:
      - No group (group_col) appears in both splits
      - The test split size is approximately test_size * len(X) by number of rows
      - If stratify_cols are provided, attempt to preserve per-bin distributions

    Parameters
    ----------
    X : DataFrame
        Feature+metadata table (must contain group_col and any stratify_cols)
    y : Series
        Target vector aligned to X
    test_size : float
        Desired fraction of rows in the test set (0 < test_size < 1)
    random_state : int, optional
        Seed for reproducibility
    group_col : str
        Column defining the grouping (e.g., patientId, sampleId)
    stratify_cols : list[str] or None
        Columns used to define stratification bins. Can be one or multiple columns.
    print_debug : bool
        If True, prints diagnostics about the split
    **kwargs : dict
        Ignored; kept for compatibility with MLPipeline.set_splitter

    Returns
    -------
    (X_train, X_test, y_train, y_test)
        Slices of input X and y
    """
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError("test_size must be in (0,1)")

    if group_col not in X.columns:
        raise KeyError(f"Required group column '{group_col}' not found in X.")

    n = len(X)
    target_test = int(round(test_size * n))
    if target_test <= 0 or target_test >= n:
        raise ValueError(
            """test_size yields degenerate split;
                         choose a value that results in non-empty train and test"""
        )

    rng = np.random.default_rng(random_state)

    # Prepare stratification bins (optional)
    if stratify_cols is None:
        bins = pd.Series(["__ALL__"] * n, index=X.index)
    else:
        if isinstance(stratify_cols, (str, bytes)):
            strat_cols = [stratify_cols]
        else:
            strat_cols = list(stratify_cols)
        missing = [c for c in strat_cols if c not in X.columns]
        if missing:
            raise KeyError(f"Stratify columns not found in X: {missing}")
        bins = X[strat_cols].apply(lambda r: _combine_strata_row(r, strat_cols), axis=1)

    # Overall counts per bin and target per bin
    bin_counts = bins.value_counts().to_dict()
    # Compute target test counts per bin (rounded). We'll reconcile totals later.
    target_bin = {b: int(round(test_size * c)) for b, c in bin_counts.items()}
    # Ensure at least 0 and no negative rounding
    for b in list(target_bin.keys()):
        target_bin[b] = max(0, min(target_bin[b], bin_counts[b]))

    # Group-level structures
    groups = X[group_col]
    unique_groups = groups.unique().tolist()
    rng.shuffle(unique_groups)

    # Precompute per-group indices and per-bin counts
    group_indices = {}
    group_sizes = {}
    group_bin_counts = {}

    for g in unique_groups:
        idx = X.index[groups == g]
        group_indices[g] = idx
        group_sizes[g] = len(idx)
        # per-bin for this group
        gb = bins.loc[idx]
        group_bin_counts[g] = gb.value_counts().to_dict()

    # Max size of any single group; used as tolerance for test size deviation
    max_group_size = max(group_sizes.values()) if len(group_sizes) > 0 else 0

    test_groups: List[Hashable] = []
    current_test_count = 0
    current_bin = {b: 0 for b in bin_counts.keys()}

    def bin_gain_if_add(g) -> int:
        """How many needed bin counts this group would satisfy right now."""
        gain = 0
        for b, cnt in group_bin_counts[g].items():
            tgt = target_bin.get(b, 0)
            cur = current_bin.get(b, 0)
            if cur < tgt:
                gain += min(cnt, tgt - cur)
        return gain

    # First pass: greedy add groups that help fill bin targets most,
    # while keeping total test size within one group size of target.
    remaining_groups = unique_groups.copy()
    remaining_groups.sort(key=lambda g: (-bin_gain_if_add(g), group_sizes[g]))
    for g in remaining_groups:
        sz = group_sizes[g]
        gain = bin_gain_if_add(g)
        if gain <= 0:
            continue
        proposed = current_test_count + sz
        # Allow adding if it keeps us under target, or if it keeps us within tolerance,
        # or if it strictly improves distance to target.
        if (
            proposed <= target_test
            or abs(proposed - target_test) <= max_group_size
            or abs(proposed - target_test) < abs(current_test_count - target_test)
        ):
            test_groups.append(g)
            current_test_count = proposed
            for b, cnt in group_bin_counts[g].items():
                current_bin[b] = current_bin.get(b, 0) + cnt
        # Early stop if we're within tolerance and further additions would worsen distance
        if abs(current_test_count - target_test) <= max_group_size:
            # Peek the best remaining candidate; if adding would worsen beyond tolerance, stop
            # (Keeps size-focused priority as per tests)
            pass

    # Second pass: if still outside tolerance below target, add groups that best improve proximity
    if (
        current_test_count < target_test
        and abs(current_test_count - target_test) > max_group_size
    ):
        remaining = [g for g in unique_groups if g not in test_groups]
        # Sort by: (absolute difference to target if added, then smaller size)
        remaining.sort(
            key=lambda g: (
                abs(target_test - (current_test_count + group_sizes[g])),
                group_sizes[g],
            )
        )
        for g in remaining:
            # Stop if within tolerance already
            if abs(current_test_count - target_test) <= max_group_size:
                break
            sz = group_sizes[g]
            proposed = current_test_count + sz
            # Add only if it improves proximity or stays within tolerance
            if (
                abs(proposed - target_test) < abs(current_test_count - target_test)
                or abs(proposed - target_test) <= max_group_size
            ):
                test_groups.append(g)
                current_test_count = proposed
                for b, cnt in group_bin_counts[g].items():
                    current_bin[b] = current_bin.get(b, 0) + cnt

    # Reduce overshoot iteratively until within tolerance (one group size) if possible.
    while current_test_count - target_test > max_group_size:
        overshoot = current_test_count - target_test
        # Try removing a group whose size best matches the overshoot
        removable = sorted(test_groups, key=lambda g: abs(group_sizes[g] - overshoot))
        removed_any = False
        for g_rem in removable:
            new_count = current_test_count - group_sizes[g_rem]
            # Remove if it improves proximity or brings us within tolerance
            if (
                abs(new_count - target_test) < abs(current_test_count - target_test)
                or abs(new_count - target_test) <= max_group_size
            ):
                test_groups.remove(g_rem)
                current_test_count = new_count
                for b, cnt in group_bin_counts[g_rem].items():
                    current_bin[b] -= cnt
                removed_any = True
                break
        if not removed_any:
            break

    # Build boolean mask for test groups
    is_test_group = groups.isin(test_groups)

    X_test = X.loc[is_test_group]
    y_test = y.loc[is_test_group]
    X_train = X.loc[~is_test_group]
    y_train = y.loc[~is_test_group]

    if print_debug:
        test_rows = len(X_test)
        train_rows = len(X_train)
        total_groups = len(unique_groups)
        test_group_count = len(test_groups)
        train_group_count = total_groups - test_group_count
        test_ratio = test_rows / n if n > 0 else 0.0

        print("Split summary:")
        print(
            f"Rows: total={n}, target_test={target_test}, test={test_rows}, "
            f"train={train_rows}, test_ratio={test_ratio:.3f}"
        )
        print(
            f"Groups: total={total_groups}, test={test_group_count}, train={train_group_count}"
        )
        if stratify_cols is not None:
            bt = bins.loc[X_test.index].value_counts()
            br = bins.loc[X_train.index].value_counts()
            print("Test bin counts:", bt.to_dict())
            print("Train bin counts:", br.to_dict())

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Empty train or test split; adjust test_size or check grouping."
        )

    return X_train, X_test, y_train, y_test


def _series_to_bool_array(series: pd.Series) -> np.ndarray:
    """Convert a row-level label series to a boolean NumPy array."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    text = pd.Series(series).astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "cancer", "positive"]).to_numpy(dtype=bool)


def _collapse_bool_label_any_positive(series: pd.Series) -> bool:
    """Collapse row-level labels to a single group label using any-positive logic."""
    vals = _series_to_bool_array(pd.Series(series))
    return bool(np.any(vals)) if vals.size > 0 else False


def make_repeated_patient_splits(
    X: pd.DataFrame,
    *,
    patient_col: str = "patientId",
    label_col: str = "target_cancer_bn",
    n_splits: int = 70,
    test_size: float = 0.3,
    random_state: Optional[int] = 32,
):
    """
    Build repeated stratified train/test patient splits.

    This helper first collapses row-level labels to a single patient-level label
    using any-positive logic, then applies StratifiedShuffleSplit on the unique
    patient table. It returns split metadata rather than row subsets, which is
    useful for repeated ROC experiments where multiple datasets must reuse the
    exact same patient-held-out split definition.
    """
    if patient_col not in X.columns:
        raise KeyError(f"Required patient column '{patient_col}' not found in X.")
    if label_col not in X.columns:
        raise KeyError(f"Required label column '{label_col}' not found in X.")
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError("test_size must be in (0,1)")
    if len(X) == 0:
        return [], pd.DataFrame(columns=[patient_col, label_col])

    patient_rows = []
    for patient_id, group_df in X.groupby(patient_col, sort=True):
        patient_rows.append(
            {
                str(patient_col): str(patient_id),
                str(label_col): _collapse_bool_label_any_positive(group_df[label_col]),
            }
        )
    patient_df = pd.DataFrame(patient_rows)
    labels = patient_df[label_col].astype(int).to_numpy()
    if np.unique(labels).size < 2:
        raise ValueError("Need at least two patient-level classes to build stratified patient splits.")

    splitter = StratifiedShuffleSplit(
        n_splits=int(n_splits),
        test_size=float(test_size),
        random_state=int(random_state) if random_state is not None else None,
    )
    patient_values = patient_df[patient_col].astype(str).to_numpy()
    splits = []
    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(patient_values, labels), start=1):
        splits.append(
            {
                "split_index": int(split_idx),
                "train_patients": patient_values[train_idx].tolist(),
                "test_patients": patient_values[test_idx].tolist(),
            }
        )
    return splits, patient_df


def patient_splitter(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    patient_col: str = "patientId",
    test_size: float = 0.25,
    random_state: Optional[int] = None,
    print_debug: bool = False,
    **kwargs,
):
    """
    Split X,y into train/test such that no patient appears in both sets,
    while stratification is performed at the patient level.

    Row-level labels are first collapsed to one label per patient using
    any-positive logic. A single StratifiedShuffleSplit is then performed on
    the unique patient table, and the resulting patient IDs are expanded back
    to row subsets.
    """
    _ = kwargs
    if patient_col not in X.columns:
        raise KeyError(f"Required patient column '{patient_col}' not found in X.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError("test_size must be in (0,1)")

    X_local = X.copy()
    y_series = pd.Series(y, index=X_local.index)
    patient_rows = []
    for patient_id, group_idx in X_local.groupby(patient_col, sort=True).groups.items():
        patient_rows.append(
            {
                str(patient_col): str(patient_id),
                "_patient_label": _collapse_bool_label_any_positive(y_series.loc[group_idx]),
            }
        )
    patient_df = pd.DataFrame(patient_rows)
    labels = patient_df["_patient_label"].astype(int).to_numpy()
    if np.unique(labels).size < 2:
        raise ValueError("Need at least two patient-level classes to build stratified patient split.")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(test_size),
        random_state=int(random_state) if random_state is not None else None,
    )
    patient_values = patient_df[patient_col].astype(str).to_numpy()
    train_idx, test_idx = next(splitter.split(patient_values, labels))
    train_patients = set(patient_values[train_idx].tolist())
    test_patients = set(patient_values[test_idx].tolist())

    is_test = X_local[patient_col].astype(str).isin(test_patients)
    X_test = X_local.loc[is_test]
    y_test = y_series.loc[is_test]
    X_train = X_local.loc[~is_test]
    y_train = y_series.loc[~is_test]

    if print_debug:
        print("Patient split summary:")
        print(
            f"Rows: total={len(X_local)}, test={len(X_test)}, train={len(X_train)}, "
            f"test_ratio={(len(X_test) / len(X_local)) if len(X_local) else 0.0:.3f}"
        )
        print(
            f"Patients: total={patient_df.shape[0]}, test={len(test_patients)}, train={len(train_patients)}"
        )
        print("Patient labels (train):", patient_df.iloc[train_idx]["_patient_label"].value_counts().to_dict())
        print("Patient labels (test):", patient_df.iloc[test_idx]["_patient_label"].value_counts().to_dict())

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Empty train or test split; adjust test_size or check patient grouping.")

    return X_train, X_test, y_train, y_test

