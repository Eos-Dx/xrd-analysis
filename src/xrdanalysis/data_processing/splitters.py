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
