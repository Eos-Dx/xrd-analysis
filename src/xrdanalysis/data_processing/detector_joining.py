"""
Detector joining pipeline: build per-type common q-range, re-integrate, and merge
PRIMARY/SECONDARY acquisitions into a single canonical row per base measurement.
Includes a scikit-learn compatible wrapper (DetectorJoiner).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from xrdanalysis.data_processing.transformers import AzimuthalIntegration

# ----------------- helpers -----------------


def _split_name_get_base_and_det(name: str) -> Tuple[str, str]:
    n = str(name)
    up = n.upper()
    if up.endswith("_PRIMARY"):
        return n[: -len("_PRIMARY")], "PRIMARY"
    if up.endswith("_SECONDARY"):
        return n[: -len("_SECONDARY")], "SECONDARY"
    return n, "SINGLE"  # explicitly mark non-suffixed measurements


def _classify_type(distance_val: float, type_rules: Dict[str, Tuple[Any, Any]]) -> str:
    for label, (op, thresh) in type_rules.items():
        if op == "<" and distance_val < thresh:
            return label
        if op == "<=" and distance_val <= thresh:
            return label
        if op == ">" and distance_val > thresh:
            return label
        if op == ">=" and distance_val >= thresh:
            return label
        if op == "==" and distance_val == thresh:
            return label
        if op == "between" and isinstance(thresh, (list, tuple)) and len(thresh) == 2:
            a, b = thresh
            if a <= distance_val <= b:
                return label
    return "OTHER"


def average_ignore_zeros(arr2d: np.ndarray) -> np.ndarray:
    """
    Average along axis=0 ignoring zeros.
    Works for any 2D array:
      - (H, W): collapses rows to (W,)
      - (N, W): averages stack of 1D profiles to (W,)
    """
    arr = np.asarray(arr2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array; got ndim={arr.ndim}")
    mask = arr != 0
    sums = arr.sum(axis=0)
    counts = mask.sum(axis=0)
    out = np.zeros_like(sums, dtype=float)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz]
    return out


def _common_q_range_tuple_for_type(
    type_df: pd.DataFrame,
) -> Optional[Tuple[float, float]]:
    """
    Compute a common q-range TUPLE for the whole type:
      - take all q_range arrays
      - q_min_all = max(starts)   (biggest smallest q)
      - q_max_all = min(ends)     (smallest biggest q)
      - return (1.1 * q_min_all, q_max_all), clamped to be valid (start < end)
    Returns tuple (q_start, q_end) or None if cannot compute.
    """
    starts: List[float] = []
    ends: List[float] = []
    for _, r in type_df.iterrows():
        q = r.get("q_range", None)
        if q is None or len(q) == 0:
            continue
        q = np.asarray(q, dtype=float)
        if q[0] > q[-1]:
            q = q[::-1]
        starts.append(float(q[0]))
        ends.append(float(q[-1]))

    if not starts or not ends:
        return None

    q_min_all = max(starts)  # biggest smallest q
    q_max_all = min(ends)  # smallest biggest  q

    if not np.isfinite(q_min_all) or not np.isfinite(q_max_all):
        return None

    q_start = 1.1 * q_min_all
    q_end = q_max_all

    if q_start >= q_end:
        return None

    return float(q_start), float(q_end)


# ----------------- main -----------------


def join_detectors(
    df: pd.DataFrame,
    *,
    name_field: Optional[str] = None,
    faulty_pixels=None,
    npt: int = 200,
    angles: int = 180,
    type_rules: Optional[Dict[str, Tuple[Any, Any]]] = None,
    calibration_mode: str = "poni",
    interpolation_q_range: Optional[Tuple[float, float]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Pipeline:
      1) First 2D azimuthal integration -> adds 'q_range' and 2D 'radial_profile_data'.
      2) Classify 'type_measurement' using rules on 'calculated_distance'.
      3) Build one common 'interpolation_q_range' PER TYPE as a TUPLE (q_start, q_end):
           q_start = 1.1 * (max of all q starts in that type)
           q_end   = (min of all q ends in that type)
         Assign this tuple to all rows of that type.
      4) Re-integrate each row on its per-type common tuple range
         (AzimuthalIntegration.transform honors 'interpolation_q_range' tuples).
      5) Group by base_meas:
         - If exactly one PRIMARY and one SECONDARY: merge (secondary overwrites non-zeros)
             polar_data = merged 2D map
             radial_profile_data = collapsed 1D (rows-avg ignoring zeros)
             drop the SECONDARY row
         - If SINGLE (no suffix) or detector-only: no merge
             polar_data = that row's 2D map
             radial_profile_data = collapsed 1D
         - If multiple SINGLE frames for same base_meas: average their 1D profiles; keep first 2D as polar_data.
      6) Return processed DataFrame (no goodness transform here).

    name_field auto-detection: uses 'meas_name' if present, else 'cal_name'.
    interpolation_q_range: If provided, uses this fixed q-range instead of computing per-type ranges.
    """
    if type_rules is None:
        type_rules = {"WAXS": ("<=", 0.05), "SAXS": (">", 0.05)}

    processed_df = df.copy().reset_index(drop=True)

    # Resolve measurement name column
    if name_field is None:
        if "meas_name" in processed_df.columns:
            name_field = "meas_name"
        elif "cal_name" in processed_df.columns:
            name_field = "cal_name"
        else:
            raise KeyError(
                "Neither 'meas_name' nor 'cal_name' found in DataFrame. Provide name_field."
            )
    elif name_field not in processed_df.columns:
        # try fallback automatically
        fallback = "cal_name" if name_field != "cal_name" else "meas_name"
        if fallback in processed_df.columns:
            name_field = fallback
        else:
            raise KeyError(f"'{name_field}' not found in DataFrame.")

    # Parse base id + detector tag
    processed_df["__name_raw"] = processed_df[name_field].astype(str)
    parsed = processed_df["__name_raw"].apply(
        lambda n: pd.Series(
            _split_name_get_base_and_det(n), index=["base_meas", "detector"]
        )
    )
    processed_df = pd.concat([processed_df, parsed], axis=1)

    # --- First integration (2D) ---
    azint2D = AzimuthalIntegration(
        calibration_mode=calibration_mode,
        faulty_pixels=faulty_pixels,
        integration_mode="2D",
        npt=npt,
        angles=angles,
    )
    processed_df = azint2D.transform(processed_df)
    if debug:
        print(f"[INFO] After initial 2D integration: {processed_df.shape}")

    # Ensure array-holding columns exist & are object dtype
    for col in [
        "q_range",
        "radial_profile_data",
        "interpolation_q_range",
        "polar_data",
    ]:
        if col not in processed_df.columns:
            processed_df[col] = None
    processed_df = processed_df.astype(
        {
            "q_range": "object",
            "radial_profile_data": "object",
            "interpolation_q_range": "object",
            "polar_data": "object",
        }
    )

    # --- Type classification (immediately after first integration) ---
    dist = pd.to_numeric(
        processed_df.get("calculated_distance", np.nan), errors="coerce"
    )
    processed_df["type_measurement"] = dist.apply(
        lambda d: _classify_type(d, type_rules)
    )
    if debug:
        print(
            "[INFO] type_measurement counts:",
            processed_df["type_measurement"].value_counts(dropna=False).to_dict(),
        )

    # --- Per-type common interpolation_q_range as TUPLE (q_start, q_end) ---
    if interpolation_q_range is not None:
        # Use manual q-range for all rows
        processed_df["interpolation_q_range"] = pd.Series(
            [tuple(interpolation_q_range) for _ in range(len(processed_df))],
            index=processed_df.index,
            dtype="object",
        )
        if debug:
            print(
                f"[INFO] Using manual interpolation_q_range: ({interpolation_q_range[0]:.6f}, {interpolation_q_range[1]:.6f})"
            )
    else:
        # Compute per-type common q-range
        for t, tdf in processed_df.groupby("type_measurement", sort=False):
            q_tuple = _common_q_range_tuple_for_type(tdf)
            if q_tuple is None:
                if debug:
                    print(
                        f"[WARN] Cannot compute common tuple q-range for type '{t}'. Skipping re-integration."
                    )
                continue
            # assign via Series (object dtype, per-row tuples)
            processed_df.loc[tdf.index, "interpolation_q_range"] = pd.Series(
                [tuple(q_tuple) for _ in range(len(tdf))],
                index=tdf.index,
                dtype="object",
            )
            if debug:
                print(
                    f"[INFO] Type '{t}': interpolation_q_range tuple = ({q_tuple[0]:.6f}, {q_tuple[1]:.6f})"
                )

    # --- Re-integrate per-row on the per-type tuple range (only rows that have it) ---
    reint_indices = processed_df.index[
        processed_df["interpolation_q_range"].notna()
    ].tolist()
    for idx in reint_indices:
        row = processed_df.loc[idx]
        q_tuple = row["interpolation_q_range"]  # (q_start, q_end)
        single = row.to_frame().T.copy()
        single.iloc[0, single.columns.get_loc("interpolation_q_range")] = q_tuple
        out = azint2D.transform(single)  # expected to honor tuple q-range
        if "q_range" in out.columns:
            processed_df.at[idx, "q_range"] = np.asarray(
                out.iloc[0]["q_range"], dtype=float
            )
        if "radial_profile_data" in out.columns:
            processed_df.at[idx, "radial_profile_data"] = out.iloc[0][
                "radial_profile_data"
            ]

    if debug:
        print(
            f"[INFO] Re-integrated rows on per-type tuple ranges: {len(reint_indices)}"
        )

    # --- Merge STRICT 1:1 (PRIMARY + SECONDARY). SINGLE / detector-only stay unmerged. ---
    drop_marks: List[int] = []

    for base_meas, gdf in processed_df.groupby("base_meas", sort=False):
        prim_rows = gdf[gdf["detector"] == "PRIMARY"]
        sec_rows = gdf[gdf["detector"] == "SECONDARY"]
        single_rows = gdf[gdf["detector"] == "SINGLE"]

        # at most one of each for parallel acquisition
        if len(prim_rows) > 1 or len(sec_rows) > 1:
            raise ValueError(
                f"Group '{base_meas}' has multiple PRIMARY/SECONDARY rows (PRIMARY={len(prim_rows)}, SECONDARY={len(sec_rows)}); expected at most one of each."
            )

        # A) strict pair -> merge
        if len(prim_rows) == 1 and len(sec_rows) == 1:
            p_idx = prim_rows.index[0]
            s_idx = sec_rows.index[0]

            p_map = np.asarray(
                processed_df.at[p_idx, "radial_profile_data"], dtype=float
            )
            s_map = np.asarray(
                processed_df.at[s_idx, "radial_profile_data"], dtype=float
            )
            if p_map.ndim == 1:
                p_map = p_map[np.newaxis, :]
            if s_map.ndim == 1:
                s_map = s_map[np.newaxis, :]

            if p_map.shape[1] != s_map.shape[1]:
                minw = min(p_map.shape[1], s_map.shape[1])
                p_map = p_map[:, :minw]
                s_map = s_map[:, :minw]

            merged_map = p_map.copy()
            merged_map[s_map != 0] = s_map[s_map != 0]

            profile_1d = average_ignore_zeros(merged_map)

            processed_df.at[p_idx, "polar_data"] = merged_map  # 2D merged
            processed_df.at[p_idx, "radial_profile_data"] = profile_1d  # 1D collapsed
            drop_marks.append(s_idx)

            if debug:
                print(f"[GROUP] {base_meas}: merged -> canonical={p_idx}")

        # B) SINGLE (no suffix)
        elif len(single_rows) >= 1 and len(prim_rows) == 0 and len(sec_rows) == 0:
            profiles_1d: List[np.ndarray] = []
            maps_2d: List[np.ndarray] = []
            for idx in single_rows.index:
                m2d = np.asarray(
                    processed_df.at[idx, "radial_profile_data"], dtype=float
                )
                if m2d.ndim == 1:
                    m2d = m2d[np.newaxis, :]
                maps_2d.append(m2d)
                profiles_1d.append(average_ignore_zeros(m2d))

            if len(profiles_1d) == 1:
                polar_1d = profiles_1d[0]
                polar_2d = maps_2d[0]
                canonical = single_rows.index[0]
            else:
                minW = min(len(p) for p in profiles_1d)
                stack = np.vstack([p[:minW] for p in profiles_1d])  # (N, W)
                polar_1d = average_ignore_zeros(stack)
                polar_2d = maps_2d[0][:, :minW]
                canonical = single_rows.index[0]
                for idx in single_rows.index[1:]:
                    drop_marks.append(idx)

            processed_df.at[canonical, "polar_data"] = polar_2d
            processed_df.at[canonical, "radial_profile_data"] = polar_1d

            if debug:
                print(f"[GROUP] {base_meas}: SINGLE -> canonical={canonical}")

        # C) Only one detector present
        elif (len(prim_rows) == 1 and len(sec_rows) == 0 and len(single_rows) == 0) or (
            len(sec_rows) == 1 and len(prim_rows) == 0 and len(single_rows) == 0
        ):
            only_idx = prim_rows.index[0] if len(prim_rows) == 1 else sec_rows.index[0]
            m2d = np.asarray(
                processed_df.at[only_idx, "radial_profile_data"], dtype=float
            )
            if m2d.ndim == 1:
                m2d = m2d[np.newaxis, :]

            polar_1d = average_ignore_zeros(m2d)
            processed_df.at[only_idx, "polar_data"] = m2d
            processed_df.at[only_idx, "radial_profile_data"] = polar_1d

            if debug:
                det = "PRIMARY" if len(prim_rows) == 1 else "SECONDARY"
                print(f"[GROUP] {base_meas}: {det}-only -> canonical={only_idx}")

        else:
            if debug:
                print(f"[WARN] {base_meas}: unexpected detector composition; skipped.")

    # Drop secondary / duplicate rows we marked
    if drop_marks:
        processed_df = processed_df.drop(index=drop_marks).reset_index(drop=True)
        if debug:
            print(
                f"[INFO] Dropped {len(drop_marks)} non-canonical rows; final rows: {processed_df.shape[0]}"
            )

    # Cleanup helper column
    processed_df = processed_df.drop(columns=["__name_raw"], errors="ignore")

    return processed_df


class DetectorJoiner(TransformerMixin):
    """
    scikit-learn compatible wrapper around join_detectors.

    Automatically picks 'meas_name' or 'cal_name' for detector suffix parsing
    unless name_field is explicitly provided.
    """

    def __init__(
        self,
        name_field: Optional[str] = None,
        faulty_pixels=None,
        npt: int = 200,
        angles: int = 180,
        type_rules: Optional[Dict[str, Tuple[Any, Any]]] = None,
        calibration_mode: str = "poni",
        interpolation_q_range: Optional[Tuple[float, float]] = None,
        debug: bool = False,
    ) -> None:
        self.name_field = name_field
        self.faulty_pixels = faulty_pixels
        self.npt = npt
        self.angles = angles
        self.type_rules = type_rules
        self.calibration_mode = calibration_mode
        self.interpolation_q_range = interpolation_q_range
        self.debug = debug

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return join_detectors(
            X,
            name_field=self.name_field,
            faulty_pixels=self.faulty_pixels,
            npt=self.npt,
            angles=self.angles,
            type_rules=self.type_rules,
            calibration_mode=self.calibration_mode,
            interpolation_q_range=self.interpolation_q_range,
            debug=self.debug,
        )
