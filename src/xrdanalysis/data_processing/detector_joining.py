"""
Detector joining pipeline: build per-type common q-range, re-integrate, and merge
PRIMARY/SECONDARY acquisitions into a single canonical row per base measurement.
Includes a scikit-learn compatible wrapper (DetectorJoiner).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

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
    interpolation_q_range: Optional[
        Union[Tuple[float, float], Dict[str, Tuple[float, float]]]
    ] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Detector joining pipeline - integrate once based on detector pairing.
    
    Logic:
      1) Parse measurement names -> base_meas, detector type (PRIMARY/SECONDARY/SINGLE)
      2) Expect 'type_measurement' column to exist (from upstream MeasurementTypeClassifier)
      3) Group by base_meas and decide integration mode:
         - If PRIMARY + SECONDARY pair: Do 2D integration, merge in polar, collapse to 1D
         - If SINGLE or single detector: Do 1D integration directly
      4) Use interpolation_q_range dict to get q-range based on type_measurement
      5) Return processed DataFrame with joined measurements

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, must contain 'type_measurement' column
    name_field : str, optional
        Column with measurement names (auto-detects 'meas_name' or 'cal_name')
    faulty_pixels : array-like, optional
        Faulty pixel coordinates
    npt : int
        Number of points for integration
    angles : int
        Number of azimuthal angles for 2D integration
    type_rules : dict, optional
        Deprecated - type_measurement should come from upstream classifier
    calibration_mode : str
        Calibration mode ('poni', etc.)
    interpolation_q_range : dict
        Dict mapping type_measurement values to (q_start, q_end) tuples
        Example: {'WAXS': (3, 21.0), 'SAXS': (1, 2)}
    debug : bool
        Print debug information
    """
    processed_df = df.copy().reset_index(drop=True)
    
    # Verify type_measurement column exists
    if "type_measurement" not in processed_df.columns:
        raise ValueError(
            "Column 'type_measurement' not found. "
            "MeasurementTypeClassifier must run before DetectorJoiner."
        )

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
        fallback = "cal_name" if name_field != "cal_name" else "meas_name"
        if fallback in processed_df.columns:
            name_field = fallback
        else:
            raise KeyError(f"'{name_field}' not found in DataFrame.")

    # Parse base id + detector tag
    processed_df["__name_raw"] = processed_df[name_field].astype(str)
    
    # Check if detector column already exists (from FaultyPixelDetector)
    if "detector" in processed_df.columns:
        # Only parse base_meas
        processed_df["base_meas"] = processed_df["__name_raw"].apply(
            lambda n: _split_name_get_base_and_det(n)[0]
        )
    else:
        # Parse both base_meas and detector
        parsed = processed_df["__name_raw"].apply(
            lambda n: pd.Series(
                _split_name_get_base_and_det(n), index=["base_meas", "detector"]
            )
        )
        processed_df = pd.concat([processed_df, parsed], axis=1)

    if debug:
        print(f"[INFO] type_measurement from upstream: {processed_df['type_measurement'].value_counts().to_dict()}")

    # Ensure array-holding columns exist
    for col in ["q_range", "radial_profile_data", "polar_data"]:
        if col not in processed_df.columns:
            processed_df[col] = None
    
    # Helper function to get q-range for a measurement
    def get_q_range(row):
        meas_type = row['type_measurement']
        if interpolation_q_range and isinstance(interpolation_q_range, dict):
            if meas_type in interpolation_q_range:
                return tuple(interpolation_q_range[meas_type])
        return None
    
    # Stats tracking
    stats = {
        "groups_total": processed_df["base_meas"].nunique(),
        "groups_merged_2d": 0,
        "groups_single_1d": 0,
        "groups_detector_only_1d": 0,
        "groups_skipped_multiple": 0,
        "rows_dropped_secondary": 0,
        "rows_removed_skipped_groups": 0,
    }
    
    drop_marks: List[int] = []
    skip_group_indices: List[int] = []

    for base_meas, gdf in processed_df.groupby("base_meas", sort=False):
        # Check for both PRIMARY/PRIM and SECONDARY/SEC naming
        prim_rows = gdf[gdf["detector"].isin(["PRIMARY", "PRIM"])]
        sec_rows = gdf[gdf["detector"].isin(["SECONDARY", "SEC"])]
        single_rows = gdf[gdf["detector"] == "SINGLE"]

        # Skip groups with multiple primaries or secondaries
        if len(prim_rows) > 1 or len(sec_rows) > 1:
            stats["groups_skipped_multiple"] += 1
            skip_group_indices.extend(gdf.index.tolist())
            if debug:
                print(f"[GROUP] {base_meas}: skipped (PRIMARY={len(prim_rows)}, SECONDARY={len(sec_rows)})"
                )
            continue

        # A) PRIMARY + SECONDARY pair -> Do 2D integration, merge, collapse
        if len(prim_rows) == 1 and len(sec_rows) == 1:
            p_idx = prim_rows.index[0]
            s_idx = sec_rows.index[0]
            p_row = processed_df.loc[p_idx]
            s_row = processed_df.loc[s_idx]
            
            # Get q-range based on type_measurement (should be same for pair)
            q_range = get_q_range(p_row)
            
            # Create 2D integrator
            azint2D = AzimuthalIntegration(
                calibration_mode=calibration_mode,
                faulty_pixels=faulty_pixels,
                integration_mode="2D",
                npt=npt,
                angles=angles,
            )
            
            # Integrate primary (2D)
            p_df = p_row.to_frame().T.copy()
            if q_range:
                p_df['interpolation_q_range'] = [q_range]
            p_integrated = azint2D.transform(p_df)
            
            # Integrate secondary (2D)
            s_df = s_row.to_frame().T.copy()
            if q_range:
                s_df['interpolation_q_range'] = [q_range]
            s_integrated = azint2D.transform(s_df)
            
            # Get 2D polar maps
            p_map = np.asarray(p_integrated.iloc[0]['radial_profile_data'], dtype=float)
            s_map = np.asarray(s_integrated.iloc[0]['radial_profile_data'], dtype=float)
            
            if p_map.ndim == 1:
                p_map = p_map[np.newaxis, :]
            if s_map.ndim == 1:
                s_map = s_map[np.newaxis, :]

            # Align widths
            if p_map.shape[1] != s_map.shape[1]:
                minw = min(p_map.shape[1], s_map.shape[1])
                p_map = p_map[:, :minw]
                s_map = s_map[:, :minw]

            # Merge (secondary overwrites non-zero)
            merged_map = p_map.copy()
            merged_map[s_map != 0] = s_map[s_map != 0]

            # Collapse to 1D
            profile_1d = average_ignore_zeros(merged_map)
            q_range_arr = np.asarray(p_integrated.iloc[0]['q_range'], dtype=float)

            # Store results in primary row
            processed_df.at[p_idx, "polar_data"] = merged_map
            processed_df.at[p_idx, "radial_profile_data"] = np.asarray(profile_1d, dtype=float)
            processed_df.at[p_idx, "q_range"] = q_range_arr
            # Preserve calculated_distance if available
            if 'calculated_distance' in p_integrated.columns:
                processed_df.at[p_idx, "calculated_distance"] = p_integrated.iloc[0]['calculated_distance']
            
            # Mark secondary for removal
            drop_marks.append(s_idx)
            stats["groups_merged_2d"] += 1
            stats["rows_dropped_secondary"] += 1

            if debug:
                print(f"[GROUP] {base_meas}: 2D merged -> canonical={p_idx}")

        # B) SINGLE or single detector -> Do 1D integration directly
        elif (len(single_rows) >= 1 and len(prim_rows) == 0 and len(sec_rows) == 0) or \
             (len(prim_rows) == 1 and len(sec_rows) == 0 and len(single_rows) == 0) or \
             (len(sec_rows) == 1 and len(prim_rows) == 0 and len(single_rows) == 0):
            
            # Combine all rows to process
            all_rows = pd.concat([single_rows, prim_rows, sec_rows])
            
            if len(all_rows) == 1:
                # Single measurement - do 1D integration
                idx = all_rows.index[0]
                row = processed_df.loc[idx]
                q_range = get_q_range(row)
                
                # Create 1D integrator
                azint1D = AzimuthalIntegration(
                    calibration_mode=calibration_mode,
                    faulty_pixels=faulty_pixels,
                    integration_mode="1D",
                    npt=npt,
                )
                
                # Integrate
                row_df = row.to_frame().T.copy()
                if q_range:
                    row_df['interpolation_q_range'] = [q_range]
                integrated = azint1D.transform(row_df)
                
                # Store results
                processed_df.at[idx, "radial_profile_data"] = np.asarray(integrated.iloc[0]['radial_profile_data'], dtype=float)
                processed_df.at[idx, "q_range"] = np.asarray(integrated.iloc[0]['q_range'], dtype=float)
                processed_df.at[idx, "polar_data"] = None  # No 2D for 1D integration
                # Preserve calculated_distance if available
                if 'calculated_distance' in integrated.columns:
                    processed_df.at[idx, "calculated_distance"] = integrated.iloc[0]['calculated_distance']
                
                if len(single_rows) > 0:
                    stats["groups_single_1d"] += 1
                else:
                    stats["groups_detector_only_1d"] += 1
                
                if debug:
                    det_type = "SINGLE" if len(single_rows) > 0 else ("PRIMARY" if len(prim_rows) > 0 else "SECONDARY")
                    print(f"[GROUP] {base_meas}: 1D {det_type} -> canonical={idx}")
            
            else:
                # Multiple SINGLE measurements - integrate each, then average
                profiles = []
                q_ranges = []
                
                azint1D = AzimuthalIntegration(
                    calibration_mode=calibration_mode,
                    faulty_pixels=faulty_pixels,
                    integration_mode="1D",
                    npt=npt,
                )
                
                for idx in all_rows.index:
                    row = processed_df.loc[idx]
                    q_range = get_q_range(row)
                    
                    row_df = row.to_frame().T.copy()
                    if q_range:
                        row_df['interpolation_q_range'] = [q_range]
                    integrated = azint1D.transform(row_df)
                    
                    profiles.append(integrated.iloc[0]['radial_profile_data'])
                    q_ranges.append(integrated.iloc[0]['q_range'])
                
                # Average profiles
                minW = min(len(p) for p in profiles)
                stack = np.vstack([np.asarray(p[:minW], dtype=float) for p in profiles])
                avg_profile = average_ignore_zeros(stack)
                
                # Keep first measurement, drop rest
                canonical = all_rows.index[0]
                processed_df.at[canonical, "radial_profile_data"] = np.asarray(avg_profile, dtype=float)
                processed_df.at[canonical, "q_range"] = np.asarray(q_ranges[0][:minW], dtype=float)
                processed_df.at[canonical, "polar_data"] = None
                # Note: calculated_distance should already be in the row from first integration
                
                for idx in all_rows.index[1:]:
                    drop_marks.append(idx)
                
                stats["groups_single_1d"] += 1
                
                if debug:
                    print(f"[GROUP] {base_meas}: 1D averaged {len(all_rows)} measurements -> canonical={canonical}")
        
        else:
            if debug:
                print(f"[WARN] {base_meas}: unexpected detector composition; skipped.")

    # Drop rows from skipped groups and secondary duplicates
    to_drop = set(skip_group_indices) | set(drop_marks)
    if to_drop:
        # adjust stats to avoid double counting if overlap
        rows_skipped = len(set(skip_group_indices))
        rows_secondary = len(set(drop_marks) - set(skip_group_indices))
        stats["rows_removed_skipped_groups"] = rows_skipped
        stats["rows_dropped_secondary"] = rows_secondary
        processed_df = processed_df.drop(index=list(to_drop)).reset_index(drop=True)
        if debug:
            print(
                f"[INFO] Skipped groups rows dropped: {rows_skipped}; Secondary/duplicates dropped: {rows_secondary}; final rows: {processed_df.shape[0]}"
            )

    # Cleanup helper column
    processed_df = processed_df.drop(columns=["__name_raw"], errors="ignore")

    # Finalize stats
    stats["final_rows"] = len(processed_df)
    if debug:
        print(
            f"[STATS] total_groups={stats['groups_total']}, merged_2d={stats['groups_merged_2d']}, single_1d={stats['groups_single_1d']}, "
            f"detector_only_1d={stats['groups_detector_only_1d']}, skipped_multiple={stats['groups_skipped_multiple']}"
        )
    try:
        processed_df.attrs["join_stats"] = stats
    except Exception:
        pass

    return processed_df


from xrdanalysis.data_processing.transformers import DetectorJoiner
