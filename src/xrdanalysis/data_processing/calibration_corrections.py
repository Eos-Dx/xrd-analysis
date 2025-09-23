"""
Calibration correction utilities:
- Build per-angle correction profiles from calibration frames (e.g., AgBH)
- Apply those profiles to sample polar_data and update radial profiles
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse average_ignore_zeros from detector_joining to avoid duplication
from xrdanalysis.data_processing.detector_joining import average_ignore_zeros

# ---------- helpers ----------


def _closest_index(q_range, target_q: float) -> int:
    q = np.asarray(q_range, dtype=float)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("q_range must be a 1D non-empty sequence.")
    return int(np.nanargmin(np.abs(q - float(target_q))))


def _per_angle_correction_from_column(
    polar_data: np.ndarray, col_idx: int
) -> np.ndarray:
    """
    Build per-angle correction:
      corr[angle] = mean(nonzero(column)) / column_value
    For zero pixels, corr = 1.0 (no scaling).
    """
    P = np.asarray(polar_data, dtype=float)
    if P.ndim != 2:
        raise ValueError("polar_data must be 2D (n_angles, n_q).")
    if not (0 <= col_idx < P.shape[1]):
        raise IndexError(
            f"col_idx {col_idx} out of bounds for polar_data with shape {P.shape}."
        )

    column = P[:, col_idx]
    nonzero = column != 0
    avg_val = column[nonzero].mean() if np.any(nonzero) else 1.0

    corr = np.ones_like(column, dtype=float)
    corr[nonzero] = avg_val / column[nonzero]
    return corr  # shape: (n_angles,)


def _per_angle_correction_from_multi_columns(
    polar_data: np.ndarray, center_col_idx: int, window_size: int = 2
) -> np.ndarray:
    """
    Build per-angle correction by averaging profiles from multiple Q-columns.

    Takes the center column and +/- window_size columns around it, computes
    correction profile for each, then averages them for more robust correction.

    Parameters:
        polar_data: 2D array (n_angles, n_q)
        center_col_idx: Index of the central Q-column to use
        window_size: Number of columns to include on each side of center

    Returns:
        Averaged correction profile (n_angles,)
    """
    P = np.asarray(polar_data, dtype=float)
    if P.ndim != 2:
        raise ValueError("polar_data must be 2D (n_angles, n_q).")
    if not (0 <= center_col_idx < P.shape[1]):
        raise IndexError(
            f"center_col_idx {center_col_idx} out of bounds for polar_data with shape {P.shape}."
        )

    n_angles, n_q = P.shape

    # Define the range of columns to use (ensure within bounds)
    start_col = max(0, center_col_idx - window_size)
    end_col = min(n_q, center_col_idx + window_size + 1)

    # Collect correction profiles from all columns in the window
    profiles = []
    for col_idx in range(start_col, end_col):
        try:
            profile = _per_angle_correction_from_column(P, col_idx)
            profiles.append(profile)
        except Exception:
            # Skip problematic columns
            continue

    if not profiles:
        # Fallback to all ones if no valid profiles
        return np.ones(n_angles, dtype=float)

    # Average all valid profiles
    profiles_array = np.array(profiles)  # shape: (n_profiles, n_angles)
    averaged_profile = np.mean(profiles_array, axis=0)

    return averaged_profile


def _apply_profile_to_polar(polar_data: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """
    Multiply polar map by profile. Accepts:
      - per-angle profile (len == n_angles) -> row-wise scaling
      - per-q profile (len == n_q)          -> column-wise scaling
    """
    P = np.asarray(polar_data, dtype=float)
    prof = np.asarray(profile, dtype=float)
    if P.ndim != 2:
        raise ValueError("polar_data must be 2D (n_angles, n_q).")
    n_angles, n_q = P.shape

    if prof.shape[0] == n_angles:
        return (P.T * prof).T
    elif prof.shape[0] == n_q:
        return P * prof
    else:
        raise ValueError(
            f"correction_profile length {prof.shape[0]} does not match n_angles ({n_angles}) or n_q ({n_q})."
        )


# ---------- step 1: compute correction_profile for AgBH in df_calib ----------


def compute_calib_correction_profiles(
    df_calib: pd.DataFrame,
    *,
    id_col: str = "id",
    calib_type_col: str = "calib_calibrationType",
    type_col: str = "type_measurement",
    q_col: str = "q_range",
    polar_col: str = "polar_data",
    out_profile_col: str = "correction_profile",
    targets: Dict[str, float] | None = None,
    inplace: bool = False,
    modify: bool = True,
    window_size: int = 2,
) -> pd.DataFrame:
    """
    Build per-angle correction_profile arrays for AgBH calibration frames.

    Parameters:
        modify: If True, applies the correction directly to polar_data and backs up
                original data to polar_data_raw. If False, only computes correction_profile
                without modifying polar_data.
        window_size: Number of Q-columns on each side of the target to include in
                     the averaged correction profile (default: 2)
    """
    if targets is None:
        targets = {"SAXS": 1.076, "WAXS": 3.228}

    out = df_calib if inplace else df_calib.copy()
    # Initialize correction_profile column
    out[out_profile_col] = None

    # Only AgBH calibration frames
    mask_agbh = out[calib_type_col].astype(str).str.upper().eq("AGBH")

    def _build_profile(row: pd.Series):
        try:
            t = str(row.get(type_col, "SAXS")).upper()
            target_q = targets.get(t, 1.076)
            q = row[q_col]
            P = row[polar_col]
            if q is None or P is None:
                return None
            idx = _closest_index(q, target_q)
            return _per_angle_correction_from_multi_columns(
                P, idx, window_size=window_size
            ).astype(float)
        except Exception:
            return None

    # Apply row-wise and assign as objects (lists or arrays)
    out.loc[mask_agbh, out_profile_col] = out.loc[mask_agbh].apply(
        _build_profile, axis=1
    )

    # If modify=True, apply correction to polar_data and backup original
    if modify:
        # Backup original polar_data for AgBH rows that have a valid profile
        valid_profile_mask = mask_agbh & out[out_profile_col].notna()
        if valid_profile_mask.any():
            out.loc[valid_profile_mask, polar_col + "_raw"] = out.loc[
                valid_profile_mask, polar_col
            ].copy()

            # Apply correction to polar_data
            def _apply_correction(row):
                profile = row[out_profile_col]
                P = np.asarray(row[polar_col], dtype=float)
                if profile is not None:
                    return _apply_profile_to_polar(P, np.asarray(profile, dtype=float))
                return P

            out.loc[valid_profile_mask, polar_col] = out.loc[valid_profile_mask].apply(
                _apply_correction, axis=1
            )

    return out


# ---------- step 2: apply those profiles to df ----------


def correct_with_calib_profiles(
    df: pd.DataFrame,
    df_calib: pd.DataFrame,
    *,
    df_calib_id_col: str = "id",
    df_calib_type_col: str = "calib_calibrationType",
    df_calib_profile_col: str = "correction_profile",
    df_id_lookup_col: str = "calib_name",
    polar_col: str = "polar_data",
    radial_out_col: str = "radial_profile_data",
    strict: bool = False,
    inplace: bool = False,
    keep: bool = False,  # backup originals if True
    raw_polar_col: str = "polar_data_raw",
    raw_radial_col: str = "radial_profile_data_raw",
) -> pd.DataFrame:
    """
    Uses df_calib[correction_profile] (AgBH only) matched by df[calib_name] == df_calib[id]
    to correct df[polar_data] and update df[radial_profile_data] via average_ignore_zeros.

    If keep=True, saves original polar_data and radial_profile_data
    into new columns (default: 'polar_data_raw', 'radial_profile_data_raw').
    """
    # Filter only AgBH calibrations with valid profile
    mask_agbh = df_calib[df_calib_type_col].astype(str).str.upper() == "AGBH"
    calib_agbh = df_calib[mask_agbh].dropna(
        subset=[df_calib_id_col, df_calib_profile_col]
    )

    # Build lookup: id -> correction_profile
    calib_map: Dict[Any, Any] = (
        calib_agbh.drop_duplicates(subset=[df_calib_id_col], keep="first")
        .set_index(df_calib_id_col)[df_calib_profile_col]
        .to_dict()
    )

    out = df if inplace else df.copy()

    def _process_row(row: pd.Series) -> pd.Series:
        key = row.get(df_id_lookup_col, None)
        profile = calib_map.get(key, None)

        if profile is None:
            if strict:
                raise KeyError(
                    f"No AgBH correction_profile for {df_id_lookup_col}='{key}'."
                )
            return row  # unchanged

        P = np.asarray(row[polar_col], dtype=float)

        if keep:
            # Save raw data before correction
            if raw_polar_col not in row.index:
                row[raw_polar_col] = P.copy()
            if raw_radial_col not in row.index:
                row[raw_radial_col] = row.get(radial_out_col, None)

        # Apply profile (per-angle scaling expected)
        P_corr = _apply_profile_to_polar(P, np.asarray(profile, dtype=float))

        row[polar_col] = P_corr
        row[radial_out_col] = average_ignore_zeros(P_corr)
        return row

    return out.apply(_process_row, axis=1)
