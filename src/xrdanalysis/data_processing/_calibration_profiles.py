"""Private numerical kernels for calibration-profile correction."""

from __future__ import annotations

import numpy as np


def _closest_index(q_range, target_q: float) -> int:
    """Return index nearest ``target_q`` on a non-empty one-dimensional grid."""
    q = np.asarray(q_range, dtype=float)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("q_range must be a 1D non-empty sequence.")
    return int(np.nanargmin(np.abs(q - float(target_q))))


def _per_angle_correction_from_column(
    polar_data: np.ndarray, col_idx: int
) -> np.ndarray:
    """Build a per-angle scale profile from one polar-map column."""
    polar = np.asarray(polar_data, dtype=float)
    if polar.ndim != 2:
        raise ValueError("polar_data must be 2D (n_angles, n_q).")
    if not 0 <= col_idx < polar.shape[1]:
        raise IndexError(
            f"col_idx {col_idx} out of bounds for polar_data with shape {polar.shape}."
        )

    column = polar[:, col_idx]
    nonzero = column != 0
    average = column[nonzero].mean() if np.any(nonzero) else 1.0

    correction = np.ones_like(column, dtype=float)
    correction[nonzero] = average / column[nonzero]
    return correction


def _per_angle_correction_from_multi_columns(
    polar_data: np.ndarray, center_col_idx: int, window_size: int = 2
) -> np.ndarray:
    """Average per-angle scale profiles around one polar-map column."""
    polar = np.asarray(polar_data, dtype=float)
    if polar.ndim != 2:
        raise ValueError("polar_data must be 2D (n_angles, n_q).")
    if not 0 <= center_col_idx < polar.shape[1]:
        raise IndexError(
            "center_col_idx "
            f"{center_col_idx} out of bounds for polar_data with shape {polar.shape}."
        )

    n_angles, n_q = polar.shape
    start_col = max(0, center_col_idx - window_size)
    end_col = min(n_q, center_col_idx + window_size + 1)
    profiles = []
    for col_idx in range(start_col, end_col):
        try:
            profiles.append(_per_angle_correction_from_column(polar, col_idx))
        except Exception:
            # Historical behavior tolerates a bad neighbouring column.
            continue

    if not profiles:
        return np.ones(n_angles, dtype=float)
    return np.mean(np.array(profiles), axis=0)


def _apply_profile_to_polar(polar_data: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Apply a per-angle or per-q profile to a two-dimensional polar map."""
    polar = np.asarray(polar_data, dtype=float)
    correction = np.asarray(profile, dtype=float)
    if polar.ndim != 2:
        raise ValueError("polar_data must be 2D (n_angles, n_q).")
    n_angles, n_q = polar.shape

    if correction.shape[0] == n_angles:
        return (polar.T * correction).T
    if correction.shape[0] == n_q:
        return polar * correction
    raise ValueError(
        f"correction_profile length {correction.shape[0]} does not match "
        f"n_angles ({n_angles}) or n_q ({n_q})."
    )
