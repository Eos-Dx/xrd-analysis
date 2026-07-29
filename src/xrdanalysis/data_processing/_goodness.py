"""Private numerical and dataframe helpers for goodness transformers."""

from __future__ import annotations

from collections.abc import Mapping
from operator import ge, gt, le, lt
from typing import Any

import numpy as np
import pandas as pd


def _select_data_column(frame: pd.DataFrame, requested_column: str) -> str:
    """Choose the requested profile column or preserve the legacy fallback."""
    if requested_column in frame.columns:
        return requested_column
    if "polar_data" in frame.columns:
        print(
            f"Warning: Column '{requested_column}' not found. "
            "Using 'polar_data' instead."
        )
        return "polar_data"
    if "radial_profile_data" in frame.columns:
        print(
            "Warning: Column "
            f"'{requested_column}' not found. Using 'radial_profile_data' instead."
        )
        return "radial_profile_data"
    raise KeyError(
        f"Neither '{requested_column}', 'polar_data', nor 'radial_profile_data' "
        "found in DataFrame columns"
    )


def _compute_hf_score(
    array: object,
    *,
    column_name: str,
    skip_bins: int,
    hf_cutoff_fraction: float,
    include_deviation: bool,
) -> tuple[float, np.ndarray | None]:
    """Compute the legacy high-frequency score and optional aligned deviation."""
    full_array = np.array(array)
    if full_array.ndim != 2:
        raise ValueError(f"Column '{column_name}' must contain 2D arrays.")

    profile = full_array[:, skip_bins:]
    n_azimuthal, n_q = profile.shape
    deviation = np.full_like(profile, np.nan, dtype=float)
    for index in range(n_q):
        values = profile[:, index]
        valid = (~np.isnan(values)) & (values != 0)
        if np.any(valid):
            mean_value = values[valid].mean()
            if mean_value != 0:
                deviation[valid, index] = (
                    (values[valid] - mean_value) / mean_value * 100.0
                )

    # FFT uses zero-filled deviations after removing their global mean.
    fft_input = np.nan_to_num(deviation, nan=0.0)
    fft_input = fft_input - fft_input.mean()
    power_shifted = np.fft.fftshift(np.abs(np.fft.fft2(fft_input)) ** 2)

    frequency_y = np.fft.fftshift(np.fft.fftfreq(n_azimuthal))
    frequency_x = np.fft.fftshift(np.fft.fftfreq(n_q))
    frequency_x_grid, frequency_y_grid = np.meshgrid(frequency_x, frequency_y)
    frequency_magnitude = np.sqrt(frequency_x_grid**2 + frequency_y_grid**2)
    high_frequency_power = power_shifted[frequency_magnitude > hf_cutoff_fraction].sum()
    total_power = power_shifted.sum()

    aligned_deviation = None
    if include_deviation:
        aligned_deviation = np.full_like(full_array, np.nan, dtype=float)
        aligned_deviation[:, skip_bins:] = deviation

    if total_power <= 0:
        return 0.0, aligned_deviation
    return float(high_frequency_power / total_power * 100.0), aligned_deviation


def transform_goodness_dataframe(
    frame: pd.DataFrame,
    *,
    column: str,
    skip_bins: int,
    hf_cutoff_fraction: float,
    output_col: str,
    save_dev: bool,
    diff_col: str,
) -> pd.DataFrame:
    """Add goodness scores and optional deviation matrices to a dataframe copy."""
    output = frame.copy()
    actual_column = _select_data_column(output, column)
    deviations: list[np.ndarray] = []

    def compute_score(array: object) -> float:
        score, deviation = _compute_hf_score(
            array,
            column_name=actual_column,
            skip_bins=skip_bins,
            hf_cutoff_fraction=hf_cutoff_fraction,
            include_deviation=save_dev,
        )
        if save_dev:
            assert deviation is not None
            deviations.append(deviation)
        return score

    # Series.apply preserves the legacy empty-frame output dtype.
    output[output_col] = output[actual_column].apply(compute_score)
    if save_dev:
        output[diff_col] = deviations
    return output


def filter_goodness_dataframe(
    frame: pd.DataFrame,
    *,
    goodness_column: str,
    type_column: str,
    thresholds: Mapping[Any, float],
    rule: str,
    default_threshold: float,
    verbose: bool,
) -> pd.DataFrame:
    """Filter goodness rows while preserving existing threshold and print rules."""
    output = frame.copy()
    if goodness_column not in output.columns:
        raise KeyError(f"Goodness column '{goodness_column}' not found in DataFrame")
    if type_column not in output.columns:
        raise KeyError(f"Type column '{type_column}' not found in DataFrame")

    comparison = {">": gt, ">=": ge, "<": lt, "<=": le}[rule]

    def meets_criteria(row: pd.Series) -> bool:
        threshold = thresholds.get(row[type_column], default_threshold)
        return comparison(row[goodness_column], threshold)

    initial_count = len(output)
    filtered = output[output.apply(meets_criteria, axis=1)]

    if verbose:
        final_count = len(filtered)
        removed_count = initial_count - final_count
        print(
            f"GoodnessFilter: {initial_count} -> {final_count} rows "
            f"({removed_count} removed, rule: '{rule}')"
        )
        for measurement_type in output[type_column].unique():
            original = len(output[output[type_column] == measurement_type])
            kept = len(filtered[filtered[type_column] == measurement_type])
            threshold = thresholds.get(measurement_type, default_threshold)
            print(
                f"  {measurement_type}: {original} -> {kept} rows "
                f"(goodness {rule} {threshold})"
            )

    return filtered
