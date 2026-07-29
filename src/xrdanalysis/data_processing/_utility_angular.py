"""Private angular-range preparation and weighted-integration support."""

from __future__ import annotations

import numpy as np


def process_angular_ranges(angles):
    """Normalize angular intervals and split ranges crossing +/-180 degrees."""
    processed_ranges = []
    for start_angle, end_angle in angles:
        start = ((start_angle + 180) % 360) - 180
        end = ((end_angle + 180) % 360) - 180
        if start >= end:
            processed_ranges.append((-180, end))
            processed_ranges.append((start, 180))
        else:
            processed_ranges.append((start, end))
    return processed_ranges


def get_angle_span(start, end):
    """Return the angular span, including an interval that wraps around."""
    if end >= start:
        return end - start
    return (end - (-180)) + (180 - start)


def prepare_angular_ranges(
    start_angle,
    end_angle,
    *,
    get_angle_span,
    process_angular_ranges,
):
    """Build processed ranges and their normalized integration weights."""
    if start_angle < -180 or end_angle > 180:
        raise ValueError("The angles must be within -180 and 180 degrees range.")

    original_span = get_angle_span(start_angle, end_angle)
    processed_ranges = process_angular_ranges([(start_angle, end_angle)])
    weights = [
        get_angle_span(range_start, range_end) / original_span
        for range_start, range_end in processed_ranges
    ]
    return {
        "processed_ranges": processed_ranges,
        "weights": weights,
        "original_span": original_span,
        "is_split": len(processed_ranges) > 1,
    }


def perform_weighted_integration(
    data,
    ai_cached,
    range_info,
    npt,
    interpolation_q_range,
    mask=None,
):
    """Integrate each angular segment and combine split-range uncertainties."""
    normalized_results = []
    for angle_range, weight in zip(
        range_info["processed_ranges"], range_info["weights"]
    ):
        result = ai_cached.integrate1d(
            data,
            npt,
            radial_range=interpolation_q_range,
            azimuth_range=angle_range,
            error_model="azimuthal",
            mask=mask,
        )
        normalized_results.append(
            {
                "intensity": result.intensity * weight,
                "sigma": result.sigma * weight,
                "std": result.std * weight,
                "radial": result.radial,
            }
        )

    if range_info["is_split"]:
        combined_sigma = sum(result["sigma"] ** 2 for result in normalized_results)
        combined_std = sum(result["std"] ** 2 for result in normalized_results)
        return (
            normalized_results[0]["radial"],
            sum(result["intensity"] for result in normalized_results),
            np.sqrt(combined_sigma),
            np.sqrt(combined_std),
        )

    result = normalized_results[0]
    return (
        result["radial"],
        result["intensity"],
        result["sigma"],
        result["std"],
    )


def unpack_rotating_angles_results(results):
    """Flatten rotating-angle analysis output into dataframe-ready columns."""
    if len(results) == 4:
        result_list, dist, center_x, center_y = results
        adjusted_distance = None
    else:
        result_list, dist, center_x, center_y, adjusted_distance = results

    col_dict = {}
    for angle, radial, intensity, sigma, std in result_list:
        col_dict[f"q_range_{angle[0]}_{angle[1]}"] = radial
        col_dict[f"radial_profile_data_{angle[0]}_{angle[1]}"] = intensity
        col_dict[f"sigma_{angle[0]}_{angle[1]}"] = sigma
        col_dict[f"std_{angle[0]}_{angle[1]}"] = std

    col_dict["calculated_distance"] = dist
    col_dict["center_x"] = center_x
    col_dict["center_y"] = center_y
    col_dict["adjusted_distance"] = adjusted_distance
    return col_dict
