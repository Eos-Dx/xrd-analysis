"""Private image and PONI geometry helpers for the public utility facade."""

from __future__ import annotations

import json
import re

import numpy as np


def extract_image_data_values(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return image values after applying the legacy binary-mask convention."""
    return data * mask


def extract_center_from_poni(poni_text: str):
    """Extract pixel-space beam-centre coordinates from a PONI document."""
    poni1 = float(re.search(r"Poni1:\s*([0-9.eE+-]+)", poni_text).group(1))
    poni2 = float(re.search(r"Poni2:\s*([0-9.eE+-]+)", poni_text).group(1))
    detector_config_str = re.search(r"Detector_config:\s*(\{.*\})", poni_text).group(1)
    detector_config = json.loads(detector_config_str)
    return poni2 / detector_config["pixel2"], poni1 / detector_config["pixel1"]


def filter_points_by_distance(row, column, distances=None, *, extract_center_from_poni):
    """Zero image pixels outside the supplied radial distance intervals."""
    ref_x, ref_y = extract_center_from_poni(row["ponifile"])
    image = row[column]
    y_coords, x_coords = np.meshgrid(
        range(image.shape[0]), range(image.shape[1]), indexing="ij"
    )
    image_distances = np.sqrt((x_coords - ref_x) ** 2 + (y_coords - ref_y) ** 2)

    masks = []
    for distance in distances:
        mask = np.ones(image.shape, dtype=bool)
        if distance[0] is not None:
            mask &= image_distances >= distance[0]
        if distance[1] is not None:
            mask &= image_distances <= distance[1]
        masks.append(mask)

    return image * np.logical_not(np.logical_or.reduce(masks))


def extract_distance_from_poni(poni_text: str):
    """Extract detector distance from a PONI document."""
    return float(re.search(r"Distance:\s*([0-9.eE+-]+)", poni_text).group(1))


def resize_image(
    row,
    column,
    ref_dist,
    *,
    extract_distance_from_poni,
    adjust_poni_centers_coef,
    substitute_poni_centers,
    cv2,
):
    """Resize an image and update beam-centre PONI fields consistently."""
    image = row[column]
    poni_str = row["ponifile"]
    distance = extract_distance_from_poni(poni_str)
    k = ref_dist * 10e-4 / distance
    adjusted_poni_centers = adjust_poni_centers_coef(poni_str, k)
    center_adjusted_poni = substitute_poni_centers(
        poni_str, adjusted_poni_centers[0], adjusted_poni_centers[1]
    )
    new_h = int(image.shape[0] * k)
    new_w = int(image.shape[1] * k)
    resized = cv2.resize(
        image.astype(np.uint16), (new_w, new_h), interpolation=cv2.INTER_CUBIC
    )
    return resized, center_adjusted_poni


def find_common_region(df, column, square=False, *, extract_center_from_poni):
    """Find the largest shared region around all reported beam centres."""
    distances_up = []
    distances_down = []
    distances_left = []
    distances_right = []
    for _, row in df.iterrows():
        image = row[column]
        center_x, center_y = extract_center_from_poni(row["ponifile"])
        distances_up.append(center_x)
        distances_down.append(image.shape[0] - center_x)
        distances_left.append(center_y)
        distances_right.append(image.shape[1] - center_y)

    max_up = min(distances_up)
    max_down = min(distances_down)
    max_left = min(distances_left)
    max_right = min(distances_right)
    if square:
        height = max_up + max_down
        width = max_left + max_right
        if height < width:
            max_left = (max_left / width) * height
            max_right = (max_right / width) * height
        else:
            max_up = (max_up / height) * width
            max_down = (max_down / height) * width

    return (
        np.floor(max_up),
        np.floor(max_down),
        np.floor(max_left),
        np.floor(max_right),
    )


def cut_common_region(
    row,
    column,
    max_up,
    max_down,
    max_left,
    max_right,
    *,
    extract_center_from_poni,
    calculate_poni_from_pixels,
    substitute_poni_centers,
):
    """Crop a shared region and translate its PONI beam-centre coordinates."""
    image = row[column]
    poni_str = row["ponifile"]
    center_x, center_y = extract_center_from_poni(poni_str)
    x_start = int(center_x - max_up)
    x_end = int(center_x + max_down)
    y_start = int(center_y - max_left)
    y_end = int(center_y + max_right)
    cropped_image = image[y_start:y_end, x_start:x_end]
    poni1, poni2 = calculate_poni_from_pixels(poni_str, max_up, max_left)
    return cropped_image, substitute_poni_centers(poni_str, poni1, poni2)
