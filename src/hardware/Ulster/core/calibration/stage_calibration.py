"""Stage calibration utilities."""

from typing import Tuple

from utils.logging import get_module_logger

logger = get_module_logger(__name__)


def calculate_pixel_to_mm_ratio(pixel_distance: float, mm_distance: float) -> float:
    """Calculate the pixel to mm conversion ratio.

    Args:
        pixel_distance: Distance in pixels
        mm_distance: Corresponding distance in mm

    Returns:
        Ratio of pixels per mm
    """
    if mm_distance == 0:
        logger.error("Cannot calculate pixel/mm ratio with zero mm distance")
        return 1.0  # Fallback ratio

    ratio = pixel_distance / mm_distance
    logger.info(
        "Calculated pixel to mm ratio",
        ratio=ratio,
        pixel_dist=pixel_distance,
        mm_dist=mm_distance,
    )
    return ratio


def validate_stage_position(
    x_mm: float, y_mm: float, limits: Tuple[float, float, float, float]
) -> bool:
    """Validate that stage position is within limits.

    Args:
        x_mm: X position in mm
        y_mm: Y position in mm
        limits: (x_min, x_max, y_min, y_max) in mm

    Returns:
        True if position is valid
    """
    x_min, x_max, y_min, y_max = limits

    valid = (x_min <= x_mm <= x_max) and (y_min <= y_mm <= y_max)

    if not valid:
        logger.warning("Stage position outside limits", x=x_mm, y=y_mm, limits=limits)

    return valid
