"""Point operations and utilities."""

from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass 
class Point:
    """Represents a measurement point."""
    x: float
    y: float
    point_id: Optional[int] = None
    point_type: str = "generated"  # "generated" or "user"
    
    
@dataclass
class Zone:
    """Represents a measurement zone."""
    center_x: float
    center_y: float
    radius: float


def sort_points_by_coordinates(points: List[Point]) -> List[Point]:
    """Sort points by x, then y coordinates for efficient measurement order."""
    return sorted(points, key=lambda p: (p.x, p.y))


def convert_pixel_to_mm(pixel_pos: Tuple[float, float], 
                       real_pos_mm: Tuple[float, float],
                       include_center: Tuple[float, float], 
                       pixel_to_mm_ratio: float) -> Tuple[float, float]:
    """Convert pixel coordinates to mm coordinates."""
    x_pixel, y_pixel = pixel_pos
    real_x_mm, real_y_mm = real_pos_mm
    
    x_mm = real_x_mm - (x_pixel - include_center[0]) / pixel_to_mm_ratio
    y_mm = real_y_mm - (y_pixel - include_center[1]) / pixel_to_mm_ratio
    
    return (x_mm, y_mm)
