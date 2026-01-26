"""Constants for zone points functionality."""

from PyQt5.QtGui import QColor


class ZonePointsConstants:
    """Constants for zone points functionality."""

    # UI Configuration
    MIN_POINTS = 2
    MAX_POINTS = 1000
    DEFAULT_POINTS = 10
    DEFAULT_SHRINK_PERCENT = 5

    # Point appearance
    POINT_RADIUS = 4
    POINT_DIAMETER = 8
    ZONE_ALPHA = 0.2

    # Sampling
    MAX_CANDIDATES = 10000

    # Coordinate defaults
    DEFAULT_REAL_X = 9.25
    DEFAULT_REAL_Y = -6.6
    COORDINATE_RANGE = (-1000.0, 1000.0)
    COORDINATE_DECIMALS = 2

    # Table columns (no Measurement column)
    TABLE_COLUMNS = ["ID", "X (px)", "Y (px)", "X (mm)", "Y (mm)"]

    # Colors
    POINT_COLOR_GENERATED = "red"
    POINT_COLOR_USER = "blue"
    POINT_COLOR_SELECTED = QColor(255, 255, 0)  # yellow
    ZONE_COLOR = "cyan"
