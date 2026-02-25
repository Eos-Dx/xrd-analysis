"""Tests for zone-point candidate geometry and shrink behavior."""

import math

from hardware.difra.gui.main_window_ext.points.zone_points_ui_builder import (
    ZonePointsGeometry,
)


class _Rect:
    def __init__(self, x: float, y: float, w: float, h: float):
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    def x(self) -> float:
        return self._x

    def y(self) -> float:
        return self._y

    def width(self) -> float:
        return self._w

    def height(self) -> float:
        return self._h


class _RectShape:
    def __init__(self, x: float, y: float, w: float, h: float):
        self._rect = _Rect(x, y, w, h)

    def boundingRect(self):
        return self._rect


class _CircleShape:
    def __init__(self, center_x: float, center_y: float, radius: float):
        self._center = (center_x, center_y)
        self._radius = radius

    def get_center(self):
        return self._center

    def get_radius(self):
        return self._radius


class _EllipseRect:
    def __init__(self, x: float, y: float, w: float, h: float):
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    def x(self):
        return self._x

    def y(self):
        return self._y

    def width(self):
        return self._w

    def height(self):
        return self._h


class FakeEllipseShape:
    def __init__(self, x: float, y: float, w: float, h: float):
        self._rect = _EllipseRect(x, y, w, h)

    def sceneBoundingRect(self):
        return self._rect


def test_rect_candidates_apply_shrink_offset():
    shape = _RectShape(0.0, 0.0, 100.0, 200.0)

    candidates, area, bounds = ZonePointsGeometry._get_rect_candidates(
        shape, shrink_factor=0.8
    )

    assert bounds == (10.0, 20.0, 90.0, 180.0)
    assert area == 80.0 * 160.0
    assert len(candidates) > 0
    assert all(10.0 <= x <= 90.0 and 20.0 <= y <= 180.0 for x, y in candidates)


def test_rect_candidates_shrink_zero_collapses_to_center():
    shape = _RectShape(5.0, 15.0, 60.0, 40.0)

    candidates, area, bounds = ZonePointsGeometry._get_rect_candidates(
        shape, shrink_factor=0.0
    )

    assert bounds == (35.0, 35.0, 35.0, 35.0)
    assert area == 0.0
    assert len(candidates) > 0
    assert all(x == 35.0 and y == 35.0 for x, y in candidates)


def test_circle_candidates_use_radial_shrink_for_resizable_circle_shape():
    shape = _CircleShape(100.0, 200.0, 50.0)
    shrink_factor = 0.8

    candidates, area, bounds = ZonePointsGeometry.get_shape_bounds_and_candidates(
        shape, shrink_factor=shrink_factor
    )

    expected_radius = 50.0 * shrink_factor
    assert bounds == (
        100.0 - expected_radius,
        200.0 - expected_radius,
        100.0 + expected_radius,
        200.0 + expected_radius,
    )
    assert math.isclose(area, math.pi * (expected_radius**2), rel_tol=1e-12)
    assert len(candidates) > 0
    assert all(
        math.hypot(x - 100.0, y - 200.0) <= expected_radius + 1e-9
        for x, y in candidates
    )


def test_ellipse_candidates_apply_shrink_to_both_radii():
    shape = FakeEllipseShape(10.0, 20.0, 100.0, 60.0)
    shrink_factor = 0.8

    candidates, area, bounds = ZonePointsGeometry.get_shape_bounds_and_candidates(
        shape, shrink_factor=shrink_factor
    )

    cx = 10.0 + 50.0
    cy = 20.0 + 30.0
    rx = 50.0 * shrink_factor
    ry = 30.0 * shrink_factor

    assert bounds == (cx - rx, cy - ry, cx + rx, cy + ry)
    assert math.isclose(area, math.pi * rx * ry, rel_tol=1e-12)
    assert len(candidates) > 0
    assert all((((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2) <= 1.0 + 1e-9 for x, y in candidates)
