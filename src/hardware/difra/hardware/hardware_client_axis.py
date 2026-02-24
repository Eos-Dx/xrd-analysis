from __future__ import annotations

from typing import Any


def normalize_axis(axis: Any) -> str:
    if isinstance(axis, str):
        value = axis.strip().lower()
        if value in {"x", "axis_x", "axis:x", "axis=1", "1"}:
            return "x"
        if value in {"y", "axis_y", "axis:y", "axis=2", "2"}:
            return "y"
    if isinstance(axis, int):
        if axis == 1:
            return "x"
        if axis == 2:
            return "y"
    raise ValueError(f"Invalid axis '{axis}'. Use axis name x/y or axis number 1/2.")
