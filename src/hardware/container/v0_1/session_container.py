"""Session container API for HDF5 container v0.1.

This module provides the session container functions as a dedicated API surface
and delegates implementation to writer.py.
"""

from .writer import (
    create_session_container,
    copy_technical_to_session,
    add_image,
    add_zone,
    add_image_mapping,
    add_point,
    update_point_status,
    get_next_measurement_counter,
    add_measurement,
    add_analytical_measurement,
    link_analytical_measurement_to_point,
    find_active_session_container,
)

__all__ = [
    "create_session_container",
    "copy_technical_to_session",
    "add_image",
    "add_zone",
    "add_image_mapping",
    "add_point",
    "update_point_status",
    "get_next_measurement_counter",
    "add_measurement",
    "add_analytical_measurement",
    "link_analytical_measurement_to_point",
    "find_active_session_container",
]
