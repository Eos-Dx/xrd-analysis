"""Session container API for NeXus/HDF5 container v0.2.

This module provides the session container functions as a dedicated API surface
and delegates implementation to writer.py.
"""

from .writer import (
    create_session_container,
    copy_technical_to_session,
    add_image,
    add_zone,
    add_image_mapping,
    append_runtime_log,
    add_point,
    update_point_status,
    get_next_measurement_counter,
    begin_measurement,
    finalize_measurement,
    fail_measurement,
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
    "append_runtime_log",
    "add_point",
    "update_point_status",
    "get_next_measurement_counter",
    "begin_measurement",
    "finalize_measurement",
    "fail_measurement",
    "add_measurement",
    "add_analytical_measurement",
    "link_analytical_measurement_to_point",
    "find_active_session_container",
]
