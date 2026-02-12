"""Legacy technical_container module (deprecated).

Use hardware.container.v0_1.technical_container instead. This module remains as a
thin compatibility layer to avoid breaking older imports.
"""

from hardware.container.v0_1.technical_container import (
    create_technical_container,
    write_detector_config,
    write_pony_datasets,
    add_technical_event,
    link_pony_to_event,
    generate_from_aux_table,
)

__all__ = [
    "create_technical_container",
    "write_detector_config",
    "write_pony_datasets",
    "add_technical_event",
    "link_pony_to_event",
    "generate_from_aux_table",
]
