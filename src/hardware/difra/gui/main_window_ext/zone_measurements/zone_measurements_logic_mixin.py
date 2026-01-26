from hardware.difra.gui.main_window_ext.zone_measurements.logic.file_mixin import (
    ZoneMeasurementsFileMixin,
)
from hardware.difra.gui.main_window_ext.zone_measurements.logic.process_mixin import (
    ZoneMeasurementsProcessMixin,
)
from hardware.difra.gui.main_window_ext.zone_measurements.logic.stage_control_mixin import (
    StageControlMixin,
)
from hardware.difra.gui.main_window_ext.zone_measurements.logic.ui_mixin import (
    ZoneMeasurementsUIMixin,
)
from hardware.difra.gui.main_window_ext.zone_measurements.logic.utils import (
    ZoneMeasurementsUtilsMixin,
)


class ZoneMeasurementsLogicMixin(
    ZoneMeasurementsUIMixin,
    StageControlMixin,
    ZoneMeasurementsProcessMixin,
    ZoneMeasurementsFileMixin,
    ZoneMeasurementsUtilsMixin,
):
    pass
