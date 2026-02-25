"""Measurement write operations extracted from SessionManager."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from hardware.difra.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class SessionManagerMeasurementOpsMixin:
    """Point/image/attenuation write operations for active sessions."""

    def replace_technical_container(
        self,
        technical_file: Path,
        auto_lock_source: bool = False,
    ) -> None:
        """Replace embedded calibration snapshot in an active unlocked session."""
        self._check_active()

        if self.is_locked():
            raise RuntimeError(
                "Cannot update technical data: session container is locked."
            )

        self.writer.copy_technical_to_session(
            technical_file=technical_file,
            session_file=self.session_path,
            auto_lock=auto_lock_source,
        )
        self.technical_container_path = Path(technical_file)

    def add_sample_image(
        self,
        image_data,
        image_index: int = 1,
        image_type: str = "sample",
    ) -> str:
        """Add sample image to session container."""
        self._check_active()

        return self.writer.add_image(
            file_path=self.session_path,
            image_index=image_index,
            image_data=image_data,
            image_type=image_type,
        )

    def add_zone(
        self,
        zone_index: int,
        geometry_px,
        shape: str,
        zone_role: str = "sample_holder",
        holder_diameter_mm: Optional[float] = None,
    ) -> str:
        """Add zone definition to session container."""
        self._check_active()

        return self.writer.add_zone(
            file_path=self.session_path,
            zone_index=zone_index,
            geometry_px=geometry_px,
            shape=shape,
            zone_role=zone_role,
            holder_diameter_mm=holder_diameter_mm,
        )

    def add_points(
        self,
        points: List[Dict],
    ) -> List[str]:
        """Add multiple measurement points to session container."""
        self._check_active()

        paths = []
        for idx, point in enumerate(points, start=1):
            path = self.writer.add_point(
                file_path=self.session_path,
                point_index=idx,
                pixel_coordinates=point["pixel_coordinates"],
                physical_coordinates_mm=point["physical_coordinates_mm"],
                point_status=point.get("point_status", "pending"),
                thickness=point.get("thickness", "unknown"),
            )
            point_uid = str(point.get("point_uid") or "").strip()
            if point_uid:
                try:
                    import h5py

                    with h5py.File(self.session_path, "a") as h5f:
                        h5f[path].attrs["point_uid"] = point_uid
                except Exception:
                    logger.warning(
                        "Failed to persist point UID into session point attrs",
                        point_index=int(idx),
                    )
            paths.append(path)

        self.log_event(
            message=f"Generated {len(points)} measurement points",
            event_type="points_generated",
            details={"count": len(points)},
        )

        logger.info("Added points to session", num_points=len(points))
        return paths

    def add_attenuation_measurement(
        self,
        measurement_data: Dict,
        detector_metadata: Dict,
        poni_alias_map: Dict,
        mode: str,  # "without" or "with"
    ) -> int:
        """Add attenuation measurement (I₀ or I) to session container."""
        self._check_active()

        ana_path = self.writer.add_analytical_measurement(
            file_path=self.session_path,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
            analysis_type="attenuation",
            analysis_role=(
                self.schema.ANALYSIS_ROLE_I0
                if mode == "without"
                else self.schema.ANALYSIS_ROLE_I
            ),
            timestamp_start=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        counter = int(ana_path.split("_")[-1])

        if mode == "without":
            self.i0_counter = counter
            logger.info("Added I₀ attenuation measurement", counter=counter)
            self.log_event(
                message="Attenuation I0 recorded",
                event_type="attenuation_i0_recorded",
                details={"counter": counter},
            )
        else:  # mode == "with"
            self.i_counter = counter
            logger.info("Added I attenuation measurement", counter=counter)
            self.log_event(
                message="Attenuation I recorded",
                event_type="attenuation_i_recorded",
                details={"counter": counter},
            )

        return counter

    def link_attenuation_to_points(
        self,
        num_points: int,
        start_point_idx: int = 1,
    ):
        """Link both I₀ and I attenuation measurements to all points."""
        self._check_active()
        if start_point_idx < 1:
            raise ValueError(
                f"start_point_idx must be >= 1, got {start_point_idx}"
            )

        if self.i0_counter is None or self.i_counter is None:
            raise RuntimeError(
                "Both I₀ and I measurements must be recorded before linking. "
                f"Current state: I₀={self.i0_counter}, I={self.i_counter}"
            )

        logger.info(
            "Linking attenuation to points",
            num_points=num_points,
            start_point_idx=start_point_idx,
            i0_counter=self.i0_counter,
            i_counter=self.i_counter,
        )

        end_point_idx = start_point_idx + num_points
        for point_idx in range(start_point_idx, end_point_idx):
            self.writer.link_analytical_measurement_to_point(
                file_path=self.session_path,
                point_index=point_idx,
                analytical_measurement_index=self.i0_counter,
            )

            self.writer.link_analytical_measurement_to_point(
                file_path=self.session_path,
                point_index=point_idx,
                analytical_measurement_index=self.i_counter,
            )

        logger.info("Attenuation linked to all points successfully")
        self.log_event(
            message="Linked attenuation analytical measurements to points",
            event_type="attenuation_linked",
            details={
                "start_point_idx": start_point_idx,
                "num_points": num_points,
                "i0_counter": self.i0_counter,
                "i_counter": self.i_counter,
            },
        )

    def begin_point_measurement(
        self,
        point_index: int,
        timestamp_start: Optional[str] = None,
    ) -> str:
        """Create an in-progress measurement record before detector capture starts."""
        self._check_active()

        existing = self._pending_measurements.get(point_index)
        if existing:
            return existing

        meas_path = self.writer.begin_measurement(
            file_path=self.session_path,
            point_index=point_index,
            timestamp_start=timestamp_start,
            measurement_status=self.schema.STATUS_IN_PROGRESS,
        )
        self._pending_measurements[point_index] = meas_path
        self.log_event(
            message="Point measurement started",
            event_type="measurement_started",
            details={
                "point_index": point_index,
                "measurement_path": meas_path,
            },
        )
        logger.info("Started point measurement", point_index=point_index, path=meas_path)
        return meas_path

    def complete_point_measurement(
        self,
        point_index: int,
        measurement_data: Dict,
        detector_metadata: Dict,
        poni_alias_map: Dict,
        raw_files: Optional[Dict] = None,
        timestamp_end: Optional[str] = None,
        measurement_status: str = None,
    ) -> str:
        """Finalize point measurement and write detector payload."""
        self._check_active()
        if measurement_status is None:
            measurement_status = self.schema.STATUS_COMPLETED

        meas_path = self._pending_measurements.pop(point_index, None)
        if meas_path:
            meas_path = self.writer.finalize_measurement(
                file_path=self.session_path,
                measurement_path=meas_path,
                measurement_data=measurement_data,
                detector_metadata=detector_metadata,
                poni_alias_map=poni_alias_map,
                raw_files=raw_files,
                timestamp_end=timestamp_end,
                measurement_status=measurement_status,
            )
        else:
            meas_path = self.writer.add_measurement(
                file_path=self.session_path,
                point_index=point_index,
                measurement_data=measurement_data,
                detector_metadata=detector_metadata,
                poni_alias_map=poni_alias_map,
                raw_files=raw_files,
                timestamp_end=timestamp_end,
                measurement_status=measurement_status,
            )

        if measurement_status == self.schema.STATUS_COMPLETED:
            self.writer.update_point_status(
                file_path=self.session_path,
                point_index=point_index,
                point_status="measured",
            )
        self.log_event(
            message="Point measurement finalized",
            event_type="measurement_finalized",
            details={
                "point_index": point_index,
                "measurement_path": meas_path,
                "status": measurement_status,
                "detector_count": len(measurement_data or {}),
            },
        )

        logger.info(
            "Completed point measurement",
            point_index=point_index,
            status=measurement_status,
            path=meas_path,
        )
        return meas_path

    def fail_point_measurement(
        self,
        point_index: int,
        reason: Optional[str] = None,
        timestamp_end: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Optional[str]:
        """Mark an in-progress point measurement as failed/aborted."""
        self._check_active()

        meas_path = self._pending_measurements.pop(point_index, None)
        if not meas_path:
            return None

        terminal_status = status or self.schema.STATUS_FAILED
        self.writer.fail_measurement(
            file_path=self.session_path,
            measurement_path=meas_path,
            failure_reason=reason,
            timestamp_end=timestamp_end,
            measurement_status=terminal_status,
        )
        self.log_event(
            message="Point measurement failed",
            event_type="measurement_failed",
            level="WARNING",
            details={
                "point_index": point_index,
                "measurement_path": meas_path,
                "status": terminal_status,
                "reason": reason or "",
            },
        )
        logger.warning(
            "Point measurement failed",
            point_index=point_index,
            status=terminal_status,
            reason=reason,
            path=meas_path,
        )
        return meas_path

    def mark_point_skipped(
        self,
        point_index: int,
        reason: Optional[str] = None,
    ) -> None:
        """Mark point as skipped and persist skip reason."""
        self._check_active()
        if self.is_locked():
            raise RuntimeError("Cannot mark point skipped: session container is locked.")

        skip_reason = str(reason or "").strip() or "user_skipped"

        # If there is an in-progress measurement for this point, terminate it first.
        pending_path = self._pending_measurements.pop(point_index, None)
        if pending_path:
            fail_measurement = getattr(self.writer, "fail_measurement", None)
            if callable(fail_measurement):
                fail_measurement(
                    file_path=self.session_path,
                    measurement_path=pending_path,
                    failure_reason=f"skipped:{skip_reason}",
                    timestamp_end=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    measurement_status=self.schema.STATUS_ABORTED,
                )

        try:
            self.writer.update_point_status(
                file_path=self.session_path,
                point_index=point_index,
                point_status=self.schema.POINT_STATUS_SKIPPED,
                skip_reason=skip_reason,
            )
        except TypeError:
            # Backward compatibility for writer versions without skip_reason support.
            self.writer.update_point_status(
                file_path=self.session_path,
                point_index=point_index,
                point_status=self.schema.POINT_STATUS_SKIPPED,
            )
        self.log_event(
            message="Point marked skipped",
            event_type="point_skipped",
            level="WARNING",
            details={"point_index": point_index, "reason": skip_reason},
        )
        logger.info(
            "Marked point as skipped",
            point_index=point_index,
            reason=skip_reason,
        )

    def delete_point(
        self,
        point_index: int,
    ) -> bool:
        """Delete an unmeasured point from session container."""
        self._check_active()
        if self.is_locked():
            raise RuntimeError("Cannot delete point: session container is locked.")

        import h5py

        point_id = self.schema.format_point_id(point_index)
        point_path = f"{self.schema.GROUP_POINTS}/{point_id}"
        measurements_point_path = f"{self.schema.GROUP_MEASUREMENTS}/{point_id}"

        with h5py.File(self.session_path, "a") as h5f:
            if point_path not in h5f:
                return False

            point_group = h5f[point_path]
            point_status = self._as_text(
                point_group.attrs.get(self.schema.ATTR_POINT_STATUS, "")
            ).strip().lower()
            measured_status = str(self.schema.POINT_STATUS_MEASURED).strip().lower()
            if point_status == measured_status:
                raise RuntimeError(
                    f"Point {point_index} is measured and cannot be deleted. Mark it skipped instead."
                )

            # Do not delete points that already contain finished measurements.
            if measurements_point_path in h5f and len(h5f[measurements_point_path].keys()) > 0:
                raise RuntimeError(
                    f"Point {point_index} has measurement records and cannot be deleted. Mark it skipped instead."
                )

            if measurements_point_path in h5f:
                del h5f[measurements_point_path]
            del h5f[point_path]

        self._pending_measurements.pop(point_index, None)
        self.log_event(
            message="Point deleted from session container",
            event_type="point_deleted",
            level="WARNING",
            details={"point_index": point_index},
        )
        logger.info("Deleted point from session container", point_index=point_index)
        return True

    def add_measurement(
        self,
        point_index: int,
        measurement_data: Dict,
        detector_metadata: Dict,
        poni_alias_map: Dict,
        raw_files: Optional[Dict] = None,
    ) -> str:
        """Backward-compatible wrapper: write completed point measurement."""
        return self.complete_point_measurement(
            point_index=point_index,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
            raw_files=raw_files,
        )
