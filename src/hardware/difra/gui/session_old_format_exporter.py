"""Export session containers into legacy folder layout used by older DIFRA flows."""

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np


@dataclass
class OldFormatExportSummary:
    """Summary for one legacy export operation."""

    export_dir: Path
    state_path: Path
    raw_file_count: int
    technical_file_count: int


class SessionOldFormatExporter:
    """Create old-style folder structure from a session container."""

    @staticmethod
    def _as_text(value: Any, default: str = "") -> str:
        if value is None:
            return default
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _safe_token(value: str, fallback: str = "unknown") -> str:
        token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (value or ""))
        token = token.strip("_")
        return token or fallback

    @staticmethod
    def _read_blob_bytes(dataset) -> bytes:
        payload = dataset[()]
        if isinstance(payload, bytes):
            return payload
        array = np.asarray(payload)
        return array.tobytes()

    @staticmethod
    def _unique_path(folder: Path, filename: str) -> Path:
        folder = Path(folder)
        candidate = folder / filename
        if not candidate.exists():
            return candidate

        stem = candidate.stem
        suffix = candidate.suffix
        idx = 2
        while True:
            alt = folder / f"{stem}_{idx}{suffix}"
            if not alt.exists():
                return alt
            idx += 1

    @classmethod
    def resolve_old_format_root(
        cls,
        *,
        config: Optional[Dict[str, Any]] = None,
        archive_folder: Optional[Path] = None,
    ) -> Path:
        cfg = config or {}
        configured = cfg.get("old_format_export_folder") or cfg.get(
            "legacy_export_folder"
        )
        if configured:
            return Path(configured)

        difra_base = cfg.get("difra_base_folder")
        if difra_base:
            return Path(difra_base) / "Old_format"

        if archive_folder is not None:
            af = Path(archive_folder)
            if af.name == "measurements" and af.parent.name == "archive":
                return af.parent.parent / "Old_format"
            return af.parent / "Old_format"

        return Path("/Data/difra/Old_format")

    @classmethod
    def _build_export_dir(
        cls,
        *,
        root: Path,
        session_id: str,
        operator_id: str,
        sample_id: str,
        study_name: str,
    ) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        date_token, time_token = timestamp.split("_", 1)
        name = (
            f"{cls._safe_token(session_id, 'session')}_"
            f"{cls._safe_token(operator_id, 'operator')}_"
            f"{cls._safe_token(sample_id, 'sample')}_"
            f"{cls._safe_token(study_name, 'study')}_"
            f"{date_token}_{time_token}"
        )
        export_dir = root / name
        idx = 2
        while export_dir.exists():
            export_dir = root / f"{name}_{idx}"
            idx += 1
        export_dir.mkdir(parents=True, exist_ok=False)
        return export_dir

    @classmethod
    def _extract_point_coordinates(
        cls,
        h5f: h5py.File,
        point_name: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        points_group = h5f.get("/entry/points")
        if points_group is None or point_name not in points_group:
            return None, None
        point_group = points_group[point_name]
        coords = point_group.attrs.get("physical_coordinates_mm")
        if coords is None:
            return None, None
        arr = np.asarray(coords).flatten().tolist()
        if len(arr) < 2:
            return None, None
        try:
            return float(arr[0]), float(arr[1])
        except Exception:
            return None, None

    @classmethod
    def _build_measurement_points(cls, h5f: h5py.File) -> list:
        points = []
        points_group = h5f.get("/entry/points")
        if points_group is None:
            return points

        for point_name in sorted(points_group.keys()):
            if not str(point_name).startswith("pt_"):
                continue
            x_mm, y_mm = cls._extract_point_coordinates(h5f, point_name)
            try:
                point_index = int(str(point_name).split("_")[-1])
            except Exception:
                point_index = len(points) + 1
            points.append(
                {
                    "point_index": point_index,
                    "unique_id": str(point_name),
                    "x": x_mm,
                    "y": y_mm,
                }
            )
        return points

    @classmethod
    def _export_measurement_raw(
        cls,
        *,
        h5f: h5py.File,
        export_dir: Path,
        sample_id: str,
    ) -> Tuple[int, Dict[str, Dict[str, Any]]]:
        measurements_group = h5f.get("/entry/measurements")
        if measurements_group is None:
            return 0, {}

        exported = 0
        measurements_meta: Dict[str, Dict[str, Any]] = {}
        sample_token = cls._safe_token(sample_id, "sample")

        for point_name in sorted(measurements_group.keys()):
            point_group = measurements_group[point_name]
            x_mm, y_mm = cls._extract_point_coordinates(h5f, point_name)

            for meas_name in sorted(point_group.keys()):
                measurement_group = point_group[meas_name]
                for det_name in sorted(measurement_group.keys()):
                    if not str(det_name).startswith("det_"):
                        continue
                    det_group = measurement_group[det_name]
                    if "processed_signal" not in det_group:
                        continue

                    detector_alias = cls._as_text(
                        det_group.attrs.get("detector_alias"),
                        str(det_name).replace("det_", "").upper(),
                    )
                    detector_id = cls._as_text(
                        det_group.attrs.get("detector_id"),
                        detector_alias,
                    )
                    alias_token = cls._safe_token(detector_alias, "detector")
                    base = f"{sample_token}_{point_name}_{meas_name}_{alias_token}"

                    npy_path = cls._unique_path(export_dir, f"{base}.npy")
                    np.save(
                        npy_path,
                        np.asarray(det_group["processed_signal"][()]),
                    )
                    exported += 1

                    integration_ms = det_group.attrs.get("integration_time_ms")
                    integration_time_s = None
                    if integration_ms is not None:
                        try:
                            integration_time_s = float(integration_ms) / 1000.0
                        except Exception:
                            integration_time_s = None

                    measurements_meta[npy_path.name] = {
                        "x": x_mm,
                        "y": y_mm,
                        "unique_id": str(point_name),
                        "base_file": sample_id,
                        "integration_time": integration_time_s,
                        "detector_alias": detector_alias,
                        "detector_id": detector_id,
                    }

                    blob_group = det_group.get("blob")
                    if blob_group is None:
                        continue

                    for blob_name in sorted(blob_group.keys()):
                        if not str(blob_name).startswith("raw_"):
                            continue
                        extension = str(blob_name)[4:] or "bin"
                        raw_path = cls._unique_path(export_dir, f"{base}.{extension}")
                        raw_path.write_bytes(cls._read_blob_bytes(blob_group[blob_name]))
                        exported += 1

        return exported, measurements_meta

    @classmethod
    def _export_technical_measurements(
        cls,
        *,
        h5f: h5py.File,
        tech_dir: Path,
    ) -> int:
        exported = 0
        tech_rows = []
        by_type_alias: Dict[str, Dict[str, str]] = {}

        technical_group = h5f.get("/entry/technical")
        if technical_group is None:
            return 0

        poni_group = technical_group.get("poni")
        if poni_group is not None:
            for poni_name in sorted(poni_group.keys()):
                poni_ds = poni_group[poni_name]
                detector_alias = cls._as_text(
                    poni_ds.attrs.get("detector_alias"),
                    str(poni_name).replace("poni_", "").upper(),
                )
                filename = f"{cls._safe_token(detector_alias, 'detector').lower()}.poni"
                poni_path = cls._unique_path(tech_dir, filename)
                value = poni_ds[()]
                if isinstance(value, bytes):
                    text = value.decode("utf-8", errors="replace")
                else:
                    text = str(value)
                poni_path.write_text(text, encoding="utf-8")
                exported += 1

        for event_name in sorted(technical_group.keys()):
            if not str(event_name).startswith("tech_evt_"):
                continue
            event_group = technical_group[event_name]
            event_type = cls._as_text(event_group.attrs.get("type"), "UNKNOWN").upper()
            event_timestamp = cls._as_text(event_group.attrs.get("timestamp"), "")
            is_primary = bool(event_group.attrs.get("is_primary", False))

            for det_name in sorted(event_group.keys()):
                if not str(det_name).startswith("det_"):
                    continue
                det_group = event_group[det_name]
                if "processed_signal" not in det_group:
                    continue

                detector_alias = cls._as_text(
                    det_group.attrs.get("detector_alias"),
                    str(det_name).replace("det_", "").upper(),
                )
                alias_token = cls._safe_token(detector_alias, "detector")
                base = f"{event_type}_{event_name}_{alias_token}"
                npy_path = cls._unique_path(tech_dir, f"{base}.npy")
                np.save(npy_path, np.asarray(det_group["processed_signal"][()]))
                exported += 1

                row = {
                    "event_id": str(event_name),
                    "type": event_type,
                    "alias": detector_alias,
                    "timestamp": event_timestamp,
                    "is_primary": bool(is_primary),
                    "file": npy_path.name,
                }
                tech_rows.append(row)

                existing = by_type_alias.setdefault(event_type, {})
                if detector_alias not in existing or is_primary:
                    existing[detector_alias] = npy_path.name

                blob_group = det_group.get("blob")
                if blob_group is None:
                    continue

                for blob_name in sorted(blob_group.keys()):
                    if not str(blob_name).startswith("raw_"):
                        continue
                    extension = str(blob_name)[4:] or "bin"
                    raw_path = cls._unique_path(tech_dir, f"{base}.{extension}")
                    raw_path.write_bytes(cls._read_blob_bytes(blob_group[blob_name]))
                    exported += 1

        meta_payload = {
            "generated_from_session_container": True,
            "generated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mapping": by_type_alias,
            "events": tech_rows,
        }
        (tech_dir / "technical_meta_legacy.json").write_text(
            json.dumps(meta_payload, indent=2),
            encoding="utf-8",
        )
        exported += 1
        return exported

    @classmethod
    def export_from_session_container(
        cls,
        session_path: Path,
        *,
        config: Optional[Dict[str, Any]] = None,
        archive_folder: Optional[Path] = None,
        target_root: Optional[Path] = None,
    ) -> OldFormatExportSummary:
        """Export legacy raw/state/technical folders from a session container."""
        source = Path(session_path)
        if not source.exists():
            raise FileNotFoundError(f"Session container not found: {source}")

        with h5py.File(source, "r") as h5f:
            sample_id = cls._as_text(h5f.attrs.get("sample_id"), "UNKNOWN")
            study_name = cls._as_text(h5f.attrs.get("study_name"), "UNSPECIFIED")
            session_id = cls._as_text(h5f.attrs.get("session_id"), source.stem)
            operator_id = cls._as_text(
                h5f.attrs.get("operator_id") or h5f.attrs.get("locked_by"),
                "unknown",
            )

            root = (
                Path(target_root)
                if target_root is not None
                else cls.resolve_old_format_root(
                    config=config,
                    archive_folder=archive_folder,
                )
            )
            root.mkdir(parents=True, exist_ok=True)
            export_dir = cls._build_export_dir(
                root=root,
                session_id=session_id,
                operator_id=operator_id,
                sample_id=sample_id,
                study_name=study_name,
            )

            raw_count, measurements_meta = cls._export_measurement_raw(
                h5f=h5f,
                export_dir=export_dir,
                sample_id=sample_id,
            )

            tech_dir = export_dir / "technical_measurements"
            tech_dir.mkdir(parents=True, exist_ok=True)
            technical_count = cls._export_technical_measurements(
                h5f=h5f,
                tech_dir=tech_dir,
            )

            state_payload: Dict[str, Any] = {}
            embedded_state = h5f.attrs.get("meta_json")
            if embedded_state is not None:
                try:
                    parsed = json.loads(cls._as_text(embedded_state, "{}"))
                    if isinstance(parsed, dict):
                        state_payload = parsed
                except Exception:
                    state_payload = {}

            if not isinstance(state_payload, dict):
                state_payload = {}

            state_payload.setdefault("sample_id", sample_id)
            state_payload.setdefault("study_name", study_name)
            state_payload.setdefault("session_id", session_id)
            state_payload.setdefault("operator_id", operator_id)
            state_payload["measurement_points"] = cls._build_measurement_points(h5f)
            state_payload["measurements_meta"] = measurements_meta
            state_payload["technical_measurements_folder"] = "technical_measurements"

            state_path = export_dir / f"{cls._safe_token(sample_id, 'sample')}_state.json"
            state_path.write_text(
                json.dumps(state_payload, indent=2),
                encoding="utf-8",
            )

        return OldFormatExportSummary(
            export_dir=export_dir,
            state_path=state_path,
            raw_file_count=raw_count,
            technical_file_count=technical_count,
        )
