"""Build h5_to_df-compatible H5 files from Difra session containers."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class DifraArchiveBuildResult:
    """Summary for one generated project/stage H5 file."""

    output_path: Path
    project: str
    stage: str
    sessions: int
    technical_groups: int
    calibration_datasets: int
    measurement_datasets: int
    skipped_sessions: int


def build_xrd_h5_by_project_stage(
    folder: str | Path,
    output_dir: str | Path,
    *,
    project: Optional[str] = None,
    stage: Optional[str] = None,
    operator: Optional[str] = None,
    overwrite: bool = True,
) -> List[DifraArchiveBuildResult]:
    """
    Find Difra session containers under ``folder`` and build one XRD-analysis
    H5 file per project/stage.

    The generated files use the structure consumed by ``h5_to_df``:
    ``<group>/calibrations`` and ``<group>/measurements_N``.

    In current Difra metadata, project is read from ``project_id`` or
    ``matadorProjectName``. Stage is read from ``study_name``.
    """
    folder = Path(folder)
    output_dir = Path(output_dir)
    sessions = list(
        find_difra_session_containers(
            folder,
            project=project,
            stage=stage,
            operator=operator,
        )
    )
    grouped: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for path in sessions:
        with h5py.File(path, "r") as h5f:
            grouped[_project_stage(h5f)].append(path)

    results: List[DifraArchiveBuildResult] = []
    for (project_name, stage_name), paths in sorted(grouped.items()):
        output_path = output_dir / (
            f"AUTO_PROJECT_{_safe_token(project_name)}__"
            f"STAGE_{_safe_token(stage_name)}__from_difra_archive.h5"
        )
        results.append(
            build_xrd_h5_for_project_stage(
                paths,
                output_path,
                project=project_name,
                stage=stage_name,
                overwrite=overwrite,
            )
        )
    return results


def build_xrd_h5_for_project_stage(
    session_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    project: Optional[str] = None,
    stage: Optional[str] = None,
    overwrite: bool = True,
) -> DifraArchiveBuildResult:
    """Build one h5_to_df-compatible H5 from selected Difra session files."""
    paths = [Path(path) for path in session_paths]
    if not paths:
        raise ValueError("No Difra session containers provided")

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    by_technical: Dict[Tuple[str, str, int, str], List[Path]] = defaultdict(list)
    first_project = project or ""
    first_stage = stage or ""
    for path in paths:
        with h5py.File(path, "r") as h5f:
            project_name, stage_name = _project_stage(h5f)
            first_project = first_project or project_name
            first_stage = first_stage or stage_name
            by_technical[_technical_key(h5f)].append(path)

    calibration_count = 0
    measurement_count = 0
    skipped_sessions = 0
    measurement_index = 0
    next_measurement_id = 1

    with h5py.File(output_path, "w") as dst:
        _write_attrs(
            dst,
            {
                "project": first_project,
                "stage": first_stage,
                "study_name": first_stage,
                "source": "Difra archived session containers",
                "format": "xrdanalysis_h5_to_df_compatible",
            },
        )
        for key, technical_paths in sorted(by_technical.items()):
            date, machine, distance_mm, technical_id = key
            group = dst.require_group(
                _safe_token(
                    f"{date}_{machine}_{distance_mm}_calibration_{technical_id}",
                    "calibration_group",
                )
            )
            group.attrs["source_technical_id"] = technical_id
            group.attrs["source_session_count"] = len(technical_paths)
            calibration_count += _copy_calibrations(technical_paths[0], group)

            for source_path in sorted(technical_paths):
                written, next_measurement_id = _copy_measurement_session(
                    source_path,
                    group,
                    measurement_index,
                    next_measurement_id,
                )
                if written:
                    measurement_count += written
                    measurement_index += 1
                else:
                    skipped_sessions += 1
                    _delete_if_exists(group, f"measurements_{measurement_index}")

    return DifraArchiveBuildResult(
        output_path=output_path,
        project=first_project,
        stage=first_stage,
        sessions=len(paths),
        technical_groups=len(by_technical),
        calibration_datasets=calibration_count,
        measurement_datasets=measurement_count,
        skipped_sessions=skipped_sessions,
    )


def find_difra_session_containers(
    folder: str | Path,
    *,
    project: Optional[str] = None,
    stage: Optional[str] = None,
    operator: Optional[str] = None,
) -> Iterable[Path]:
    """Yield Difra ``session_*.nxs.h5`` containers matching optional filters."""
    root = Path(folder)
    for path in sorted(root.rglob("session_*.nxs.h5")):
        try:
            with h5py.File(path, "r") as h5f:
                if _as_text(h5f.attrs.get("container_type")) != "session":
                    continue
                project_name, stage_name = _project_stage(h5f)
                operator_name = _as_text(h5f.attrs.get("operator_id"))
        except Exception:
            continue
        if project is not None and project_name != project:
            continue
        if stage is not None and stage_name != stage:
            continue
        if operator is not None and operator_name != operator:
            continue
        yield path


def _copy_calibrations(source_path: Path, group: h5py.Group) -> int:
    written = 0
    with h5py.File(source_path, "r") as src:
        distance_cm = src.attrs.get("distance_cm")
        distance_token = _distance_token(distance_cm)
        technical_id = _technical_key(src)[3]
        poni_by_alias = _poni_map(src)
        calibration_group = group.require_group("calibrations")
        _write_attrs(
            calibration_group,
            {
                "calib_unique_id": technical_id,
                "distanceInMM": _distance_mm(distance_cm),
                "machineName": _machine_token(src.attrs.get("machine_name")),
                "matrixResolution": "M256X256",
                "name": group.name.rsplit("/", 1)[-1],
                "pixelSize": 55,
                "source": "Difra archive",
            },
        )

        technical = src.get("/entry/technical")
        if not isinstance(technical, h5py.Group):
            return 0

        type_order = {"DARK": "001", "EMPTY": "002", "AGBH": "003", "BACKGROUND": "004"}
        type_prefix = {
            "DARK": "DC",
            "EMPTY": "Empty",
            "AGBH": "AgBH",
            "BACKGROUND": "Bg",
        }
        for event_name in sorted(technical.keys()):
            event = technical.get(event_name)
            if not isinstance(event, h5py.Group) or not event_name.startswith(
                "tech_evt_"
            ):
                continue
            event_type = _as_text(event.attrs.get("type"), "TECH").upper()
            day, clock = _datetime_tokens(
                event.attrs.get("timestamp") or src.attrs.get("creation_timestamp")
            )
            for det_name in sorted(event.keys()):
                det = event.get(det_name)
                if not isinstance(det, h5py.Group) or not det_name.startswith("det_"):
                    continue
                signal = det.get("processed_signal")
                if not isinstance(signal, h5py.Dataset):
                    continue
                alias = _detector_alias(det_name)
                exposure_s = float(det.attrs.get("integration_time_ms", 0.0)) / 1000.0
                frames = int(det.attrs.get("n_frames", 1) or 1)
                dataset_name = _safe_token(
                    f"{type_prefix.get(event_type, event_type)}_{distance_token}_"
                    f"{type_order.get(event_type, '999')}_{day}_{clock}_"
                    f"{exposure_s:.6f}s_{frames}frames_{alias}",
                    "calibration",
                )
                if dataset_name in calibration_group:
                    continue
                ds = calibration_group.create_dataset(
                    dataset_name,
                    data=signal[...],
                    compression="lzf",
                    shuffle=True,
                )
                _write_attrs(
                    ds,
                    {
                        "calibrationType": event_type,
                        "detectorType": alias,
                        "detector_alias": alias,
                        "exposure": exposure_s,
                        "measurement_timestamp": _as_text(event.attrs.get("timestamp")),
                        "metadata": _raw_dsc_text(det),
                        "ponifile": poni_by_alias.get(alias, ""),
                        "unique_id": f"{technical_id}_{event_name}_{alias}",
                    },
                )
                written += 1
    return written


def _copy_measurement_session(
    source_path: Path,
    group: h5py.Group,
    measurement_index: int,
    measurement_id_start: int,
) -> Tuple[int, int]:
    written = 0
    measurement_id = measurement_id_start
    with h5py.File(source_path, "r") as src:
        sample_id = _as_text(src.attrs.get("sample_id") or src.attrs.get("specimenId"))
        project, stage = _project_stage(src)
        distance_cm = src.attrs.get("distance_cm")
        distance_token = _distance_token(distance_cm)
        poni_by_alias = _poni_map(src)
        measurement_group = group.require_group(f"measurements_{measurement_index}")
        _write_attrs(
            measurement_group,
            {
                "sample_id": sample_id,
                "specimenId": _as_text(src.attrs.get("specimenId") or sample_id),
                "patientDBId": _leading_int(sample_id),
                "measurementsGroupId": _second_int(sample_id),
                "project_id": project,
                "stage": stage,
                "study_name": stage,
                "operator_id": _as_text(src.attrs.get("operator_id")),
                "machineName": _machine_token(src.attrs.get("machine_name")),
                "distanceInMM": _distance_mm(distance_cm),
                "acquisition_date": _as_text(src.attrs.get("acquisition_date")),
                "container_id": _as_text(src.attrs.get("container_id")),
                "source_container_path": str(source_path),
            },
        )

        measurements = src.get("/entry/measurements")
        points = src.get("/entry/points")
        if not isinstance(measurements, h5py.Group):
            return written, measurement_id

        for point_name in sorted(measurements.keys()):
            point_group = measurements.get(point_name)
            if not isinstance(point_group, h5py.Group):
                continue
            x, y = _point_xy(
                points.get(point_name) if isinstance(points, h5py.Group) else None
            )
            for meas_name in sorted(point_group.keys()):
                meas = point_group.get(meas_name)
                if not isinstance(meas, h5py.Group) or not meas_name.startswith(
                    "meas_"
                ):
                    continue
                timestamp = meas.attrs.get("timestamp_start") or meas.attrs.get(
                    "timestamp_end"
                )
                day, clock = _datetime_tokens(timestamp)
                for det_name in sorted(meas.keys()):
                    det = meas.get(det_name)
                    if not isinstance(det, h5py.Group) or not det_name.startswith(
                        "det_"
                    ):
                        continue
                    signal = det.get("processed_signal")
                    if not isinstance(signal, h5py.Dataset):
                        continue
                    alias = _detector_alias(det_name)
                    exposure_s = (
                        float(det.attrs.get("integration_time_ms", 0.0)) / 1000.0
                    )
                    dataset_name = _unique_dataset_name(
                        measurement_group,
                        _safe_token(
                            f"{sample_id}_{distance_token}_{x:.2f}_{y:.2f}_"
                            f"{day}_{clock}_{alias}",
                            "measurement",
                        ),
                    )
                    ds = measurement_group.create_dataset(
                        dataset_name,
                        data=signal[...],
                        compression="lzf",
                        shuffle=True,
                    )
                    _write_attrs(
                        ds,
                        {
                            "detector_alias": alias,
                            "detector_id": _as_text(det.attrs.get("detector_id")),
                            "exposure": exposure_s,
                            "measurement_timestamp": _as_text(timestamp),
                            "metadata": _raw_dsc_text(det),
                            "ponifile": poni_by_alias.get(alias, ""),
                            "unique_id": (
                                f"{_as_text(src.attrs.get('container_id'))}_"
                                f"{point_name}_{meas_name}_{alias}"
                            ),
                            "measurement_id": measurement_id,
                            "x": x,
                            "y": y,
                            "point_ref": point_name,
                            "source_measurement_path": det.name,
                        },
                    )
                    written += 1
                    measurement_id += 1
    return written, measurement_id


def _project_stage(h5f: h5py.File) -> Tuple[str, str]:
    project = _as_text(
        h5f.attrs.get("project_id") or h5f.attrs.get("matadorProjectName")
    )
    stage = _as_text(h5f.attrs.get("study_name"))
    return project or "unknown_project", stage or "unknown_stage"


def _technical_key(h5f: h5py.File) -> Tuple[str, str, int, str]:
    technical = h5f.get("/entry/technical")
    technical_id = (
        _as_text(technical.attrs.get("source_container_id"))
        if isinstance(technical, h5py.Group)
        else "unknown"
    )
    date = _as_text(h5f.attrs.get("acquisition_date")).strip()[:10] or "unknown-date"
    return (
        date,
        _machine_token(h5f.attrs.get("machine_name")),
        _distance_mm(h5f.attrs.get("distance_cm")),
        technical_id or "unknown",
    )


def _poni_map(h5f: h5py.File) -> Dict[str, str]:
    result: Dict[str, str] = {}
    poni_root = h5f.get("/entry/technical/poni")
    if not isinstance(poni_root, h5py.Group):
        return result
    for name, node in poni_root.items():
        if not isinstance(node, h5py.Dataset):
            continue
        text = _as_text(node[()])
        alias = _as_text(node.attrs.get("detector_alias")).upper()
        if alias and text:
            result[alias] = text
            if alias.startswith("DET_"):
                result[alias[4:]] = text
        if name.lower() in {"poni_primary", "poni_secondary", "poni_saxs", "poni_waxs"}:
            result[name.split("_", 1)[1].upper()] = text
    return result


def _raw_dsc_text(detector_group: h5py.Group) -> str:
    blob = detector_group.get("blob")
    raw = blob.get("raw_dsc") if isinstance(blob, h5py.Group) else None
    if not isinstance(raw, h5py.Dataset):
        return ""
    data = raw[...]
    if isinstance(data, np.ndarray):
        data = data.astype(np.uint8).tobytes()
    return _as_text(data)


def _point_xy(point_group: Optional[h5py.Group]) -> Tuple[float, float]:
    if not isinstance(point_group, h5py.Group):
        return np.nan, np.nan
    coords = point_group.attrs.get("physical_coordinates_mm")
    if coords is None:
        return np.nan, np.nan
    arr = np.asarray(coords, dtype=float)
    x = float(arr[0]) if arr.size > 0 else np.nan
    y = float(arr[1]) if arr.size > 1 else np.nan
    return x, y


def _datetime_tokens(value: Any) -> Tuple[str, str]:
    text = _as_text(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y%m%d_%H%M%S"):
        try:
            dt = datetime.strptime(text[: len(fmt)], fmt)
            return dt.strftime("%Y%m%d"), dt.strftime("%H%M%S")
        except Exception:
            pass
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 14:
        return digits[:8], digits[8:14]
    if len(digits) >= 8:
        return digits[:8], digits[8:14].ljust(6, "0")
    return "unknown", "000000"


def _write_attrs(obj: h5py.Group | h5py.Dataset, attrs: Dict[str, Any]) -> None:
    for key, value in attrs.items():
        if value is None:
            continue
        try:
            obj.attrs[str(key)] = value
        except TypeError:
            obj.attrs[str(key)] = _as_text(value)


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _as_text(value.item(), default)
    return str(value)


def _safe_token(value: Any, fallback: str = "unknown") -> str:
    text = _as_text(value).strip()
    token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or fallback


def _distance_mm(distance_cm: Any) -> int:
    try:
        return int(round(float(distance_cm) * 10.0))
    except Exception:
        return 0


def _distance_token(distance_cm: Any) -> str:
    try:
        value = float(distance_cm)
    except Exception:
        return "unknowncm"
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}cm"
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", "p") + "cm"


def _machine_token(machine_name: Any) -> str:
    text = _as_text(machine_name).upper()
    if "XENA" in text:
        return "XENA"
    if "MOLI" in text:
        return "MOLI"
    return _safe_token(text, "MACHINE").upper()


def _detector_alias(det_name: str) -> str:
    token = str(det_name or "").upper()
    return token[4:] if token.startswith("DET_") else token


def _leading_int(value: Any) -> Optional[int]:
    match = re.match(r"^\s*(\d+)", _as_text(value))
    return int(match.group(1)) if match else None


def _second_int(value: Any) -> Optional[int]:
    match = re.match(r"^\s*\d+__([0-9]+)", _as_text(value))
    return int(match.group(1)) if match else None


def _unique_dataset_name(group: h5py.Group, name: str) -> str:
    if name not in group:
        return name
    suffix = 2
    while f"{name}_{suffix}" in group:
        suffix += 1
    return f"{name}_{suffix}"


def _delete_if_exists(group: h5py.Group, name: str) -> None:
    if name in group:
        del group[name]
