import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np


def _load_module():
    path = Path("src/xrdanalysis/data_processing/difra_archive_to_h5.py")
    spec = importlib.util.spec_from_file_location("difra_archive_to_h5", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_difra_session(path: Path, *, project: str, stage: str, sample_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5f:
        h5f.attrs["container_type"] = "session"
        h5f.attrs["project_id"] = project
        h5f.attrs["study_name"] = stage
        h5f.attrs["operator_id"] = "jennifer_nicell"
        h5f.attrs["sample_id"] = sample_id
        h5f.attrs["specimenId"] = sample_id
        h5f.attrs["container_id"] = path.stem.split("_")[1]
        h5f.attrs["machine_name"] = "Ulster (Xena)"
        h5f.attrs["distance_cm"] = 2.0
        h5f.attrs["acquisition_date"] = "2026-04-30"
        technical = h5f.require_group("/entry/technical")
        technical.attrs["source_container_id"] = "tech_001"
        poni = h5f.require_group("/entry/technical/poni")
        for alias in ("primary", "secondary"):
            ds = poni.create_dataset(f"poni_{alias}", data="Distance: 0.02\nPoni1: 0\n")
            ds.attrs["detector_alias"] = alias.upper()
        event = h5f.require_group("/entry/technical/tech_evt_000001")
        event.attrs["type"] = "AGBH"
        event.attrs["timestamp"] = "2026-04-30 12:00:00"
        for det_name in ("det_primary", "det_secondary"):
            det = event.require_group(det_name)
            det.attrs["integration_time_ms"] = 300000.0
            det.attrs["n_frames"] = 1
            det.create_dataset("processed_signal", data=np.ones((4, 4)))
            blob = det.require_group("blob")
            blob.create_dataset("raw_dsc", data=np.frombuffer(b"meta", dtype=np.uint8))

        point = h5f.require_group("/entry/points/pt_001")
        point.attrs["physical_coordinates_mm"] = np.array([1.2, -3.4])
        meas = h5f.require_group("/entry/measurements/pt_001/meas_000000001")
        meas.attrs["timestamp_start"] = "2026-04-30 13:00:00"
        for det_name in ("det_primary", "det_secondary"):
            det = meas.require_group(det_name)
            det.attrs["detector_id"] = det_name
            det.attrs["integration_time_ms"] = 60000.0
            det.create_dataset("processed_signal", data=np.full((4, 4), 2.0))
            blob = det.require_group("blob")
            blob.create_dataset("raw_dsc", data=np.frombuffer(b"meta", dtype=np.uint8))


def test_build_xrd_h5_by_project_stage(tmp_path):
    mod = _load_module()
    archive = tmp_path / "archive"
    _write_difra_session(
        archive / "a" / "session_aaa_SAMPLE_A_20260430.nxs.h5",
        project="Project A",
        stage="Stage 1",
        sample_id="123__456_A",
    )
    _write_difra_session(
        archive / "b" / "session_bbb_SAMPLE_B_20260430.nxs.h5",
        project="Project B",
        stage="Stage 2",
        sample_id="124__457_B",
    )

    results = mod.build_xrd_h5_by_project_stage(archive, tmp_path / "out")

    assert len(results) == 2
    assert {r.project for r in results} == {"Project A", "Project B"}
    assert {r.measurement_datasets for r in results} == {2}
    assert {r.calibration_datasets for r in results} == {2}

    output = next(r.output_path for r in results if r.project == "Project A")
    with h5py.File(output, "r") as h5f:
        group = next(iter(h5f.values()))
        assert "calibrations" in group
        assert "measurements_0" in group
        measurements = group["measurements_0"]
        assert len(measurements) == 2
        row = next(iter(measurements.values()))
        assert row.shape == (4, 4)
        assert row.attrs["detector_alias"] in {"PRIMARY", "SECONDARY"}
        assert measurements.attrs["project_id"] == "Project A"
        assert measurements.attrs["stage"] == "Stage 1"
