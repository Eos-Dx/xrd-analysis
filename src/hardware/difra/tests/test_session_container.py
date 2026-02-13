"""Tests for session HDF5 container generation and management.

Tests the complete workflow:
1. Create session container with metadata
2. Copy technical data from technical container
3. Add images and zones
4. Add points and measurements
5. Link analytical measurements
6. Validate HDF5 structure and content
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

# Add the project src root to the path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.container.v0_1 import schema, session_container, technical_container
from hardware.difra.hardware.detectors import DummyDetectorController


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def demo_config():
    """Configuration for DEMO mode."""
    return {
        "DEV": True,
        "detectors": [
            {
                "id": "PRIMARY",
                "alias": "PRIMARY",
                "name": "SAXS Demo",
                "type": "dummy",
                "width": 256,
                "height": 256,
                "size": (256, 256),
                "pixel_size_um": 75.0,
                "faulty_pixels": None,
            },
            {
                "id": "SECONDARY",
                "alias": "SECONDARY",
                "name": "WAXS Demo",
                "type": "dummy",
                "width": 256,
                "height": 256,
                "size": (256, 256),
                "pixel_size_um": 75.0,
                "faulty_pixels": None,
            },
        ],
        "dev_active_detectors": ["PRIMARY", "SECONDARY"],
        "active_detectors": ["PRIMARY", "SECONDARY"],
    }


@pytest.fixture
def demo_detectors(demo_config):
    """Create DEMO detector controllers."""
    detectors = {}
    for det_config in demo_config["detectors"]:
        det_id = det_config["id"]
        if det_id in demo_config["dev_active_detectors"]:
            controller = DummyDetectorController(
                alias=det_id, size=det_config["size"]
            )
            detectors[det_id] = controller
    return detectors


@pytest.fixture
def demo_poni_files(temp_output_dir):
    """Create demo PONI files."""
    poni_dir = temp_output_dir / "poni"
    poni_dir.mkdir()

    poni_content_template = """# Detector: {detector}
PixelSize1: 7.500000e-05
PixelSize2: 7.500000e-05
Distance: {distance}
Poni1: 0.012345
Poni2: 0.023456
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 1.54e-10
"""

    poni_files = {}
    for detector, distance in [("PRIMARY", 0.17), ("SECONDARY", 0.02)]:
        poni_path = poni_dir / f"{detector.lower()}_demo.poni"
        content = poni_content_template.format(detector=detector, distance=distance)
        poni_path.write_text(content)
        poni_files[detector] = poni_path

    return poni_files


@pytest.fixture
def technical_container_file(temp_output_dir, demo_config, demo_poni_files):
    """Create a technical container for testing."""
    # Create synthetic technical measurements
    aux_measurements = {}
    for tech_type in ["DARK", "EMPTY", "BACKGROUND", "AGBH"]:
        aux_measurements[tech_type] = {}
        for det_id in ["PRIMARY", "SECONDARY"]:
            filename = temp_output_dir / f"{tech_type}_{det_id}.npy"
            data = np.random.poisson(20, size=(256, 256)).astype(np.float32)
            np.save(filename, data)
            aux_measurements[tech_type][det_id] = str(filename)

    # Create PONI data dict
    poni_data = {}
    for alias, poni_path in demo_poni_files.items():
        content = poni_path.read_text()
        poni_data[alias] = (content, poni_path.name)

    # Generate technical container with per-detector distances matching PONI files
    container_id, file_path = technical_container.generate_from_aux_table(
        folder=temp_output_dir,
        aux_measurements=aux_measurements,
        poni_data=poni_data,
        detector_config=demo_config["detectors"],
        active_detector_ids=demo_config["dev_active_detectors"],
        distances_cm={"PRIMARY": 17.0, "SECONDARY": 2.0},  # Match PONI distances
    )

    return file_path


def test_create_session_container(temp_output_dir):
    """Test creating a new session container."""
    container_id, file_path = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_001",
        operator_id="operator_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    assert Path(file_path).exists()
    assert schema.validate_container_id(container_id)

    with h5py.File(file_path, "r") as f:
        assert f.attrs["container_id"] == container_id
        assert f.attrs["container_type"] == schema.CONTAINER_TYPE_SESSION
        assert f.attrs["sample_id"] == "SAMPLE_001"
        assert f.attrs["study_name"] == "UNSPECIFIED"
        assert f.attrs["operator_id"] == "operator_1"
        assert f.attrs["machine_name"] == "DIFRA_01"
        assert f.attrs["beam_energy_keV"] == 12.5


def test_create_session_container_with_study_name(temp_output_dir):
    """Test creating session container with explicit study_name."""
    _container_id, file_path = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_003",
        study_name="STUDY_A",
        operator_id="operator_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    with h5py.File(file_path, "r") as f:
        assert f.attrs.get("study_name") == "STUDY_A"


def test_create_session_container_with_patient_id(temp_output_dir):
    """Test creating session container with optional patient_id."""
    container_id, file_path = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_002",
        operator_id="operator_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
        patient_id="PATIENT_123",
    )

    with h5py.File(file_path, "r") as f:
        assert f.attrs.get("patient_id") == "PATIENT_123"


def test_copy_technical_to_session(temp_output_dir, technical_container_file):
    """Test copying technical group from technical to session container."""
    session_id, session_file = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_001",
        operator_id="op_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_file, session_file=session_file
    )

    with h5py.File(session_file, "r") as f:
        assert "/technical" in f
        assert "/technical/config" in f
        assert "/technical/poni" in f


def test_add_image(temp_output_dir, technical_container_file):
    """Test adding an image to session container."""
    session_id, session_file = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_001",
        operator_id="op_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    # Create synthetic image
    image_data = np.random.rand(512, 512).astype(np.float32)

    image_path = session_container.add_image(
        file_path=session_file,
        image_index=1,
        image_data=image_data,
        image_type=schema.IMAGE_TYPE_SAMPLE,
    )

    with h5py.File(session_file, "r") as f:
        assert image_path in f
        assert f[image_path].attrs["image_type"] == schema.IMAGE_TYPE_SAMPLE
        assert "data" in f[image_path]


def test_add_zone(temp_output_dir, technical_container_file):
    """Test adding zones to session container."""
    session_id, session_file = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_001",
        operator_id="op_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    # Add sample_holder zone
    holder_geometry = [[100, 100], [400, 100], [400, 400], [100, 400]]
    zone_path = session_container.add_zone(
        file_path=session_file,
        zone_index=1,
        zone_role=schema.ZONE_ROLE_SAMPLE_HOLDER,
        geometry_px=holder_geometry,
        holder_diameter_mm=25.0,
    )

    with h5py.File(session_file, "r") as f:
        assert zone_path in f
        assert f[zone_path].attrs["zone_role"] == schema.ZONE_ROLE_SAMPLE_HOLDER
        assert f[zone_path].attrs["holder_diameter_mm"] == 25.0


def test_add_image_mapping(temp_output_dir, technical_container_file):
    """Test adding pixel-to-mm mapping."""
    session_id, session_file = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_001",
        operator_id="op_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    mapping = session_container.add_image_mapping(
        file_path=session_file,
        sample_holder_zone_id="zone_001",
        pixel_to_mm_conversion={"scale": 0.1, "offset_x": 0.0, "offset_y": 0.0},
    )

    with h5py.File(session_file, "r") as f:
        assert mapping in f
        mapping_data = json.loads(f[mapping][()].decode("utf-8"))
        assert mapping_data["sample_holder_zone_id"] == "zone_001"


def test_add_point(temp_output_dir, technical_container_file):
    """Test adding a measurement point."""
    session_id, session_file = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_001",
        operator_id="op_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    point_path = session_container.add_point(
        file_path=session_file,
        point_index=1,
        pixel_coordinates=[100.5, 200.5],
        physical_coordinates_mm=[10.0, 20.0],
    )

    with h5py.File(session_file, "r") as f:
        assert point_path in f
        px_coords = f[point_path].attrs["pixel_coordinates"]
        mm_coords = f[point_path].attrs["physical_coordinates_mm"]
        assert np.allclose(px_coords, [100.5, 200.5])
        assert np.allclose(mm_coords, [10.0, 20.0])


def test_add_measurement(temp_output_dir, technical_container_file):
    """Test adding a measurement to a point."""
    session_id, session_file = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_001",
        operator_id="op_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_file, session_file=session_file
    )

    # Add point first
    session_container.add_point(
        file_path=session_file,
        point_index=1,
        pixel_coordinates=[100.5, 200.5],
        physical_coordinates_mm=[10.0, 20.0],
    )

    # Add measurement
    measurement_data = {
        "PRIMARY": np.random.rand(256, 256).astype(np.float32),
        "SECONDARY": np.random.rand(256, 256).astype(np.float32),
    }

    detector_metadata = {
        "PRIMARY": {"integration_time_ms": 100.0, "beam_energy_keV": 12.5},
        "SECONDARY": {"integration_time_ms": 100.0, "beam_energy_keV": 12.5},
    }

    poni_alias_map = {"PRIMARY": "PRIMARY", "SECONDARY": "SECONDARY"}

    meas_path = session_container.add_measurement(
        file_path=session_file,
        point_index=1,
        measurement_data=measurement_data,
        detector_metadata=detector_metadata,
        poni_alias_map=poni_alias_map,
    )

    with h5py.File(session_file, "r") as f:
        assert meas_path in f
        assert f[meas_path].attrs["measurement_counter"] == 1
        assert f[meas_path].attrs["measurement_status"] == schema.STATUS_COMPLETED
        assert "/measurements/pt_001/meas_000000001" in f


def test_measurement_counter_increments(temp_output_dir, technical_container_file):
    """Test that measurement counter increments properly."""
    session_id, session_file = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_001",
        operator_id="op_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_file, session_file=session_file
    )

    # Add multiple measurements
    for pt_idx in range(1, 4):
        session_container.add_point(
            file_path=session_file,
            point_index=pt_idx,
            pixel_coordinates=[100.0, 100.0],
            physical_coordinates_mm=[10.0, 10.0],
        )

        for meas_idx in range(1, 3):
            measurement_data = {
                "PRIMARY": np.random.rand(256, 256).astype(np.float32),
            }
            detector_metadata = {"PRIMARY": {"integration_time_ms": 100.0}}
            poni_alias_map = {"PRIMARY": "PRIMARY"}

            session_container.add_measurement(
                file_path=session_file,
                point_index=pt_idx,
                measurement_data=measurement_data,
                detector_metadata=detector_metadata,
                poni_alias_map=poni_alias_map,
            )

    with h5py.File(session_file, "r") as f:
        # Should have 6 measurements total (3 points * 2 measurements each)
        assert f.attrs["measurement_counter"] == 6


def test_add_analytical_measurement(temp_output_dir, technical_container_file):
    """Test adding an analytical measurement."""
    session_id, session_file = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_001",
        operator_id="op_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_file, session_file=session_file
    )

    measurement_data = {
        "PRIMARY": np.random.rand(256, 256).astype(np.float32),
    }

    detector_metadata = {"PRIMARY": {"integration_time_ms": 100.0}}
    poni_alias_map = {"PRIMARY": "PRIMARY"}

    ana_path = session_container.add_analytical_measurement(
        file_path=session_file,
        measurement_data=measurement_data,
        detector_metadata=detector_metadata,
        poni_alias_map=poni_alias_map,
        analysis_type=schema.ANALYSIS_TYPE_ATTENUATION,
    )

    with h5py.File(session_file, "r") as f:
        assert ana_path in f
        assert f[ana_path].attrs["analysis_type"] == schema.ANALYSIS_TYPE_ATTENUATION


def test_link_analytical_measurement_to_point(temp_output_dir, technical_container_file):
    """Test linking analytical measurement to a point."""
    session_id, session_file = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_001",
        operator_id="op_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_file, session_file=session_file
    )

    # Add point
    session_container.add_point(
        file_path=session_file,
        point_index=1,
        pixel_coordinates=[100.0, 100.0],
        physical_coordinates_mm=[10.0, 10.0],
    )

    # Add analytical measurement
    measurement_data = {
        "PRIMARY": np.random.rand(256, 256).astype(np.float32),
    }
    detector_metadata = {"PRIMARY": {"integration_time_ms": 100.0}}
    poni_alias_map = {"PRIMARY": "PRIMARY"}

    session_container.add_analytical_measurement(
        file_path=session_file,
        measurement_data=measurement_data,
        detector_metadata=detector_metadata,
        poni_alias_map=poni_alias_map,
        analysis_type=schema.ANALYSIS_TYPE_ATTENUATION,
    )

    # Link to point
    session_container.link_analytical_measurement_to_point(
        file_path=session_file, point_index=1, analytical_measurement_index=1
    )

    with h5py.File(session_file, "r") as f:
        point_path = "/points/pt_001"
        assert schema.ATTR_ANALYTICAL_MEASUREMENT_REFS in f[point_path].attrs


def test_update_point_status(temp_output_dir):
    """Test updating point status."""
    session_id, session_file = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_001",
        operator_id="op_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    session_container.add_point(
        file_path=session_file,
        point_index=1,
        pixel_coordinates=[100.0, 100.0],
        physical_coordinates_mm=[10.0, 10.0],
        point_status=schema.POINT_STATUS_PENDING,
    )

    session_container.update_point_status(
        file_path=session_file, point_index=1, point_status=schema.POINT_STATUS_MEASURED
    )

    with h5py.File(session_file, "r") as f:
        point_path = "/points/pt_001"
        assert f[point_path].attrs["point_status"] == schema.POINT_STATUS_MEASURED


def test_find_active_session_container(temp_output_dir):
    """Test finding most recent session container."""
    # Create first session
    _, session_file_1 = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_001",
        operator_id="op_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    # Create second session
    _, session_file_2 = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_002",
        operator_id="op_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    # Most recent should be session 2
    found = session_container.find_active_session_container(temp_output_dir)
    assert found == session_file_2

    # With sample_id filter
    found = session_container.find_active_session_container(
        temp_output_dir, sample_id="SAMPLE_001"
    )
    assert found == session_file_1


def test_complete_session_workflow(temp_output_dir, technical_container_file):
    """Test complete session workflow: create, add image, zones, points, measurements."""
    session_id, session_file = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_001",
        operator_id="op_1",
        site_id="site_A",
        machine_name="DIFRA_01",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    # Copy technical data
    session_container.copy_technical_to_session(
        technical_file=technical_container_file, session_file=session_file
    )

    # Add image
    image_data = np.random.rand(512, 512).astype(np.float32)
    session_container.add_image(
        file_path=session_file,
        image_index=1,
        image_data=image_data,
        image_type=schema.IMAGE_TYPE_SAMPLE,
    )

    # Add zones
    session_container.add_zone(
        file_path=session_file,
        zone_index=1,
        zone_role=schema.ZONE_ROLE_SAMPLE_HOLDER,
        geometry_px=[[100, 100], [400, 100], [400, 400], [100, 400]],
        holder_diameter_mm=25.0,
    )

    # Add mapping
    session_container.add_image_mapping(
        file_path=session_file,
        sample_holder_zone_id="zone_001",
        pixel_to_mm_conversion={"scale": 0.1},
    )

    # Add points and measurements
    for pt_idx in range(1, 4):
        session_container.add_point(
            file_path=session_file,
            point_index=pt_idx,
            pixel_coordinates=[100.0 + pt_idx * 50, 100.0],
            physical_coordinates_mm=[10.0 + pt_idx * 5, 10.0],
        )

        measurement_data = {
            "PRIMARY": np.random.rand(256, 256).astype(np.float32),
            "SECONDARY": np.random.rand(256, 256).astype(np.float32),
        }

        detector_metadata = {
            "PRIMARY": {"integration_time_ms": 100.0, "beam_energy_keV": 12.5},
            "SECONDARY": {"integration_time_ms": 100.0, "beam_energy_keV": 12.5},
        }

        poni_alias_map = {"PRIMARY": "PRIMARY", "SECONDARY": "SECONDARY"}

        session_container.add_measurement(
            file_path=session_file,
            point_index=pt_idx,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
        )

        # Update point status
        session_container.update_point_status(
            file_path=session_file,
            point_index=pt_idx,
            point_status=schema.POINT_STATUS_MEASURED,
        )

    # Verify complete structure
    with h5py.File(session_file, "r") as f:
        # Root attributes
        assert f.attrs["sample_id"] == "SAMPLE_001"
        assert f.attrs["operator_id"] == "op_1"

        # Technical data
        assert "/technical" in f
        assert "/technical/config" in f
        assert "/technical/poni" in f

        # Images
        assert "/images/img_001" in f

        # Zones
        assert "/images/zones/zone_001" in f

        # Mapping
        assert "/images/mapping/mapping" in f

        # Points
        for pt_idx in range(1, 4):
            point_path = f"/points/pt_{pt_idx:03d}"
            assert point_path in f
            assert f[point_path].attrs["point_status"] == schema.POINT_STATUS_MEASURED

        # Measurements
        for pt_idx in range(1, 4):
            meas_path = f"/measurements/pt_{pt_idx:03d}/meas_{pt_idx:09d}"
            assert meas_path in f
