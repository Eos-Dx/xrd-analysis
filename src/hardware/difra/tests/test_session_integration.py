"""End-to-end integration tests for session container workflow.

Simulates complete GUI measurement workflow:
1. Create session container with technical data
2. Add sample image, zones, and mapping
3. Add measurement points
4. Simulate measurement capture and writing
5. Validate complete container structure
"""

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

from hardware.container.v0_2 import (
    schema,
    session_container,
    validator,
    technical_container,
    measurement_counter,
    utils,
)
from hardware.difra.gui.session_measurement_handler import SessionMeasurementHandler


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
    }


@pytest.fixture
def technical_container_file(temp_output_dir, demo_config):
    """Create a technical container for testing."""
    from hardware.difra.hardware.detectors import DummyDetectorController

    # Create PONI files
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

    poni_data = {}
    for detector, distance in [("PRIMARY", 0.17), ("SECONDARY", 0.02)]:
        poni_path = poni_dir / f"{detector.lower()}_demo.poni"
        content = poni_content_template.format(detector=detector, distance=distance)
        poni_path.write_text(content)
        poni_data[detector] = (content, poni_path.name)

    # Create synthetic technical measurements
    aux_measurements = {}
    for tech_type in ["DARK", "EMPTY", "BACKGROUND", "AGBH"]:
        aux_measurements[tech_type] = {}
        for det_id in ["PRIMARY", "SECONDARY"]:
            filename = temp_output_dir / f"{tech_type}_{det_id}.npy"
            data = np.random.poisson(20, size=(256, 256)).astype(np.float32)
            np.save(filename, data)
            aux_measurements[tech_type][det_id] = str(filename)

    # Generate technical container
    container_id, file_path = technical_container.generate_from_aux_table(
        folder=temp_output_dir,
        aux_measurements=aux_measurements,
        poni_data=poni_data,
        detector_config=demo_config["detectors"],
        active_detector_ids=demo_config["dev_active_detectors"],
        distances_cm={"PRIMARY": 17.0, "SECONDARY": 2.0},
    )

    return file_path


def test_session_handler_complete_workflow(temp_output_dir, technical_container_file):
    """Test SessionMeasurementHandler with complete workflow."""
    handler = SessionMeasurementHandler(
        session_folder=temp_output_dir / "sessions",
        technical_container_file=technical_container_file,
        sample_id="SAMPLE_TEST_001",
        operator_id="test_operator",
        site_id="test_site",
        machine_name="TEST_DIFRA",
        beam_energy_keV=12.5,
    )

    # Create session
    session_file = handler.create_session()
    assert Path(session_file).exists()

    # Add image
    image_data = np.random.rand(512, 512).astype(np.float32)
    handler.add_image(image_data=image_data)

    # Add zones
    handler.add_zone(
        zone_index=1,
        zone_role=schema.ZONE_ROLE_SAMPLE_HOLDER,
        geometry_px=[[100, 100], [400, 100], [400, 400], [100, 400]],
        holder_diameter_mm=25.0,
    )

    # Add mapping
    handler.add_image_mapping(
        sample_holder_zone_id="zone_001",
        pixel_to_mm_conversion={"scale": 0.1, "offset_x": 0.0, "offset_y": 0.0},
    )

    # Add points and measurements
    for pt_idx in range(1, 4):
        handler.add_point(
            point_index=pt_idx,
            pixel_coordinates=[100.0 + pt_idx * 50, 100.0],
            physical_coordinates_mm=[10.0 + pt_idx * 5, 10.0],
        )

        # Simulate measurement
        measurement_data = {
            "PRIMARY": np.random.rand(256, 256).astype(np.float32),
            "SECONDARY": np.random.rand(256, 256).astype(np.float32),
        }

        detector_metadata = {
            "PRIMARY": {"integration_time_ms": 100.0, "beam_energy_keV": 12.5},
            "SECONDARY": {"integration_time_ms": 100.0, "beam_energy_keV": 12.5},
        }
        raw_files = {
            "PRIMARY": {
                "raw_txt": b"primary txt blob",
                "raw_dsc": b"primary dsc blob",
            },
            "SECONDARY": {
                "raw_txt": b"secondary txt blob",
                "raw_dsc": b"secondary dsc blob",
            },
        }

        poni_alias_map = {"PRIMARY": "PRIMARY", "SECONDARY": "SECONDARY"}

        handler.add_measurement(
            point_index=pt_idx,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
            raw_files=raw_files,
        )

        handler.update_point_status(
            point_index=pt_idx, point_status=schema.POINT_STATUS_MEASURED
        )

    # Verify complete structure
    with h5py.File(session_file, "r") as f:
        assert f.attrs["sample_id"] == "SAMPLE_TEST_001"
        assert f.attrs["operator_id"] == "test_operator"
        assert schema.GROUP_CALIBRATION_SNAPSHOT in f
        assert "/entry/images/img_001" in f
        assert "/entry/images/zones/zone_001" in f
        assert "/entry/images/mapping/mapping" in f
        assert f"{schema.GROUP_MEASUREMENTS}/pt_001/meas_000000001/det_primary/blob/raw_txt" in f
        assert "/entry/measurements/pt_001/meas_000000001/det_secondary/blob/raw_dsc" in f
        assert f.attrs["measurement_counter"] == 3


def test_measurement_counter_persistence(temp_output_dir, technical_container_file):
    """Test measurement counter persists across accesses."""
    session_id, session_file = session_container.create_session_container(
        folder=temp_output_dir,
        sample_id="SAMPLE_002",
        operator_id="op",
        site_id="site",
        machine_name="DIFRA",
        beam_energy_keV=12.5,
        acquisition_date="2024-02-09",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_file, session_file=session_file
    )

    # Get counter multiple times
    counter1 = measurement_counter.MeasurementCounter(session_file)
    assert counter1.get_current() == 0

    val1 = counter1.get_next()
    assert val1 == 1

    val2 = counter1.get_next()
    assert val2 == 2

    # Verify persistence with new counter object
    counter2 = measurement_counter.MeasurementCounter(session_file)
    assert counter2.get_current() == 2

    val3 = counter2.get_next()
    assert val3 == 3


def test_session_validator_complete_container(
    temp_output_dir, technical_container_file
):
    """Test validator on complete, valid container."""
    handler = SessionMeasurementHandler(
        session_folder=temp_output_dir / "sessions",
        technical_container_file=technical_container_file,
        sample_id="SAMPLE_VALID",
        operator_id="op",
        site_id="site",
        machine_name="DIFRA",
        beam_energy_keV=12.5,
    )

    session_file = handler.create_session()

    # Add minimal required content
    handler.add_image(image_data=np.random.rand(256, 256).astype(np.float32))
    handler.add_point(
        point_index=1,
        pixel_coordinates=[100.0, 100.0],
        physical_coordinates_mm=[10.0, 10.0],
    )

    # Should be valid (or have only warnings, not errors)
    container_validator = validator.SessionContainerValidator(session_file)
    is_valid, errors = container_validator.validate()

    error_count = sum(1 for e in errors if e.severity == "ERROR")
    assert error_count == 0, f"Validation errors: {[e for e in errors if e.severity == 'ERROR']}"


def test_session_validator_detects_missing_technical():
    """Test validator detects missing technical group."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create empty session container (without technical data)
        session_id, session_file = session_container.create_session_container(
            folder=tmpdir,
            sample_id="SAMPLE_INVALID",
            operator_id="op",
            site_id="site",
            machine_name="DIFRA",
            beam_energy_keV=12.5,
            acquisition_date="2024-02-09",
        )

        # Don't copy technical data - this should be invalid
        container_validator = validator.SessionContainerValidator(session_file)
        is_valid, errors = container_validator.validate()

        # Should have errors
        error_count = sum(1 for e in errors if e.severity == "ERROR")
        assert error_count > 0


def test_session_validator_detects_missing_processed_signal():
    """Test validator detects missing processed_signal in measurements."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        session_id, session_file = session_container.create_session_container(
            folder=tmpdir,
            sample_id="SAMPLE_INCOMPLETE",
            operator_id="op",
            site_id="site",
            machine_name="DIFRA",
            beam_energy_keV=12.5,
            acquisition_date="2024-02-09",
        )

        # Add point
        session_container.add_point(
            file_path=session_file,
            point_index=1,
            pixel_coordinates=[100.0, 100.0],
            physical_coordinates_mm=[10.0, 10.0],
        )

    # Create measurement group without processed_signal
        utils.create_group_if_missing(
            file_path=session_file,
            group_path=f"{schema.GROUP_MEASUREMENTS}/pt_001/meas_000000001/det_primary",
        )

        # Validate
        container_validator = validator.SessionContainerValidator(session_file)
        is_valid, errors = container_validator.validate()

    # Should detect missing processed_signal
    has_processed_signal_error = any(
        "processed_signal" in e.message for e in errors if e.severity == "ERROR"
    )
    assert has_processed_signal_error


def test_session_multiple_images_zones(temp_output_dir, technical_container_file):
    """Test session with multiple images and zones."""
    handler = SessionMeasurementHandler(
        session_folder=temp_output_dir / "sessions",
        technical_container_file=technical_container_file,
        sample_id="SAMPLE_MULTI",
        operator_id="op",
        site_id="site",
        machine_name="DIFRA",
        beam_energy_keV=12.5,
    )

    session_file = handler.create_session()

    # Add multiple images
    for i in range(1, 4):
        img_data = np.random.rand(256, 256).astype(np.float32)
        handler.add_image(image_data=img_data, image_index=i)

    # Add multiple zones
    for i in range(1, 4):
        if i == 1:
            role = schema.ZONE_ROLE_SAMPLE_HOLDER
        elif i == 2:
            role = schema.ZONE_ROLE_INCLUDE
        else:
            role = schema.ZONE_ROLE_EXCLUDE

        handler.add_zone(
            zone_index=i,
            zone_role=role,
            geometry_px=[[100, 100], [200, 200], [300, 300]],
            holder_diameter_mm=25.0 if i == 1 else None,
        )

    # Verify in HDF5
    with h5py.File(session_file, "r") as f:
        assert "/entry/images/img_001" in f
        assert "/entry/images/img_002" in f
        assert "/entry/images/img_003" in f
        assert "/entry/images/zones/zone_001" in f
        assert "/entry/images/zones/zone_002" in f
        assert "/entry/images/zones/zone_003" in f

        assert (
            f["/entry/images/zones/zone_001"].attrs["zone_role"]
            == schema.ZONE_ROLE_SAMPLE_HOLDER
        )
        assert (
            f["/entry/images/zones/zone_002"].attrs["zone_role"]
            == schema.ZONE_ROLE_INCLUDE
        )


def test_session_analytical_measurement_workflow(
    temp_output_dir, technical_container_file
):
    """Test session with analytical measurements."""
    handler = SessionMeasurementHandler(
        session_folder=temp_output_dir / "sessions",
        technical_container_file=technical_container_file,
        sample_id="SAMPLE_ANA",
        operator_id="op",
        site_id="site",
        machine_name="DIFRA",
        beam_energy_keV=12.5,
    )

    session_file = handler.create_session()

    # Add point
    handler.add_point(
        point_index=1,
        pixel_coordinates=[100.0, 100.0],
        physical_coordinates_mm=[10.0, 10.0],
    )

    # Add analytical measurement
    ana_data = {"PRIMARY": np.random.rand(256, 256).astype(np.float32)}
    ana_meta = {"PRIMARY": {"integration_time_ms": 50.0}}
    poni_map = {"PRIMARY": "PRIMARY"}

    handler.add_analytical_measurement(
        measurement_data=ana_data,
        detector_metadata=ana_meta,
        poni_alias_map=poni_map,
        analysis_type=schema.ANALYSIS_TYPE_ATTENUATION,
        analysis_role=schema.ANALYSIS_ROLE_I0,
    )

    # Link to point
    handler.link_analytical_measurement_to_point(
        point_index=1, analytical_measurement_index=1
    )

    # Verify
    with h5py.File(session_file, "r") as f:
        assert "/entry/analytical_measurements/ana_000000001" in f
        ana = f["/entry/analytical_measurements/ana_000000001"]
        assert ana.attrs["analysis_type"] == schema.ANALYSIS_TYPE_ATTENUATION
        assert ana.attrs[schema.ATTR_ANALYSIS_ROLE] == schema.ANALYSIS_ROLE_I0
        assert schema.ATTR_POINT_IDS in ana.attrs

        # Check ID-based link
        point = f["/entry/points/pt_001"]
        assert schema.ATTR_ANALYTICAL_MEASUREMENT_IDS in point.attrs
        assert schema.ATTR_ANALYTICAL_MEASUREMENT_REFS in point.attrs
        assert schema.ATTR_POINT_REFS in ana.attrs
