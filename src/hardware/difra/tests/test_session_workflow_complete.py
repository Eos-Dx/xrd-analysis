"""Comprehensive Step-by-Step Session Container Workflow Tests.

Tests the complete end-to-end workflow from session creation to finalization,
including error handling, archiving, and replacement scenarios.

Test Structure:
1. Setup: Create technical container with calibration data
2. Session Creation: Create new session with sample image
3. Zone Definition: Add measurement zones (sample holder, include, exclude)
4. Point Generation: Add measurement points in zones
5. Measurements: Record detector measurements at each point
6. Attenuation: Record I₀ and I measurements
7. Validation: Verify HDF5 structure and data integrity
8. Error Handling: Test session replacement with error marking
9. Archiving: Test session archiving workflow
10. Finalization: Lock container for upload
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

# Add project root to path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.container.v0_1 import (
    container_manager,
    schema,
    writer as session_container,
    technical_container,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def demo_config():
    """Demo configuration with two detectors (SAXS, WAXS)."""
    return {
        "detectors": [
            {
                "id": "det_saxs",
                "alias": "SAXS",
                "name": "SAXS Detector",
                "size": {"width": 256, "height": 256},
                "pixel_size_um": [75.0, 75.0],
            },
            {
                "id": "det_waxs",
                "alias": "WAXS",
                "name": "WAXS Detector",
                "size": {"width": 256, "height": 256},
                "pixel_size_um": [55.0, 55.0],
            },
        ],
        "active_detectors": ["det_saxs", "det_waxs"],
    }


@pytest.fixture
def demo_poni_files(temp_dir):
    """Create demo PONI files for both detectors."""
    poni_dir = temp_dir / "poni"
    poni_dir.mkdir()

    poni_template = """# PONI calibration file
poni_version: 2.1
Detector: Detector
Distance: {distance}
Poni1: 0.012345
Poni2: 0.023456
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 1.54e-10
PixelSize1: {pixel_size}
PixelSize2: {pixel_size}
"""

    poni_files = {}
    distances = {"SAXS": 1.00, "WAXS": 0.17}  # meters
    pixel_sizes = {"SAXS": 7.5e-05, "WAXS": 5.5e-05}

    for alias, distance in distances.items():
        poni_path = poni_dir / f"{alias.lower()}_demo.poni"
        content = poni_template.format(
            distance=distance, pixel_size=pixel_sizes[alias]
        )
        poni_path.write_text(content)
        poni_files[alias] = poni_path

    return poni_files


@pytest.fixture
def technical_container_path(temp_dir, demo_config, demo_poni_files):
    """Create technical container with all required calibration measurements."""
    print("\n=== STEP 0: Creating Technical Container ===")

    # Create technical measurements for all required types
    tech_measurements = {}
    for tech_type in ["DARK", "EMPTY", "BACKGROUND", "AGBH"]:
        tech_measurements[tech_type] = {}
        for det_config in demo_config["detectors"]:
            alias = det_config["alias"]
            size = det_config["size"]

            # Create synthetic data
            if tech_type == "DARK":
                data = np.random.poisson(5, size=(size["height"], size["width"]))
            elif tech_type == "EMPTY":
                data = np.random.poisson(10, size=(size["height"], size["width"]))
            elif tech_type == "BACKGROUND":
                data = np.random.poisson(20, size=(size["height"], size["width"]))
            else:  # AGBH
                # Create ring pattern for AgBH
                y, x = np.ogrid[: size["height"], : size["width"]]
                center_y, center_x = size["height"] // 2, size["width"] // 2
                r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                # Add rings at specific radii
                rings = (
                    (np.abs(r - 30) < 2).astype(float) * 100
                    + (np.abs(r - 60) < 2).astype(float) * 80
                    + (np.abs(r - 90) < 2).astype(float) * 60
                )
                data = np.random.poisson(50 + rings)

            # Save as .npy file
            filename = temp_dir / f"{tech_type}_{alias}.npy"
            np.save(filename, data.astype(np.float32))
            tech_measurements[tech_type][alias] = str(filename)

    # Load PONI data
    poni_data = {}
    for alias, poni_path in demo_poni_files.items():
        content = poni_path.read_text()
        poni_data[alias] = (content, poni_path.name)

    # Generate technical container
    container_id, file_path = technical_container.generate_from_aux_table(
        folder=temp_dir,
        aux_measurements=tech_measurements,
        poni_data=poni_data,
        detector_config=demo_config["detectors"],
        active_detector_ids=demo_config["active_detectors"],
        distances_cm={"SAXS": 100.0, "WAXS": 17.0},
    )

    print(f"  ✓ Technical container created: {Path(file_path).name}")
    print(f"  ✓ Container ID: {container_id}")
    print(f"  ✓ Calibration types: {', '.join(tech_measurements.keys())}")

    # Lock the technical container
    container_manager.lock_technical_container(Path(file_path), locked_by="test_operator", notes="Test technical container")
    print(f"  ✓ Container locked and ready for use")

    return file_path


# ============================================================================
# Test 1: Session Creation
# ============================================================================


def test_step1_create_session(temp_dir, technical_container_path):
    """Step 1: Create new session container with metadata."""
    print("\n=== STEP 1: Create Session Container ===")

    # Create session folder
    session_folder = temp_dir / "sessions"
    session_folder.mkdir()

    # Create session
    session_id, session_path = session_container.create_session_container(
        folder=session_folder,
        sample_id="TEST_SAMPLE_001",
        operator_id="test_operator",
        site_id="LAB_A",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-11",
        patient_id="PATIENT_XYZ",
    )

    print(f"  ✓ Session created: {Path(session_path).name}")
    print(f"  ✓ Session ID: {session_id}")
    print(f"  ✓ Sample ID: TEST_SAMPLE_001")

    # Verify session container structure
    assert Path(session_path).exists(), "Session file should exist"

    with h5py.File(session_path, "r") as f:
        # Check root attributes
        assert f.attrs["container_type"] == schema.CONTAINER_TYPE_SESSION
        assert f.attrs["session_id"] == session_id
        assert f.attrs["sample_id"] == "TEST_SAMPLE_001"
        assert f.attrs["operator_id"] == "test_operator"
        assert f.attrs["patient_id"] == "PATIENT_XYZ"
        assert f.attrs["beam_energy_keV"] == 17.5

        # Check required groups exist
        assert "/images" in f
        assert "/images/zones" in f
        assert "/images/mapping" in f
        assert "/points" in f
        assert "/measurements" in f
        assert "/analytical_measurements" in f

        print(f"  ✓ All required groups created")
        print(f"  ✓ Root attributes verified")

    # Copy technical data
    session_container.copy_technical_to_session(
        technical_file=technical_container_path, session_file=session_path
    )

    # Verify technical data was copied
    with h5py.File(session_path, "r") as f:
        assert "/technical" in f
        assert "/technical/poni" in f
        print(f"  ✓ Technical calibration data copied")

        # Count technical events
        tech_group = f["/technical"]
        tech_events = [k for k in tech_group.keys() if k.startswith("tech_evt_")]
        print(f"  ✓ Technical events: {len(tech_events)}")

    return session_id, session_path


# ============================================================================
# Test 2: Add Sample Image
# ============================================================================


def test_step2_add_sample_image(temp_dir, technical_container_path):
    """Step 2: Add sample image to session container."""
    print("\n=== STEP 2: Add Sample Image ===")

    # Create session first
    session_folder = temp_dir / "sessions"
    session_folder.mkdir()

    session_id, session_path = session_container.create_session_container(
        folder=session_folder,
        sample_id="TEST_SAMPLE_002",
        operator_id="test_operator",
        site_id="LAB_A",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-11",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_path, session_file=session_path
    )

    # Create synthetic sample image (grayscale)
    image_height, image_width = 800, 600
    image_data = np.random.randint(0, 255, size=(image_height, image_width), dtype=np.uint8)

    # Add image to session
    image_path = session_container.add_image(
        file_path=session_path,
        image_index=1,
        image_data=image_data,
        image_type="sample",
    )

    print(f"  ✓ Sample image added: {image_path}")
    print(f"  ✓ Image size: {image_width}x{image_height}")

    # Verify image stored correctly
    with h5py.File(session_path, "r") as f:
        assert "/images/img_001" in f
        img_group = f["/images/img_001"]
        assert "data" in img_group

        stored_data = img_group["data"][:]
        assert stored_data.shape == (image_height, image_width)
        assert np.array_equal(stored_data, image_data)

        assert img_group.attrs["image_type"] == "sample"
        print(f"  ✓ Image data integrity verified")

    return session_id, session_path, (image_width, image_height)


# ============================================================================
# Test 3: Define Measurement Zones
# ============================================================================


def test_step3_define_zones(temp_dir, technical_container_path):
    """Step 3: Define measurement zones (sample holder, include, exclude)."""
    print("\n=== STEP 3: Define Measurement Zones ===")

    # Setup session with image
    session_folder = temp_dir / "sessions"
    session_folder.mkdir()

    session_id, session_path = session_container.create_session_container(
        folder=session_folder,
        sample_id="TEST_SAMPLE_003",
        operator_id="test_operator",
        site_id="LAB_A",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-11",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_path, session_file=session_path
    )

    image_data = np.random.randint(0, 255, size=(800, 600), dtype=np.uint8)
    session_container.add_image(
        file_path=session_path, image_index=1, image_data=image_data
    )

    # Define sample holder zone (circular)
    holder_geometry = [300.0, 400.0, 200.0, 200.0]  # x, y, width, height
    holder_diameter_mm = 25.4  # 1 inch

    zone1_path = session_container.add_zone(
        file_path=session_path,
        zone_index=1,
        zone_role="sample_holder",
        geometry_px=holder_geometry,
        shape="circle",
        holder_diameter_mm=holder_diameter_mm,
    )

    print(f"  ✓ Sample holder zone: {zone1_path}")
    print(f"    - Shape: circle")
    print(f"    - Diameter: {holder_diameter_mm} mm")

    # Define include zone (polygon)
    include_geometry = [
        [250.0, 350.0],
        [550.0, 350.0],
        [550.0, 650.0],
        [250.0, 650.0],
    ]

    zone2_path = session_container.add_zone(
        file_path=session_path,
        zone_index=2,
        zone_role="include",
        geometry_px=include_geometry,
        shape="polygon",
    )

    print(f"  ✓ Include zone: {zone2_path}")
    print(f"    - Shape: polygon")
    print(f"    - Vertices: {len(include_geometry)}")

    # Define exclude zone (circular - e.g., air bubble)
    exclude_geometry = [380.0, 420.0, 40.0, 40.0]

    zone3_path = session_container.add_zone(
        file_path=session_path,
        zone_index=3,
        zone_role="exclude",
        geometry_px=exclude_geometry,
        shape="circle",
    )

    print(f"  ✓ Exclude zone: {zone3_path}")
    print(f"    - Shape: circle")
    print(f"    - Purpose: exclude air bubble")

    # Add pixel-to-mm mapping
    pixel_to_mm_conversion = {
        "scale_x": holder_diameter_mm / holder_geometry[2],  # mm per pixel
        "scale_y": holder_diameter_mm / holder_geometry[3],
        "method": "holder_calibration",
    }

    mapping_path = session_container.add_image_mapping(
        file_path=session_path,
        sample_holder_zone_id="zone_001",
        pixel_to_mm_conversion=pixel_to_mm_conversion,
    )

    print(f"  ✓ Pixel-to-mm mapping: {mapping_path}")
    print(
        f"    - Scale: {pixel_to_mm_conversion['scale_x']:.4f} mm/px"
    )

    # Verify zones
    with h5py.File(session_path, "r") as f:
        assert "/images/zones/zone_001" in f
        assert "/images/zones/zone_002" in f
        assert "/images/zones/zone_003" in f

        zone1 = f["/images/zones/zone_001"]
        assert zone1.attrs["zone_role"] == "sample_holder"
        assert zone1.attrs["holder_diameter_mm"] == holder_diameter_mm

        zone2 = f["/images/zones/zone_002"]
        assert zone2.attrs["zone_role"] == "include"

        zone3 = f["/images/zones/zone_003"]
        assert zone3.attrs["zone_role"] == "exclude"

        print(f"  ✓ All zones verified in HDF5")

    return session_id, session_path


# ============================================================================
# Test 4: Add Measurement Points
# ============================================================================


def test_step4_add_points(temp_dir, technical_container_path):
    """Step 4: Add measurement points with coordinates."""
    print("\n=== STEP 4: Add Measurement Points ===")

    # Setup session with image and zones
    session_folder = temp_dir / "sessions"
    session_folder.mkdir()

    session_id, session_path = session_container.create_session_container(
        folder=session_folder,
        sample_id="TEST_SAMPLE_004",
        operator_id="test_operator",
        site_id="LAB_A",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-11",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_path, session_file=session_path
    )

    # Add points in grid pattern
    points = []
    for i in range(5):  # 5 points
        pixel_x = 300.0 + i * 50.0
        pixel_y = 400.0
        mm_x = pixel_x * 0.127  # Assuming scale
        mm_y = pixel_y * 0.127

        point_path = session_container.add_point(
            file_path=session_path,
            point_index=i + 1,
            pixel_coordinates=[pixel_x, pixel_y],
            physical_coordinates_mm=[mm_x, mm_y],
            point_status="pending",
        )

        points.append((point_path, pixel_x, pixel_y, mm_x, mm_y))
        print(f"  ✓ Point {i+1}: px=({pixel_x:.1f}, {pixel_y:.1f}), mm=({mm_x:.2f}, {mm_y:.2f})")

    # Verify points
    with h5py.File(session_path, "r") as f:
        for i in range(5):
            point_path = f"/points/pt_{i+1:03d}"
            assert point_path in f
            point = f[point_path]
            assert point.attrs["point_status"] == "pending"
            assert len(point.attrs["pixel_coordinates"]) == 2
            assert len(point.attrs["physical_coordinates_mm"]) == 2

    print(f"  ✓ {len(points)} points added and verified")

    return session_id, session_path


# ============================================================================
# Test 5: Record Measurements
# ============================================================================


def test_step5_record_measurements(temp_dir, technical_container_path):
    """Step 5: Record detector measurements at each point."""
    print("\n=== STEP 5: Record Measurements ===")

    # Setup session
    session_folder = temp_dir / "sessions"
    session_folder.mkdir()

    session_id, session_path = session_container.create_session_container(
        folder=session_folder,
        sample_id="TEST_SAMPLE_005",
        operator_id="test_operator",
        site_id="LAB_A",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-11",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_path, session_file=session_path
    )

    # Add 3 points
    for i in range(3):
        session_container.add_point(
            file_path=session_path,
            point_index=i + 1,
            pixel_coordinates=[300.0 + i * 50.0, 400.0],
            physical_coordinates_mm=[15.0 + i * 2.5, 20.0],
            point_status="pending",
        )

    # Record measurements for each point
    poni_alias_map = {"SAXS": "det_saxs", "WAXS": "det_waxs"}

    for point_idx in range(1, 4):
        # Generate synthetic detector data
        measurement_data = {
            "det_saxs": np.random.poisson(100, size=(256, 256)).astype(np.float32),
            "det_waxs": np.random.poisson(200, size=(256, 256)).astype(np.float32),
        }

        detector_metadata = {
            "det_saxs": {
                "integration_time_ms": 1000.0,
                "beam_energy_keV": 17.5,
                "detector_id": "det_saxs",
            },
            "det_waxs": {
                "integration_time_ms": 500.0,
                "beam_energy_keV": 17.5,
                "detector_id": "det_waxs",
            },
        }

        meas_path = session_container.add_measurement(
            file_path=session_path,
            point_index=point_idx,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
        )

        print(f"  ✓ Point {point_idx} measured: {meas_path}")

        # Update point status
        session_container.update_point_status(
            file_path=session_path, point_index=point_idx, point_status="measured"
        )

    # Verify measurements
    with h5py.File(session_path, "r") as f:
        assert "/measurements/pt_001" in f
        assert "/measurements/pt_002" in f
        assert "/measurements/pt_003" in f

        # Check first measurement
        meas = f["/measurements/pt_001/meas_000000001"]
        assert "det_saxs" in meas
        assert "det_waxs" in meas

        saxs_data = meas[f"det_saxs/{schema.DATASET_PROCESSED_SIGNAL}"][:]
        assert saxs_data.shape == (256, 256)
        assert saxs_data.dtype == np.float32

        print(f"  ✓ All measurements verified")
        print(f"  ✓ Data integrity confirmed")

    return session_id, session_path


# ============================================================================
# Test 6: Attenuation Measurements
# ============================================================================


def test_step6_attenuation(temp_dir, technical_container_path):
    """Step 6: Record attenuation measurements (I₀ and I)."""
    print("\n=== STEP 6: Attenuation Measurements ===")

    # Setup session
    session_folder = temp_dir / "sessions"
    session_folder.mkdir()

    session_id, session_path = session_container.create_session_container(
        folder=session_folder,
        sample_id="TEST_SAMPLE_006",
        operator_id="test_operator",
        site_id="LAB_A",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-11",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_path, session_file=session_path
    )

    poni_alias_map = {"SAXS": "det_saxs", "WAXS": "det_waxs"}

    # Record I₀ (without sample at loading position)
    i0_data = {
        "det_saxs": np.random.poisson(500, size=(256, 256)).astype(np.float32),
        "det_waxs": np.random.poisson(800, size=(256, 256)).astype(np.float32),
    }

    i0_metadata = {
        "det_saxs": {
            "integration_time_ms": 100.0,
            "beam_energy_keV": 17.5,
            "detector_id": "det_saxs",
        },
        "det_waxs": {
            "integration_time_ms": 100.0,
            "beam_energy_keV": 17.5,
            "detector_id": "det_waxs",
        },
    }

    i0_path = session_container.add_analytical_measurement(
        file_path=session_path,
        measurement_data=i0_data,
        detector_metadata=i0_metadata,
        poni_alias_map=poni_alias_map,
        analysis_type="attenuation_i0",
    )

    print(f"  ✓ I₀ (without sample) recorded: {i0_path}")

    # Record I (with sample)
    i_data = {
        "det_saxs": np.random.poisson(300, size=(256, 256)).astype(np.float32),
        "det_waxs": np.random.poisson(450, size=(256, 256)).astype(np.float32),
    }

    i_metadata = i0_metadata.copy()

    i_path = session_container.add_analytical_measurement(
        file_path=session_path,
        measurement_data=i_data,
        detector_metadata=i_metadata,
        poni_alias_map=poni_alias_map,
        analysis_type="attenuation_i",
    )

    print(f"  ✓ I (with sample) recorded: {i_path}")

    # Link to a point (e.g., first measurement point)
    session_container.add_point(
        file_path=session_path,
        point_index=1,
        pixel_coordinates=[300.0, 400.0],
        physical_coordinates_mm=[15.0, 20.0],
    )

    session_container.link_analytical_measurement_to_point(
        file_path=session_path, point_index=1, analytical_measurement_index=1
    )

    print(f"  ✓ I₀ linked to point pt_001")

    # Verify attenuation data
    with h5py.File(session_path, "r") as f:
        assert "/analytical_measurements/ana_000000001" in f
        assert "/analytical_measurements/ana_000000002" in f

        i0_meas = f["/analytical_measurements/ana_000000001"]
        assert i0_meas.attrs["analysis_type"] == "attenuation_i0"

        i_meas = f["/analytical_measurements/ana_000000002"]
        assert i_meas.attrs["analysis_type"] == "attenuation_i"

        print(f"  ✓ Attenuation measurements verified")

    return session_id, session_path


# ============================================================================
# Test 7: Error Handling - Session with Error Marking
# ============================================================================


def test_step7_error_marking(temp_dir, technical_container_path):
    """Step 7: Test error marking and archiving of session containers."""
    print("\n=== STEP 7: Error Marking & Archiving ===")

    # Create session folder
    session_folder = temp_dir / "sessions"
    session_folder.mkdir()

    # Create a session
    session_id, session_path = session_container.create_session_container(
        folder=session_folder,
        sample_id="ERROR_SAMPLE",
        operator_id="test_operator",
        site_id="LAB_A",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-11",
    )

    print(f"  ✓ Created session: {session_id}")

    # Mark session as created by error
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    error_reason = "Wrong sample loaded by mistake"

    with h5py.File(session_path, "a") as f:
        f.attrs["created_by_error"] = True
        f.attrs["error_reason"] = error_reason
        f.attrs["archived_timestamp"] = timestamp

    print(f"  ✓ Marked as error: {error_reason}")

    # Verify error attributes
    with h5py.File(session_path, "r") as f:
        assert f.attrs["created_by_error"] == True
        assert f.attrs["error_reason"] == error_reason
        assert "archived_timestamp" in f.attrs

    print(f"  ✓ Error attributes verified")

    # Archive the session
    archive_base = session_folder / "session_archive"
    archive_folder = archive_base / f"{session_id}_{timestamp}"
    archive_folder.mkdir(parents=True)

    archived_path = archive_folder / Path(session_path).name
    shutil.move(str(session_path), str(archived_path))

    print(f"  ✓ Session archived to: {archive_folder.name}")

    # Verify archived file
    assert archived_path.exists()
    assert not Path(session_path).exists()

    with h5py.File(archived_path, "r") as f:
        assert f.attrs["created_by_error"] == True

    print(f"  ✓ Archived session verified")

    return session_id, archived_path


# ============================================================================
# Test 8: Container Locking
# ============================================================================


def test_step8_lock_container(temp_dir, technical_container_path):
    """Step 8: Lock/finalize session container."""
    print("\n=== STEP 8: Lock/Finalize Session Container ===")

    # Create and populate session
    session_folder = temp_dir / "sessions"
    session_folder.mkdir()

    session_id, session_path = session_container.create_session_container(
        folder=session_folder,
        sample_id="FINAL_SAMPLE",
        operator_id="test_operator",
        site_id="LAB_A",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-11",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_path, session_file=session_path
    )

    # Add a measurement
    session_container.add_point(
        file_path=session_path,
        point_index=1,
        pixel_coordinates=[300.0, 400.0],
        physical_coordinates_mm=[15.0, 20.0],
    )

    poni_alias_map = {"SAXS": "det_saxs", "WAXS": "det_waxs"}
    measurement_data = {
        "det_saxs": np.random.poisson(100, size=(256, 256)).astype(np.float32),
    }
    detector_metadata = {
        "det_saxs": {
            "integration_time_ms": 1000.0,
            "beam_energy_keV": 17.5,
            "detector_id": "det_saxs",
        },
    }

    session_container.add_measurement(
        file_path=session_path,
        point_index=1,
        measurement_data=measurement_data,
        detector_metadata=detector_metadata,
        poni_alias_map=poni_alias_map,
    )

    print(f"  ✓ Session populated with data")

    # Check container is not locked
    assert not container_manager.is_container_locked(session_path)
    print(f"  ✓ Container is unlocked (editable)")

    # Lock the container
    container_manager.lock_container(Path(session_path), user_id="test_operator")

    print(f"  ✓ Container locked")

    # Verify locked status
    assert container_manager.is_container_locked(session_path)

    with h5py.File(session_path, "r") as f:
        assert f.attrs["locked"] == True
        assert f.attrs["locked_by"] == "test_operator"
        assert "locked_timestamp" in f.attrs

    print(f"  ✓ Lock attributes verified")
    print(f"  ✓ Container ready for upload")

    return session_id, session_path


# ============================================================================
# Test 9: Complete End-to-End Workflow
# ============================================================================


def test_step9_complete_workflow(temp_dir, technical_container_path):
    """Step 9: Complete end-to-end session workflow."""
    print("\n=== STEP 9: Complete End-to-End Workflow ===")

    session_folder = temp_dir / "sessions"
    session_folder.mkdir()

    # 1. Create session
    session_id, session_path = session_container.create_session_container(
        folder=session_folder,
        sample_id="COMPLETE_TEST",
        operator_id="test_operator",
        site_id="LAB_A",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-11",
    )
    print(f"  ✓ Step 1: Session created")

    # 2. Copy technical data
    session_container.copy_technical_to_session(
        technical_file=technical_container_path, session_file=session_path
    )
    print(f"  ✓ Step 2: Technical data copied")

    # 3. Add sample image
    image_data = np.random.randint(0, 255, size=(800, 600), dtype=np.uint8)
    session_container.add_image(
        file_path=session_path, image_index=1, image_data=image_data
    )
    print(f"  ✓ Step 3: Sample image added")

    # 4. Define zones
    session_container.add_zone(
        file_path=session_path,
        zone_index=1,
        zone_role="sample_holder",
        geometry_px=[300.0, 400.0, 200.0, 200.0],
        shape="circle",
        holder_diameter_mm=25.4,
    )
    print(f"  ✓ Step 4: Zones defined")

    # 5. Add points
    for i in range(3):
        session_container.add_point(
            file_path=session_path,
            point_index=i + 1,
            pixel_coordinates=[300.0 + i * 50.0, 400.0],
            physical_coordinates_mm=[15.0 + i * 2.5, 20.0],
        )
    print(f"  ✓ Step 5: Points added")

    # 6. Record measurements
    poni_alias_map = {"SAXS": "det_saxs", "WAXS": "det_waxs"}
    for point_idx in range(1, 4):
        measurement_data = {
            "det_saxs": np.random.poisson(100, size=(256, 256)).astype(np.float32),
            "det_waxs": np.random.poisson(200, size=(256, 256)).astype(np.float32),
        }
        detector_metadata = {
            "det_saxs": {
                "integration_time_ms": 1000.0,
                "beam_energy_keV": 17.5,
                "detector_id": "det_saxs",
            },
            "det_waxs": {
                "integration_time_ms": 500.0,
                "beam_energy_keV": 17.5,
                "detector_id": "det_waxs",
            },
        }
        session_container.add_measurement(
            file_path=session_path,
            point_index=point_idx,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
        )
        session_container.update_point_status(
            file_path=session_path, point_index=point_idx, point_status="measured"
        )
    print(f"  ✓ Step 6: Measurements recorded")

    # 7. Record attenuation
    i0_data = {
        "det_saxs": np.random.poisson(500, size=(256, 256)).astype(np.float32),
    }
    i0_metadata = {
        "det_saxs": {
            "integration_time_ms": 100.0,
            "beam_energy_keV": 17.5,
            "detector_id": "det_saxs",
        },
    }
    session_container.add_analytical_measurement(
        file_path=session_path,
        measurement_data=i0_data,
        detector_metadata=i0_metadata,
        poni_alias_map=poni_alias_map,
        analysis_type="attenuation_i0",
    )
    print(f"  ✓ Step 7: Attenuation recorded")

    # 8. Lock container
    container_manager.lock_container(Path(session_path), user_id="test_operator")
    print(f"  ✓ Step 8: Container locked")

    # 9. Verify complete structure
    with h5py.File(session_path, "r") as f:
        assert "/technical" in f
        assert "/images/img_001" in f
        assert "/images/zones/zone_001" in f
        assert "/points/pt_001" in f
        assert "/points/pt_002" in f
        assert "/points/pt_003" in f
        assert "/measurements/pt_001/meas_000000001" in f
        assert "/measurements/pt_002/meas_000000002" in f
        assert "/measurements/pt_003/meas_000000003" in f
        # Analytical measurement uses shared counter (after 3 regular measurements)
        assert "/analytical_measurements/ana_000000004" in f
        assert f.attrs["locked"] == True

    print(f"  ✓ Step 9: Complete structure verified")
    print(f"\n=== ALL WORKFLOW STEPS COMPLETED SUCCESSFULLY ===")

    return session_id, session_path


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
