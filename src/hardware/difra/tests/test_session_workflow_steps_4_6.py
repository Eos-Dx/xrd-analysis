"""Session workflow tests for steps 4-6 (points, measurements, attenuation)."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _session_workflow_complete_shared import *  # noqa: F401,F403
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

# ============================================================================
# Test 7: Error Handling - Session with Error Marking
# ============================================================================


