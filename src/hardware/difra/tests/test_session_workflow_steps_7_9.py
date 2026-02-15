"""Session workflow tests for steps 7-9 (error marking, lock, complete flow)."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _session_workflow_complete_shared import *  # noqa: F401,F403
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
