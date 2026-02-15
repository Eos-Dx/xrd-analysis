"""Technical workflow tests: primary assignment and end-to-end workflow checks."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _technical_workflow_shared import *  # noqa: F401,F403
def test_set_measurement_primary_status(
    temp_dir, valid_poni_files, sample_measurements, demo_config
):
    """Test marking measurements as primary or supplementary."""
    container_path = create_valid_container(
        temp_dir, valid_poni_files, sample_measurements, demo_config
    )

    # Mark event 1 as supplementary
    container_manager.set_measurement_primary_status(
        container_path, event_index=1, is_primary=False, note="Backup measurement"
    )

    # Verify it was set
    with h5py.File(container_path, "r") as f:
        event_path = "technical/tech_evt_001"
        assert not f[event_path].attrs["is_primary"]
        assert f[event_path].attrs["supplementary_note"] == "Backup measurement"

    print(f"✅ Primary status set correctly")


def test_get_primary_measurements(
    temp_dir, valid_poni_files, sample_measurements, demo_config
):
    """Test retrieving primary measurements from container."""
    container_path = create_valid_container(
        temp_dir, valid_poni_files, sample_measurements, demo_config
    )

    # All measurements should be primary by default
    primary_events = container_manager.get_primary_measurements(container_path)

    assert len(primary_events) > 0, "Should have primary events"
    assert "DARK" in primary_events
    assert "EMPTY" in primary_events
    assert "BACKGROUND" in primary_events
    assert "AGBH" in primary_events

    print(f"✅ Primary measurements retrieved: {primary_events}")


# ==================== Integration: Full Workflow ====================

def test_complete_workflow_with_validation_and_locking(
    temp_dir, valid_poni_files, sample_measurements, demo_config
):
    """Test complete workflow: create → validate → lock → verify."""
    # Step 1: Create container
    container_path = create_valid_container(
        temp_dir, valid_poni_files, sample_measurements, demo_config
    )
    assert container_path.exists()

    # Step 2: Validate container
    is_valid, errors, warnings = technical_validator.validate_technical_container(
        str(container_path)
    )
    assert is_valid, f"Container should be valid: {errors}"

    # Step 3: Lock container
    operator_id = "workflow_test@example.com"
    container_manager.lock_technical_container(
        container_path, locked_by=operator_id, notes="Production ready"
    )

    # Step 4: Verify lock
    assert container_manager.is_container_locked(container_path)
    lock_info = container_manager.get_lock_info(container_path)
    assert lock_info["locked_by"] == operator_id
    assert lock_info["locked_notes"] == "Production ready"

    # Step 5: Verify read-only
    locked_perms = container_path.stat().st_mode
    assert not (locked_perms & stat.S_IWUSR)

    print(f"✅ Complete workflow validated successfully")
    print(f"   Container: {container_path.name}")
    print(f"   Locked by: {lock_info['locked_by']}")
    print(f"   Timestamp: {lock_info['locked_timestamp']}")


def test_workflow_fails_on_invalid_container(temp_dir, demo_config):
    """Test that workflow correctly rejects invalid containers."""
    # Create invalid container (missing required measurements)
    container_path = temp_dir / "invalid.h5"

    with h5py.File(container_path, "w") as f:
        f.attrs["container_id"] = "invalid123"
        f.attrs["container_type"] = "technical"
        f.attrs["schema_version"] = "0.1"
        f.attrs["creation_timestamp"] = "2024-01-01T00:00:00Z"
        f.attrs["distance_cm"] = 100.0
        f.create_group("technical")

    # Validation should fail
    is_valid, errors, warnings = technical_validator.validate_technical_container(
        str(container_path)
    )
    assert not is_valid, "Invalid container should fail validation"

    # Should not be able to lock invalid container (but no built-in check, so will succeed)
    # In production, UI should prevent locking invalid containers
    print(f"✅ Workflow correctly identifies invalid container with {len(errors)} errors")


# ==================== Business Logic Validation Tests ====================

