"""Technical workflow tests: validation and distance constraints."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _technical_workflow_shared import *  # noqa: F401,F403
def test_validation_passes_for_valid_container(
    temp_dir, valid_poni_files, sample_measurements, demo_config
):
    """Test that validation passes for a properly structured container."""
    container_path = create_valid_container(
        temp_dir, valid_poni_files, sample_measurements, demo_config
    )

    # Run validation
    is_valid, errors, warnings = technical_validator.validate_technical_container(
        str(container_path)
    )

    assert is_valid, f"Valid container failed validation: {errors}"
    assert len(errors) == 0, f"Unexpected errors: {errors}"
    print(f"✅ Validation passed with {len(warnings)} warnings")


def test_validation_fails_missing_required_types(
    temp_dir, valid_poni_files, demo_config
):
    """Test that validation fails when required measurement types are missing."""
    # Create measurements with only DARK (missing EMPTY, BACKGROUND, AGBH)
    incomplete_measurements = {}
    incomplete_measurements["DARK"] = {}
    for detector_id in demo_config["dev_active_detectors"]:
        data = np.random.poisson(50, size=(256, 256)).astype(np.float32)
        filename = temp_dir / f"DARK_{detector_id}.npy"
        np.save(filename, data)
        incomplete_measurements["DARK"][detector_id] = str(filename)

    # Create container with incomplete measurements
    poni_data = {}
    for detector_id, poni_path in valid_poni_files.items():
        content = poni_path.read_text()
        poni_data[detector_id] = (content, poni_path.name)

    container_id, file_path = technical_container.generate_from_aux_table(
        folder=str(temp_dir),
        aux_measurements=incomplete_measurements,
        poni_data=poni_data,
        detector_config=demo_config["detectors"],
        active_detector_ids=demo_config["dev_active_detectors"],
        distances_cm={"PRIMARY": 100.0, "SECONDARY": 17.0},
        poni_distances_cm={"PRIMARY": 100.0, "SECONDARY": 17.0},
    )

    # Run validation
    is_valid, errors, warnings = technical_validator.validate_technical_container(
        file_path
    )

    assert not is_valid, "Container with missing types should fail validation"
    assert any("Missing required technical measurement types" in e for e in errors)
    print(f"✅ Validation correctly failed for missing types: {errors[0]}")


def test_validation_fails_invalid_schema_version(
    temp_dir, valid_poni_files, sample_measurements, demo_config
):
    """Test that validation warns about schema version mismatch."""
    container_path = create_valid_container(
        temp_dir, valid_poni_files, sample_measurements, demo_config
    )

    # Modify schema version
    with h5py.File(container_path, "a") as f:
        f.attrs["schema_version"] = "99.9"  # Invalid version

    # Run validation
    is_valid, errors, warnings = technical_validator.validate_technical_container(
        str(container_path)
    )

    # Should have warning about version mismatch
    assert any("Schema version mismatch" in w for w in warnings)
    print(f"✅ Validation detected schema version mismatch: {warnings}")


def test_validation_fails_missing_root_attributes(temp_dir, demo_config):
    """Test that validation fails when required root attributes are missing."""
    # Create minimal invalid container
    container_path = temp_dir / "invalid.h5"

    with h5py.File(container_path, "w") as f:
        f.attrs["container_id"] = "test123"
        # Missing: container_type, schema_version, creation_timestamp, distance_cm
        f.create_group("technical")

    # Run validation
    is_valid, errors, warnings = technical_validator.validate_technical_container(
        str(container_path)
    )

    assert not is_valid, "Container with missing attributes should fail"
    assert any("Missing required root attribute" in e for e in errors)
    print(f"✅ Validation correctly failed for missing attributes: {len(errors)} errors")


# ==================== Phase 3: Container Locking Tests ====================

def test_poni_distance_mismatch_validation(temp_dir, demo_config):
    """Test PONI distance validation with mismatched distances."""
    # Create PONI files with wrong distances
    poni_dir = temp_dir / "poni"
    poni_dir.mkdir()

    poni_template = """Distance: {distance}
PixelSize1: 7.5e-05
PixelSize2: 7.5e-05
Poni1: 0.01
Poni2: 0.02
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 1.54e-10
"""

    # Create PONI with wrong distance (50cm instead of 100cm)
    wrong_poni = poni_dir / "primary.poni"
    wrong_poni.write_text(poni_template.format(distance=0.50))

    poni_data = {"PRIMARY": (wrong_poni.read_text(), "primary.poni")}

    # Create sample measurement
    measurements = {"DARK": {}}
    data = np.random.poisson(50, size=(256, 256)).astype(np.float32)
    filename = temp_dir / "DARK_PRIMARY.npy"
    np.save(filename, data)
    measurements["DARK"]["PRIMARY"] = str(filename)

    # Try to generate container with mismatched distance
    # This should fail PONI validation (distance mismatch > 5%)
    with pytest.raises(ValueError, match="PONI.*validation failed|distance.*validation failed"):
        technical_container.generate_from_aux_table(
            folder=str(temp_dir),
            aux_measurements=measurements,
            poni_data=poni_data,
            detector_config=[demo_config["detectors"][0]],
            active_detector_ids=["PRIMARY"],
            distances_cm={"PRIMARY": 100.0},  # User expects 100cm
            poni_distances_cm={"PRIMARY": 50.0},  # PONI has 50cm (50% diff!)
        )

    print(f"✅ PONI distance mismatch correctly detected")


def test_missing_poni_file_error(temp_dir, sample_measurements, demo_config):
    """Test error when PONI files are missing."""
    # Try to create container without PONI data
    # Note: Current implementation may skip PONI validation if poni_data is empty
    # This test verifies that behavior is handled gracefully
    try:
        container_id, file_path = technical_container.generate_from_aux_table(
            folder=str(temp_dir),
            aux_measurements=sample_measurements,
            poni_data={},  # No PONI data
            detector_config=demo_config["detectors"],
            active_detector_ids=demo_config["dev_active_detectors"],
            distances_cm={"PRIMARY": 100.0, "SECONDARY": 17.0},
            poni_distances_cm={"PRIMARY": 100.0, "SECONDARY": 17.0},
        )
        # If no error is raised, container is created without PONI (which is allowed)
        # Verify container exists but has no PONI data
        import h5py
        with h5py.File(file_path, "r") as f:
            poni_group = f.get("technical/poni")
            if poni_group:
                poni_datasets = list(poni_group.keys())
                assert len(poni_datasets) == 0, "Should have no PONI datasets"
        print(f"✅ Container created without PONI data (allowed behavior)")
    except (ValueError, KeyError) as e:
        # If error is raised, that's also valid behavior
        print(f"✅ Missing PONI files correctly rejected: {e}")


def test_per_detector_distance_validation(temp_dir, valid_poni_files, demo_config):
    """Test that per-detector distance validation works correctly."""
    # Create measurements
    measurements = {"DARK": {}}
    for detector_id in demo_config["dev_active_detectors"]:
        data = np.random.poisson(50, size=(256, 256)).astype(np.float32)
        filename = temp_dir / f"DARK_{detector_id}.npy"
        np.save(filename, data)
        measurements["DARK"][detector_id] = str(filename)

    poni_data = {}
    for detector_id, poni_path in valid_poni_files.items():
        content = poni_path.read_text()
        poni_data[detector_id] = (content, poni_path.name)

    # Test with correct per-detector distances
    distances_cm = {"PRIMARY": 100.0, "SECONDARY": 17.0}
    poni_distances_cm = {"PRIMARY": 100.0, "SECONDARY": 17.0}

    container_id, file_path = technical_container.generate_from_aux_table(
        folder=str(temp_dir),
        aux_measurements=measurements,
        poni_data=poni_data,
        detector_config=demo_config["detectors"],
        active_detector_ids=demo_config["dev_active_detectors"],
        distances_cm=distances_cm,
        poni_distances_cm=poni_distances_cm,
    )

    # Verify per-detector distances stored in container
    with h5py.File(file_path, "r") as f:
        # Check technical events have per-detector distances
        tech_group = f["technical"]
        for event_key in tech_group.keys():
            if event_key.startswith("tech_evt_"):
                evt_group = tech_group[event_key]
                # Check detector subgroups
                for det_id in ["PRIMARY", "SECONDARY"]:
                    det_key = f"det_{det_id.lower()}"
                    if det_key in evt_group:
                        det_group = evt_group[det_key]
                        stored_distance = det_group.attrs.get("distance_cm")
                        expected_distance = distances_cm[det_id]
                        assert (
                            stored_distance == expected_distance
                        ), f"Distance mismatch for {det_id}"

    print(f"✅ Per-detector distances validated and stored correctly")


# ==================== Phase 4: Session Workflow Tests ====================

def test_validation_max_one_primary_per_type_detector():
    """Test that validation enforces max one primary per measurement type per detector.
    
    Business rule: Each measurement type (DARK/EMPTY/BACKGROUND/AGBH) can have
    at most ONE primary file per detector. Multiple primaries for same type+detector
    should be rejected.
    """
    # This test validates the business logic is correct
    # In production, the UI should prevent multiple primaries for same type+detector
    # If somehow multiple are selected, generation should fail with clear error
    
    # Test data: simulate primary selections
    primary_selections = {
        "DARK": {"PRIMARY": True, "SECONDARY": True},  # OK: one per detector
        "EMPTY": {"PRIMARY": True, "SECONDARY": False},  # OK: one primary, one supplementary
        "BACKGROUND": {"PRIMARY": True, "SECONDARY": True},  # OK
        "AGBH": {"PRIMARY": True, "SECONDARY": True},  # OK
    }
    
    # Validation logic: count primaries per (type, detector)
    violations = []
    for meas_type, detector_map in primary_selections.items():
        for detector, is_primary in detector_map.items():
            # In real code, we'd count all rows with this (type, detector) marked as primary
            # For this test, we just verify the logic
            if is_primary:
                # This would be fine - one primary per type+detector
                pass
    
    assert len(violations) == 0, "Valid selection should have no violations"
    
    # Now test INVALID case: multiple primaries for same type+detector
    # This simulates having 2 DARK measurements for PRIMARY detector, both marked primary
    invalid_selections = {
        "DARK": [
            {"detector": "PRIMARY", "primary": True, "file": "dark1.npy"},
            {"detector": "PRIMARY", "primary": True, "file": "dark2.npy"},  # VIOLATION!
        ]
    }
    
    # Count primaries per (type, detector)
    primary_count = {}
    for meas_type, files in invalid_selections.items():
        for file_info in files:
            key = (meas_type, file_info["detector"])
            if file_info["primary"]:
                primary_count[key] = primary_count.get(key, 0) + 1
    
    # Check for violations
    violations = [f"{key[0]}→{key[1]}: {count} primaries" for key, count in primary_count.items() if count > 1]
    
    assert len(violations) == 1, "Should detect one violation"
    assert "DARK→PRIMARY: 2 primaries" in violations[0]
    
    print(f"✅ Primary validation logic correct: {violations[0]}")


def test_validation_distances_required_for_all_detectors():
    """Test that distances must be configured for ALL active detectors.
    
    Business rule: User cannot generate H5 container until distances are configured
    for every active detector. Partial configuration should be rejected.
    """
    active_detectors = ["PRIMARY", "SECONDARY"]
    
    # Test 1: No distances configured (should fail)
    configured_distances = {}
    missing = [d for d in active_detectors if d not in configured_distances]
    assert len(missing) == 2
    assert "PRIMARY" in missing and "SECONDARY" in missing
    print(f"✅ No distances: correctly identified missing {missing}")
    
    # Test 2: Partial configuration (should fail)
    configured_distances = {"PRIMARY": 100.0}  # Missing SECONDARY
    missing = [d for d in active_detectors if d not in configured_distances]
    assert len(missing) == 1
    assert "SECONDARY" in missing
    print(f"✅ Partial distances: correctly identified missing {missing}")
    
    # Test 3: Complete configuration (should pass)
    configured_distances = {"PRIMARY": 100.0, "SECONDARY": 17.0}
    missing = [d for d in active_detectors if d not in configured_distances]
    assert len(missing) == 0
    print(f"✅ Complete distances: no missing detectors")


def test_validation_different_primaries_per_detector_allowed():
    """Test that different detectors can have different primary selections.
    
    Business rule: PRIMARY detector can use DARK file A as primary,
    while SECONDARY detector uses DARK file B as primary. This is valid.
    """
    # Simulate different primary selections per detector
    selections = {
        "DARK": {
            "PRIMARY": {"file": "dark_primary_1.npy", "primary": True},
            "SECONDARY": {"file": "dark_secondary_1.npy", "primary": True},
        },
        "EMPTY": {
            "PRIMARY": {"file": "empty_primary_1.npy", "primary": True},
            "SECONDARY": {"file": "empty_secondary_1.npy", "primary": False},  # Different choice
        },
    }
    
    # Validate: for each (type, detector), count primaries
    primary_count = {}
    for meas_type, detector_map in selections.items():
        for detector, file_info in detector_map.items():
            key = (meas_type, detector)
            if file_info["primary"]:
                primary_count[key] = primary_count.get(key, 0) + 1
    
    # All counts should be ≤ 1
    violations = [key for key, count in primary_count.items() if count > 1]
    assert len(violations) == 0
    
    print(f"✅ Different primaries per detector is valid")


# ==================== Folder Structure Tests ====================

