"""Technical workflow tests: folder layout and H5 loading behavior."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _technical_workflow_shared import *  # noqa: F401,F403
def test_folder_structure_creation(temp_dir):
    """Test that folder helper functions create correct difra folder structure."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from hardware.difra.gui.main_window_ext.technical_measurements import (
        _get_difra_base_folder,
        _get_technical_storage_folder,
        _get_technical_archive_folder,
        _get_measurement_default_folder,
    )
    
    # Create config pointing to temp directory
    config = {
        "difra_base_folder": str(temp_dir / "difra"),
    }
    
    # Test base folder creation
    base_folder = _get_difra_base_folder(config)
    assert Path(base_folder).exists()
    assert Path(base_folder).name == "difra"
    
    # Test technical storage folder
    tech_folder = _get_technical_storage_folder(config)
    assert Path(tech_folder).exists()
    assert Path(tech_folder).name == "technical"
    assert Path(tech_folder).parent.name == "difra"
    
    # Test technical archive folder
    archive_folder = _get_technical_archive_folder(config)
    assert Path(archive_folder).exists()
    assert "archive" in str(archive_folder)
    assert "technical" in str(archive_folder)
    
    # Test measurements folder
    meas_folder = _get_measurement_default_folder(config)
    assert Path(meas_folder).exists()
    assert Path(meas_folder).name == "measurements"
    
    print(f"✅ Folder structure created correctly:")
    print(f"   Base: {base_folder}")
    print(f"   Technical: {tech_folder}")
    print(f"   Archive: {archive_folder}")
    print(f"   Measurements: {meas_folder}")


def test_folder_structure_matches_config(temp_dir):
    """Test that folders match the paths specified in config."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from hardware.difra.gui.main_window_ext.technical_measurements import (
        _get_technical_storage_folder,
        _get_technical_archive_folder,
        _get_measurement_default_folder,
    )
    
    # Explicit config paths
    config = {
        "technical_folder": str(temp_dir / "custom" / "technical"),
        "technical_archive_folder": str(temp_dir / "custom" / "archive" / "technical"),
        "measurements_folder": str(temp_dir / "custom" / "measurements"),
    }
    
    tech_folder = _get_technical_storage_folder(config)
    archive_folder = _get_technical_archive_folder(config)
    meas_folder = _get_measurement_default_folder(config)
    
    # Verify paths match config
    assert tech_folder == config["technical_folder"]
    assert archive_folder == config["technical_archive_folder"]
    assert meas_folder == config["measurements_folder"]
    
    # Verify folders were created
    assert Path(tech_folder).exists()
    assert Path(archive_folder).exists()
    assert Path(meas_folder).exists()
    
    print(f"✅ Config paths correctly applied and folders created")


# ==================== Raw Data Archiving Tests ====================

def test_load_h5_imports_correctly():
    """Test that Load H5 can import the correct validator module."""
    try:
        from hardware.container.v0_1.technical_validator import validate_technical_container
        print(f"✅ Load H5 validator import successful")
        assert validate_technical_container is not None
    except ImportError as e:
        pytest.fail(f"Failed to import validator: {e}")


def test_load_h5_with_valid_container(
    temp_dir, valid_poni_files, sample_measurements, demo_config
):
    """Test loading a valid H5 container."""
    from hardware.container.v0_1.technical_validator import validate_technical_container
    
    # Create container
    container_path = create_valid_container(
        temp_dir, valid_poni_files, sample_measurements, demo_config
    )
    
    # Validate it can be loaded
    is_valid, errors, warnings = validate_technical_container(str(container_path))
    
    assert is_valid, f"Container should be valid for loading: {errors}"
    assert container_path.exists()
    
    # Verify we can read the container
    with h5py.File(container_path, "r") as f:
        assert "technical" in f
        assert "container_id" in f.attrs
    
    print(f"✅ Container loaded and validated successfully")
    print(f"   Errors: {len(errors)}, Warnings: {len(warnings)}")


def test_load_h5_with_invalid_container(temp_dir):
    """Test loading an invalid H5 container shows appropriate errors."""
    from hardware.container.v0_1.technical_validator import validate_technical_container
    
    # Create invalid container
    invalid_h5 = temp_dir / "invalid.h5"
    with h5py.File(invalid_h5, "w") as f:
        f.attrs["container_id"] = "invalid"
        # Missing required attributes and groups
    
    # Validate - should fail
    is_valid, errors, warnings = validate_technical_container(str(invalid_h5))
    
    assert not is_valid, "Invalid container should fail validation"
    assert len(errors) > 0, "Should have validation errors"
    
    print(f"✅ Invalid container correctly identified")
    print(f"   Errors: {len(errors)}")


if __name__ == "__main__":
    # Run tests
    import sys

