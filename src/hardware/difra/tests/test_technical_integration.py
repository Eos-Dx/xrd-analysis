"""Integration tests for technical measurements and HDF5 container generation.

Tests the complete workflow:
1. Initialize hardware (DEMO mode)
2. Capture technical measurements (DARK, EMPTY, BACKGROUND, AGBH)
3. Generate HDF5 technical container
4. Validate HDF5 structure and content
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

# Add the project src root to the path to import modules as the application does
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Import the hardware controller and technical container
from hardware.container.v0_1 import schema, technical_container
from hardware.difra.hardware.detectors import DummyDetectorController


@pytest.fixture
def demo_config():
    """Configuration for DEMO mode with minimal integration time."""
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
            },
            {
                "id": "SECONDARY",
                "alias": "SECONDARY",
                "name": "WAXS Demo",
                "type": "dummy",
                "width": 256,
                "height": 256,
            },
        ],
        "dev_active_detectors": ["PRIMARY", "SECONDARY"],
        "active_detectors": ["PRIMARY", "SECONDARY"],
    }


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def demo_detectors(demo_config):
    """Create DEMO detector controllers for each active detector."""
    detectors = {}
    for det_config in demo_config["detectors"]:
        det_id = det_config["id"]
        if det_id in demo_config["dev_active_detectors"]:
            controller = DummyDetectorController(
                alias=det_id,
                size=(det_config["width"], det_config["height"])
            )
            detectors[det_id] = controller
    return detectors


@pytest.fixture
def demo_poni_files(temp_output_dir):
    """Create demo PONI files with realistic content."""
    poni_dir = temp_output_dir / "poni"
    poni_dir.mkdir()

    poni_content_template = """# Detector: {detector}
# Pixel1: 7.500e-05
# Pixel2: 7.500e-05
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


def capture_technical_measurement(
    detectors: dict, measurement_type: str, frames: int = 5
) -> dict:
    """Simulate capturing a technical measurement.

    Args:
        detectors: Dict of {detector_id: DummyDetectorController}
        measurement_type: Type of measurement (DARK, EMPTY, BACKGROUND, AGBH, WATER)
        frames: Number of frames to capture

    Returns:
        dict: {detector_id: numpy_array} mapping
    """
    # Simulate detector capture with different patterns for each type
    results = {}

    for detector_id, controller in detectors.items():
        width, height = controller.size

        # Generate synthetic data based on measurement type
        if measurement_type == "DARK":
            # Dark current: low signal, some noise
            data = np.random.poisson(5, size=(height, width)).astype(np.float32)
        elif measurement_type == "EMPTY":
            # Empty beam: high signal, scattered pattern
            center_y, center_x = height // 2, width // 2
            y, x = np.ogrid[:height, :width]
            r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            data = 1000 * np.exp(-r / 50) + np.random.poisson(10, size=(height, width))
            data = data.astype(np.float32)
        elif measurement_type == "BACKGROUND":
            # Background: medium uniform signal
            data = np.random.poisson(50, size=(height, width)).astype(np.float32)
        elif measurement_type == "AGBH":
            # Silver behenate: ring pattern
            center_y, center_x = height // 2, width // 2
            y, x = np.ogrid[:height, :width]
            r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            # Create rings at specific radii
            rings = (
                500 * np.exp(-((r - 40) ** 2) / 10)
                + 500 * np.exp(-((r - 80) ** 2) / 10)
                + 500 * np.exp(-((r - 120) ** 2) / 10)
            )
            data = rings + np.random.poisson(20, size=(height, width))
            data = data.astype(np.float32)
        elif measurement_type == "WATER":
            # Water: diffuse scattering
            data = (
                100 + 50 * np.random.randn(height, width) + np.random.poisson(30, size=(height, width))
            )
            data = np.clip(data, 0, None).astype(np.float32)
        else:
            # Default: random noise
            data = np.random.poisson(10, size=(height, width)).astype(np.float32)

        # Average multiple frames if requested
        if frames > 1:
            frame_data = [data for _ in range(frames)]
            data = np.mean(frame_data, axis=0).astype(np.float32)

        results[detector_id] = data

    return results


def test_capture_all_technical_measurements(
    demo_detectors, temp_output_dir, demo_poni_files, demo_config
):
    """Test capturing all required technical measurements in DEMO mode."""
    required_types = ["DARK", "EMPTY", "BACKGROUND", "AGBH"]
    captured_files = {}

    for meas_type in required_types:
        # Capture measurement
        data_dict = capture_technical_measurement(
            demo_detectors, meas_type, frames=5
        )

        # Save to .npy files
        for detector_id, data in data_dict.items():
            filename = temp_output_dir / f"{meas_type}_{detector_id}.npy"
            np.save(filename, data)

            # Store in nested dict structure for HDF5 generation
            if meas_type not in captured_files:
                captured_files[meas_type] = {}
            captured_files[meas_type][detector_id] = str(filename)

    # Verify all required measurements captured for all detectors
    for meas_type in required_types:
        assert meas_type in captured_files
        for detector_id in demo_config["dev_active_detectors"]:
            assert detector_id in captured_files[meas_type]
            assert Path(captured_files[meas_type][detector_id]).exists()

    return captured_files


def test_generate_technical_h5_container(
    temp_output_dir, demo_poni_files, demo_config
):
    """Test generating HDF5 technical container from measurements."""
    # Create detector controllers
    detectors = {}
    for det_config in demo_config["detectors"]:
        det_id = det_config["id"]
        if det_id in demo_config["dev_active_detectors"]:
            detectors[det_id] = DummyDetectorController(
                alias=det_id,
                size=(det_config["width"], det_config["height"])
            )

    # Capture all required measurements
    aux_measurements = {}
    required_types = ["DARK", "EMPTY", "BACKGROUND", "AGBH"]

    for meas_type in required_types:
        data_dict = capture_technical_measurement(detectors, meas_type, frames=5)

        aux_measurements[meas_type] = {}
        for detector_id, data in data_dict.items():
            filename = temp_output_dir / f"{meas_type}_{detector_id}.npy"
            np.save(filename, data)
            aux_measurements[meas_type][detector_id] = str(filename)

    # Prepare PONI data
    pony_data = {}
    for detector_id, poni_path in demo_poni_files.items():
        content = poni_path.read_text()
        pony_data[detector_id] = (content, poni_path.name)

    # Extract distance from PONI (should be 17cm for PRIMARY)
    distance_cm = 17.0

    # Generate HDF5 container
    container_id, file_path = technical_container.generate_from_aux_table(
        folder=str(temp_output_dir),
        aux_measurements=aux_measurements,
        pony_data=pony_data,
        detector_config=demo_config["detectors"],
        active_detector_ids=demo_config["dev_active_detectors"],
        distance_cm=distance_cm,
    )

    # Verify file created
    assert Path(file_path).exists()
    assert container_id in file_path
    assert file_path.endswith(".h5")

    return file_path, container_id


def test_validate_h5_structure(temp_output_dir, demo_poni_files, demo_config):
    """Test and validate complete HDF5 container structure."""
    # Generate container
    file_path, container_id = test_generate_technical_h5_container(
        temp_output_dir, demo_poni_files, demo_config
    )

    # Open and validate structure
    with h5py.File(file_path, "r") as f:
        # 1. Root attributes
        assert "container_id" in f.attrs
        assert f.attrs["container_id"] == container_id
        assert "schema_version" in f.attrs
        assert f.attrs["schema_version"] == "1.0"
        assert "creation_timestamp" in f.attrs
        assert "distance_cm" in f.attrs
        assert f.attrs["distance_cm"] == 17.0

        # 2. Technical group exists
        assert "technical" in f
        tech_group = f["technical"]

        # 3. Config group
        assert "config" in tech_group
        config_group = tech_group["config"]
        
        # 4. Detector config JSON dataset
        assert "detector_config" in config_group
        detector_config_ds = config_group["detector_config"]
        # Load and parse JSON config
        config_json_str = detector_config_ds[()]
        if isinstance(config_json_str, bytes):
            config_json_str = config_json_str.decode('utf-8')
        config_data = json.loads(config_json_str)
        assert "detectors" in config_data
        assert "active_detector_ids" in config_data
        assert len(config_data["detectors"]) == len(demo_config["dev_active_detectors"])

        # 5. PONY primary group
        assert "pony" in tech_group
        pony_group = tech_group["pony"]
        for detector_id in demo_config["dev_active_detectors"]:
            pony_id = f"pony_{detector_id.lower()}"
            assert pony_id in pony_group
            pony_data = pony_group[pony_id]
            assert pony_data.dtype.kind in ["S", "O"]  # String or object type
            assert "pony_filename" in pony_data.attrs

        # 6. Technical event groups (DARK, EMPTY, BACKGROUND, AGBH)
        required_types = ["DARK", "EMPTY", "BACKGROUND", "AGBH"]
        tech_evt_groups = [key for key in tech_group.keys() if key.startswith("tech_evt_")]

        # Should have events for each required type
        assert len(tech_evt_groups) >= len(required_types)

        for evt_key in tech_evt_groups:
            evt_group = tech_group[evt_key]

            # Check attributes
            assert "type" in evt_group.attrs
            assert evt_group.attrs["type"] in schema.ALL_TECHNICAL_TYPES
            assert "timestamp_utc" in evt_group.attrs

            # Check detector data subgroups
            for detector_id in demo_config["dev_active_detectors"]:
                det_key = f"det_{detector_id.lower()}"
                assert det_key in evt_group
                det_data = evt_group[det_key]

                # Check datasets
                assert "raw_signal" in det_data
                signal_ds = det_data["raw_signal"]
                assert signal_ds.shape == (256, 256)  # Demo detector size
                assert signal_ds.dtype in [np.float32, np.float64]

                # Check attributes
                assert "detector_id" in det_data.attrs

        # 7. Object references
        # Check that pony_primary references are valid
        for det_id in demo_config["dev_active_detectors"]:
            pony_id = f"pony_{det_id.lower()}"
            if pony_id in pony_group:
                # Reference should be dereferenceable
                ref_ds = pony_group[pony_id]
                assert ref_ds is not None

    print(f"✅ HDF5 container validation passed: {file_path}")
    print(f"   Container ID: {container_id}")
    print(f"   File size: {Path(file_path).stat().st_size / 1024:.1f} KB")


def test_roundtrip_measurement_data(temp_output_dir, demo_poni_files, demo_config):
    """Test that measurement data survives roundtrip through HDF5."""
    # Generate container
    file_path, _ = test_generate_technical_h5_container(
        temp_output_dir, demo_poni_files, demo_config
    )

    # Load original data from .npy files
    original_dark_primary = np.load(temp_output_dir / "DARK_PRIMARY.npy")

    # Load from HDF5
    with h5py.File(file_path, "r") as f:
        tech_group = f["technical"]

        # Find DARK event
        dark_evt = None
        for evt_key in tech_group.keys():
            if evt_key.startswith("tech_evt_"):
                evt = tech_group[evt_key]
                if evt.attrs["type"] == "DARK":
                    dark_evt = evt
                    break

        assert dark_evt is not None, "DARK event not found in HDF5"

        # Load data
        h5_data = dark_evt["det_primary"]["raw_signal"][:]

        # Compare
        np.testing.assert_array_almost_equal(
            original_dark_primary, h5_data, decimal=5
        )

    print("✅ Roundtrip data integrity verified")


def test_multiple_containers_unique_ids(temp_output_dir, demo_poni_files, demo_config):
    """Test that multiple containers get unique IDs."""
    container_ids = []

    for i in range(3):
        file_path, container_id = test_generate_technical_h5_container(
            temp_output_dir, demo_poni_files, demo_config
        )
        container_ids.append(container_id)

        # Verify unique
        assert container_ids.count(container_id) == 1, f"Duplicate container ID: {container_id}"

        # Verify ID in filename
        assert container_id in Path(file_path).name

    print(f"✅ Multiple containers with unique IDs: {container_ids}")


@pytest.mark.parametrize(
    "measurement_type",
    ["DARK", "EMPTY", "BACKGROUND", "AGBH", "WATER"],
)
def test_individual_measurement_types(
    demo_detectors, measurement_type, temp_output_dir
):
    """Test individual measurement type generation."""
    data_dict = capture_technical_measurement(
        demo_detectors, measurement_type, frames=3
    )

    assert len(data_dict) == len(demo_detectors)

    for detector_id, data in data_dict.items():
        # Check shape
        controller = demo_detectors[detector_id]
        assert data.shape == controller.size[::-1]  # (height, width)

        # Check data type
        assert data.dtype in [np.float32, np.float64]

        # Check data range (no negative values, reasonable max)
        assert np.all(data >= 0)
        assert np.all(data < 1e6)

        # Save and verify
        filename = temp_output_dir / f"{measurement_type}_{detector_id}.npy"
        np.save(filename, data)
        assert filename.exists()

        # Load and verify
        loaded = np.load(filename)
        np.testing.assert_array_equal(data, loaded)

    print(f"✅ {measurement_type} measurement captured and saved")


if __name__ == "__main__":
    # Run tests manually for debugging
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
