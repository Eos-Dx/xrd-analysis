"""Tests for SessionManager GUI integration module."""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

# Add project src to path
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hardware.difra.gui.session_manager import SessionManager
from hardware.container.v0_2 import schema
from hardware.container.v0_2.technical_container import generate_from_aux_table
from hardware.container.v0_2.container_manager import lock_container


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def technical_container(temp_dir):
    """Create and lock a technical container."""
    # Create minimal technical container
    poni_content = """Detector: AdvaPIX
PixelSize1: 5.500e-05
PixelSize2: 5.500e-05
Distance: 0.170000
Poni1: 0.014025
Poni2: 0.014025
"""
    
    # Create dummy measurement files
    dark_file = temp_dir / "dark.npy"
    np.save(dark_file, np.random.rand(256, 256).astype(np.float32))
    
    tech_id, tech_path = generate_from_aux_table(
        folder=temp_dir,
        aux_measurements={"DARK": {"DET1": str(dark_file)}},
        poni_data={"DET1": (poni_content, "DET1_17cm.poni")},
        detector_config=[{
            "id": "DET1",
            "alias": "DET1",
            "type": "AdvaPIX",
            "size": [256, 256],
            "pixel_size_um": 55.0,
        }],
        active_detector_ids=["DET1"],
        distances_cm=17.0,
        validate_poni=True,
    )
    
    # Lock it
    lock_container(tech_path)
    
    return Path(tech_path)


def test_session_manager_create_session(temp_dir, technical_container):
    """Test creating a new session."""
    manager = SessionManager(config={"technical_folder": str(temp_dir)})
    
    # Initially no session
    assert not manager.is_session_active()
    
    # Create session
    session_id, session_path = manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_001",
        distance_cm=17.0,
        operator_id="test_operator",
    )
    
    # Session is now active
    assert manager.is_session_active()
    assert session_path.exists()
    assert manager.sample_id == "TEST_SAMPLE_001"
    assert manager.study_name == "UNSPECIFIED"
    assert manager.session_id == session_id


def test_session_manager_create_session_with_study(temp_dir, technical_container):
    """Test creating a session with explicit study_name."""
    manager = SessionManager(config={"technical_folder": str(temp_dir)})
    session_id, session_path = manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_002",
        study_name="STUDY_X",
        distance_cm=17.0,
        operator_id="test_operator",
    )

    assert session_path.exists()
    assert manager.session_id == session_id
    assert manager.study_name == "STUDY_X"

    with h5py.File(session_path, "r") as session_file:
        assert session_file.attrs.get(schema.ATTR_STUDY_NAME) == "STUDY_X"


def test_session_manager_add_points(temp_dir, technical_container):
    """Test adding points to session."""
    manager = SessionManager(config={"technical_folder": str(temp_dir)})
    manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_001",
        distance_cm=17.0,
    )
    
    # Add points
    points = [
        {
            "pixel_coordinates": [100, 200],
            "physical_coordinates_mm": [10.0, 20.0],
        },
        {
            "pixel_coordinates": [150, 250],
            "physical_coordinates_mm": [15.0, 25.0],
        },
    ]
    
    paths = manager.add_points(points)
    assert len(paths) == 2


def test_session_manager_attenuation_workflow(temp_dir, technical_container):
    """Test complete attenuation workflow."""
    manager = SessionManager(config={"technical_folder": str(temp_dir)})
    manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_001",
        distance_cm=17.0,
    )
    
    # Add points
    points = [
        {
            "pixel_coordinates": [100, 200],
            "physical_coordinates_mm": [10.0, 20.0],
        },
    ]
    manager.add_points(points)
    
    # Add I₀ measurement (without sample)
    i0_data = {"DET1": np.random.randint(800, 1000, (256, 256), dtype=np.uint16)}
    i0_metadata = {"DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5}}
    
    i0_counter = manager.add_attenuation_measurement(
        measurement_data=i0_data,
        detector_metadata=i0_metadata,
        poni_alias_map={"DET1": "DET1"},
        mode="without",
    )
    
    assert i0_counter == 1
    assert manager.i0_counter == 1
    
    # Add I measurement (with sample)
    i_data = {"DET1": np.random.randint(400, 600, (256, 256), dtype=np.uint16)}
    i_metadata = {"DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5}}
    
    i_counter = manager.add_attenuation_measurement(
        measurement_data=i_data,
        detector_metadata=i_metadata,
        poni_alias_map={"DET1": "DET1"},
        mode="with",
    )
    
    assert i_counter == 2
    assert manager.i_counter == 2
    
    # Link to points
    manager.link_attenuation_to_points(num_points=1)
    
    # Get session info
    info = manager.get_session_info()
    assert info["attenuation_complete"] is True


def test_session_manager_link_attenuation_start_point(temp_dir, technical_container):
    """Test linking attenuation measurements starting from a specific point index."""
    manager = SessionManager(config={"technical_folder": str(temp_dir)})
    manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_001",
        distance_cm=17.0,
    )

    manager.add_points(
        [
            {"pixel_coordinates": [100, 200], "physical_coordinates_mm": [10.0, 20.0]},
            {"pixel_coordinates": [110, 210], "physical_coordinates_mm": [11.0, 21.0]},
            {"pixel_coordinates": [120, 220], "physical_coordinates_mm": [12.0, 22.0]},
        ]
    )

    i0_data = {"DET1": np.random.randint(800, 1000, (256, 256), dtype=np.uint16)}
    i0_metadata = {"DET1": {"integration_time_ms": 50.0}}
    i_data = {"DET1": np.random.randint(400, 600, (256, 256), dtype=np.uint16)}
    i_metadata = {"DET1": {"integration_time_ms": 50.0}}
    poni_alias_map = {"DET1": "DET1"}

    manager.add_attenuation_measurement(
        measurement_data=i0_data,
        detector_metadata=i0_metadata,
        poni_alias_map=poni_alias_map,
        mode="without",
    )
    manager.add_attenuation_measurement(
        measurement_data=i_data,
        detector_metadata=i_metadata,
        poni_alias_map=poni_alias_map,
        mode="with",
    )

    manager.link_attenuation_to_points(num_points=1, start_point_idx=2)

    with h5py.File(manager.session_path, "r") as session_file:
        pt1 = session_file[f"{schema.GROUP_POINTS}/pt_001"]
        pt2 = session_file[f"{schema.GROUP_POINTS}/pt_002"]
        pt3 = session_file[f"{schema.GROUP_POINTS}/pt_003"]

        assert schema.ATTR_ANALYTICAL_MEASUREMENT_IDS in pt1.attrs
        assert schema.ATTR_ANALYTICAL_MEASUREMENT_IDS in pt2.attrs
        assert schema.ATTR_ANALYTICAL_MEASUREMENT_IDS in pt3.attrs

        refs_pt1 = pt1.attrs[schema.ATTR_ANALYTICAL_MEASUREMENT_IDS]
        refs_pt2 = pt2.attrs[schema.ATTR_ANALYTICAL_MEASUREMENT_IDS]
        refs_pt3 = pt3.attrs[schema.ATTR_ANALYTICAL_MEASUREMENT_IDS]
        assert len(refs_pt1) == 0
        assert len(refs_pt2) == 2
        assert len(refs_pt3) == 0


def test_session_manager_add_measurement(temp_dir, technical_container):
    """Test adding regular measurements."""
    manager = SessionManager(config={"technical_folder": str(temp_dir)})
    manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_001",
        distance_cm=17.0,
    )
    
    # Add point
    points = [{"pixel_coordinates": [100, 200], "physical_coordinates_mm": [10.0, 20.0]}]
    manager.add_points(points)
    
    # Add measurement at point 1
    meas_data = {"DET1": np.random.randint(0, 100, (256, 256), dtype=np.uint16)}
    meas_metadata = {"DET1": {"integration_time_ms": 1000.0, "beam_energy_keV": 17.5}}
    
    meas_path = manager.add_measurement(
        point_index=1,
        measurement_data=meas_data,
        detector_metadata=meas_metadata,
        poni_alias_map={"DET1": "DET1"},
    )
    
    assert "meas_" in meas_path


def test_session_manager_close_session(temp_dir, technical_container):
    """Test closing session."""
    manager = SessionManager(config={"technical_folder": str(temp_dir)})
    manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_001",
        distance_cm=17.0,
    )
    
    assert manager.is_session_active()
    
    manager.close_session()
    
    assert not manager.is_session_active()
    assert manager.session_path is None
    assert manager.sample_id is None


def test_session_manager_requires_active_session(temp_dir):
    """Test that operations require active session."""
    manager = SessionManager(config={"technical_folder": str(temp_dir)})
    
    # Should raise without active session
    with pytest.raises(RuntimeError, match="No active session"):
        manager.add_points([])
    
    with pytest.raises(RuntimeError, match="No active session"):
        manager.add_measurement(1, {}, {}, {})


def test_session_manager_get_session_info(temp_dir, technical_container):
    """Test getting session info."""
    manager = SessionManager(config={"technical_folder": str(temp_dir)})
    
    # No active session
    info = manager.get_session_info()
    assert info["active"] is False
    
    # Create session
    manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_001",
        distance_cm=17.0,
    )
    
    info = manager.get_session_info()
    assert info["active"] is True
    assert info["sample_id"] == "TEST_SAMPLE_001"
    assert info["attenuation_complete"] is False


def test_session_manager_replace_technical_container(temp_dir, technical_container):
    """Test replacing embedded technical data in an active unlocked session."""
    manager = SessionManager(config={"technical_folder": str(temp_dir)})
    _session_id, session_path = manager.create_session(
        folder=temp_dir,
        sample_id="TEST_SAMPLE_SWAP",
        distance_cm=17.0,
    )

    # Create second locked technical container to swap in.
    dark_file = temp_dir / "dark_swap.npy"
    np.save(dark_file, np.random.rand(64, 64).astype(np.float32))
    _new_id, new_tech_path = generate_from_aux_table(
        folder=temp_dir,
        aux_measurements={"DARK": {"DET1": str(dark_file)}},
        poni_data={
            "DET1": (
                "Detector: AdvaPIX\nPixelSize1: 5.500e-05\nPixelSize2: 5.500e-05\nDistance: 0.170000\n",
                "DET1_swap.poni",
            )
        },
        detector_config=[
            {
                "id": "DET1",
                "alias": "DET1",
                "type": "AdvaPIX",
                "size": [64, 64],
                "pixel_size_um": 55.0,
            }
        ],
        active_detector_ids=["DET1"],
        distances_cm=17.0,
        validate_poni=True,
    )
    lock_container(new_tech_path)

    manager.replace_technical_container(Path(new_tech_path))

    assert manager.technical_container_path == Path(new_tech_path)
    with h5py.File(session_path, "r") as session_file:
        source_file = session_file[schema.GROUP_CALIBRATION_SNAPSHOT].attrs.get(
            "source_file", ""
        )
        if isinstance(source_file, bytes):
            source_file = source_file.decode("utf-8")
        assert source_file == str(new_tech_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
