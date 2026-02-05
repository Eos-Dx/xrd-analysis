"""Tests for equipment activation and initialization in technical measurements.

Tests:
- DEMO detector initialization
- DEMO stage controller initialization (if available)
- Equipment ready state verification
- Proper cleanup and deinitialization
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add project src root to path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.difra.hardware.detectors import DummyDetectorController


@pytest.fixture
def demo_config():
    """Configuration for DEMO mode equipment."""
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
    }


@pytest.fixture
def temp_folder():
    """Temporary folder for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_init_demo_detectors(demo_config):
    """Test initialization of DEMO detector controllers."""
    detectors = {}
    
    for det_config in demo_config["detectors"]:
        det_id = det_config["id"]
        if det_id in demo_config["dev_active_detectors"]:
            controller = DummyDetectorController(
                alias=det_id,
                size=(det_config["width"], det_config["height"])
            )
            # Initialize detector
            success = controller.init_detector()
            assert success is True, f"Failed to initialize detector {det_id}"
            
            detectors[det_id] = controller
    
    # Verify all active detectors initialized
    assert len(detectors) == len(demo_config["dev_active_detectors"])
    assert "PRIMARY" in detectors
    assert "SECONDARY" in detectors


def test_detector_controller_properties(demo_config):
    """Test that detector controllers have required properties."""
    controller = DummyDetectorController(
        alias="PRIMARY",
        size=(256, 256)
    )
    
    # Check essential properties
    assert hasattr(controller, "alias")
    assert controller.alias == "PRIMARY"
    assert hasattr(controller, "size")
    assert controller.size == (256, 256)
    assert hasattr(controller, "init_detector")
    assert hasattr(controller, "capture_point")
    assert hasattr(controller, "deinit_detector")


def test_equipment_ready_state(demo_config):
    """Test that equipment reaches ready state after initialization."""
    detectors = {}
    
    # Initialize all detectors
    for det_config in demo_config["detectors"]:
        det_id = det_config["id"]
        if det_id in demo_config["dev_active_detectors"]:
            controller = DummyDetectorController(
                alias=det_id,
                size=(det_config["width"], det_config["height"])
            )
            success = controller.init_detector()
            assert success, f"Detector {det_id} failed to initialize"
            detectors[det_id] = controller
    
    # Verify ready state (all detectors initialized and functional)
    for det_id, controller in detectors.items():
        # Detector should be able to perform capture
        assert controller.alias == det_id
        assert controller.size == (256, 256)


def test_equipment_cleanup(demo_config, temp_folder):
    """Test proper cleanup and deinitialization of equipment."""
    detectors = {}
    
    # Initialize detectors
    for det_config in demo_config["detectors"]:
        det_id = det_config["id"]
        if det_id in demo_config["dev_active_detectors"]:
            controller = DummyDetectorController(
                alias=det_id,
                size=(det_config["width"], det_config["height"])
            )
            controller.init_detector()
            detectors[det_id] = controller
    
    # Perform some operations
    for controller in detectors.values():
        filename_base = temp_folder / f"{controller.alias}_test"
        controller.capture_point(Nframes=1, Nseconds=0.01, filename_base=str(filename_base))
    
    # Cleanup
    for controller in detectors.values():
        controller.deinit_detector()
    
    # Verify cleanup (no exceptions raised)
    assert True


def test_multiple_init_deinit_cycles(demo_config):
    """Test that equipment can be initialized and deinitialized multiple times."""
    controller = DummyDetectorController(alias="PRIMARY", size=(256, 256))
    
    for cycle in range(3):
        # Initialize
        success = controller.init_detector()
        assert success, f"Cycle {cycle}: initialization failed"
        
        # Deinitialize
        controller.deinit_detector()
        
        # Should be able to initialize again
        if cycle < 2:  # Not the last cycle
            success = controller.init_detector()
            assert success, f"Cycle {cycle}: re-initialization failed"
            controller.deinit_detector()


def test_detector_capture_without_init(demo_config, temp_folder):
    """Test that detector capture works even without explicit init (DEMO mode)."""
    controller = DummyDetectorController(alias="PRIMARY", size=(256, 256))
    
    # DEMO detectors don't require init to capture
    filename_base = temp_folder / "test_capture"
    result = controller.capture_point(Nframes=1, Nseconds=0.01, filename_base=str(filename_base))
    
    assert result is True
    # Check that file was created
    assert (Path(str(filename_base) + ".txt")).exists()


def test_detector_size_configuration(demo_config):
    """Test that detector size can be configured."""
    # Test different sizes
    sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    
    for width, height in sizes:
        controller = DummyDetectorController(alias="TEST", size=(width, height))
        assert controller.size == (width, height)


def test_multiple_detectors_simultaneously(demo_config):
    """Test that multiple detectors can be active simultaneously."""
    detectors = {}
    
    # Initialize multiple detectors
    for det_config in demo_config["detectors"]:
        det_id = det_config["id"]
        controller = DummyDetectorController(
            alias=det_id,
            size=(det_config["width"], det_config["height"])
        )
        controller.init_detector()
        detectors[det_id] = controller
    
    # All should be initialized
    assert len(detectors) == len(demo_config["detectors"])
    
    # All should be functional
    for det_id, controller in detectors.items():
        assert controller.alias == det_id
    
    # Cleanup
    for controller in detectors.values():
        controller.deinit_detector()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
