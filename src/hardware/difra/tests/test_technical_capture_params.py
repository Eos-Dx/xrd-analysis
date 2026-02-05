"""Tests for detector capture with varying parameters.

Tests integration times, frame counts, detector aliases, and data quality.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add project src root to path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.difra.hardware.detectors import DummyDetectorController


@pytest.fixture
def temp_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def demo_detector():
    return DummyDetectorController(alias="PRIMARY", size=(256, 256))


@pytest.mark.parametrize("integration_time", [0.1, 1.0, 5.0, 10.0])
def test_capture_varying_integration_times(demo_detector, temp_folder, integration_time):
    """Test capture with different integration times."""
    filename_base = temp_folder / f"test_{integration_time}s"
    
    result = demo_detector.capture_point(
        Nframes=1,
        Nseconds=integration_time,
        filename_base=str(filename_base)
    )
    
    assert result is True
    output_file = Path(str(filename_base) + ".txt")
    assert output_file.exists()
    
    # Load and verify data
    data = np.loadtxt(output_file)
    assert data.shape == (256, 256)
    assert np.all(np.isfinite(data))


@pytest.mark.parametrize("frame_count", [1, 5, 10])
def test_capture_varying_frame_counts(demo_detector, temp_folder, frame_count):
    """Test capture with different frame counts."""
    filename_base = temp_folder / f"test_{frame_count}frames"
    
    result = demo_detector.capture_point(
        Nframes=frame_count,
        Nseconds=0.1,
        filename_base=str(filename_base)
    )
    
    assert result is True
    output_file = Path(str(filename_base) + ".txt")
    assert output_file.exists()
    
    # Data should be integrated across frames
    data = np.loadtxt(output_file)
    assert data.shape == (256, 256)


@pytest.mark.parametrize("alias", ["PRIMARY", "SECONDARY", "WAXS", "SAXS"])
def test_capture_all_detector_aliases(temp_folder, alias):
    """Test capture with different detector aliases."""
    controller = DummyDetectorController(alias=alias, size=(256, 256))
    filename_base = temp_folder / f"test_{alias}"
    
    result = controller.capture_point(
        Nframes=1,
        Nseconds=0.1,
        filename_base=str(filename_base)
    )
    
    assert result is True
    assert controller.alias == alias


def test_capture_data_quality(demo_detector, temp_folder):
    """Test that captured data meets quality criteria."""
    filename_base = temp_folder / "quality_test"
    
    demo_detector.capture_point(
        Nframes=5,
        Nseconds=1.0,
        filename_base=str(filename_base)
    )
    
    data = np.loadtxt(str(filename_base) + ".txt")
    
    # Quality checks
    assert np.all(np.isfinite(data)), "Data contains NaN or Inf values"
    # Note: DEMO detector adds noise which can produce negative values
    # This is realistic - real detectors can have negative background-subtracted values
    assert data.std() > 0, "Data has no variation (constant)"
    assert np.abs(data).max() < 1e10, "Data values unreasonably large"


def test_capture_sequential_measurements(demo_detector, temp_folder):
    """Test multiple sequential captures."""
    for i in range(5):
        filename_base = temp_folder / f"seq_{i}"
        result = demo_detector.capture_point(
            Nframes=1,
            Nseconds=0.1,
            filename_base=str(filename_base)
        )
        assert result is True
        assert (Path(str(filename_base) + ".txt")).exists()


def test_capture_with_short_integration(demo_detector, temp_folder):
    """Test very short integration times (0.01s)."""
    filename_base = temp_folder / "short_integration"
    
    result = demo_detector.capture_point(
        Nframes=1,
        Nseconds=0.01,
        filename_base=str(filename_base)
    )
    
    assert result is True
    data = np.loadtxt(str(filename_base) + ".txt")
    assert data.shape == (256, 256)


def test_capture_with_many_frames(demo_detector, temp_folder):
    """Test integration with many frames (simulates averaging)."""
    filename_base = temp_folder / "many_frames"
    
    result = demo_detector.capture_point(
        Nframes=100,
        Nseconds=0.01,  # Keep total time reasonable
        filename_base=str(filename_base)
    )
    
    assert result is True
    data = np.loadtxt(str(filename_base) + ".txt")
    # Integrated data should be larger due to summing frames
    assert np.mean(data) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
