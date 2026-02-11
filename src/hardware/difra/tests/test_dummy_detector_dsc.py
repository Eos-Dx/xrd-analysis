"""Test .dsc file generation in DummyDetectorController."""
import sys
import os
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from hardware.difra.hardware.detectors import DummyDetectorController


def test_dummy_detector_generates_dsc_file():
    """Test that DummyDetectorController generates .dsc files alongside .txt files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create detector
        detector = DummyDetectorController(alias="TEST_DETECTOR", size=(256, 256))
        
        # Capture data
        filename_base = str(tmpdir / "test_measurement")
        success = detector.capture_point(Nframes=5, Nseconds=1.0, filename_base=filename_base)
        
        assert success, "Capture should succeed"
        
        # Verify .txt file exists
        txt_file = Path(f"{filename_base}.txt")
        assert txt_file.exists(), f"Expected .txt file at {txt_file}"
        
        # Verify .dsc file exists
        dsc_file = Path(f"{filename_base}.dsc")
        assert dsc_file.exists(), f"Expected .dsc file at {dsc_file}"
        
        # Verify .dsc content
        dsc_content = dsc_file.read_text()
        assert "[F0]" in dsc_content, ".dsc should contain [F0] section header"
        assert "Type=i16" in dsc_content, ".dsc should contain Type field"
        assert "Acq time=" in dsc_content, ".dsc should contain acquisition time"
        assert "Frames=5" in dsc_content, ".dsc should contain frame count"
        assert "DEMO MODE" in dsc_content, ".dsc should indicate demo mode"
        assert "TEST_DETECTOR" in dsc_content, ".dsc should contain detector alias"
        
        print(f"✅ .dsc file generated successfully")
        print(f"   .txt: {txt_file.stat().st_size} bytes")
        print(f"   .dsc: {dsc_file.stat().st_size} bytes")
        print(f"\n.dsc content preview:\n{dsc_content[:200]}...")


def test_dsc_file_metadata_accuracy():
    """Test that .dsc file contains accurate metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create detector with specific parameters
        detector = DummyDetectorController(alias="SAXS", size=(512, 512))
        
        # Capture with specific parameters
        Nframes = 10
        Nseconds = 0.5
        filename_base = str(tmpdir / "metadata_test")
        detector.capture_point(Nframes=Nframes, Nseconds=Nseconds, filename_base=filename_base)
        
        # Read .dsc file
        dsc_file = Path(f"{filename_base}.dsc")
        dsc_content = dsc_file.read_text()
        
        # Verify dimensions
        assert "width=512" in dsc_content, ".dsc should reflect detector width"
        assert "height=512" in dsc_content, ".dsc should reflect detector height"
        
        # Verify timing
        expected_time = Nframes * Nseconds
        assert f"Acq time={expected_time:.6f}" in dsc_content, ".dsc should contain correct acquisition time"
        assert f"Frames={Nframes}" in dsc_content, ".dsc should contain correct frame count"
        
        # Verify detector ID
        assert "SAXS" in dsc_content, ".dsc should contain detector alias"
        
        print(f"✅ .dsc metadata accuracy verified")
        print(f"   Expected acq time: {expected_time:.6f}s")
        print(f"   Frames: {Nframes}")
        print(f"   Detector: SAXS (512x512)")


def test_multiple_captures_unique_dsc_files():
    """Test that multiple captures create separate .dsc files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        detector = DummyDetectorController(alias="WAXS", size=(256, 256))
        
        # Capture multiple measurements
        files_created = []
        for i in range(3):
            filename_base = str(tmpdir / f"capture_{i}")
            detector.capture_point(Nframes=1, Nseconds=0.1, filename_base=filename_base)
            
            txt_file = Path(f"{filename_base}.txt")
            dsc_file = Path(f"{filename_base}.dsc")
            
            assert txt_file.exists()
            assert dsc_file.exists()
            
            files_created.append((txt_file, dsc_file))
        
        # Verify all files are unique
        all_files = [f for pair in files_created for f in pair]
        assert len(all_files) == len(set(all_files)), "All files should be unique"
        
        print(f"✅ Multiple captures created {len(files_created)} unique .txt/.dsc pairs")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
