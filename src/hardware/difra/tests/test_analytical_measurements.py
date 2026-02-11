"""Tests for analytical measurements (attenuation) in DIFRA HDF5 containers.

Tests verify:
- Adding analytical measurements with analysis_type="attenuation"
- Linking analytical measurements to points
- Per-detector metadata (integration time, beam energy)
- PONY references from analytical measurements
- Multiple analytical measurements per session
- Validation of analytical measurement structure
"""

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

from hardware.container.v0_1 import schema, validator, writer


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def session_container_with_technical(temp_dir):
    """Create session container with technical data copied."""
    from hardware.container.v0_1.technical_container import (
        create_technical_container,
        write_detector_config,
        write_pony_datasets,
    )
    
    # Create minimal technical container
    tech_id, tech_path = create_technical_container(
        folder=temp_dir,
        distance_cm=17.0,
    )
    
    # Add detector config
    detector_config = [
        {
            "id": "DET1",
            "alias": "DET1",
            "type": "AdvaPIX",
            "size": [256, 256],
            "pixel_size_um": 55.0,
        },
        {
            "id": "DET2",
            "alias": "DET2",
            "type": "AdvaPIX",
            "size": [256, 256],
            "pixel_size_um": 55.0,
        },
    ]
    
    write_detector_config(tech_path, detector_config, ["DET1", "DET2"])
    
    # Add PONI data
    poni_content = """Detector: AdvaPIX
PixelSize1: 5.500e-05
PixelSize2: 5.500e-05
Distance: 0.170000
Poni1: 0.014025
Poni2: 0.014025
Rot1: 0.000000
Rot2: 0.000000
Rot3: 0.000000
Detector_config: {"pixel1": 5.5e-05, "pixel2": 5.5e-05, "max_shape": [256, 256]}
"""
    
    pony_data = {
        "DET1": (poni_content, "DET1_17cm.poni"),
        "DET2": (poni_content, "DET2_17cm.poni"),
    }
    
    write_pony_datasets(tech_path, pony_data, 17.0)
    
    # Lock technical container
    from hardware.container.v0_1.container_manager import lock_container
    lock_container(tech_path)
    
    # Create session container
    session_id, session_path = writer.create_session_container(
        folder=temp_dir,
        sample_id="SAMPLE_001",
        operator_id="test_operator",
        site_id="test_site",
        machine_name="DIFRA-01",
        beam_energy_keV=17.5,
        acquisition_date="2024-01-15",
    )
    
    # Copy technical data to session
    writer.copy_technical_to_session(
        technical_file=tech_path,
        session_file=session_path,
        auto_lock=False,  # Already locked
    )
    
    # Add a test point
    writer.add_point(
        file_path=session_path,
        point_index=1,
        pixel_coordinates=[100, 200],
        physical_coordinates_mm=[10.0, 20.0],
    )
    
    yield session_path


class TestAnalyticalMeasurementBasics:
    """Test basic analytical measurement creation."""
    
    def test_add_attenuation_measurement(self, session_container_with_technical):
        """Test adding attenuation measurement to session."""
        session_file = session_container_with_technical
        
        # Create synthetic attenuation data (without sample)
        attenuation_data = {
            "DET1": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
            "DET2": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
        }
        
        detector_metadata = {
            "DET1": {
                "integration_time_ms": 50.0,
                "beam_energy_keV": 17.5,
            },
            "DET2": {
                "integration_time_ms": 50.0,
                "beam_energy_keV": 17.5,
            },
        }
        
        pony_alias_map = {"DET1": "DET1", "DET2": "DET2"}
        
        # Add analytical measurement
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=attenuation_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
            timestamp_start="2024-01-15 10:30:00",
        )
        
        # Verify structure
        with h5py.File(session_file, "r") as f:
            assert ana_path in f
            ana_group = f[ana_path]
            
            # Check attributes
            assert ana_group.attrs[schema.ATTR_ANALYSIS_TYPE] == "attenuation"
            assert ana_group.attrs[schema.ATTR_MEASUREMENT_COUNTER] == 1
            assert ana_group.attrs[schema.ATTR_TIMESTAMP_START] == "2024-01-15 10:30:00"
            assert ana_group.attrs[schema.ATTR_MEASUREMENT_STATUS] == schema.STATUS_COMPLETED
            
            # Check detector groups exist
            assert f"{ana_path}/det_det1" in f
            assert f"{ana_path}/det_det2" in f
            
            # Check raw_signal datasets
            det1_group = f[f"{ana_path}/det_det1"]
            det1_signal = det1_group[schema.DATASET_RAW_SIGNAL]
            assert det1_signal.shape == (256, 256)
            # Attributes are on the detector group, not the dataset
            assert det1_group.attrs[schema.ATTR_DETECTOR_ID] == "DET1"
            assert det1_group.attrs[schema.ATTR_INTEGRATION_TIME_MS] == 50.0
            assert det1_group.attrs[schema.ATTR_BEAM_ENERGY_KEV] == 17.5
    
    def test_multiple_analytical_measurements(self, session_container_with_technical):
        """Test adding multiple analytical measurements to same session."""
        session_file = session_container_with_technical
        
        pony_alias_map = {"DET1": "DET1", "DET2": "DET2"}
        
        # Add "without sample" measurement
        without_data = {
            "DET1": np.random.randint(800, 1000, size=(256, 256), dtype=np.uint16),
            "DET2": np.random.randint(800, 1000, size=(256, 256), dtype=np.uint16),
        }
        
        detector_metadata = {
            "DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
            "DET2": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
        }
        
        without_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=without_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
            timestamp_start="2024-01-15 10:30:00",
        )
        
        # Add "with sample" measurement
        with_data = {
            "DET1": np.random.randint(400, 600, size=(256, 256), dtype=np.uint16),
            "DET2": np.random.randint(400, 600, size=(256, 256), dtype=np.uint16),
        }
        
        with_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=with_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
            timestamp_start="2024-01-15 10:31:00",
        )
        
        # Verify both exist and have correct counters
        with h5py.File(session_file, "r") as f:
            assert without_path == "/analytical_measurements/ana_000000001"
            assert with_path == "/analytical_measurements/ana_000000002"
            
            assert f[without_path].attrs[schema.ATTR_MEASUREMENT_COUNTER] == 1
            assert f[with_path].attrs[schema.ATTR_MEASUREMENT_COUNTER] == 2
    
    def test_analytical_measurement_pony_references(self, session_container_with_technical):
        """Test that analytical measurements correctly reference PONY files."""
        session_file = session_container_with_technical
        
        attenuation_data = {
            "DET1": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
        }
        
        detector_metadata = {
            "DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
        }
        
        pony_alias_map = {"DET1": "DET1"}
        
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=attenuation_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
        )
        
        # Verify PONY reference
        with h5py.File(session_file, "r") as f:
            det_group = f[f"{ana_path}/det_det1"]
            
            # Check PONY reference exists
            assert schema.ATTR_PONY_REF in det_group.attrs
            
            # Dereference and verify it points to correct PONY
            pony_ref = det_group.attrs[schema.ATTR_PONY_REF]
            pony_group = f[pony_ref]
            
            assert pony_group.name == "/technical/pony/pony_det1"
            # PONY content is stored as dataset data, not an attribute
            assert schema.ATTR_DETECTOR_ID in pony_group.attrs


class TestAnalyticalMeasurementPointLinking:
    """Test linking analytical measurements to points."""
    
    def test_link_single_analytical_measurement(self, session_container_with_technical):
        """Test linking analytical measurement to a point."""
        session_file = session_container_with_technical
        
        # Add analytical measurement
        attenuation_data = {
            "DET1": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
        }
        
        detector_metadata = {
            "DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
        }
        
        pony_alias_map = {"DET1": "DET1"}
        
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=attenuation_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
        )
        
        # Link to point 1
        writer.link_analytical_measurement_to_point(
            file_path=session_file,
            point_index=1,
            analytical_measurement_index=1,
        )
        
        # Verify bidirectional link
        with h5py.File(session_file, "r") as f:
            point_group = f["/points/pt_001"]
            ana_group = f["/analytical_measurements/ana_000000001"]
            
            # Check Point → Analytical measurement reference
            assert schema.ATTR_ANALYTICAL_MEASUREMENT_REFS in point_group.attrs
            point_refs = point_group.attrs[schema.ATTR_ANALYTICAL_MEASUREMENT_REFS]
            assert len(point_refs) == 1
            assert f[point_refs[0]].name == "/analytical_measurements/ana_000000001"
            
            # Check Analytical measurement → Point reference (bidirectional)
            assert schema.ATTR_POINT_REFS in ana_group.attrs
            ana_refs = ana_group.attrs[schema.ATTR_POINT_REFS]
            assert len(ana_refs) == 1
            assert f[ana_refs[0]].name == "/points/pt_001"
    
    def test_link_multiple_analytical_measurements_to_point(self, session_container_with_technical):
        """Test linking multiple analytical measurements to same point."""
        session_file = session_container_with_technical
        
        pony_alias_map = {"DET1": "DET1"}
        detector_metadata = {
            "DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
        }
        
        # Add 3 analytical measurements
        for i in range(3):
            data = {
                "DET1": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
            }
            
            writer.add_analytical_measurement(
                file_path=session_file,
                measurement_data=data,
                detector_metadata=detector_metadata,
                pony_alias_map=pony_alias_map,
                analysis_type="attenuation",
            )
        
        # Link all to point 1
        for i in range(1, 4):
            writer.link_analytical_measurement_to_point(
                file_path=session_file,
                point_index=1,
                analytical_measurement_index=i,
            )
        
        # Verify all links
        with h5py.File(session_file, "r") as f:
            point_group = f["/points/pt_001"]
            refs = point_group.attrs[schema.ATTR_ANALYTICAL_MEASUREMENT_REFS]
            
            assert len(refs) == 3
            
            # Verify each reference points to correct analytical measurement
            for i, ref in enumerate(refs, start=1):
                ana_group = f[ref]
                expected_name = f"/analytical_measurements/ana_{i:09d}"
                assert ana_group.name == expected_name
    
    def test_link_analytical_to_multiple_points(self, session_container_with_technical):
        """Test linking same analytical measurement to multiple points."""
        session_file = session_container_with_technical
        
        # Add second and third points
        writer.add_point(
            session_file, 2, 
            pixel_coordinates=[150, 250], 
            physical_coordinates_mm=[15.0, 25.0]
        )
        writer.add_point(
            session_file, 3, 
            pixel_coordinates=[200, 300], 
            physical_coordinates_mm=[20.0, 30.0]
        )
        
        # Add analytical measurement
        data = {
            "DET1": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
        }
        
        detector_metadata = {
            "DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
        }
        
        pony_alias_map = {"DET1": "DET1"}
        
        writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
        )
        
        # Link to all 3 points
        for point_idx in [1, 2, 3]:
            writer.link_analytical_measurement_to_point(
                file_path=session_file,
                point_index=point_idx,
                analytical_measurement_index=1,
            )
        
        # Verify all points reference the same analytical measurement
        with h5py.File(session_file, "r") as f:
            for point_idx in [1, 2, 3]:
                point_path = f"/points/pt_{point_idx:03d}"
                point_group = f[point_path]
                
                refs = point_group.attrs[schema.ATTR_ANALYTICAL_MEASUREMENT_REFS]
                assert len(refs) == 1
                
                ana_group = f[refs[0]]
                assert ana_group.name == "/analytical_measurements/ana_000000001"


class TestAnalyticalMeasurementMetadata:
    """Test per-detector metadata in analytical measurements."""
    
    def test_per_detector_beam_energy_variation(self, session_container_with_technical):
        """Test that each detector can have different beam energy in analytical measurement."""
        session_file = session_container_with_technical
        
        attenuation_data = {
            "DET1": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
            "DET2": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
        }
        
        # Different beam energies per detector
        detector_metadata = {
            "DET1": {
                "integration_time_ms": 50.0,
                "beam_energy_keV": 17.0,  # Different energy
            },
            "DET2": {
                "integration_time_ms": 50.0,
                "beam_energy_keV": 18.0,  # Different energy
            },
        }
        
        pony_alias_map = {"DET1": "DET1", "DET2": "DET2"}
        
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=attenuation_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
        )
        
        # Verify per-detector beam energies
        with h5py.File(session_file, "r") as f:
            det1 = f[f"{ana_path}/det_det1"]
            det2 = f[f"{ana_path}/det_det2"]
            
            assert det1.attrs[schema.ATTR_BEAM_ENERGY_KEV] == 17.0
            assert det2.attrs[schema.ATTR_BEAM_ENERGY_KEV] == 18.0
    
    def test_different_integration_times(self, session_container_with_technical):
        """Test different integration times per detector."""
        session_file = session_container_with_technical
        
        attenuation_data = {
            "DET1": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
            "DET2": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
        }
        
        # Different integration times
        detector_metadata = {
            "DET1": {
                "integration_time_ms": 10.0,
                "beam_energy_keV": 17.5,
            },
            "DET2": {
                "integration_time_ms": 100.0,  # 10x longer
                "beam_energy_keV": 17.5,
            },
        }
        
        pony_alias_map = {"DET1": "DET1", "DET2": "DET2"}
        
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=attenuation_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
        )
        
        # Verify per-detector integration times
        with h5py.File(session_file, "r") as f:
            det1 = f[f"{ana_path}/det_det1"]
            det2 = f[f"{ana_path}/det_det2"]
            
            assert det1.attrs[schema.ATTR_INTEGRATION_TIME_MS] == 10.0
            assert det2.attrs[schema.ATTR_INTEGRATION_TIME_MS] == 100.0
    
    def test_optional_metadata_fields(self, session_container_with_technical):
        """Test analytical measurements work with minimal metadata."""
        session_file = session_container_with_technical
        
        attenuation_data = {
            "DET1": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
        }
        
        # Minimal metadata (no beam_energy_keV)
        detector_metadata = {
            "DET1": {
                "integration_time_ms": 50.0,
            },
        }
        
        pony_alias_map = {"DET1": "DET1"}
        
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=attenuation_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
        )
        
        # Verify measurement created successfully
        with h5py.File(session_file, "r") as f:
            det_group = f[f"{ana_path}/det_det1"]
            
            # integration_time_ms should be present
            assert det_group.attrs[schema.ATTR_INTEGRATION_TIME_MS] == 50.0
            
            # beam_energy_keV should not be present
            assert schema.ATTR_BEAM_ENERGY_KEV not in det_group.attrs


class TestAnalyticalMeasurementValidation:
    """Test validation of analytical measurements."""
    
    def test_validate_session_with_analytical_measurements(self, session_container_with_technical):
        """Test that session with analytical measurements passes validation."""
        session_file = session_container_with_technical
        
        # Add analytical measurement
        data = {
            "DET1": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
        }
        
        detector_metadata = {
            "DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
        }
        
        pony_alias_map = {"DET1": "DET1"}
        
        writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
        )
        
        # Validate container - validator returns tuple (is_valid, summary)
        is_valid, summary = validator.validate_session_container(session_file)
        assert is_valid is True
    
    def test_validate_analytical_measurement_structure(self, session_container_with_technical):
        """Test that analytical measurements have correct structure."""
        session_file = session_container_with_technical
        
        # Add analytical measurement
        data = {
            "DET1": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
            "DET2": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
        }
        
        detector_metadata = {
            "DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
            "DET2": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
        }
        
        pony_alias_map = {"DET1": "DET1", "DET2": "DET2"}
        
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
        )
        
        # Validate structure manually
        with h5py.File(session_file, "r") as f:
            ana_group = f[ana_path]
            
            # Check required attributes
            required_attrs = [
                schema.ATTR_MEASUREMENT_COUNTER,
                schema.ATTR_TIMESTAMP_START,
                schema.ATTR_MEASUREMENT_STATUS,
                schema.ATTR_ANALYSIS_TYPE,
            ]
            
            for attr in required_attrs:
                assert attr in ana_group.attrs, f"Missing required attribute: {attr}"
            
            # Check detector groups have raw_signal datasets
            for det_name in ["det_det1", "det_det2"]:
                det_group = ana_group[det_name]
                assert schema.DATASET_RAW_SIGNAL in det_group
                
                # Check dataset is 2D array
                raw_signal = det_group[schema.DATASET_RAW_SIGNAL]
                assert len(raw_signal.shape) == 2


class TestAnalyticalMeasurementIntegration:
    """Integration tests combining analytical measurements with regular measurements."""
    
    def test_session_with_both_regular_and_analytical(self, session_container_with_technical):
        """Test session containing both regular and analytical measurements."""
        session_file = session_container_with_technical
        
        # Add regular measurement
        regular_data = {
            "DET1": np.random.randint(0, 100, size=(256, 256), dtype=np.uint16),
        }
        
        detector_metadata = {
            "DET1": {"integration_time_ms": 1000.0, "beam_energy_keV": 17.5},
        }
        
        pony_alias_map = {"DET1": "DET1"}
        
        meas_path = writer.add_measurement(
            file_path=session_file,
            measurement_data=regular_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            point_index=1,
        )
        
        # Add analytical measurement
        analytical_data = {
            "DET1": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
        }
        
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=analytical_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
        )
        
        # Link analytical to point (counter 2 since regular used counter 1)
        writer.link_analytical_measurement_to_point(
            file_path=session_file,
            point_index=1,
            analytical_measurement_index=2,
        )
        
        # Verify both exist and counters are independent
        with h5py.File(session_file, "r") as f:
            # Regular measurement in /measurements/pt_001/meas_NNNNNNNNN
            assert "/measurements/pt_001/meas_" in meas_path
            meas_group = f[meas_path]
            assert schema.ATTR_MEASUREMENT_COUNTER in meas_group.attrs
            
            # Analytical measurement in /analytical_measurements  
            # Counter is shared, so analytical gets counter 2 after regular measurement
            assert "/analytical_measurements/ana_" in ana_path
            ana_group = f[ana_path]
            assert schema.ATTR_ANALYSIS_TYPE in ana_group.attrs
            
            # Point references both
            point_group = f["/points/pt_001"]
            
            # Regular measurement via point_ref
            assert schema.ATTR_POINT_REF in meas_group.attrs
            
            # Analytical measurement via list
            assert schema.ATTR_ANALYTICAL_MEASUREMENT_REFS in point_group.attrs
            refs = point_group.attrs[schema.ATTR_ANALYTICAL_MEASUREMENT_REFS]
            assert len(refs) == 1
    
    def test_attenuation_workflow(self, session_container_with_technical):
        """Test complete attenuation workflow: without sample -> with sample -> link to points."""
        session_file = session_container_with_technical
        
        # Add 3 more points
        for i in range(2, 5):
            writer.add_point(
                file_path=session_file,
                point_index=i,
                pixel_coordinates=[100 * i, 200 * i],
                physical_coordinates_mm=[10.0 * i, 20.0 * i],
            )
        
        pony_alias_map = {"DET1": "DET1", "DET2": "DET2"}
        
        # Step 1: Measure without sample (I0)
        without_data = {
            "DET1": np.random.randint(900, 1000, size=(256, 256), dtype=np.uint16),
            "DET2": np.random.randint(900, 1000, size=(256, 256), dtype=np.uint16),
        }
        
        detector_metadata = {
            "DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
            "DET2": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
        }
        
        writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=without_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
            timestamp_start="2024-01-15 10:00:00",
        )
        
        # Step 2: Measure with sample (I)
        with_data = {
            "DET1": np.random.randint(400, 500, size=(256, 256), dtype=np.uint16),
            "DET2": np.random.randint(400, 500, size=(256, 256), dtype=np.uint16),
        }
        
        writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=with_data,
            detector_metadata=detector_metadata,
            pony_alias_map=pony_alias_map,
            analysis_type="attenuation",
            timestamp_start="2024-01-15 10:01:00",
        )
        
        # Step 3: Link both to all points (attenuation correction applies to all)
        for point_idx in range(1, 5):
            # Link I0 (counter 1)
            writer.link_analytical_measurement_to_point(
                file_path=session_file,
                point_index=point_idx,
                analytical_measurement_index=1,
            )
            # Link I (counter 2)
            writer.link_analytical_measurement_to_point(
                file_path=session_file,
                point_index=point_idx,
                analytical_measurement_index=2,
            )
        
        # Verify workflow completed correctly
        with h5py.File(session_file, "r") as f:
            # Check 2 analytical measurements exist
            assert "/analytical_measurements/ana_000000001" in f
            assert "/analytical_measurements/ana_000000002" in f
            
            # Check all points have both references
            for point_idx in range(1, 5):
                point_path = f"/points/pt_{point_idx:03d}"
                point_group = f[point_path]
                
                refs = point_group.attrs[schema.ATTR_ANALYTICAL_MEASUREMENT_REFS]
                assert len(refs) == 2
                
                # Verify references point to correct analytical measurements
                assert f[refs[0]].name == "/analytical_measurements/ana_000000001"
                assert f[refs[1]].name == "/analytical_measurements/ana_000000002"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
