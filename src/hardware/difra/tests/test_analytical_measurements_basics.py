"""Basic analytical measurement tests."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _analytical_measurements_shared import session_container_with_technical
from _analytical_measurements_shared import h5py, np, schema, writer
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
        
        poni_alias_map = {"DET1": "DET1", "DET2": "DET2"}
        
        # Add analytical measurement
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=attenuation_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
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
            
            # Check processed_signal datasets
            det1_group = f[f"{ana_path}/det_det1"]
            det1_signal = det1_group[schema.DATASET_PROCESSED_SIGNAL]
            assert det1_signal.shape == (256, 256)
            # Attributes are on the detector group, not the dataset
            assert det1_group.attrs[schema.ATTR_DETECTOR_ID] == "DET1"
            assert det1_group.attrs[schema.ATTR_INTEGRATION_TIME_MS] == 50.0
            assert det1_group.attrs[schema.ATTR_BEAM_ENERGY_KEV] == 17.5
    
    def test_multiple_analytical_measurements(self, session_container_with_technical):
        """Test adding multiple analytical measurements to same session."""
        session_file = session_container_with_technical
        
        poni_alias_map = {"DET1": "DET1", "DET2": "DET2"}
        
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
            poni_alias_map=poni_alias_map,
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
            poni_alias_map=poni_alias_map,
            analysis_type="attenuation",
            timestamp_start="2024-01-15 10:31:00",
        )
        
        # Verify both exist and have correct counters
        with h5py.File(session_file, "r") as f:
            assert without_path == "/analytical_measurements/ana_000000001"
            assert with_path == "/analytical_measurements/ana_000000002"
            
            assert f[without_path].attrs[schema.ATTR_MEASUREMENT_COUNTER] == 1
            assert f[with_path].attrs[schema.ATTR_MEASUREMENT_COUNTER] == 2
    
    def test_analytical_measurement_poni_references(self, session_container_with_technical):
        """Test that analytical measurements correctly reference PONI files."""
        session_file = session_container_with_technical
        
        attenuation_data = {
            "DET1": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
        }
        
        detector_metadata = {
            "DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
        }
        
        poni_alias_map = {"DET1": "DET1"}
        
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=attenuation_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
            analysis_type="attenuation",
        )
        
        # Verify PONI reference
        with h5py.File(session_file, "r") as f:
            det_group = f[f"{ana_path}/det_det1"]
            
            # Check PONI reference exists
            assert schema.ATTR_PONI_REF in det_group.attrs
            
            # Dereference and verify it points to correct PONI
            poni_ref = det_group.attrs[schema.ATTR_PONI_REF]
            poni_group = f[poni_ref]
            
            assert poni_group.name == "/technical/poni/poni_det1"
            # PONI content is stored as dataset data, not an attribute
            assert schema.ATTR_DETECTOR_ID in poni_group.attrs


