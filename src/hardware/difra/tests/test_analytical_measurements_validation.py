"""Analytical measurement validation tests."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _analytical_measurements_shared import session_container_with_technical
from _analytical_measurements_shared import h5py, np, schema, validator, writer
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
        
        poni_alias_map = {"DET1": "DET1"}
        
        writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
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
        
        poni_alias_map = {"DET1": "DET1", "DET2": "DET2"}
        
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
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
            
            # Check detector groups have processed_signal datasets
            for det_name in ["det_det1", "det_det2"]:
                det_group = ana_group[det_name]
                assert schema.DATASET_PROCESSED_SIGNAL in det_group
                
                # Check dataset is 2D array
                processed_signal = det_group[schema.DATASET_PROCESSED_SIGNAL]
                assert len(processed_signal.shape) == 2


