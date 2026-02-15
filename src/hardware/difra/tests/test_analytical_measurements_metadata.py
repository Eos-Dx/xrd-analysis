"""Analytical measurement metadata tests."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _analytical_measurements_shared import session_container_with_technical
from _analytical_measurements_shared import h5py, np, schema, writer
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
        
        poni_alias_map = {"DET1": "DET1", "DET2": "DET2"}
        
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=attenuation_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
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
        
        poni_alias_map = {"DET1": "DET1", "DET2": "DET2"}
        
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=attenuation_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
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
        
        poni_alias_map = {"DET1": "DET1"}
        
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=attenuation_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
            analysis_type="attenuation",
        )
        
        # Verify measurement created successfully
        with h5py.File(session_file, "r") as f:
            det_group = f[f"{ana_path}/det_det1"]
            
            # integration_time_ms should be present
            assert det_group.attrs[schema.ATTR_INTEGRATION_TIME_MS] == 50.0
            
            # beam_energy_keV should not be present
            assert schema.ATTR_BEAM_ENERGY_KEV not in det_group.attrs


