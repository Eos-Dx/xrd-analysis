"""Analytical measurement integration tests."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _analytical_measurements_shared import session_container_with_technical
from _analytical_measurements_shared import h5py, np, schema, writer
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
        
        poni_alias_map = {"DET1": "DET1"}
        
        meas_path = writer.add_measurement(
            file_path=session_file,
            measurement_data=regular_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
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
            poni_alias_map=poni_alias_map,
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
        
        poni_alias_map = {"DET1": "DET1", "DET2": "DET2"}
        
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
            poni_alias_map=poni_alias_map,
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
            poni_alias_map=poni_alias_map,
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


