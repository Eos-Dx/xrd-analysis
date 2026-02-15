"""Analytical measurement point-linking tests."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _analytical_measurements_shared import session_container_with_technical
from _analytical_measurements_shared import h5py, np, schema, writer
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
        
        poni_alias_map = {"DET1": "DET1"}
        
        ana_path = writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=attenuation_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
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
        
        poni_alias_map = {"DET1": "DET1"}
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
                poni_alias_map=poni_alias_map,
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
        
        poni_alias_map = {"DET1": "DET1"}
        
        writer.add_analytical_measurement(
            file_path=session_file,
            measurement_data=data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
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


