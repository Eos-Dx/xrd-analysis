#!/usr/bin/env python3
"""Standalone test for technical container HDF5 writer.

This script tests the technical container generation without requiring
the full DIFRA GUI or Qt dependencies.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add src root to path
src_root = Path(__file__).resolve().parents[5]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from hardware.difra.data.hdf5 import io, schema_v1, technical_container


def create_dummy_detector_data(size=(256, 256)):
    """Create dummy detector data matching typical Pixet output."""
    # Simulate 2D Gaussian spot
    x, y = np.arange(size[0]), np.arange(size[1])
    X, Y = np.meshgrid(x, y)
    x0, y0 = size[0] // 2, size[1] // 2
    sigma = 20
    amp = 5000
    data = amp * np.exp(-(((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2)))
    data += np.random.normal(scale=amp * 0.05, size=data.shape)
    return data.astype(np.float32)


def test_basic_container_creation():
    """Test creating an empty technical container."""
    print("\n=== Test 1: Basic Container Creation ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        container_id, file_path = technical_container.create_technical_container(
            folder=tmpdir,
            distance_cm=17.0
        )
        
        print(f"✓ Created container ID: {container_id}")
        print(f"✓ File path: {file_path}")
        
        # Verify file exists
        assert Path(file_path).exists(), "Container file not created"
        print("✓ Container file exists")
        
        # Verify root attributes
        with io.open_h5_append(file_path) as f:
            assert f.attrs[schema_v1.ATTR_CONTAINER_ID] == container_id
            assert f.attrs[schema_v1.ATTR_CONTAINER_TYPE] == schema_v1.CONTAINER_TYPE_TECHNICAL
            assert f.attrs[schema_v1.ATTR_SCHEMA_VERSION] == schema_v1.SCHEMA_VERSION
            assert f.attrs[schema_v1.ATTR_DISTANCE_CM] == 17.0
            print("✓ Root attributes verified")
            
            # Verify groups exist
            assert schema_v1.GROUP_TECHNICAL in f
            assert schema_v1.GROUP_TECHNICAL_CONFIG in f
            assert schema_v1.GROUP_TECHNICAL_PONY in f
            print("✓ Required groups exist")
    
    print("✓ Test 1 PASSED\n")


def test_detector_config_writing():
    """Test writing detector configuration."""
    print("=== Test 2: Detector Config Writing ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        container_id, file_path = technical_container.create_technical_container(
            folder=tmpdir,
            distance_cm=17.0
        )
        
        # Mock detector config
        detector_config = [
            {
                "id": "MiniPIX G08-W0299",
                "alias": "PRIMARY",
                "type": "Pixet",
                "size": {"width": 256, "height": 256},
                "pixel_size_um": [55, 55],
                "faulty_pixels": "faulty_pixels_primary.npy"
            },
            {
                "id": "MiniPIX G05-W0339",
                "alias": "SECONDARY",
                "type": "Pixet",
                "size": {"width": 256, "height": 256},
                "pixel_size_um": [55, 55],
                "faulty_pixels": "faulty_pixels_secondary.npy"
            }
        ]
        active_ids = ["MiniPIX G08-W0299", "MiniPIX G05-W0339"]
        
        technical_container.write_detector_config(
            file_path=file_path,
            detectors_config=detector_config,
            active_detector_ids=active_ids
        )
        
        print("✓ Detector config written")
        
        # Verify config was written
        with io.open_h5_append(file_path) as f:
            config_path = f"{schema_v1.GROUP_TECHNICAL_CONFIG}/detector_config"
            assert config_path in f
            config_json = f[config_path][()]
            
            import json
            config = json.loads(config_json)
            
            assert len(config["detectors"]) == 2
            assert config["detectors"][0]["alias"] == "PRIMARY"
            assert config["detectors"][0]["role"] == "det_primary"
            assert config["detectors"][1]["alias"] == "SECONDARY"
            assert config["detectors"][1]["role"] == "det_secondary"
            print("✓ Config structure verified")
            print(f"  - PRIMARY role: {config['detectors'][0]['role']}")
            print(f"  - SECONDARY role: {config['detectors'][1]['role']}")
    
    print("✓ Test 2 PASSED\n")


def test_pony_writing():
    """Test writing PONY calibration data."""
    print("=== Test 3: PONY Data Writing ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        container_id, file_path = technical_container.create_technical_container(
            folder=tmpdir,
            distance_cm=17.0
        )
        
        # Mock PONY data
        pony_content = """# Nota: C-Order, 1 refers to the Y axis, 2 to the X axis
poni_version: 2.1
Detector: Detector
Detector_config: {"pixel1": 5.5e-05, "pixel2": 5.5e-05, "max_shape": [256, 256], "orientation": 3}
Distance: 0.17
Poni1: 0.007
Poni2: 0.0008
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 1.5406e-10
"""
        
        pony_data = {
            "PRIMARY": (pony_content, "primary_17cm.poni"),
            "SECONDARY": (pony_content.replace("0.17", "0.023"), "secondary_2cm.poni")
        }
        
        technical_container.write_pony_datasets(
            file_path=file_path,
            pony_data=pony_data,
            distance_cm=17.0,
            operator_confirmed=True
        )
        
        print("✓ PONY datasets written")
        
        # Verify PONY data
        with io.open_h5_append(file_path) as f:
            # Check PRIMARY pony
            pony_primary_path = f"{schema_v1.GROUP_TECHNICAL_PONY}/pony_primary"
            assert pony_primary_path in f
            pony_primary = f[pony_primary_path][()]
            # h5py returns bytes for string datasets
            if isinstance(pony_primary, bytes):
                pony_primary = pony_primary.decode('utf-8')
            assert "Distance: 0.17" in pony_primary
            assert f[pony_primary_path].attrs[schema_v1.ATTR_DETECTOR_ID] == "PRIMARY"
            assert f[pony_primary_path].attrs[schema_v1.ATTR_DISTANCE_CM] == 17.0
            assert f[pony_primary_path].attrs[schema_v1.ATTR_PONY_OPERATOR_CONFIRMED] == True
            print("✓ PRIMARY PONY verified")
            
            # Check SECONDARY pony
            pony_secondary_path = f"{schema_v1.GROUP_TECHNICAL_PONY}/pony_secondary"
            assert pony_secondary_path in f
            pony_secondary = f[pony_secondary_path][()]
            if isinstance(pony_secondary, bytes):
                pony_secondary = pony_secondary.decode('utf-8')
            assert "Distance: 0.023" in pony_secondary
            print("✓ SECONDARY PONY verified")
    
    print("✓ Test 3 PASSED\n")


def test_technical_event_writing():
    """Test adding technical measurement events."""
    print("=== Test 4: Technical Event Writing ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        container_id, file_path = technical_container.create_technical_container(
            folder=tmpdir,
            distance_cm=17.0
        )
        
        # Create dummy measurement data
        measurements = {
            "PRIMARY": {
                "data": create_dummy_detector_data((256, 256)),
                "detector_id": "MiniPIX G08-W0299",
                "timestamp": "2026-02-05 09:00:00"
            },
            "SECONDARY": {
                "data": create_dummy_detector_data((256, 256)),
                "detector_id": "MiniPIX G05-W0339",
                "timestamp": "2026-02-05 09:00:00"
            }
        }
        
        event_path = technical_container.add_technical_event(
            file_path=file_path,
            event_index=1,
            technical_type="AGBH",
            measurements=measurements,
            timestamp="2026-02-05 09:00:00",
            distance_cm=17.0
        )
        
        print(f"✓ Technical event created: {event_path}")
        
        # Verify event structure
        with io.open_h5_append(file_path) as f:
            # Check event exists
            assert event_path in f
            print(f"✓ Event group exists")
            
            # Check PRIMARY detector data
            primary_path = f"{event_path}/det_primary"
            assert primary_path in f
            raw_signal_path = f"{primary_path}/{schema_v1.DATASET_RAW_SIGNAL}"
            assert raw_signal_path in f
            
            raw_data = f[raw_signal_path][()]
            assert raw_data.shape == (256, 256)
            assert raw_data.dtype == np.float32
            print(f"✓ PRIMARY raw_signal shape: {raw_data.shape}, dtype: {raw_data.dtype}")
            
            # Check attributes
            assert f[primary_path].attrs[schema_v1.ATTR_TECHNICAL_TYPE] == "AGBH"
            assert f[primary_path].attrs[schema_v1.ATTR_DISTANCE_CM] == 17.0
            assert f[primary_path].attrs[schema_v1.ATTR_DETECTOR_ID] == "MiniPIX G08-W0299"
            print("✓ PRIMARY attributes verified")
            
            # Check SECONDARY detector data
            secondary_path = f"{event_path}/det_secondary"
            assert secondary_path in f
            print("✓ SECONDARY detector data exists")
    
    print("✓ Test 4 PASSED\n")


def test_full_generation_from_aux_table():
    """Test complete generation from aux table structure (mimics UI workflow)."""
    print("=== Test 5: Full Generation from Aux Table ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy .npy files for all required types
        aux_measurements = {}
        
        for tech_type in ["DARK", "EMPTY", "BACKGROUND", "AGBH"]:
            aux_measurements[tech_type] = {}
            
            for alias in ["PRIMARY", "SECONDARY"]:
                # Create dummy .npy file
                npy_path = Path(tmpdir) / f"{tech_type.lower()}_{alias.lower()}.npy"
                data = create_dummy_detector_data((256, 256))
                np.save(npy_path, data)
                aux_measurements[tech_type][alias] = str(npy_path)
        
        print("✓ Created dummy measurement files")
        
        # Mock PONY data
        pony_content = "# PONY v2.1\nDistance: 0.17\n"
        pony_data = {
            "PRIMARY": (pony_content, "primary.poni"),
            "SECONDARY": (pony_content, "secondary.poni")
        }
        
        # Mock detector config
        detector_config = [
            {
                "id": "MiniPIX G08-W0299",
                "alias": "PRIMARY",
                "type": "Pixet",
                "size": {"width": 256, "height": 256},
                "pixel_size_um": [55, 55],
                "faulty_pixels": None
            },
            {
                "id": "MiniPIX G05-W0339",
                "alias": "SECONDARY",
                "type": "Pixet",
                "size": {"width": 256, "height": 256},
                "pixel_size_um": [55, 55],
                "faulty_pixels": None
            }
        ]
        active_ids = ["MiniPIX G08-W0299", "MiniPIX G05-W0339"]
        
        # Generate container
        container_id, file_path = technical_container.generate_from_aux_table(
            folder=tmpdir,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=active_ids,
            distance_cm=17.0
        )
        
        print(f"✓ Container generated: {Path(file_path).name}")
        print(f"✓ Container ID: {container_id}")
        
        # Verify complete structure
        with io.open_h5_append(file_path) as f:
            # Check root attrs
            assert f.attrs[schema_v1.ATTR_CONTAINER_ID] == container_id
            print("✓ Container ID in root attrs")
            
            # Check config
            config_path = f"{schema_v1.GROUP_TECHNICAL_CONFIG}/detector_config"
            assert config_path in f
            print("✓ Detector config present")
            
            # Check PONY
            assert f"{schema_v1.GROUP_TECHNICAL_PONY}/pony_primary" in f
            assert f"{schema_v1.GROUP_TECHNICAL_PONY}/pony_secondary" in f
            print("✓ PONY datasets present")
            
            # Check technical events
            event_count = 0
            for key in f[schema_v1.GROUP_TECHNICAL].keys():
                if key.startswith("tech_evt_"):
                    event_count += 1
                    # Verify each event has both detectors
                    event = f[f"{schema_v1.GROUP_TECHNICAL}/{key}"]
                    assert "det_primary" in event
                    assert "det_secondary" in event
            
            assert event_count == 4  # DARK, EMPTY, BACKGROUND, AGBH
            print(f"✓ Found {event_count} technical events")
            
            # List all groups for inspection
            print("\nContainer structure:")
            def print_structure(group, indent=0):
                for key in group.keys():
                    obj = group[key]
                    prefix = "  " * indent
                    if hasattr(obj, 'keys'):
                        print(f"{prefix}📁 {key}/")
                        print_structure(obj, indent + 1)
                    else:
                        print(f"{prefix}📄 {key} {obj.shape if hasattr(obj, 'shape') else ''}")
            
            print_structure(f)
    
    print("\n✓ Test 5 PASSED\n")


def test_object_references():
    """Test HDF5 object reference functionality."""
    print("=== Test 6: Object References ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        container_id, file_path = technical_container.create_technical_container(
            folder=tmpdir,
            distance_cm=17.0
        )
        
        # Add a technical event
        measurements = {
            "PRIMARY": {
                "data": create_dummy_detector_data((256, 256)),
                "detector_id": "PRIMARY",
                "timestamp": "2026-02-05 09:00:00"
            }
        }
        
        event_path = technical_container.add_technical_event(
            file_path=file_path,
            event_index=1,
            technical_type="AGBH",
            measurements=measurements,
            timestamp="2026-02-05 09:00:00",
            distance_cm=17.0
        )
        
        # Write PONY
        pony_data = {"PRIMARY": ("# PONY data", "primary.poni")}
        technical_container.write_pony_datasets(
            file_path=file_path,
            pony_data=pony_data,
            distance_cm=17.0
        )
        
        # Link PONY to event
        technical_container.link_pony_to_event(
            file_path=file_path,
            pony_alias="PRIMARY",
            event_index=1
        )
        
        print("✓ Created reference from PONY to event")
        
        # Verify reference
        with io.open_h5_append(file_path) as f:
            pony_path = f"{schema_v1.GROUP_TECHNICAL_PONY}/pony_primary"
            pony = f[pony_path]
            
            assert schema_v1.ATTR_PONY_DERIVED_FROM in pony.attrs
            ref = pony.attrs[schema_v1.ATTR_PONY_DERIVED_FROM]
            
            # Dereference
            target = f[ref]
            assert target.name == event_path
            print(f"✓ Reference resolved: {pony_path} -> {target.name}")
    
    print("✓ Test 6 PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("DIFRA HDF5 Technical Container Writer Tests")
    print("="*60)
    
    try:
        test_basic_container_creation()
        test_detector_config_writing()
        test_pony_writing()
        test_technical_event_writing()
        test_full_generation_from_aux_table()
        test_object_references()
        
        print("="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60 + "\n")
        return 0
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
