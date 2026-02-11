"""Comprehensive tests for container management features.

Tests:
- PONI distance parsing and validation
- Container locking (HDF5 + OS permissions)
- Archive management with user confirmation
- Primary/supplementary measurement marking
- Find active containers
- Session container lock checking
"""

import os
import stat
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

# Add project src to path
import sys
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hardware.container.v0_1 import (
    schema,
    writer,
    technical_container,
    container_manager,
)


# ==================== PONI Distance Validation Tests ====================

def test_parse_poni_distance_valid():
    """Test parsing distance from valid PONI content."""
    poni_content = """PixelSize1: 7.5e-05
PixelSize2: 7.5e-05
Distance: 0.17
Poni1: 0.012345
Poni2: 0.023456
Wavelength: 1.54e-10"""
    
    distance = schema.parse_poni_distance(poni_content)
    assert distance == 0.17


def test_parse_poni_distance_missing():
    """Test parsing PONI without Distance field raises error."""
    poni_content = """PixelSize1: 7.5e-05
PixelSize2: 7.5e-05
Poni1: 0.012345"""
    
    with pytest.raises(ValueError, match="Distance field not found"):
        schema.parse_poni_distance(poni_content)


def test_validate_poni_distance_exact_match():
    """Test PONI validation with exact distance match."""
    poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
    
    # Should not raise
    schema.validate_poni_distance(poni_content, user_distances_cm=17.0)


def test_validate_poni_distance_within_tolerance():
    """Test PONI validation within 5% tolerance."""
    poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
    
    # 17.5 cm is 2.9% deviation - should pass
    schema.validate_poni_distance(poni_content, user_distances_cm=17.5)
    
    # 17.8 cm is 4.7% deviation - should pass
    schema.validate_poni_distance(poni_content, user_distances_cm=17.8)


def test_validate_poni_distance_exceeds_tolerance():
    """Test PONI validation fails when exceeding 5% tolerance."""
    poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
    
    # 20 cm is 17.6% deviation - should fail
    with pytest.raises(ValueError, match="validation failed"):
        schema.validate_poni_distance(poni_content, user_distances_cm=20.0)


def test_validate_poni_distance_custom_tolerance():
    """Test PONI validation with custom tolerance."""
    poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
    
    # 18 cm is 5.9% deviation
    # Should fail with 5% tolerance
    with pytest.raises(ValueError):
        schema.validate_poni_distance(poni_content, user_distances_cm=18.0, tolerance_percent=5.0)
    
    # Should pass with 10% tolerance
    schema.validate_poni_distance(poni_content, user_distances_cm=18.0, tolerance_percent=10.0)


# ==================== Container Locking Tests ====================

def test_container_initially_unlocked():
    """Test that newly created containers are unlocked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        # Create technical container
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus', 
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {
            'DARK': {'PRIMARY': str(folder / 'dark.npy')},
        }
        
        # Create test data
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        # Should be unlocked initially
        assert not container_manager.is_container_locked(Path(tech_file))


def test_lock_container():
    """Test locking a container sets HDF5 attribute and OS permissions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        # Create minimal technical container
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        
        # Lock it
        container_manager.lock_container(tech_path, user_id='test_user')
        
        # Verify HDF5 attribute
        with h5py.File(tech_path, 'r') as f:
            assert f.attrs.get('locked', False) == True
            assert 'locked_timestamp' in f.attrs
            assert f.attrs.get('locked_by') == 'test_user'
        
        # Verify container reports as locked
        assert container_manager.is_container_locked(tech_path)
        
        # Verify OS read-only permissions
        file_perms = tech_path.stat().st_mode
        # User write should be removed
        assert not (file_perms & stat.S_IWUSR)


def test_lock_already_locked_container():
    """Test that locking an already locked container raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        
        # Lock it once
        container_manager.lock_container(tech_path)
        
        # Try to lock again - should raise
        with pytest.raises(RuntimeError, match="already locked"):
            container_manager.lock_container(tech_path)


def test_unlock_container():
    """Test unlocking a container (administrative)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        
        # Lock and then unlock
        container_manager.lock_container(tech_path)
        assert container_manager.is_container_locked(tech_path)
        
        container_manager.unlock_container(tech_path)
        assert not container_manager.is_container_locked(tech_path)


# ==================== Archive Management Tests ====================

def test_archive_locked_container():
    """Test archiving a locked container."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        
        # Lock it
        container_manager.lock_container(tech_path)
        
        # Archive with confirmation
        archived_path = container_manager.archive_technical_container(
            folder, tech_path, user_confirmed=True
        )
        
        # Original should be gone
        assert not tech_path.exists()
        
        # Archived should exist
        assert archived_path.exists()
        assert 'archive' in str(archived_path)
        assert 'archived_' in archived_path.name


def test_archive_requires_confirmation():
    """Test archiving requires user confirmation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        container_manager.lock_container(tech_path)
        
        # Without confirmation - should raise
        with pytest.raises(RuntimeError, match="user confirmation"):
            container_manager.archive_technical_container(
                folder, tech_path, user_confirmed=False
            )


def test_archive_unlocked_container_fails():
    """Test cannot archive unlocked container."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        # Don't lock it
        
        with pytest.raises(RuntimeError, match="unlocked container"):
            container_manager.archive_technical_container(
                folder, tech_path, user_confirmed=True
            )


def test_find_active_container_by_distance():
    """Test finding active container by distance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        poni_content_17 = "Distance: 0.17\nPixelSize1: 7.5e-05"
        poni_content_20 = "Distance: 0.20\nPixelSize1: 7.5e-05"
        
        # Create container at 17cm
        pony_data = {'PRIMARY': (poni_content_17, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark1.npy')}}
        np.save(folder / 'dark1.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id_17, tech_file_17 = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        # Create container at 20cm
        pony_data = {'PRIMARY': (poni_content_20, 'primary.poni')}
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark2.npy')}}
        np.save(folder / 'dark2.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id_20, tech_file_20 = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=20.0,
        )
        
        # Find 17cm container
        found = container_manager.find_active_technical_container(folder, distances_cm=17.0)
        assert found == Path(tech_file_17)
        
        # Find 20cm container
        found = container_manager.find_active_technical_container(folder, distances_cm=20.0)
        assert found == Path(tech_file_20)


def test_find_active_excludes_archived():
    """Test find_active_container excludes archived containers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        
        # Should find it
        found = container_manager.find_active_technical_container(folder, distances_cm=17.0)
        assert found == tech_path
        
        # Lock and archive
        container_manager.lock_container(tech_path)
        container_manager.archive_technical_container(folder, tech_path, user_confirmed=True)
        
        # Should NOT find it (archived)
        found = container_manager.find_active_technical_container(folder, distances_cm=17.0)
        assert found is None


# ==================== Primary/Supplementary Tests ====================

def test_set_measurement_primary_status():
    """Test marking measurements as primary/supplementary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        
        # Create container with 2 DARK measurements
        aux_measurements = {
            'DARK': {'PRIMARY': str(folder / 'dark1.npy')},
            'EMPTY': {'PRIMARY': str(folder / 'empty.npy')},
        }
        np.save(folder / 'dark1.npy', np.random.rand(256, 256).astype(np.float32))
        np.save(folder / 'empty.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        
        # Mark event 1 as primary
        container_manager.set_measurement_primary_status(tech_path, event_index=1, is_primary=True)
        
        # Mark event 2 as supplementary
        container_manager.set_measurement_primary_status(
            tech_path, event_index=2, is_primary=False, note="verification measurement"
        )
        
        # Verify attributes
        with h5py.File(tech_path, 'r') as f:
            assert f['/technical/tech_evt_001'].attrs['is_primary'] == True
            assert f['/technical/tech_evt_002'].attrs['is_primary'] == False
            assert 'supplementary_note' in f['/technical/tech_evt_002'].attrs


def test_cannot_modify_locked_container():
    """Test cannot mark measurements in locked container."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        
        # Lock it
        container_manager.lock_container(tech_path)
        
        # Try to modify - should raise
        with pytest.raises(RuntimeError, match="locked container"):
            container_manager.set_measurement_primary_status(tech_path, event_index=1, is_primary=True)


def test_get_primary_measurements():
    """Test getting list of primary measurements."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {
            'DARK': {'PRIMARY': str(folder / 'dark.npy')},
            'EMPTY': {'PRIMARY': str(folder / 'empty.npy')},
        }
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        np.save(folder / 'empty.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        
        # Mark statuses
        container_manager.set_measurement_primary_status(tech_path, 1, is_primary=True)
        container_manager.set_measurement_primary_status(tech_path, 2, is_primary=False)
        
        # Get primary measurements
        primary = container_manager.get_primary_measurements(tech_path)
        
        assert 'DARK' in primary
        assert 1 in primary['DARK']
        assert 'EMPTY' not in primary or 2 not in primary.get('EMPTY', [])


# ==================== Session Container Lock Check Tests ====================

def test_copy_technical_locks_unlocked_container():
    """Test copy_technical_to_session locks unlocked container with auto_lock."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        # Create technical container
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        
        # Create session container
        session_id, session_file = writer.create_session_container(
            folder=folder / 'sessions',
            sample_id='SAMPLE_001',
            operator_id='test_user',
            site_id='test_site',
            machine_name='DIFRA_TEST',
            beam_energy_keV=12.5,
            acquisition_date='2026-02-10',
        )
        
        # Copy with auto_lock
        writer.copy_technical_to_session(tech_file, session_file, auto_lock=True)
        
        # Technical container should now be locked
        assert container_manager.is_container_locked(tech_path)


def test_copy_technical_user_confirm_lock():
    """Test copy_technical_to_session with user confirmation callback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        
        session_id, session_file = writer.create_session_container(
            folder=folder / 'sessions',
            sample_id='SAMPLE_001',
            operator_id='test_user',
            site_id='test_site',
            machine_name='DIFRA_TEST',
            beam_energy_keV=12.5,
            acquisition_date='2026-02-10',
        )
        
        # Mock user confirmation - returns True
        def user_confirm(tech_file):
            return True
        
        writer.copy_technical_to_session(tech_file, session_file, user_confirm_lock=user_confirm)
        
        # Should be locked
        assert container_manager.is_container_locked(tech_path)


def test_copy_technical_already_locked():
    """Test copy_technical_to_session with already locked container."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        
        # Lock it first
        container_manager.lock_container(tech_path)
        
        # Create session
        session_id, session_file = writer.create_session_container(
            folder=folder / 'sessions',
            sample_id='SAMPLE_001',
            operator_id='test_user',
            site_id='test_site',
            machine_name='DIFRA_TEST',
            beam_energy_keV=12.5,
            acquisition_date='2026-02-10',
        )
        
        # Should work without prompting (already locked)
        writer.copy_technical_to_session(tech_file, session_file)
        
        # Should still be locked
        assert container_manager.is_container_locked(tech_path)


# ==================== Integration Tests ====================

def test_poni_validation_in_generate_from_aux_table():
    """Test PONI validation is enforced during container generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        # PONI with wrong distance
        poni_content = "Distance: 0.20\nPixelSize1: 7.5e-05"  # 20cm, not 17cm
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        # Should raise ValueError due to PONI mismatch
        with pytest.raises(ValueError, match="validation failed"):
            technical_container.generate_from_aux_table(
                folder=folder,
                aux_measurements=aux_measurements,
                pony_data=pony_data,
                detector_config=detector_config,
                active_detector_ids=['PRIMARY'],
                distances_cm=17.0,  # User says 17cm, but PONI says 20cm
                validate_poni=True,
            )


def test_poni_validation_can_be_disabled():
    """Test PONI validation can be disabled for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        # PONI with wrong distance
        poni_content = "Distance: 0.20\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        # Should work with validation disabled
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
            validate_poni=False,  # Disabled
        )
        
        assert Path(tech_file).exists()


def test_locked_container_reused_by_multiple_sessions():
    """Test one locked technical container can be used by multiple sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        # Create and lock technical container
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        pony_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            pony_data=pony_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        tech_path = Path(tech_file)
        container_manager.lock_container(tech_path)
        
        # Create multiple sessions using same technical container
        sessions = []
        for i in range(3):
            session_id, session_file = writer.create_session_container(
                folder=folder / 'sessions',
                sample_id=f'SAMPLE_{i:03d}',
                operator_id='test_user',
                site_id='test_site',
                machine_name='DIFRA_TEST',
                beam_energy_keV=12.5,
                acquisition_date='2026-02-10',
            )
            
            writer.copy_technical_to_session(tech_file, session_file)
            sessions.append(session_file)
        
        # All sessions should exist
        for session in sessions:
            assert Path(session).exists()
        
        # Technical container still locked
        assert container_manager.is_container_locked(tech_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
