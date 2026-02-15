"""Container management tests: PONI validation and locking behavior."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _container_management_shared import *  # noqa: F401,F403
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
    schema.validate_poni_distance(poni_content, user_distance_cm=17.0)


def test_validate_poni_distance_within_tolerance():
    """Test PONI validation within 5% tolerance."""
    poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
    
    # 17.5 cm is 2.9% deviation - should pass
    schema.validate_poni_distance(poni_content, user_distance_cm=17.5)
    
    # 17.8 cm is 4.7% deviation - should pass
    schema.validate_poni_distance(poni_content, user_distance_cm=17.8)


def test_validate_poni_distance_exceeds_tolerance():
    """Test PONI validation fails when exceeding 5% tolerance."""
    poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
    
    # 20 cm is 17.6% deviation - should fail
    with pytest.raises(ValueError, match="validation failed"):
        schema.validate_poni_distance(poni_content, user_distance_cm=20.0)


def test_validate_poni_distance_custom_tolerance():
    """Test PONI validation with custom tolerance."""
    poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
    
    # 18 cm is 5.9% deviation
    # Should fail with 5% tolerance
    with pytest.raises(ValueError):
        schema.validate_poni_distance(poni_content, user_distance_cm=18.0, tolerance_percent=5.0)
    
    # Should pass with 10% tolerance
    schema.validate_poni_distance(poni_content, user_distance_cm=18.0, tolerance_percent=10.0)


# ==================== Container Locking Tests ====================

def test_container_initially_unlocked():
    """Test that newly created containers are unlocked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        # Create technical container
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        poni_data = {'PRIMARY': (poni_content, 'primary.poni')}
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
            poni_data=poni_data,
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
        poni_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            poni_data=poni_data,
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
        poni_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            poni_data=poni_data,
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
        poni_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            poni_data=poni_data,
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
