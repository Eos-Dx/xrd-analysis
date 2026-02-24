"""Container management tests: archiving and primary/supplementary flags."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _container_management_shared import *  # noqa: F401,F403
def test_archive_locked_container():
    """Test archiving a locked container."""
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
        poni_data = {'PRIMARY': (poni_content_17, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark1.npy')}}
        np.save(folder / 'dark1.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id_17, tech_file_17 = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            poni_data=poni_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=17.0,
        )
        
        # Create container at 20cm
        poni_data = {'PRIMARY': (poni_content_20, 'primary.poni')}
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark2.npy')}}
        np.save(folder / 'dark2.npy', np.random.rand(256, 256).astype(np.float32))
        
        tech_id_20, tech_file_20 = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            poni_data=poni_data,
            detector_config=detector_config,
            active_detector_ids=['PRIMARY'],
            distances_cm=20.0,
        )
        
        # Find 17cm container
        found = container_manager.find_active_technical_container(folder, distance_cm=17.0)
        assert found == Path(tech_file_17)
        
        # Find 20cm container
        found = container_manager.find_active_technical_container(folder, distance_cm=20.0)
        assert found == Path(tech_file_20)


def test_find_active_excludes_archived():
    """Test find_active_container excludes archived containers."""
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
        
        # Should find it
        found = container_manager.find_active_technical_container(folder, distance_cm=17.0)
        assert found == tech_path
        
        # Lock and archive
        container_manager.lock_container(tech_path)
        container_manager.archive_technical_container(folder, tech_path, user_confirmed=True)
        
        # Should NOT find it (archived)
        found = container_manager.find_active_technical_container(folder, distance_cm=17.0)
        assert found is None


def test_archive_technical_data_files_includes_poni_by_default():
    """Default technical raw archive patterns must include .poni files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        container_path = folder / "technical_demo.nxs.h5"
        container_path.write_text("demo", encoding="utf-8")

        (folder / "capture.txt").write_text("txt", encoding="utf-8")
        (folder / "capture.dsc").write_text("dsc", encoding="utf-8")
        np.save(folder / "capture.npy", np.array([1, 2, 3], dtype=np.int16))
        (folder / "primary.poni").write_text("Distance: 0.17\n", encoding="utf-8")
        (folder / "capture_state.json").write_text('{"ok": true}', encoding="utf-8")

        archive_folder = folder / "archive_payload"
        archived_count = container_manager.archive_technical_data_files(
            container_path=container_path,
            archive_folder=archive_folder,
            file_patterns=None,
        )

        assert archived_count == 5
        assert (archive_folder / "capture.txt").exists()
        assert (archive_folder / "capture.dsc").exists()
        assert (archive_folder / "capture.npy").exists()
        assert (archive_folder / "primary.poni").exists()
        assert (archive_folder / "capture_state.json").exists()
        assert not (folder / "primary.poni").exists()


# ==================== Primary/Supplementary Tests ====================

def test_set_measurement_primary_status():
    """Test marking measurements as primary/supplementary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        poni_data = {'PRIMARY': (poni_content, 'primary.poni')}
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
            poni_data=poni_data,
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
            event_1 = f[f"{schema.GROUP_TECHNICAL}/tech_evt_000001"]
            event_2 = f[f"{schema.GROUP_TECHNICAL}/tech_evt_000002"]
            assert event_1.attrs['is_primary'] == True
            assert event_2.attrs['is_primary'] == False
            assert 'supplementary_note' in event_2.attrs


def test_cannot_modify_locked_container():
    """Test cannot mark measurements in locked container."""
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
        poni_data = {'PRIMARY': (poni_content, 'primary.poni')}
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
            poni_data=poni_data,
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
