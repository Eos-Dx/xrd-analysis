"""Container management tests: session-copy lock checks and ZIP bundle behavior."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _container_management_shared import *  # noqa: F401,F403
def test_copy_technical_locks_unlocked_container():
    """Test copy_technical_to_session locks unlocked container with auto_lock."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        
        # Create technical container
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
        poni_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        # Should raise ValueError due to PONI mismatch
        with pytest.raises(ValueError, match="validation failed"):
            technical_container.generate_from_aux_table(
                folder=folder,
                aux_measurements=aux_measurements,
                poni_data=poni_data,
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
        poni_data = {'PRIMARY': (poni_content, 'primary.poni')}
        detector_config = [{'id': 'PRIMARY', 'alias': 'PRIMARY', 'type': 'Pilatus',
                           'size': [256, 256], 'pixel_size_um': 172.0}]
        aux_measurements = {'DARK': {'PRIMARY': str(folder / 'dark.npy')}}
        np.save(folder / 'dark.npy', np.random.rand(256, 256).astype(np.float32))
        
        # Should work with validation disabled
        tech_id, tech_file = technical_container.generate_from_aux_table(
            folder=folder,
            aux_measurements=aux_measurements,
            poni_data=poni_data,
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


# ==================== ZIP Bundle Tests ====================

def test_create_container_bundle_preserves_operator_structure():
    """Test bundle exporter keeps operator folder structure and container."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        session_folder = base / "sessions"
        session_folder.mkdir(parents=True, exist_ok=True)

        session_id, session_file = writer.create_session_container(
            folder=session_folder,
            sample_id="SAMPLE_BUNDLE_001",
            operator_id="test_user",
            site_id="test_site",
            machine_name="DIFRA_TEST",
            beam_energy_keV=12.5,
            acquisition_date="2026-02-13",
        )

        operator_folder = base / "operator_run_001"
        nested_dir = operator_folder / "raw" / "det_primary"
        nested_dir.mkdir(parents=True, exist_ok=True)
        (nested_dir / "frame_001.txt").write_text("1 2 3\n4 5 6\n", encoding="utf-8")
        (nested_dir / "frame_001.dsc").write_text("[F0]\nType=i16\n", encoding="utf-8")
        (operator_folder / "state" / "sample_state.json").parent.mkdir(
            parents=True, exist_ok=True
        )
        (operator_folder / "state" / "sample_state.json").write_text(
            '{"sample_id":"SAMPLE_BUNDLE_001"}', encoding="utf-8"
        )

        bundle_zip = create_container_bundle(
            container_file=session_file,
            source_folder=operator_folder,
            output_zip=base / "session_bundle.zip",
            source_arcname=operator_folder.name,
        )

        assert bundle_zip.exists()

        with zipfile.ZipFile(bundle_zip, "r") as zf:
            names = set(zf.namelist())

        assert Path(session_file).name in names
        assert "operator_run_001/raw/det_primary/frame_001.txt" in names
        assert "operator_run_001/raw/det_primary/frame_001.dsc" in names
        assert "operator_run_001/state/sample_state.json" in names


def test_open_container_bundle_loads_session_container():
    """Test opener can load session container directly from ZIP bundle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        session_folder = base / "sessions"
        session_folder.mkdir(parents=True, exist_ok=True)

        session_id, session_file = writer.create_session_container(
            folder=session_folder,
            sample_id="SAMPLE_BUNDLE_002",
            operator_id="test_user",
            site_id="test_site",
            machine_name="DIFRA_TEST",
            beam_energy_keV=12.5,
            acquisition_date="2026-02-13",
        )

        payload = base / "operator_payload"
        payload.mkdir(parents=True, exist_ok=True)
        (payload / "notes.txt").write_text("bundle payload", encoding="utf-8")

        bundle_zip = create_container_bundle(
            container_file=session_file,
            source_folder=payload,
            output_zip=base / "bundle_open_test.zip",
        )

        opened = open_container_bundle(bundle_zip, validate=False)
        metadata = opened.get_metadata()

        assert metadata.get("container_type") == schema.CONTAINER_TYPE_SESSION
        assert metadata.get("sample_id") == "SAMPLE_BUNDLE_002"

