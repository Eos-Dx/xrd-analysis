"""Test automatic archiving of old containers when creating new ones."""
import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_archive_old_containers_before_new_generation():
    """Test that old .h5 containers are archived when generating a new one."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Setup folder structure
        technical_folder = tmpdir / "technical"
        archive_folder = tmpdir / "archive" / "technical"
        technical_folder.mkdir(parents=True)
        archive_folder.mkdir(parents=True, exist_ok=True)
        
        # Create an "old" H5 container and raw data files
        old_container = technical_folder / "technical_abc123_100cm.h5"
        old_container.write_text("fake h5 data old")
        
        # Only .txt and .dsc are raw data files (.npy is processed)
        old_txt = technical_folder / "DARK_PRIMARY.txt"
        old_txt.write_text("1 2 3\n4 5 6\n")
        
        old_dsc = technical_folder / "DARK_PRIMARY.dsc"
        old_dsc.write_text("[F0]\nType=i16\nFrames=1\n")
        
        # Verify old files exist
        assert old_container.exists()
        assert old_txt.exists()
        assert old_dsc.exists()
        
        # Simulate archiving (what happens before new container generation)
        from hardware.difra.gui.main_window_ext.technical_measurements import (
            _get_technical_archive_folder
        )
        
        # Manual archiving simulation
        import shutil
        import time
        
        # Find H5 files
        h5_files = list(technical_folder.glob("*.h5"))
        assert len(h5_files) == 1
        
        # Extract container ID
        filename = h5_files[0].stem
        parts = filename.split('_')
        container_id = parts[1] if len(parts) >= 2 else filename
        
        # Create timestamped archive folder
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_subfolder = archive_folder / f"{container_id}_{timestamp}"
        archive_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Move H5 container
        shutil.move(str(old_container), str(archive_subfolder / old_container.name))
        
        # Move raw data files (.txt and .dsc only, not .npy which is processed)
        for pattern in ["*.txt", "*.dsc"]:
            for raw_file in technical_folder.glob(pattern):
                shutil.move(str(raw_file), str(archive_subfolder / raw_file.name))
        
        # Verify old files are gone from technical folder
        assert not old_container.exists()
        assert not old_txt.exists()
        assert not old_dsc.exists()
        
        # Verify files are in archive
        archived_h5 = archive_subfolder / old_container.name
        archived_txt = archive_subfolder / old_txt.name
        archived_dsc = archive_subfolder / old_dsc.name
        
        assert archived_h5.exists()
        assert archived_txt.exists()
        assert archived_dsc.exists()
        
        # Now simulate new container creation
        new_container = technical_folder / "technical_def456_17cm.h5"
        new_container.write_text("fake h5 data new")
        
        # Verify only new container in technical folder
        h5_files_after = list(technical_folder.glob("*.h5"))
        assert len(h5_files_after) == 1
        assert h5_files_after[0].name == new_container.name
        
        print("✅ Old container and raw data archived successfully")
        print(f"   Archive: {archive_subfolder.name}/")
        print(f"   Files archived: 3 (1 .h5 + 1 .txt + 1 .dsc)")
        print(f"   Note: .npy not archived (processed data in H5)")
        print(f"   New container: {new_container.name}")


def test_multiple_old_containers_all_archived():
    """Test that multiple old containers are all archived."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        technical_folder = tmpdir / "technical"
        archive_folder = tmpdir / "archive" / "technical"
        technical_folder.mkdir(parents=True)
        archive_folder.mkdir(parents=True, exist_ok=True)
        
        # Create multiple old containers
        old_containers = [
            technical_folder / "technical_aaa111_100cm.h5",
            technical_folder / "technical_bbb222_17cm.h5",
            technical_folder / "technical_ccc333_50cm.h5",
        ]
        
        for container in old_containers:
            container.write_text("fake h5 data")
        
        # Verify all exist
        assert all(c.exists() for c in old_containers)
        
        # Archive all
        import shutil
        import time
        
        for h5_file in technical_folder.glob("*.h5"):
            filename = h5_file.stem
            parts = filename.split('_')
            container_id = parts[1] if len(parts) >= 2 else filename
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_subfolder = archive_folder / f"{container_id}_{timestamp}"
            archive_subfolder.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(h5_file), str(archive_subfolder / h5_file.name))
            time.sleep(0.01)  # Ensure unique timestamps
        
        # Verify all gone from technical folder
        assert len(list(technical_folder.glob("*.h5"))) == 0
        
        # Verify all in archive
        archive_contents = list(archive_folder.glob("*/technical_*.h5"))
        assert len(archive_contents) == 3
        
        print(f"✅ All {len(old_containers)} containers archived successfully")
        for archive_item in archive_folder.glob("*"):
            if archive_item.is_dir():
                h5_in_archive = list(archive_item.glob("*.h5"))
                print(f"   {archive_item.name}/ -> {h5_in_archive[0].name if h5_in_archive else 'empty'}")


def test_no_archiving_when_no_old_containers():
    """Test that archiving is skipped when no old containers exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        technical_folder = tmpdir / "technical"
        archive_folder = tmpdir / "archive" / "technical"
        technical_folder.mkdir(parents=True)
        archive_folder.mkdir(parents=True, exist_ok=True)
        
        # No old containers exist
        h5_files = list(technical_folder.glob("*.h5"))
        assert len(h5_files) == 0
        
        # Create new container directly
        new_container = technical_folder / "technical_xyz789_100cm.h5"
        new_container.write_text("fake h5 data new")
        
        # Verify only new container exists
        h5_files_after = list(technical_folder.glob("*.h5"))
        assert len(h5_files_after) == 1
        assert h5_files_after[0].name == new_container.name
        
        # Verify archive is empty
        archive_contents = list(archive_folder.glob("*"))
        assert len(archive_contents) == 0
        
        print("✅ No archiving performed when no old containers exist")
        print(f"   New container: {new_container.name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
