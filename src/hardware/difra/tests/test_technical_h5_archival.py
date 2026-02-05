"""Tests for technical HDF5 container archival system."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add the project src root to the path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.difra.utils.technical_h5_archival import (
    TechnicalH5Archival,
    format_archival_summary,
)


@pytest.fixture
def temp_folder():
    """Create a temporary folder for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_containers(temp_folder):
    """Create sample HDF5 containers for testing."""
    containers = []
    for i in range(3):
        container = temp_folder / f"technical_abc{i:03d}def456.h5"
        container.write_text(f"Mock HDF5 content {i}")
        containers.append(container)
    return containers


@pytest.fixture
def sample_npy_files(temp_folder):
    """Create sample .npy files for testing."""
    import numpy as np
    
    measurement_types = ["DARK", "EMPTY", "BACKGROUND", "AGBH"]
    aliases = ["PRIMARY", "SECONDARY"]
    files = []
    
    for mtype in measurement_types:
        for alias in aliases:
            filename = temp_folder / f"{mtype}_{alias}_20240205_123456.npy"
            np.save(filename, np.random.rand(256, 256))
            files.append(filename)
    
    return files


def test_find_h5_containers_empty(temp_folder):
    """Test finding containers in empty folder."""
    containers = TechnicalH5Archival.find_h5_containers(str(temp_folder))
    assert len(containers) == 0


def test_find_h5_containers(sample_containers, temp_folder):
    """Test finding existing containers."""
    containers = TechnicalH5Archival.find_h5_containers(str(temp_folder))
    assert len(containers) == 3
    assert all(c.name.startswith("technical_") for c in containers)
    assert all(c.suffix == ".h5" for c in containers)


def test_get_storage_folder(temp_folder):
    """Test getting/creating storage folder."""
    storage = TechnicalH5Archival.get_storage_folder(str(temp_folder))
    
    assert storage.exists()
    assert storage.is_dir()
    assert storage.name == TechnicalH5Archival.STORAGE_SUBFOLDER
    assert storage.parent == temp_folder


def test_archive_container(sample_containers, temp_folder):
    """Test archiving a single container."""
    container = sample_containers[0]
    storage = TechnicalH5Archival.get_storage_folder(str(temp_folder))
    
    success, archived_path, error = TechnicalH5Archival.archive_container(
        container, storage, add_timestamp=False
    )
    
    assert success
    assert error is None
    assert archived_path is not None
    assert archived_path.exists()
    assert not container.exists()  # Original moved
    assert archived_path.parent == storage


def test_archive_container_with_timestamp(sample_containers, temp_folder):
    """Test archiving with timestamp in filename."""
    container = sample_containers[0]
    storage = TechnicalH5Archival.get_storage_folder(str(temp_folder))
    
    success, archived_path, error = TechnicalH5Archival.archive_container(
        container, storage, add_timestamp=True
    )
    
    assert success
    assert "archived_" in archived_path.name
    assert archived_path.suffix == ".h5"


def test_archive_nonexistent_container(temp_folder):
    """Test archiving a non-existent container."""
    fake_container = temp_folder / "fake_technical_123.h5"
    storage = TechnicalH5Archival.get_storage_folder(str(temp_folder))
    
    success, archived_path, error = TechnicalH5Archival.archive_container(
        fake_container, storage
    )
    
    assert not success
    assert archived_path is None
    assert error is not None
    assert "not found" in error.lower()


def test_cleanup_npy_files(sample_npy_files, temp_folder):
    """Test cleaning up .npy files."""
    measurement_types = ["DARK", "EMPTY"]
    aliases = ["PRIMARY", "SECONDARY"]
    
    # Verify files exist before cleanup
    assert len(list(temp_folder.glob("*.npy"))) == 8  # 4 types * 2 aliases
    
    removed = TechnicalH5Archival.cleanup_npy_files(
        str(temp_folder), measurement_types, aliases
    )
    
    # Should remove 2 types * 2 aliases = 4 files
    assert removed == 4
    
    # Verify only DARK and EMPTY files were removed
    remaining = list(temp_folder.glob("*.npy"))
    assert len(remaining) == 4  # BACKGROUND and AGBH remain
    assert all("BACKGROUND" in f.name or "AGBH" in f.name for f in remaining)


def test_cleanup_npy_files_empty_folder(temp_folder):
    """Test cleanup on folder with no .npy files."""
    removed = TechnicalH5Archival.cleanup_npy_files(
        str(temp_folder), ["DARK"], ["PRIMARY"]
    )
    assert removed == 0


def test_archive_all_and_cleanup(sample_containers, sample_npy_files, temp_folder):
    """Test archiving all containers and cleaning up files."""
    measurement_types = ["DARK", "EMPTY", "BACKGROUND", "AGBH"]
    aliases = ["PRIMARY", "SECONDARY"]
    
    archived, cleaned, errors = TechnicalH5Archival.archive_all_and_cleanup(
        str(temp_folder),
        measurement_types=measurement_types,
        aliases=aliases,
        add_timestamp=True,
    )
    
    # Verify counts
    assert archived == 3  # 3 containers
    assert cleaned == 8  # 4 types * 2 aliases
    assert len(errors) == 0
    
    # Verify containers moved
    assert len(list(temp_folder.glob("technical_*.h5"))) == 0
    storage = temp_folder / TechnicalH5Archival.STORAGE_SUBFOLDER
    assert len(list(storage.glob("*.h5"))) == 3
    
    # Verify .npy files removed
    assert len(list(temp_folder.glob("*.npy"))) == 0


def test_archive_all_without_cleanup(sample_containers, sample_npy_files, temp_folder):
    """Test archiving containers without cleaning up .npy files."""
    archived, cleaned, errors = TechnicalH5Archival.archive_all_and_cleanup(
        str(temp_folder),
        measurement_types=None,  # No cleanup
        aliases=None,
        add_timestamp=True,
    )
    
    # Verify containers archived
    assert archived == 3
    assert cleaned == 0  # No cleanup requested
    assert len(errors) == 0
    
    # Verify .npy files remain
    assert len(list(temp_folder.glob("*.npy"))) == 8


def test_archive_all_empty_folder(temp_folder):
    """Test archiving when no containers exist."""
    archived, cleaned, errors = TechnicalH5Archival.archive_all_and_cleanup(
        str(temp_folder)
    )
    
    assert archived == 0
    assert cleaned == 0
    assert len(errors) == 0


def test_format_archival_summary_success():
    """Test formatting successful archival summary."""
    summary = format_archival_summary(3, 8, [])
    
    assert "3 HDF5 container(s)" in summary
    assert "8 measurement file(s)" in summary
    assert "✓" in summary
    assert "⚠" not in summary


def test_format_archival_summary_with_errors():
    """Test formatting summary with errors."""
    errors = ["Error 1", "Error 2", "Error 3"]
    summary = format_archival_summary(2, 5, errors)
    
    assert "2 HDF5 container(s)" in summary
    assert "5 measurement file(s)" in summary
    assert "⚠ 3 error(s)" in summary
    assert "Error 1" in summary


def test_format_archival_summary_many_errors():
    """Test formatting summary with many errors (truncation)."""
    errors = [f"Error {i}" for i in range(10)]
    summary = format_archival_summary(1, 1, errors)
    
    assert "⚠ 10 error(s)" in summary
    assert "... and 7 more" in summary


def test_format_archival_summary_no_operation():
    """Test formatting summary when nothing happened."""
    summary = format_archival_summary(0, 0, [])
    assert "No containers found" in summary


def test_storage_folder_creation_idempotent(temp_folder):
    """Test that storage folder creation is idempotent."""
    storage1 = TechnicalH5Archival.get_storage_folder(str(temp_folder))
    storage2 = TechnicalH5Archival.get_storage_folder(str(temp_folder))
    
    assert storage1 == storage2
    assert storage1.exists()


def test_archive_preserves_content(sample_containers, temp_folder):
    """Test that archiving preserves file content."""
    container = sample_containers[0]
    original_content = container.read_text()
    
    storage = TechnicalH5Archival.get_storage_folder(str(temp_folder))
    success, archived_path, _ = TechnicalH5Archival.archive_container(
        container, storage, add_timestamp=False
    )
    
    assert success
    archived_content = archived_path.read_text()
    assert archived_content == original_content


def test_multiple_archival_sessions(sample_containers, temp_folder):
    """Test running multiple archival sessions with timestamps."""
    import time
    
    storage = TechnicalH5Archival.get_storage_folder(str(temp_folder))
    
    # First archival
    for i, container in enumerate(sample_containers):
        success, _, _ = TechnicalH5Archival.archive_container(
            container, storage, add_timestamp=True
        )
        assert success
        if i < len(sample_containers) - 1:
            time.sleep(0.01)  # Ensure different timestamps
    
    # Verify all archived with unique names
    archived_files = list(storage.glob("*.h5"))
    assert len(archived_files) == 3
    # Check that timestamps make filenames unique
    names = [f.name for f in archived_files]
    assert len(names) == len(set(names))  # All unique


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
