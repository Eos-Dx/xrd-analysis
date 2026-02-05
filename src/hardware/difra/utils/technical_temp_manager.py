"""Temporary folder management for technical measurements.

Provides a staging area for technical measurement files that:
1. Stores captured .npy files temporarily
2. Copies files into HDF5 container
3. Cleans up temporary files after successful container creation
4. Preserves the old flat-file format in the user-specified folder
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TechnicalTempManager:
    """Manager for temporary technical measurement files."""

    def __init__(self, base_temp_dir: Optional[str] = None):
        """Initialize temporary folder manager.

        Args:
            base_temp_dir: Optional base directory for temp files.
                          If None, uses system temp directory.
        """
        self.base_temp_dir = base_temp_dir
        self._active_session_dir: Optional[Path] = None
        self._staged_files: Dict[str, str] = {}  # measurement_id -> file_path
        logger.debug(f"TechnicalTempManager initialized with base: {base_temp_dir}")

    def create_session_dir(self, session_id: Optional[str] = None) -> Path:
        """Create a new temporary session directory for technical measurements.

        Args:
            session_id: Optional session identifier for the directory name

        Returns:
            Path to the created session directory
        """
        if session_id:
            dir_name = f"difra_technical_{session_id}"
        else:
            import uuid
            dir_name = f"difra_technical_{uuid.uuid4().hex[:8]}"

        if self.base_temp_dir:
            base = Path(self.base_temp_dir)
            base.mkdir(parents=True, exist_ok=True)
            session_dir = base / dir_name
        else:
            # Use system temp directory
            session_dir = Path(tempfile.gettempdir()) / dir_name

        session_dir.mkdir(parents=True, exist_ok=True)
        self._active_session_dir = session_dir

        logger.info(f"Created technical measurement session directory: {session_dir}")
        return session_dir

    def get_session_dir(self) -> Optional[Path]:
        """Get the current active session directory.

        Returns:
            Path to active session directory, or None if not created
        """
        return self._active_session_dir

    def stage_file(self, source_path: str, measurement_id: str) -> str:
        """Stage a measurement file in the temporary session directory.

        Args:
            source_path: Path to the source .npy file
            measurement_id: Identifier for this measurement (e.g., "DARK_PRIMARY")

        Returns:
            Path to the staged file

        Raises:
            ValueError: If no active session directory
            FileNotFoundError: If source file doesn't exist
        """
        if not self._active_session_dir:
            raise ValueError("No active session directory. Call create_session_dir() first.")

        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Create staged file with measurement ID
        staged_path = self._active_session_dir / f"{measurement_id}.npy"

        # Copy file to staging area
        shutil.copy2(source, staged_path)
        self._staged_files[measurement_id] = str(staged_path)

        logger.debug(f"Staged file: {measurement_id} -> {staged_path}")
        return str(staged_path)

    def get_staged_files(self) -> Dict[str, str]:
        """Get all currently staged files.

        Returns:
            Dictionary mapping measurement_id to staged file path
        """
        return self._staged_files.copy()

    def cleanup_session(self, preserve_files: Optional[List[str]] = None):
        """Clean up the current session directory.

        Args:
            preserve_files: Optional list of file paths to preserve (won't be deleted)
        """
        if not self._active_session_dir:
            logger.warning("No active session directory to clean up")
            return

        preserve_set = set(preserve_files or [])

        try:
            # Remove staged files not in preserve list
            removed_count = 0
            for measurement_id, file_path in self._staged_files.items():
                if file_path not in preserve_set:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            removed_count += 1
                            logger.debug(f"Removed staged file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove staged file {file_path}: {e}")

            # Try to remove the session directory if empty
            try:
                remaining = list(self._active_session_dir.iterdir())
                if not remaining:
                    self._active_session_dir.rmdir()
                    logger.info(f"Removed empty session directory: {self._active_session_dir}")
                else:
                    logger.info(
                        f"Session directory not empty ({len(remaining)} files remaining): "
                        f"{self._active_session_dir}"
                    )
            except Exception as e:
                logger.warning(f"Failed to remove session directory: {e}")

            logger.info(f"Cleaned up {removed_count} staged files from session")

        finally:
            self._staged_files.clear()
            self._active_session_dir = None

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old temporary session directories.

        Args:
            max_age_hours: Maximum age in hours for keeping temp directories
        """
        import time

        if self.base_temp_dir:
            base = Path(self.base_temp_dir)
        else:
            base = Path(tempfile.gettempdir())

        if not base.exists():
            return

        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        removed_count = 0

        try:
            for item in base.iterdir():
                if item.is_dir() and item.name.startswith("difra_technical_"):
                    try:
                        # Check directory age
                        mtime = item.stat().st_mtime
                        if mtime < cutoff_time:
                            shutil.rmtree(item)
                            removed_count += 1
                            logger.info(f"Removed old session directory: {item}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old session {item}: {e}")

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old session directories")

        except Exception as e:
            logger.error(f"Error during old sessions cleanup: {e}", exc_info=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup session."""
        self.cleanup_session()
        return False


def get_technical_temp_dir() -> Path:
    """Get the base temporary directory for technical measurements.

    Returns:
        Path to the base temporary directory
    """
    # Use a subdirectory in system temp to keep things organized
    temp_base = Path(tempfile.gettempdir()) / "difra_technical"
    temp_base.mkdir(parents=True, exist_ok=True)
    return temp_base
