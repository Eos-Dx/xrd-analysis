"""HDF5 container archival system for technical measurements.

Manages archival of completed technical HDF5 containers when starting new measurements.
Moves containers to a 'storage' subfolder and cleans up temporary files.
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class TechnicalH5Archival:
    """Manager for archiving completed technical HDF5 containers."""

    STORAGE_SUBFOLDER = "storage_technical"
    H5_PATTERN = "technical_*.h5"

    @staticmethod
    def find_h5_containers(folder: str) -> List[Path]:
        """Find all technical HDF5 containers in the given folder.

        Args:
            folder: Path to search for containers

        Returns:
            List of Path objects for found containers
        """
        folder_path = Path(folder)
        if not folder_path.exists():
            return []

        containers = list(folder_path.glob("technical_*.h5"))
        logger.debug(f"Found {len(containers)} technical containers in {folder}")
        return containers

    @staticmethod
    def get_storage_folder(base_folder: str) -> Path:
        """Get or create the storage archive folder.

        Args:
            base_folder: Base folder where measurements are saved

        Returns:
            Path to storage folder
        """
        storage_path = Path(base_folder) / TechnicalH5Archival.STORAGE_SUBFOLDER
        storage_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Storage folder: {storage_path}")
        return storage_path

    @staticmethod
    def archive_container(
        container_path: Path, storage_folder: Path, add_timestamp: bool = True
    ) -> Tuple[bool, Optional[Path], Optional[str]]:
        """Archive a single HDF5 container to storage folder.

        Args:
            container_path: Path to the container to archive
            storage_folder: Destination storage folder
            add_timestamp: Whether to add timestamp to archived filename

        Returns:
            Tuple of (success, archived_path, error_message)
        """
        try:
            if not container_path.exists():
                return False, None, f"Container not found: {container_path}"

            # Build archived filename
            if add_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = container_path.stem  # filename without extension
                archived_name = f"{base_name}_archived_{timestamp}.h5"
            else:
                archived_name = container_path.name

            archived_path = storage_folder / archived_name

            # Move file to storage
            shutil.move(str(container_path), str(archived_path))

            logger.info(f"Archived container: {container_path.name} -> {archived_path}")
            return True, archived_path, None

        except Exception as e:
            error_msg = f"Failed to archive {container_path.name}: {e}"
            logger.error(error_msg, exc_info=True)
            return False, None, error_msg

    @staticmethod
    def cleanup_npy_files(folder: str, measurement_types: List[str], aliases: List[str]) -> int:
        """Clean up .npy files associated with archived containers.

        Args:
            folder: Folder containing .npy files
            measurement_types: List of measurement types (e.g., ["DARK", "EMPTY", "BACKGROUND", "AGBH"])
            aliases: List of detector aliases (e.g., ["PRIMARY", "SECONDARY"])

        Returns:
            Number of files removed
        """
        folder_path = Path(folder)
        if not folder_path.exists():
            return 0

        removed_count = 0

        # Pattern: TYPE_ALIAS_*.npy (e.g., DARK_PRIMARY_20240205_123456.npy)
        for meas_type in measurement_types:
            for alias in aliases:
                pattern = f"{meas_type}_{alias}*.npy"
                for npy_file in folder_path.glob(pattern):
                    try:
                        npy_file.unlink()
                        removed_count += 1
                        logger.debug(f"Removed .npy file: {npy_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {npy_file.name}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} .npy measurement files")

        return removed_count

    @staticmethod
    def archive_all_and_cleanup(
        folder: str,
        measurement_types: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        add_timestamp: bool = True,
    ) -> Tuple[int, int, List[str]]:
        """Archive all containers in folder and optionally clean up .npy files.

        Args:
            folder: Folder containing containers and .npy files
            measurement_types: List of measurement types for .npy cleanup (None = no cleanup)
            aliases: List of detector aliases for .npy cleanup (None = no cleanup)
            add_timestamp: Whether to add timestamp to archived filenames

        Returns:
            Tuple of (archived_count, cleaned_npy_count, errors)
        """
        # Find containers
        containers = TechnicalH5Archival.find_h5_containers(folder)
        if not containers:
            logger.debug("No containers to archive")
            return 0, 0, []

        # Get storage folder
        storage_folder = TechnicalH5Archival.get_storage_folder(folder)

        # Archive containers
        archived_count = 0
        errors = []

        for container in containers:
            success, archived_path, error = TechnicalH5Archival.archive_container(
                container, storage_folder, add_timestamp
            )
            if success:
                archived_count += 1
            else:
                errors.append(error)

        # Clean up .npy files if requested
        cleaned_npy_count = 0
        if measurement_types and aliases:
            cleaned_npy_count = TechnicalH5Archival.cleanup_npy_files(
                folder, measurement_types, aliases
            )

        logger.info(
            f"Archival complete: {archived_count} containers archived, "
            f"{cleaned_npy_count} .npy files cleaned, {len(errors)} errors"
        )

        return archived_count, cleaned_npy_count, errors


def format_archival_summary(
    archived_count: int, cleaned_count: int, errors: List[str]
) -> str:
    """Format a user-friendly summary of archival operation.

    Args:
        archived_count: Number of containers archived
        cleaned_count: Number of .npy files cleaned
        errors: List of error messages

    Returns:
        Formatted summary string
    """
    lines = []

    if archived_count > 0:
        lines.append(f"✓ {archived_count} HDF5 container(s) moved to storage")

    if cleaned_count > 0:
        lines.append(f"✓ {cleaned_count} measurement file(s) cleaned up")

    if errors:
        lines.append(f"\n⚠ {len(errors)} error(s):")
        for error in errors[:3]:  # Show max 3 errors
            lines.append(f"  • {error}")
        if len(errors) > 3:
            lines.append(f"  ... and {len(errors) - 3} more")

    if not lines:
        lines.append("No containers found to archive")

    return "\n".join(lines)
