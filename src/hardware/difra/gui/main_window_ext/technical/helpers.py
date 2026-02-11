"""Helper functions for technical measurements."""
import logging
import platform
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_technical_temp_folder(config=None):
    """Get the technical temp folder path with platform-specific defaults.
    
    Args:
        config: Optional config dict from global.json
    
    Returns:
        Path to technical temp folder (created if it doesn't exist)
    """
    # Check config first
    if config and config.get("technical_temp_folder"):
        temp_path = Path(config["technical_temp_folder"])
    else:
        # Platform-specific defaults
        system = platform.system()
        if system == "Darwin":  # macOS
            temp_path = Path.home() / "dev" / "Data" / "tech_temp"
        elif system == "Windows":
            temp_path = Path("C:/dev/Data/tech_temp")
        else:  # Linux or other
            temp_path = Path.home() / "dev" / "Data" / "tech_temp"
    
    # Create directory if it doesn't exist
    try:
        temp_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Technical temp folder: {temp_path}")
        return str(temp_path)
    except Exception as e:
        # Fallback to system temp if creation fails
        logger.warning(f"Failed to create technical temp folder {temp_path}: {e}")
        fallback = Path(tempfile.gettempdir()) / "difra_technical"
        fallback.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using fallback temp folder: {fallback}")
        return str(fallback)


def _get_difra_base_folder(config=None):
    """Get the DIFRA base folder path.
    
    Args:
        config: Optional config dict from global.json
    
    Returns:
        Path to difra base folder (created if it doesn't exist)
    """
    # Check config first
    if config and config.get("difra_base_folder"):
        difra_path = Path(config["difra_base_folder"])
    else:
        # Platform-specific defaults: ~/dev/Data/difra
        system = platform.system()
        if system == "Darwin":  # macOS
            difra_path = Path.home() / "dev" / "Data" / "difra"
        elif system == "Windows":
            difra_path = Path("C:/dev/Data/difra")
        else:  # Linux or other
            difra_path = Path.home() / "dev" / "Data" / "difra"
    
    # Create directory if it doesn't exist
    try:
        difra_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"DIFRA base folder: {difra_path}")
        return str(difra_path)
    except Exception as e:
        logger.warning(f"Failed to create DIFRA base folder {difra_path}: {e}")
        # Fallback to home directory
        return str(Path.home())


def _get_technical_storage_folder(config=None):
    """Get the technical storage folder path (difra/technical).
    
    Args:
        config: Optional config dict from global.json
    
    Returns:
        Path to technical storage folder (created if it doesn't exist)
    """
    # Check config first
    if config and config.get("technical_folder"):
        storage_path = Path(config["technical_folder"])
    else:
        # Get base difra folder and append 'technical' subfolder
        difra_base = _get_difra_base_folder(config)
        storage_path = Path(difra_base) / "technical"
    
    # Create directory if it doesn't exist
    try:
        storage_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Technical storage folder: {storage_path}")
        return str(storage_path)
    except Exception as e:
        logger.warning(f"Failed to create technical storage folder {storage_path}: {e}")
        # Fallback to temp folder
        return _get_technical_temp_folder(config)


def _get_technical_archive_folder(config=None):
    """Get the technical archive folder path (difra/archive/technical).
    
    For archiving raw measurement data after container locking.
    
    Args:
        config: Optional config dict from global.json
    
    Returns:
        Path to technical archive folder (created if it doesn't exist)
    """
    # Check config first
    if config and config.get("technical_archive_folder"):
        archive_path = Path(config["technical_archive_folder"])
    else:
        # Get base difra folder and append 'archive/technical' subfolder
        difra_base = _get_difra_base_folder(config)
        archive_path = Path(difra_base) / "archive" / "technical"
    
    # Create directory if it doesn't exist
    try:
        archive_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Technical archive folder: {archive_path}")
        return str(archive_path)
    except Exception as e:
        logger.warning(f"Failed to create technical archive folder {archive_path}: {e}")
        # Fallback to home directory
        return str(Path.home())


def _get_measurement_default_folder(config=None):
    """Get the measurement default folder path (difra/measurements).
    
    Args:
        config: Optional config dict from global.json
    
    Returns:
        Path to measurement default folder (created if it doesn't exist)
    """
    # Check config first
    if config and config.get("measurements_folder"):
        meas_path = Path(config["measurements_folder"])
    else:
        # Get base difra folder and append 'measurements' subfolder
        difra_base = _get_difra_base_folder(config)
        meas_path = Path(difra_base) / "measurements"
    
    # Create directory if it doesn't exist
    try:
        meas_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Measurement default folder: {meas_path}")
        return str(meas_path)
    except Exception as e:
        logger.warning(f"Failed to create measurement default folder {meas_path}: {e}")
        # Fallback to home directory
        return str(Path.home())


def _get_default_folder(config=None):
    """Get the default folder for technical measurements UI.
    
    Returns the technical storage folder as the default.
    
    Args:
        config: Optional config dict from global.json
    
    Returns:
        Path to default folder
    """
    # Check explicit default_folder in config
    if config and config.get("default_folder"):
        return config["default_folder"]
    
    # Otherwise use technical storage folder
    return _get_technical_storage_folder(config)
