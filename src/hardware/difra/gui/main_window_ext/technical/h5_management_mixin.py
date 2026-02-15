"""H5 Management Mixin - Container validation, locking, archiving, and loading."""
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

from hardware.difra.gui.container_api import (
    get_container_manager,
    get_schema,
    get_technical_validator,
)

# Import Qt for type hints and usage
try:
    from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox
except ImportError:
    # Test stubs
    class QFileDialog:
        @staticmethod
        def getOpenFileName(*args, **kwargs):
            return "", ""
    
    class QInputDialog:
        @staticmethod
        def getText(*args, **kwargs):
            return "", False
    
    class QMessageBox:
        Yes, No = 1, 0
        
        @staticmethod
        def question(*args, **kwargs):
            return QMessageBox.Yes
        
        @staticmethod
        def information(*args, **kwargs):
            pass
        
        @staticmethod
        def warning(*args, **kwargs):
            return QMessageBox.Yes
        
        @staticmethod
        def critical(*args, **kwargs):
            pass
from hardware.difra.gui.main_window_ext.technical.h5_management_locking_mixin import H5ManagementLockingMixin
from hardware.difra.gui.main_window_ext.technical.h5_management_loading_mixin import H5ManagementLoadingMixin


class H5ManagementMixin(H5ManagementLockingMixin, H5ManagementLoadingMixin):
    """Mixin for H5 container loading/validation/lock operations."""

    pass
