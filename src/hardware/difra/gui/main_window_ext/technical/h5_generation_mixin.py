"""H5 Generation Mixin - Technical container and metadata generation."""
import json
import logging
import os
import re
import uuid

logger = logging.getLogger(__name__)

from hardware.difra.gui.container_api import get_schema, get_technical_container

# Import Qt for type hints and usage
try:
    from PyQt5.QtWidgets import QComboBox, QDialog, QFileDialog, QInputDialog, QMessageBox, QCheckBox
    from PyQt5.QtCore import Qt
except ImportError:
    # Test stubs
    class QComboBox:
        def __init__(self, *args, **kwargs):
            pass
    
    class QDialog:
        Accepted = 1
        Rejected = 0
    
    class QFileDialog:
        @staticmethod
        def getOpenFileName(*args, **kwargs):
            return "", ""
    
    class QInputDialog:
        @staticmethod
        def getDouble(*args, **kwargs):
            return 17.0, True
    
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
            pass
        
        @staticmethod
        def critical(*args, **kwargs):
            pass
    
    class QCheckBox:
        def __init__(self, *args, **kwargs):
            pass
        
        def isChecked(self):
            return False
    
    class Qt:
        UserRole = 32
from hardware.difra.gui.main_window_ext.technical.h5_generation_meta_mixin import H5GenerationMetaMixin
from hardware.difra.gui.main_window_ext.technical.h5_generation_container_mixin import H5GenerationContainerMixin


class H5GenerationMixin(H5GenerationMetaMixin, H5GenerationContainerMixin):
    """Mixin for H5 container and metadata generation operations."""

    pass
