import base64
import hashlib
import json
import os
import shutil
import string
from pathlib import Path
from urllib.parse import unquote, urlparse

from PyQt5.QtCore import QRectF, QTimer
from PyQt5.QtGui import QColor, QPen, QPixmap
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsRectItem

from hardware.difra.gui.image_view_ext.point_editing_extension import null_dict
from hardware.difra.gui.main_window_ext.state_saver_io_mixin import StateSaverIOMixin
from hardware.difra.gui.main_window_ext.state_saver_restore_mixin import StateSaverRestoreMixin


class StateSaverMixin(StateSaverIOMixin, StateSaverRestoreMixin):
    _AUTOSAVE_DRIVE = StateSaverIOMixin._AUTOSAVE_DRIVE
    AUTO_STATE_FILE = StateSaverIOMixin.AUTO_STATE_FILE
    PREV_STATE_FILE = StateSaverIOMixin.PREV_STATE_FILE

    pass
