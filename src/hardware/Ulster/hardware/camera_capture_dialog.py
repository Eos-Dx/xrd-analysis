import sys
from PyQt5.QtWidgets import (
    QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QFileDialog, QMessageBox, QComboBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage

import cv2
import os

class CameraCaptureDialog(QDialog):
    def __init__(self, parent=None, default_folder=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Capture")

        # Robustly set default folder
        if not default_folder or not os.path.isdir(default_folder):
            default_folder = os.getcwd()

        self.layout = QVBoxLayout(self)

        # --- Camera selection dropdown ---
        self.camera_indices = self.get_available_cameras()
        if not self.camera_indices:
            QMessageBox.critical(self, "Camera Error", "No available cameras found!")
            self.reject()
            return

        self.camera_select = QComboBox()
        for idx in self.camera_indices:
            self.camera_select.addItem(f"Camera {idx}", idx)
        self.layout.addWidget(QLabel("Select camera:"))
        self.layout.addWidget(self.camera_select)

        # Camera and preview
        self.camera = None
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(480, 360)
        self.layout.addWidget(self.preview_label)

        # File controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("File name:"))
        self.filename_edit = QLineEdit("capture.jpg")
        controls_layout.addWidget(self.filename_edit)
        controls_layout.addWidget(QLabel("Folder:"))
        self.folder_edit = QLineEdit(default_folder)
        self.folder_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse…")
        controls_layout.addWidget(self.folder_edit)
        controls_layout.addWidget(browse_btn)
        self.layout.addLayout(controls_layout)

        browse_btn.clicked.connect(self.select_folder)

        # Capture/save button
        capture_btn = QPushButton("Capture && Save")
        capture_btn.clicked.connect(self.capture_and_save)
        self.layout.addWidget(capture_btn)

        # Variables
        self.current_frame = None

        # Connect camera selection and open first camera
        self.camera_select.currentIndexChanged.connect(self.open_selected_camera)
        self.open_selected_camera()  # Open initial camera

        # Start timer for real-time preview
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS

    @staticmethod
    def get_available_cameras(max_tested=5):
        """Returns a list of available camera indices (0, 1, ...)."""
        available = []
        for idx in range(max_tested):
            cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                available.append(idx)
                cap.release()
        return available

    def open_selected_camera(self):
        idx = self.camera_select.currentData()
        if self.camera is not None:
            self.camera.release()
        self.camera = cv2.VideoCapture(idx)
        if not self.camera.isOpened():
            QMessageBox.critical(self, "Error", f"Could not open camera {idx}.")
            self.reject()
            return
        self.current_frame = None  # Clear last frame

    def update_frame(self):
        if self.camera is None:
            return
        ret, frame = self.camera.read()
        if not ret:
            return
        self.current_frame = frame  # Store BGR frame for saving!
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.preview_label.width(), self.preview_label.height(),
            Qt.KeepAspectRatio
        )
        self.preview_label.setPixmap(pixmap)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)

    def capture_and_save(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Image", "No frame to save!")
            return
        filename = self.filename_edit.text().strip()
        folder = self.folder_edit.text().strip()
        if not filename:
            QMessageBox.warning(self, "Filename", "Please enter a filename.")
            return
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Folder", "Selected folder does not exist.")
            return
        full_path = os.path.join(folder, filename)
        cv2.imwrite(full_path, self.current_frame)
        self.selected_image_path = full_path
        self.accept()

    def closeEvent(self, event):
        self.timer.stop()
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
        super().closeEvent(event)
