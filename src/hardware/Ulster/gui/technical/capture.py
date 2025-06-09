from PyQt5.QtCore import  QThread, pyqtSignal
import os
import numpy as np
import seaborn as sns
from PyQt5.QtWidgets import QDialog, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from xrdanalysis.data_processing.azimuthal_integration import (
    initialize_azimuthal_integrator_df,
    initialize_azimuthal_integrator_poni_text
)

class CaptureWorker(QThread):
    # emit (success: bool, txt_filename: str)
    finished = pyqtSignal(bool, str)

    def __init__(self, detector_controller, integration_time, txt_filename, parent=None):
        super().__init__(parent)
        self.detector_controller = detector_controller
        self.integration_time = integration_time
        self.txt_filename = txt_filename

    def run(self):
        success = self.detector_controller.capture_point(
            1,
            self.integration_time,
            self.txt_filename
        )
        self.finished.emit(success, self.txt_filename)


def validate_folder(path: str) -> str:
    if not path:
        path = os.getcwd()
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        path = os.getcwd()
    if not os.access(path, os.W_OK):
        path = os.getcwd()
    return path


def show_measurement_window(
    measurement_filename: str,
    mask: np.ndarray,
    poni_text: str = None,
    parent=None
) -> QDialog:
    """
    Opens a dialog window displaying the raw 2D image and its azimuthal integration.

    Parameters:
    - measurement_filename: Path to the .txt or .npy data file.
    - mask: 2D numpy array mask to apply during integration.
    - poni_text: Optional PONI file contents as text. If provided, uses PONI integrator.
    - parent: Optional Qt parent for the dialog.

    Returns:
    - The QDialog instance (which has already been shown).
    """
    # Load data
    data = np.load(measurement_filename)

    # Choose integrator
    if poni_text:
        ai = initialize_azimuthal_integrator_poni_text(poni_text)
    else:
        # Fallback: manual integration parameters
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        center_row, center_column = max_idx
        pixel_size = 55e-6
        wavelength = 1.54
        sample_distance_mm = 100.0
        ai = initialize_azimuthal_integrator_df(
            pixel_size,
            center_column,
            center_row,
            wavelength,
            sample_distance_mm
        )

    # Perform integration
    npt = 200
    try:
        result = ai.integrate1d(
            data,
            npt,
            unit="q_nm^-1",
            error_model="azimuthal",
            mask=mask
        )
        radial = result.radial
        intensity = result.intensity
        cake, _, _ = ai.integrate2d(
            data,
            200,
            npt_azim=180,
            mask=mask
        )
    except Exception as e:
        print(f"Error integrating data: {e}")
        return None

    # Create dialog and layout
    dialog = QDialog(parent)
    dialog.setWindowTitle(f"Azimuthal Integration: {os.path.basename(measurement_filename)}")
    layout = QHBoxLayout(dialog)

    # Create figure and canvas
    fig = Figure(figsize=(6, 6))
    canvas = FigureCanvas(fig)

    # Top-left: raw 2D heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    sns.heatmap(data, robust=True, square=True, ax=ax1)
    ax1.set_title("2D Image")

    # Top-right: 1D integration
    ax2 = fig.add_subplot(2, 2, 2)
    sns.lineplot(x=radial, y=intensity, marker='o', ax=ax2)
    ax2.set_title("Azimuthal Integration")
    ax2.set_xlabel("q (nm^-1)")
    ax2.set_ylabel("Intensity")
    ax2.set_yscale("log")

    # Bottom-left: cake representation
    ax3 = fig.add_subplot(2, 2, 3)
    sns.heatmap(cake[:, 30:], robust=True, square=True, ax=ax3)
    ax3.set_title("Cake Representation")

    # Bottom-right: deviation map
    cake2 = cake[:, 30:]
    mask_zero = (cake2 == 0)
    col_sums = cake2.sum(axis=0)
    valid_counts = (~mask_zero).sum(axis=0)
    col_means = np.divide(col_sums, valid_counts, where=valid_counts > 0)
    pct_dev = (cake2 - col_means[np.newaxis, :]) / col_means[np.newaxis, :] * 100

    ax4 = fig.add_subplot(2, 2, 4)
    sns.heatmap(pct_dev, robust=True, square=True, ax=ax4)
    ax4.set_title("Deviation (%)")

    # Show the canvas
    layout.addWidget(canvas)
    dialog.resize(700, 700)
    dialog.show()

    return dialog