import os

import numpy as np
import seaborn as sns
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog, QHBoxLayout

from xrdanalysis.data_processing.azimuthal_integration import (
    initialize_azimuthal_integrator_df,
    initialize_azimuthal_integrator_poni_text,
)


class CaptureWorker(QThread):
    # Emits (success: bool, result_files: dict) e.g. {"WAXS": path1, "SAXS": path2}
    finished = pyqtSignal(bool, dict)

    def __init__(
        self,
        detector_controller,
        integration_time,
        txt_filename_base,
        parent=None,
    ):
        super().__init__(parent)
        self.detector_controller = detector_controller
        self.integration_time = integration_time
        self.txt_filename_base = (
            txt_filename_base  # Only the base; suffixes will be added
        )

    def run(self):
        # Perform dual acquisition
        success = self.detector_controller.capture_point(
            1, self.integration_time, self.txt_filename_base
        )
        # Collect the expected filenames for both detectors
        result_files = {}
        for name in ["WAXS", "SAXS"]:
            result_files[name] = f"{self.txt_filename_base}_{name}.txt"
        self.finished.emit(success, result_files)


def validate_folder(path: str):
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
    parent=None,
    columns_to_remove: int = 30,
    goodness: float = 0.0,
):
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
            sample_distance_mm,
        )

    # Perform integration
    npt = 200
    try:
        result = ai.integrate1d(
            data, npt, unit="q_nm^-1", error_model="azimuthal", mask=mask
        )
        radial = result.radial
        intensity = result.intensity
        std = result.std
        sigma = result.sigma
        cake, _, _ = ai.integrate2d(data, 200, npt_azim=180, mask=mask)
    except Exception as e:
        print(f"Error integrating data: {e}")
        return None

    # Create dialog and layout
    try:
        dialog = QDialog(parent)
        dialog.setWindowTitle(
            f"Azimuthal Integration: {os.path.basename(measurement_filename)}"
        )
        layout = QHBoxLayout(dialog)
    except Exception as e:
        raise e

    # Create figure and canvas
    fig = Figure(figsize=(6, 6))
    canvas = FigureCanvas(fig)

    # Top-left: raw 2D heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    sns.heatmap(data, robust=True, square=True, ax=ax1)
    ax1.set_title("2D Image")

    # Top-right: 1D integration
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Top-right: 1D integration
    ax2 = fig.add_subplot(2, 2, 2)

    # 1) main curve with σ‐errorbars in “candle” style
    ax2.errorbar(
        radial,
        intensity,
        yerr=sigma,
        fmt="-o",  # line + circle marker
        markersize=3,
        linewidth=1,
        ecolor="black",  # error‐bar color
        capsize=3,  # horizontal bar at ends
        capthick=1,  # thickness of caps
        label="Intensity ± σ",
    )

    # 2) extend x-limits by 30%
    xmin, xmax = radial.min(), radial.max()
    ax2.set_xlim(xmin, xmax * 1.3)

    ax2.set_yscale("log")
    ax2.set_title("Azimuthal Integration")
    ax2.set_xlabel("q (nm⁻¹)")
    ax2.set_ylabel("Intensity")
    ax2.legend(loc="upper right", fontsize="small")

    # 3) inset for std (top-left)
    ax_std = inset_axes(
        ax2,
        width="30%",  # 30% of ax2 width
        height="30%",  # 30% of ax2 height
        bbox_to_anchor=(0.05, -0.2, 1, 1),  # x0, y0, w, h in axes fraction
        bbox_transform=ax2.transAxes,
    )
    ax_std.plot(radial, std, "-", linewidth=1)
    ax_std.set_title("std", fontsize="x-small")
    ax_std.tick_params(labelsize="x-small", axis="both", which="both")

    # 4) inset for SNR = I / σ (below the std inset)
    snr = intensity / sigma
    ax_snr = inset_axes(
        ax2,
        width="30%",
        height="30%",
        bbox_to_anchor=(0.05, -0.5, 1, 1),
        bbox_transform=ax2.transAxes,
    )
    ax_snr.plot(radial, snr, "-", linewidth=1)
    ax_snr.set_title("SNR", fontsize="x-small")
    ax_snr.tick_params(labelsize="x-small", axis="both", which="both")

    # Bottom-left: cake representation
    ax3 = fig.add_subplot(2, 2, 3)
    sns.heatmap(cake[:, 30:], robust=True, square=True, ax=ax3)
    ax3.set_title("Cake Representation")

    # Bottom-right: deviation map
    cake2 = cake[:, columns_to_remove:]
    mask_zero = cake2 == 0
    col_sums = cake2.sum(axis=0)
    valid_counts = (~mask_zero).sum(axis=0)
    col_means = np.divide(col_sums, valid_counts, where=valid_counts > 0)
    pct_dev = (
        (cake2 - col_means[np.newaxis, :]) / col_means[np.newaxis, :] * 100
    )

    ax4 = fig.add_subplot(2, 2, 4)
    sns.heatmap(pct_dev, robust=True, square=True, ax=ax4)
    ax4.set_title(f"Deviation (%), goodness: {goodness}")

    # Show the canvas
    layout.addWidget(canvas)
    dialog.resize(700, 700)
    dialog.show()

    return dialog


def compute_hf_score_from_cake(
    measurement_filename: np.ndarray,
    poni_text: str = None,
    mask=None,
    hf_cutoff_fraction: float = 0.2,
    skip_bins: int = 30,
):
    """
    Compute the percentage of power in 'high' spatial frequencies
    from a 2D 'cake' integration array.
    """
    try:
        data = np.load(measurement_filename)
    except Exception as e:
        print(e)
        return -1

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
            sample_distance_mm,
        )

    # Perform integration
    npt = 200
    try:
        result = ai.integrate1d(
            data, npt, unit="q_nm^-1", error_model="azimuthal", mask=mask
        )
        radial = result.radial
        intensity = result.intensity
        cake, _, _ = ai.integrate2d(data, 200, npt_azim=180, mask=mask)
    except Exception as e:
        print(f"Error integrating data: {e}")
        return None

    # 1) Skip low-q bins
    Z = cake[:, skip_bins:]
    n_az, n_q = Z.shape

    # 2) Percent deviation per bin
    Z_norm = np.full_like(Z, np.nan, dtype=float)
    for j in range(n_q):
        col = Z[:, j]
        valid = col != 0
        if np.any(valid):
            mean_val = col[valid].mean()
            if mean_val != 0:
                Z_norm[valid, j] = (col[valid] - mean_val) / mean_val * 100

    # 3) Prepare for FFT
    Z_fft = np.nan_to_num(Z_norm, nan=0.0)
    Z_fft -= Z_fft.mean()

    # 4) FFT → power spectrum → shift
    F = np.fft.fft2(Z_fft)
    P = np.abs(F) ** 2
    P_shift = np.fft.fftshift(P)

    # 5) Build normalized frequency grid
    fy = np.fft.fftshift(np.fft.fftfreq(n_az))
    fx = np.fft.fftshift(np.fft.fftfreq(n_q))
    FX, FY = np.meshgrid(fx, fy)
    FreqMag = np.sqrt(FX**2 + FY**2)

    # 6) High-freq mask + fraction
    mask_hf = FreqMag > hf_cutoff_fraction
    P_high = P_shift[mask_hf].sum()
    P_total = P_shift.sum()
    return float((P_high / P_total) * 100) if P_total > 0 else 0.0
