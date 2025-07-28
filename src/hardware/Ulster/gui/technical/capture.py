import os

import numpy as np
import seaborn as sns
from pathlib import Path
import shutil
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog, QHBoxLayout

from xrdanalysis.data_processing.azimuthal_integration import (
    initialize_azimuthal_integrator_df,
    initialize_azimuthal_integrator_poni_text,
)
from PyQt5.QtCore import QObject, pyqtSignal

class CaptureWorker(QObject):
    # Emits: (success: bool, result_files: dict)
    finished = pyqtSignal(bool, dict)

    def __init__(self, detector_controller, integration_time, txt_filename_base, parent=None):
        super().__init__(parent)
        self.detector_controller = detector_controller  # dict: {alias: controller}
        self.integration_time = integration_time
        self.txt_filename_base = txt_filename_base

    def start(self):
        # Run in main thread for minimal test (real code: moveToThread)
        result_files = {}
        try:
            for alias, controller in self.detector_controller.items():
                # File naming: base_alias.txt (could be improved)
                filename = f"{self.txt_filename_base}_{alias}.txt"
                # You may need to adapt this call to match your controller API:
                success = controller.capture_point(
                    Nframes=1,  # or use frames param if needed
                    Nseconds=self.integration_time,
                    filename_base=filename.replace(".txt", "")
                )
                if success:
                    result_files[alias] = filename
                else:
                    print(f"Acquisition failed for {alias}")
            ok = len(result_files) == len(self.detector_controller)
        except Exception as e:
            print(f"CaptureWorker error: {e}")
            ok = False
        self.finished.emit(ok, result_files)


def move_and_convert_measurement_file(src_file, alias_folder):
    """
    Move the .txt and .dsc file to alias_folder, convert .txt to .npy.
    Args:
        src_file: str or Path, path to the original .txt file
        alias_folder: str or Path, target directory for detector alias (will be created if needed)
    Returns:
        str: Path to the saved .npy file
    """
    src_file = Path(src_file)
    alias_folder = Path(alias_folder)
    alias_folder.mkdir(parents=True, exist_ok=True)

    # Move .txt file
    dest_txt = alias_folder / src_file.name
    try:
        shutil.move(str(src_file), str(dest_txt))
    except Exception as e:
        print(f"[move_and_convert_measurement_file] Error moving .txt: {e}")
        dest_txt = src_file  # fallback

    # Move .dsc file (if present)
    dsc_file = src_file.with_suffix('.dsc')
    if dsc_file.exists():
        dest_dsc = alias_folder / dsc_file.name
        try:
            shutil.move(str(dsc_file), str(dest_dsc))
        except Exception as e:
            print(f"[move_and_convert_measurement_file] Error moving .dsc: {e}")

    # Convert to .npy in alias folder
    try:
        data = np.loadtxt(dest_txt)
        npy_file = dest_txt.with_suffix('.npy')
        np.save(npy_file, data)
    except Exception as e:
        print(f"[move_and_convert_measurement_file] Error converting to .npy: {e}")
        npy_file = dest_txt  # fallback

    return str(npy_file)


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
