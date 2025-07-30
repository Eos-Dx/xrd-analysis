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

from xrdanalysis.data_processing.utility_functions import create_mask


from PyQt5.QtCore import QObject, pyqtSignal
import threading

class CaptureWorker(QObject):
    finished = pyqtSignal(bool, dict)  # Or as appropriate for your use

    def __init__(self, detector_controller, integration_time, txt_filename_base, parent=None):
        super().__init__(parent)
        self.detector_controller = detector_controller
        self.integration_time = integration_time
        self.txt_filename_base = txt_filename_base

    def run(self):
        threads = {}
        results = {}

        def run_capture(alias, controller):
            try:
                success = controller.capture_point(
                    Nframes=1,
                    Nseconds=self.integration_time,
                    filename_base=self.txt_filename_base + f"_{alias}"
                )
                results[alias] = self.txt_filename_base + f"_{alias}.txt" if success else None
            except Exception as e:
                print(f"Error in capture for {alias}: {e}")
                results[alias] = None

        for alias, controller in self.detector_controller.items():
            t = threading.Thread(target=run_capture, args=(alias, controller))
            threads[alias] = t
            t.start()

        for t in threads.values():
            t.join()

        overall_success = all(r is not None for r in results.values())
        self.finished.emit(overall_success, results)



from pathlib import Path
import shutil
import numpy as np

def move_and_convert_measurement_file(src_file, alias_folder):
    """
    Move .txt and .dsc (of any extension style) to alias_folder, convert .txt to .npy.
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

    # Move both .dsc and .txt.dsc (Pixet style)
    candidates = [
        src_file.with_suffix('.dsc'),                    # aux_001_..._SAXS.dsc
        src_file.parent / (src_file.name + '.dsc')       # aux_001_..._SAXS.txt.dsc
    ]
    for dsc_candidate in candidates:
        if dsc_candidate.exists():
            dest_dsc = alias_folder / dsc_candidate.name
            try:
                shutil.move(str(dsc_candidate), str(dest_dsc))
                print(f"Moved {dsc_candidate} → {dest_dsc}")
            except Exception as e:
                print(f"[move_and_convert_measurement_file] Error moving {dsc_candidate}: {e}")

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
    center=None,                  # <-- NEW
    integration_radius=None,      # <-- NEW
):
    """
    Opens a dialog window displaying the raw 2D image and its azimuthal integration.
    Optionally overlays the beam center and integration region.
    """
    import matplotlib.pyplot as plt

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
    sns.heatmap(data, robust=True, square=True, ax=ax1, cbar=False)
    ax1.set_title("2D Image")

    # === Overlay beam center and integration region ===
    if center is not None:
        cy, cx = center
        print(cx, cy)
        # Mark center
        ax1.plot([cx], [cy], marker="x", color="red", markersize=10, label="Beam center")
        # Mark integration region
        if integration_radius is not None and integration_radius > 0:
            from matplotlib.patches import Circle
            circ = Circle((cx, cy), integration_radius, edgecolor="red", facecolor="none", lw=2, ls="--", label="Integration area")
            ax1.add_patch(circ)
        ax1.legend(fontsize="x-small")

    # Top-right: 1D integration
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.errorbar(
        radial,
        intensity,
        yerr=sigma,
        fmt="-o",
        markersize=3,
        linewidth=1,
        ecolor="black",
        capsize=3,
        capthick=1,
        label="Intensity ± σ",
    )
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
        width="30%",
        height="30%",
        bbox_to_anchor=(0.05, -0.2, 1, 1),
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
