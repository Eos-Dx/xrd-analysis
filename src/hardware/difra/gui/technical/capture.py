import os
import shutil
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QDialog, QHBoxLayout

from hardware.difra.gui.container_api import get_container_version
from xrdanalysis.data_processing.azimuthal_integration import (
    initialize_azimuthal_integrator_df,
    initialize_azimuthal_integrator_poni_text,
)
from xrdanalysis.data_processing.utility_functions import create_mask

logger = logging.getLogger(__name__)


def _load_measurement_array(measurement_filename: str) -> np.ndarray:
    value = str(measurement_filename or "").strip()
    if value.startswith("h5ref://"):
        # Format: h5ref://<absolute-container-path>#<dataset_path>
        import h5py

        payload = value[len("h5ref://") :]
        container_path, sep, dataset_path = payload.partition("#")
        if not sep or not container_path or not dataset_path:
            raise ValueError(f"Invalid H5 reference: {measurement_filename}")

        container = Path(container_path)
        if not container.exists():
            raise FileNotFoundError(f"H5 container does not exist: {container}")

        with h5py.File(container, "r") as h5f:
            if dataset_path not in h5f:
                raise KeyError(
                    f"Dataset not found in container: {container}#{dataset_path}"
                )
            data = h5f[dataset_path][()]
            arr = np.asarray(data, dtype=float)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {arr.shape}")
            return arr

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"Measurement file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        loaders = (np.loadtxt, np.load)
    elif suffix == ".npy":
        loaders = (np.load, np.loadtxt)
    else:
        loaders = (np.load, np.loadtxt)

    last_error = None
    for loader in loaders:
        try:
            data = loader(path)
            arr = np.asarray(data, dtype=float)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {arr.shape}")
            return arr
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to load measurement file '{path}': {last_error}")


def _place_raw_capture_file(src_raw: str, target_txt: Path, allow_move: bool = True) -> None:
    """Place raw detector output at target path, preferring move over copy."""
    src_path = Path(src_raw)
    target_txt = Path(target_txt)
    target_txt.parent.mkdir(parents=True, exist_ok=True)
    src_dsc = src_path.with_suffix(".dsc")
    dst_dsc = target_txt.with_suffix(".dsc")

    if src_path.resolve() == target_txt.resolve():
        if src_dsc.exists() and not dst_dsc.exists():
            shutil.copy2(src_dsc, dst_dsc)
        return

    moved = False
    if allow_move:
        try:
            shutil.move(str(src_path), str(target_txt))
            moved = True
        except Exception:
            moved = False

    if not moved:
        shutil.copy2(src_path, target_txt)

    if src_dsc.exists():
        if moved:
            try:
                shutil.move(str(src_dsc), str(dst_dsc))
            except Exception:
                shutil.copy2(src_dsc, dst_dsc)
        else:
            shutil.copy2(src_dsc, dst_dsc)


class CaptureWorker(QObject):
    finished = pyqtSignal(bool, dict)  # success, {alias: converted_file_path}

    def __init__(
        self,
        detector_controller,
        integration_time,
        txt_filename_base,
        parent=None,
        frames: int = 1,
        naming_mode: str = "normal",  # normal | attenuation_with | attenuation_without
        continuous_movement_controller=None,
        stage_controller=None,
        hardware_client=None,
        enable_continuous_movement: bool = False,
        movement_radius: float = 2.0,
        container_version: str = None,  # Container version for format conversion
    ):
        super().__init__(parent)
        self.detector_controller = detector_controller
        self.integration_time = integration_time
        self.txt_filename_base = txt_filename_base
        self.frames = frames
        self.naming_mode = naming_mode
        self.continuous_movement_controller = continuous_movement_controller
        self.stage_controller = stage_controller
        self.hardware_client = hardware_client
        self.enable_continuous_movement = enable_continuous_movement
        self.movement_radius = movement_radius
        self.container_version = container_version or get_container_version(None)
        self._stop_requested = False
        self.error_messages = []

    def _record_error(self, message: str, exc: Exception = None) -> None:
        self.error_messages.append(message)
        if exc is None:
            logger.error(message)
        else:
            logger.error("%s: %s", message, exc, exc_info=True)

    def run(self):
        results = {}
        movement_started = False

        # Determine if continuous movement should be used (checkbox-driven only)
        is_continuous_movement = (
            self.enable_continuous_movement
            and self.continuous_movement_controller
            and self.stage_controller
        )

        try:
            # Start continuous movement when enabled by the checkbox
            if is_continuous_movement:
                # Get current stage position as center
                try:
                    center_x, center_y = self.stage_controller.get_xy_position()
                except Exception:
                    if self.hardware_client is not None:
                        center_x, center_y = self.hardware_client.get_xy_position()
                    else:
                        raise

                # Configure movement for the full acquisition duration (frames × integration time)
                total_duration = float(self.integration_time) * max(int(self.frames), 1)
                self.continuous_movement_controller.configure(
                    self.movement_radius, total_duration
                )

                movement_started = self.continuous_movement_controller.start_movement(
                    center_x, center_y
                )

                if movement_started:
                    logger.info(
                        "Started continuous movement for technical measurement "
                        "(center: %.3f, %.3f, radius: %.3fmm)",
                        center_x,
                        center_y,
                        float(self.movement_radius),
                    )
                else:
                    message = (
                        "Failed to start continuous movement for technical measurement"
                    )
                    logger.warning(message)
                    self.error_messages.append(message)

            if self.hardware_client is None:
                raise RuntimeError(
                    "Hardware client is required for capture; direct detector calls are disabled in GUI."
                )

            raw_outputs = self.hardware_client.capture_exposure(
                exposure_s=float(self.integration_time),
                frames=max(int(self.frames), 1),
                timeout_s=max(30.0, float(self.integration_time) * max(int(self.frames), 1) + 30.0),
            )

            source_usage = Counter()
            fallback_single = next(iter(raw_outputs.values())) if len(raw_outputs) == 1 else None
            for alias in self.detector_controller.keys():
                src_raw = raw_outputs.get(alias) or fallback_single
                if not src_raw:
                    continue
                try:
                    source_usage[str(Path(src_raw).resolve())] += 1
                except Exception:
                    source_usage[str(src_raw)] += 1

            for alias, controller in self.detector_controller.items():
                if self._stop_requested:
                    results[alias] = None
                    continue
                try:
                    if self.naming_mode == "attenuation_with":
                        base = f"{self.txt_filename_base}__{alias}_ATTENUATION"
                    elif self.naming_mode == "attenuation_without":
                        base = f"{self.txt_filename_base}__{alias}_ATTENUATION0"
                    else:
                        base = f"{self.txt_filename_base}_{alias}"

                    src_raw = raw_outputs.get(alias)
                    if src_raw is None and len(raw_outputs) == 1:
                        src_raw = next(iter(raw_outputs.values()))
                    if not src_raw:
                        self._record_error(
                            f"No raw output for detector '{alias}'. "
                            f"Available output aliases: {sorted(raw_outputs.keys())}"
                        )
                        results[alias] = None
                        continue

                    src_path = Path(src_raw)
                    target_txt = Path(base + ".txt")
                    key = str(src_path.resolve())
                    allow_move = source_usage.get(key, 0) <= 1
                    _place_raw_capture_file(src_raw=src_raw, target_txt=target_txt, allow_move=allow_move)
                    if key in source_usage and source_usage[key] > 0:
                        source_usage[key] -= 1

                    converted_file = controller.convert_to_container_format(
                        str(target_txt), self.container_version
                    )
                    results[alias] = converted_file
                    logger.info(
                        "Converted technical capture for %s: %s -> %s",
                        alias,
                        target_txt.name,
                        Path(converted_file).name,
                    )
                except Exception as e:
                    self._record_error(
                        f"Error while processing detector '{alias}' output",
                        e,
                    )
                    results[alias] = None

        except Exception as e:
            self._record_error("Error during capture operation", e)
            results = {alias: None for alias in self.detector_controller.keys()}

        finally:
            # Stop continuous movement if it was started
            if movement_started and self.continuous_movement_controller:
                try:
                    self.continuous_movement_controller.stop_movement(
                        return_to_origin=True
                    )
                    logger.info(
                        "Stopped continuous movement and returned to original position"
                    )
                except Exception as e:
                    self._record_error("Error stopping continuous movement", e)

        overall_success = (
            all(r is not None for r in results.values()) and not self._stop_requested
        )
        if not overall_success and not self.error_messages:
            self.error_messages.append("Capture failed without explicit error details.")
        self.finished.emit(overall_success, results)

    def stop(self):
        """Request the capture operation to stop."""
        self._stop_requested = True

        # Stop continuous movement immediately if active
        if (
            self.continuous_movement_controller
            and self.continuous_movement_controller.is_moving()
        ):
            try:
                self.continuous_movement_controller.stop_movement(return_to_origin=True)
                logger.info("Stopped continuous movement due to capture stop request")
            except Exception as e:
                self._record_error(
                    "Error stopping continuous movement during stop request", e
                )

def validate_folder(path: str):
    if not path:
        path = os.getcwd()
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        path = os.getcwd()
    if not os.access(path, os.W_OK):
        path = os.getcwd()
    return Path(path)


def show_measurement_window(
    measurement_filename: str,
    mask: np.ndarray,
    poni_text: str = None,
    parent=None,
    columns_to_remove: int = 30,
    goodness: float = 0.0,
    center=None,  # <-- NEW
    integration_radius=None,  # <-- NEW
):
    """
    Opens a dialog window displaying the raw 2D image and its azimuthal integration.
    Optionally overlays the beam center and integration region.
    """
    import matplotlib.pyplot as plt

    # Load data
    data = _load_measurement_array(measurement_filename)

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
        radial = np.asarray(result.radial, dtype=float).reshape(-1)
        intensity = np.asarray(result.intensity, dtype=float).reshape(-1)
        std = np.asarray(result.std, dtype=float).reshape(-1)
        sigma = np.asarray(result.sigma, dtype=float).reshape(-1)

        min_len = min(radial.size, intensity.size, std.size, sigma.size)
        if min_len <= 0:
            raise ValueError("Integration produced empty radial/intensity arrays")
        radial = radial[:min_len]
        intensity = intensity[:min_len]
        std = std[:min_len]
        sigma = sigma[:min_len]

        finite = np.isfinite(radial) & np.isfinite(intensity)
        if not np.any(finite):
            raise ValueError("Integration produced only NaN/Inf values")
        radial = radial[finite]
        intensity = intensity[finite]
        std = std[finite]
        sigma = sigma[finite]

        cake, _, _ = ai.integrate2d(data, 200, npt_azim=180, mask=mask)
    except Exception as e:
        raise RuntimeError(f"Error integrating data for '{measurement_filename}': {e}")

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
        ax1.plot(
            [cx],
            [cy],
            marker="x",
            color="red",
            markersize=10,
            label="Beam center",
        )
        if integration_radius is not None and integration_radius > 0:
            from matplotlib.patches import Circle

            circ = Circle(
                (cx, cy),
                integration_radius,
                edgecolor="red",
                facecolor="none",
                lw=3,
                ls="--",
                label="Integration area",
            )
            ax1.add_patch(circ)

    # Top-right: 1D integration
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.errorbar(
        radial,
        intensity,
        yerr=np.where(np.isfinite(sigma) & (sigma >= 0), sigma, np.nan),
        fmt="-o",
        markersize=3,
        linewidth=1,
        ecolor="black",
        capsize=3,
        capthick=1,
        label="Intensity ± σ",
    )
    xmin = float(np.nanmin(radial))
    xmax = float(np.nanmax(radial))
    xright = xmax * 1.3 if xmax > 0 else xmax + 1.0
    if (not np.isfinite(xmin)) or (not np.isfinite(xright)) or (xright <= xmin):
        xleft = 0.0
        xright = max(1.0, abs(xmax)) * 1.3
    else:
        xleft = xmin
    ax2.set_xlim(xleft, xright)
    if np.any(np.isfinite(intensity) & (intensity > 0)):
        ax2.set_yscale("log")
    else:
        ax2.set_yscale("linear")
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
    safe_sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    snr = np.divide(
        intensity,
        safe_sigma,
        out=np.full_like(intensity, np.nan, dtype=float),
        where=np.isfinite(safe_sigma),
    )
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
    pct_dev = (cake2 - col_means[np.newaxis, :]) / col_means[np.newaxis, :] * 100

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
        data = _load_measurement_array(measurement_filename)
    except Exception as e:
        print(f"Failed to load measurement '{measurement_filename}': {e}")
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
