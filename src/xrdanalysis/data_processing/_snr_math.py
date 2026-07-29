"""Private numerical kernels used by :class:`SNRTransformer`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedSigma:
    """Finite, positive sigma samples aligned with a profile when available."""

    sigma: np.ndarray | None
    q: np.ndarray | None
    is_valid: bool


@dataclass(frozen=True)
class SNRMetrics:
    """Scalar SNR outputs and the method label written by the transformer."""

    noise_std: float
    snr_linear: float
    snr_db: float
    method_used: str


def ensure_uniform_grid(
    q: np.ndarray,
    intensity: np.ndarray,
    n_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a profile onto a uniformly spaced q grid when requested."""
    q = np.asarray(q, float)
    intensity = np.asarray(intensity, float)
    if n_points is None:
        n_points = len(intensity)
    if n_points <= 1:
        return q, intensity
    q_uniform = np.linspace(np.nanmin(q), np.nanmax(q), int(n_points))
    return q_uniform, np.interp(q_uniform, q, intensity)


def normalize_by_surface(
    q: np.ndarray,
    intensity: np.ndarray,
    integrate: Callable[[np.ndarray, np.ndarray], float],
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Normalize by integrated area, with the legacy median fallback."""
    area = integrate(intensity, q)
    if not np.isfinite(area) or abs(area) < eps:
        median = (
            float(np.nanmedian(intensity[np.isfinite(intensity)]))
            if np.isfinite(intensity).any()
            else np.nan
        )
        scale = median if (np.isfinite(median) and median != 0.0) else 1.0
        return intensity / scale, {"norm": "median", "scale": scale, "area": area}
    return intensity / area, {"norm": "area", "scale": area, "area": area}


def smooth_signal(
    y: np.ndarray,
    window_frac: float,
    polyorder: int,
    savgol_filter: Callable[..., np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, int | str | None]]:
    """Smooth a profile, falling back to reflected moving-average smoothing."""
    y = np.asarray(y, float)
    n = len(y)
    if n <= 4:
        return y.copy(), {"method": "identity", "win": None, "poly": None}

    window = max(5, int(round(window_frac * n)))
    if window % 2 == 0:
        window += 1
    if window >= n:
        window = max(5, n - 1 if (n - 1) % 2 else n - 2)
    poly = min(polyorder, window - 1)

    if savgol_filter is not None and window > poly and window <= n:
        try:
            smoothed = savgol_filter(
                y, window_length=window, polyorder=poly, mode="interp"
            )
            return smoothed, {"method": "savgol", "win": window, "poly": poly}
        except Exception:
            pass

    # Preserve the historical fallback when SciPy is unavailable or rejects a window.
    padding = window // 2
    padded = np.pad(y, (padding, padding), mode="reflect")
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed, {"method": "movavg", "win": window, "poly": None}


def prepare_sigma(
    sigma_raw: object,
    q: np.ndarray,
    intensity: np.ndarray,
) -> PreparedSigma:
    """Prepare sigma samples using the transformer's historical truncation rules."""
    if sigma_raw is None:
        return PreparedSigma(None, None, False)

    sigma = np.asarray(sigma_raw, float)
    if sigma.ndim == 0:
        sigma = np.asarray([float(sigma)], dtype=float)
    sample_count = min(len(sigma), len(q), len(intensity))
    if sample_count < 2:
        return PreparedSigma(sigma, None, False)

    sigma = sigma[:sample_count]
    sigma_q = q[:sample_count]
    sigma_intensity = intensity[:sample_count]
    finite = np.isfinite(sigma_q) & np.isfinite(sigma_intensity) & np.isfinite(sigma)
    sigma = sigma[finite]
    sigma_q = sigma_q[finite]
    sigma_intensity = sigma_intensity[finite]
    return PreparedSigma(
        sigma,
        sigma_q,
        bool(np.sum(sigma > 0) >= 2 and len(sigma) >= 2),
    )


def calculate_snr_metrics(
    *,
    snr_method: str,
    enforce_common_q: bool,
    q_uniform: np.ndarray,
    intensity_uniform: np.ndarray,
    prepared_sigma: PreparedSigma,
    normalize: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, object]],
    smooth: Callable[[np.ndarray], tuple[np.ndarray, object]],
) -> SNRMetrics:
    """Select the Poisson or residual metric path without changing its rules."""
    use_poisson = snr_method in {"poisson", "auto"} and prepared_sigma.is_valid

    if use_poisson:
        # is_valid guarantees these arrays are present and contain two samples.
        sigma = prepared_sigma.sigma
        sigma_q = prepared_sigma.q
        assert sigma is not None and sigma_q is not None
        if enforce_common_q:
            sigma_uniform = np.interp(q_uniform, sigma_q, sigma)
        else:
            sigma_uniform = sigma[: len(intensity_uniform)]

        valid = (
            np.isfinite(intensity_uniform)
            & np.isfinite(sigma_uniform)
            & (sigma_uniform > 0)
        )
        if int(np.sum(valid)) >= 2:
            snr_q = np.abs(intensity_uniform[valid]) / (sigma_uniform[valid] + 1e-12)
            snr_linear = float(np.sqrt(np.mean(np.square(snr_q))))
            snr_db = float(20.0 * np.log10(snr_linear + 1e-12))
            noise_std = float(np.sqrt(np.mean(np.square(sigma_uniform[valid]))))
            return SNRMetrics(noise_std, snr_linear, snr_db, "poisson")
        return SNRMetrics(np.nan, np.nan, np.nan, "poisson_invalid_sigma")

    if snr_method == "poisson":
        return SNRMetrics(np.nan, np.nan, np.nan, "poisson_missing_sigma")

    # Inject transformer adapters so smoothing and normalization contracts
    # stay with the public class.
    intensity_normalized, _ = normalize(q_uniform, intensity_uniform)
    intensity_smoothed, _ = smooth(intensity_normalized)
    residual = intensity_normalized - intensity_smoothed
    if residual.size <= 1:
        return SNRMetrics(np.nan, np.nan, np.nan, "residual")

    noise_std = float(np.nanstd(residual, ddof=1))
    signal_power = float(np.nanvar(intensity_smoothed, ddof=1))
    noise_power = float(np.nanvar(residual, ddof=1))
    if np.isfinite(signal_power) and np.isfinite(noise_power) and noise_power > 0:
        snr_linear = signal_power / noise_power
        snr_db = 10.0 * float(np.log10(snr_linear))
    else:
        snr_linear, snr_db = np.nan, np.nan
    return SNRMetrics(noise_std, snr_linear, snr_db, "residual")
