"""Private numerical kernels used by :class:`SNRTransformer`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedSigma:
    """Finite sigma samples, plus optional q/intensity alignment metadata."""

    sigma: np.ndarray | None
    q: np.ndarray | None
    is_valid: bool
    intensity: np.ndarray | None = None


@dataclass(frozen=True)
class SNRMetrics:
    """Scalar SNR outputs and the method label written by the transformer."""

    noise_std: float
    snr_linear: float
    snr_db: float
    method_used: str


@dataclass(frozen=True)
class SNRRowResult:
    """Scalar metrics and optional legacy profile arrays for one input row."""

    metrics: SNRMetrics
    smoothed: np.ndarray | None
    residual: np.ndarray | None


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


def prepare_regridded_sigma(
    sigma_raw: object,
    q: np.ndarray,
    intensity: np.ndarray,
) -> PreparedSigma:
    """Prepare sigma for the historical q-grid interpolation path.

    The legacy implementation discarded samples with non-finite q before it
    interpolated sigma.  Keep that rule isolated here so restored transformers
    retain their previous numerical results.
    """
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


def prepare_native_sigma(
    sigma_raw: object,
    intensity: np.ndarray,
) -> PreparedSigma:
    """Prepare native intensity/sigma pairs for the Poisson scalar contract.

    Poisson SNR is defined pointwise for aligned intensity and sigma samples.
    q is deliberately absent: q spacing, order, and finite values do not
    affect this scalar calculation.
    """
    if sigma_raw is None:
        return PreparedSigma(None, None, False)

    sigma = np.asarray(sigma_raw, float)
    if sigma.ndim == 0:
        sigma = np.asarray([float(sigma)], dtype=float)
    sample_count = min(len(sigma), len(intensity))
    if sample_count < 2:
        return PreparedSigma(sigma, None, False)

    sigma = sigma[:sample_count]
    aligned_intensity = np.asarray(intensity[:sample_count], dtype=float)
    finite = np.isfinite(aligned_intensity) & np.isfinite(sigma)
    sigma = sigma[finite]
    return PreparedSigma(
        sigma,
        None,
        bool(np.sum(sigma > 0) >= 2 and len(sigma) >= 2),
        aligned_intensity[finite],
    )


def _poisson_metrics(
    intensity: np.ndarray,
    sigma: np.ndarray,
) -> SNRMetrics | None:
    """Return RMS Poisson SNR metrics for valid aligned samples."""
    valid = np.isfinite(intensity) & np.isfinite(sigma) & (sigma > 0)
    if int(np.sum(valid)) < 2:
        return None

    snr_q = np.abs(intensity[valid]) / (sigma[valid] + 1e-12)
    snr_linear = float(np.sqrt(np.mean(np.square(snr_q))))
    snr_db = float(20.0 * np.log10(snr_linear + 1e-12))
    noise_std = float(np.sqrt(np.mean(np.square(sigma[valid]))))
    return SNRMetrics(noise_std, snr_linear, snr_db, "poisson")


def calculate_snr_metrics(
    *,
    snr_method: str,
    enforce_common_q: bool,
    regrid_poisson: bool,
    q_uniform: np.ndarray,
    intensity_uniform: np.ndarray,
    native_sigma: PreparedSigma,
    regridded_sigma: PreparedSigma,
    normalize: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, object]],
    smooth: Callable[[np.ndarray], tuple[np.ndarray, object]],
) -> SNRMetrics:
    """Select native/regridded Poisson or legacy residual SNR metrics."""
    poisson_sigma = regridded_sigma if regrid_poisson else native_sigma
    use_poisson = snr_method in {"poisson", "auto"} and poisson_sigma.is_valid

    if use_poisson:
        sigma = poisson_sigma.sigma
        assert sigma is not None
        if regrid_poisson and enforce_common_q:
            sigma_q = poisson_sigma.q
            assert sigma_q is not None
            sigma_uniform = np.interp(q_uniform, sigma_q, sigma)
            result = _poisson_metrics(intensity_uniform, sigma_uniform)
        else:
            intensity = intensity_uniform if regrid_poisson else poisson_sigma.intensity
            assert intensity is not None
            result = _poisson_metrics(intensity[: len(sigma)], sigma)
        if result is not None:
            return result
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


def calculate_snr_row(
    *,
    q_raw: object,
    intensity_raw: object,
    sigma_raw: object,
    snr_method: str,
    enforce_common_q: bool,
    regrid_poisson: bool,
    n_points: int | None,
    uniform_grid: Callable[
        [np.ndarray, np.ndarray, int | None], tuple[np.ndarray, np.ndarray]
    ],
    normalize: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, object]],
    smooth: Callable[[np.ndarray], tuple[np.ndarray, object]],
) -> SNRRowResult | None:
    """Calculate one row while keeping native and legacy Poisson paths explicit."""
    intensity_native = np.asarray(intensity_raw, float)
    if intensity_native.ndim != 1 or len(intensity_native) < 2:
        return None

    native_sigma = prepare_native_sigma(sigma_raw, intensity_native)
    q_values = np.asarray(q_raw, float)
    if q_values.ndim != 1:
        q = np.asarray([], dtype=float)
        intensity = np.asarray([], dtype=float)
    else:
        sample_count = min(len(q_values), len(intensity_native))
        q = q_values[:sample_count]
        intensity = intensity_native[:sample_count]
        finite_qi = np.isfinite(q) & np.isfinite(intensity)
        q = q[finite_qi]
        intensity = intensity[finite_qi]

    if len(intensity) < 2:
        if (
            snr_method == "residual"
            or regrid_poisson
            or (snr_method == "auto" and not native_sigma.is_valid)
        ):
            return None
        metrics = calculate_snr_metrics(
            snr_method=snr_method,
            enforce_common_q=False,
            regrid_poisson=False,
            q_uniform=np.asarray([], dtype=float),
            intensity_uniform=np.asarray([], dtype=float),
            native_sigma=native_sigma,
            regridded_sigma=prepare_regridded_sigma(sigma_raw, q, intensity),
            normalize=normalize,
            smooth=smooth,
        )
        smoothed, _ = smooth(intensity_native)
        return SNRRowResult(metrics, smoothed, intensity_native - smoothed)

    if enforce_common_q:
        q_uniform, intensity_uniform = uniform_grid(q, intensity, n_points)
    else:
        q_uniform, intensity_uniform = q, intensity
    smoothed_uniform, _ = smooth(intensity_uniform)
    metrics = calculate_snr_metrics(
        snr_method=snr_method,
        enforce_common_q=enforce_common_q,
        regrid_poisson=regrid_poisson,
        q_uniform=q_uniform,
        intensity_uniform=intensity_uniform,
        native_sigma=native_sigma,
        regridded_sigma=prepare_regridded_sigma(sigma_raw, q, intensity),
        normalize=normalize,
        smooth=smooth,
    )
    smoothed = (
        np.interp(q, q_uniform, smoothed_uniform)
        if enforce_common_q
        else smoothed_uniform
    )
    residual = (
        intensity - smoothed
        if len(smoothed) == len(intensity)
        else np.full_like(intensity, np.nan)
    )
    return SNRRowResult(metrics, smoothed, residual)
