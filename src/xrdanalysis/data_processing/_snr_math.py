"""Private numerical kernels used by :class:`SNRTransformer`."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


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
