# SNR Transformer (xrd-analysis)

This document summarizes the **SNRTransformer** implementation and includes the exact code from the project.

## Description (plain language)

The SNR transformer estimates signal-to-noise from a 1D diffraction profile (q vs intensity) by:

- Optionally interpolating each curve onto a uniform q-grid.
- Normalizing intensity by the area under the curve (or median if the area is invalid).
- Smoothing the normalized signal (Savitzky–Golay if available, otherwise moving average).
- Treating the residuals (original − smoothed) as noise.
- Computing signal power vs noise power, then converting to dB:
  - `snr_linear = var(smoothed) / var(residual)`
  - `snr_db = 10 * log10(snr_linear)`

Outputs written to the DataFrame:

- `noise_std`, `snr_linear`, `snr_db`, and `snr` (alias)
- Optional arrays: `radial_profile_data_snr` (smoothed) and `radial_profile_residual`

## Code (verbatim from project)

```python
class SNRTransformer(TransformerMixin):
    """
    Compute signal-to-noise metrics from 1D azimuthal integration results.

    For each row, this transformer:
    - optionally re-interpolates (q, I) to a common, uniformly spaced q-grid
    - normalizes the intensity by its area (fallback to median scaling)
    - smooths the normalized intensity using Savitzky–Golay (fallback to moving average)
    - computes residual = I_norm - I_smooth
    - computes noise_std, snr_linear = var(I_smooth)/var(residual), and snr_db
    - writes a denoised 1D profile to `radial_profile_data_snr` (smoothed in original scale)
    - writes scalar SNR in dB to column `snr`

    Parameters
    ----------
    x_column : str
        Column with the q-range array. Defaults to 'q_range'.
    y_column : str
        Column with the 1D intensity array. Defaults to 'radial_profile_data'.
    window_frac : float
        Fraction of the number of points to set SavGol window length. Default 0.04.
    polyorder : int
        Polynomial order for SavGol. Default 2.
    enforce_common_q : bool
        If True, re-interpolate to a uniformly spaced q grid per row. Default True.
    n_points : int | None
        If set, number of points for the uniform grid. Defaults to len(I) when None.
    save_smoothed : bool
        If True, saves smoothed and residual arrays in the DataFrame.
    smoothed_col : str
        Column name for smoothed intensity when saved. Defaults to 'radial_profile_data_snr'.
    residual_col : str
        Column name for residual intensity when saved. Defaults to 'radial_profile_residual'.
    snr_col : str
        Column name to write SNR in dB. Defaults to 'snr'.
    """

    def __init__(
        self,
        x_column: str = "q_range",
        y_column: str = "radial_profile_data",
        window_frac: float = 0.04,
        polyorder: int = 2,
        enforce_common_q: bool = True,
        n_points: int = None,
        save_smoothed: bool = True,
        smoothed_col: str = "radial_profile_data_snr",
        residual_col: str = "radial_profile_residual",
        snr_col: str = "snr",
    ) -> None:
        self.x_column = x_column
        self.y_column = y_column
        self.window_frac = float(window_frac)
        self.polyorder = int(polyorder)
        self.enforce_common_q = bool(enforce_common_q)
        self.n_points = n_points
        self.save_smoothed = bool(save_smoothed)
        self.smoothed_col = smoothed_col
        self.residual_col = residual_col
        self.snr_col = snr_col

        # Optional Savitzky–Golay import
        try:
            from scipy.signal import savgol_filter as _sg
        except Exception:
            _sg = None
        self._savgol = _sg

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def _ensure_uniform_grid(
        self, q: np.ndarray, intensity: np.ndarray, n_points: int | None
    ):
        q = np.asarray(q, float)
        intensity = np.asarray(intensity, float)
        if n_points is None:
            n_points = len(intensity)
        if n_points <= 1:
            return q, intensity
        q_uniform = np.linspace(np.nanmin(q), np.nanmax(q), int(n_points))
        intensity_uniform = np.interp(q_uniform, q, intensity)
        return q_uniform, intensity_uniform

    def _normalize_by_surface(
        self, q: np.ndarray, intensity: np.ndarray, eps: float = 1e-12
    ):
        area = float(np.trapz(intensity, q))
        if not np.isfinite(area) or abs(area) < eps:
            med = (
                float(np.nanmedian(intensity[np.isfinite(intensity)]))
                if np.isfinite(intensity).any()
                else np.nan
            )
            scale = med if (np.isfinite(med) and med != 0.0) else 1.0
            return intensity / scale, {"norm": "median", "scale": scale, "area": area}
        return intensity / area, {"norm": "area", "scale": area, "area": area}

    def _smooth(self, y: np.ndarray):
        y = np.asarray(y, float)
        n = len(y)
        if n <= 4:
            return y.copy(), {"method": "identity", "win": None, "poly": None}
        w = max(5, int(round(self.window_frac * n)))
        if w % 2 == 0:
            w += 1
        if w >= n:
            w = max(5, n - 1 if (n - 1) % 2 else n - 2)
        poly = min(self.polyorder, w - 1)

        if self._savgol is not None and w > poly and w <= n:
            try:
                y_sm = self._savgol(y, window_length=w, polyorder=poly, mode="interp")
                return y_sm, {"method": "savgol", "win": w, "poly": poly}
            except Exception:
                pass
        # Fallback: centered moving average with reflect padding
        pad = w // 2
        xp = np.pad(y, (pad, pad), mode="reflect")
        kern = np.ones(w, dtype=float) / float(w)
        y_sm = np.convolve(xp, kern, mode="valid")
        return y_sm, {"method": "movavg", "win": w, "poly": None}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Prepare output columns
        df["noise_std"] = np.nan
        df["snr_linear"] = np.nan
        df["snr_db"] = np.nan
        df[self.snr_col] = np.nan  # alias for snr in dB as requested
        if self.save_smoothed:
            df[self.smoothed_col] = None
            df[self.residual_col] = None
            df[self.smoothed_col].astype(object)
            df[self.residual_col].astype(object)

        for i, row in df.iterrows():
            q = np.asarray(row.get(self.x_column), float)
            intensity = np.asarray(row.get(self.y_column), float)
            if q is None or intensity is None or len(intensity) < 2:
                continue

            # Ensure uniform grid if requested
            if self.enforce_common_q:
                q_u, intensity_u = self._ensure_uniform_grid(
                    q, intensity, self.n_points
                )
            else:
                q_u, intensity_u = q, intensity

            # Normalize for SNR metric
            intensity_norm, _ = self._normalize_by_surface(q_u, intensity_u)

            # Smooth for SNR (normalized domain)
            intensity_sm_norm, _ = self._smooth(intensity_norm)

            # Residuals & metrics
            resid_norm = intensity_norm - intensity_sm_norm
            if resid_norm.size > 1:
                noise_std = float(np.nanstd(resid_norm, ddof=1))
                sig_pow = float(np.nanvar(intensity_sm_norm, ddof=1))
                noi_pow = float(np.nanvar(resid_norm, ddof=1))
                if np.isfinite(sig_pow) and np.isfinite(noi_pow) and noi_pow > 0:
                    snr_lin = sig_pow / noi_pow
                    snr_db = 10.0 * float(np.log10(snr_lin))
                else:
                    snr_lin, snr_db = np.nan, np.nan
            else:
                noise_std, snr_lin, snr_db = np.nan, np.nan, np.nan

            # Also produce a smoothed version in the original intensity scale
            intensity_sm_u, _ = self._smooth(intensity_u)
            # Map smoothed uniform-grid curve back to original q sampling if needed
            if self.enforce_common_q:
                intensity_sm_out = np.interp(q, q_u, intensity_sm_u)
            else:
                intensity_sm_out = intensity_sm_u
            resid_out = (
                intensity - intensity_sm_out
                if intensity_sm_out is not None
                and len(intensity_sm_out) == len(intensity)
                else np.full_like(intensity, np.nan)
            )

            # Assign outputs
            df.at[i, "noise_std"] = noise_std
            df.at[i, "snr_linear"] = snr_lin
            df.at[i, "snr_db"] = snr_db
            df.at[i, self.snr_col] = snr_db  # requested alias

            if self.save_smoothed:
                df.at[i, self.smoothed_col] = intensity_sm_out
                df.at[i, self.residual_col] = resid_out
```
