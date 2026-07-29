"""Private numerical kernels for spectrokinetic decomposition."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def as_1d_float(arr: ArrayLike, name: str) -> np.ndarray:
    """Return a validated one-dimensional float array."""
    out = np.asarray(arr, dtype=float)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {out.shape}.")
    return out


def as_2d_float(arr: ArrayLike, name: str) -> np.ndarray:
    """Return a validated two-dimensional float array."""
    out = np.asarray(arr, dtype=float)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {out.shape}.")
    return out


def normalize_nonneg_vector(
    nonneg: Union[bool, Sequence[bool]], n_cols: int
) -> np.ndarray:
    """Normalize a scalar or vector non-negativity constraint."""
    if isinstance(nonneg, (bool, np.bool_)):
        return np.full(n_cols, bool(nonneg), dtype=bool)
    vec = np.asarray(nonneg, dtype=bool)
    if vec.ndim != 1 or vec.size != n_cols:
        raise ValueError(f"nonneg_s must have length {n_cols}; got shape {vec.shape}.")
    return vec


def normalize_bool_vector(flag: Union[bool, Sequence[bool]], n_cols: int) -> np.ndarray:
    """Normalize a scalar or vector Boolean flag."""
    if isinstance(flag, (bool, np.bool_)):
        return np.full(n_cols, bool(flag), dtype=bool)
    vec = np.asarray(flag, dtype=bool)
    if vec.ndim != 1 or vec.size != n_cols:
        raise ValueError(f"Boolean vector must have length {n_cols}, got {vec.shape}.")
    return vec


def mask_to_bool(mask: Optional[ArrayLike], n: int) -> np.ndarray:
    """Convert an optional mask into the historical Boolean form."""
    if mask is None:
        return np.ones(n, dtype=bool)
    mask_array = np.asarray(mask)
    if mask_array.ndim != 1 or mask_array.size != n:
        raise ValueError(f"Mask must have length {n}; got shape {mask_array.shape}.")
    if mask_array.dtype == bool:
        return mask_array.copy()
    if np.issubdtype(mask_array.dtype, np.floating):
        return ~np.isnan(mask_array)
    return mask_array.astype(bool)


def compute_lof(model: np.ndarray, data: np.ndarray) -> float:
    """Compute lack-of-fit percentage, matching SK-Ana convention."""
    model = as_2d_float(model, "model")
    data = as_2d_float(data, "data")
    denom = float(np.sum(data**2))
    if denom <= 0:
        return 100.0
    return float(100.0 * np.sqrt(np.sum((data - model) ** 2) / denom))


def convolve_spectrum(spectrum: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian convolution to a 1D spectrum."""
    spectrum = as_1d_float(spectrum, "spectrum")
    if not np.isfinite(sigma) or sigma <= 0:
        return spectrum.copy()
    kernel_size = max(3, int(np.ceil(3 * sigma)))
    x = np.arange(-kernel_size, kernel_size + 1, dtype=float)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:
        return spectrum.copy()
    kernel /= kernel_sum
    padded = np.pad(spectrum, (kernel_size, kernel_size), mode="edge")
    conv = np.convolve(padded, kernel, mode="same")
    return conv[kernel_size : kernel_size + spectrum.size]


def safe_lstsq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve least squares with the historical zero-vector fallback."""
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        sol = np.zeros(a.shape[1], dtype=float)
    return np.asarray(sol, dtype=float)


def safe_solve(a: np.ndarray, b: np.ndarray, nonneg: bool) -> np.ndarray:
    """Solve constrained or unconstrained least squares."""
    if nonneg:
        try:
            sol, _ = nnls(a, b)
        except Exception:
            sol = np.zeros(a.shape[1], dtype=float)
    else:
        sol = safe_lstsq(a, b)
    return np.asarray(sol, dtype=float)


def moving_average(y: np.ndarray, span: float) -> np.ndarray:
    """Apply the historical reflected moving average."""
    if span <= 0:
        return y.copy()
    n = y.size
    if n <= 2:
        return y.copy()
    window = max(3, int(round(span * n)))
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window <= 1:
        return y.copy()
    padding = window // 2
    padded = np.pad(y, (padding, padding), mode="reflect")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def enforce_unimodal(y: np.ndarray) -> np.ndarray:
    """Project a vector onto a unimodal shape via isotonic segments."""
    y = as_1d_float(y, "y")
    if y.size <= 2:
        return y.copy()
    mode_idx = int(np.nanargmax(y))
    left_x = np.arange(mode_idx + 1, dtype=float)
    right_x = np.arange(y.size - mode_idx, dtype=float)
    iso_inc = IsotonicRegression(increasing=True, out_of_bounds="clip")
    left = iso_inc.fit_transform(left_x, y[: mode_idx + 1])
    right_rev = iso_inc.fit_transform(right_x, y[mode_idx:][::-1])
    right = right_rev[::-1]
    out = y.copy()
    out[: mode_idx + 1] = left
    out[mode_idx:] = right
    out[mode_idx] = 0.5 * (left[-1] + right[0])
    return out


def normalize_column(col: np.ndarray, sum_norm: bool, norm_mode: str) -> np.ndarray:
    """Apply the historical component normalization rule."""
    denom = (
        float(np.sum(np.abs(col)))
        if sum_norm or norm_mode == "l1"
        else float(np.max(np.abs(col)))
    )
    if denom <= 0:
        return col
    return col / denom


def optimize_broadening_single(
    data_row: np.ndarray,
    c_row: np.ndarray,
    s: np.ndarray,
    g_init: np.ndarray,
    sigma_max: Optional[float] = None,
    broadening_vec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Three-stage grid refinement for broadening sigmas."""
    data_row = as_1d_float(data_row, "data_row")
    c_row = as_1d_float(c_row, "c_row")
    s = as_2d_float(s, "s")
    g_init = as_1d_float(g_init, "g_init")
    n_components = s.shape[1]
    if c_row.size != n_components:
        raise ValueError("c_row length must match number of components.")
    if g_init.size != n_components:
        raise ValueError("g_init length must match number of components.")
    sigma_max = float(max(0.10 * s.shape[0] if sigma_max is None else sigma_max, 1e-12))
    broadening_vec = (
        np.ones(n_components, dtype=bool)
        if broadening_vec is None
        else normalize_bool_vector(broadening_vec, n_components)
    )

    def eval_error(g_vec: np.ndarray) -> float:
        recon = np.zeros_like(data_row, dtype=float)
        for j in range(n_components):
            recon += c_row[j] * convolve_spectrum(s[:, j], float(g_vec[j]))
        return float(np.sum((data_row - recon) ** 2))

    g_opt = g_init.copy()
    for k in range(n_components):
        if not broadening_vec[k]:
            continue
        best_sigma, best_error = float(g_opt[k]), np.inf
        for fraction in (1.0, 0.1, 0.02):
            if fraction == 1.0:
                grid = np.linspace(0.0, sigma_max, num=10)
            else:
                spread = sigma_max * fraction
                grid = np.linspace(
                    max(0.0, best_sigma - spread),
                    min(sigma_max, best_sigma + spread),
                    num=10,
                )
            for sigma_test in grid:
                g_tmp = g_opt.copy()
                g_tmp[k] = sigma_test
                error = eval_error(g_tmp)
                if error < best_error:
                    best_error, best_sigma = error, float(sigma_test)
        g_opt[k] = best_sigma
    return g_opt


def solve_c(
    s: np.ndarray,
    data: np.ndarray,
    c: np.ndarray,
    nonneg_c: bool = True,
    null_c: Optional[np.ndarray] = None,
    close_c: bool = False,
    w_close_c: float = 0.0,
    g: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve for C given S and data, row-wise."""
    s, data, c = as_2d_float(s, "s"), as_2d_float(data, "data"), as_2d_float(c, "c")
    if data.shape[0] != c.shape[0]:
        raise ValueError("C row count must match data row count.")
    if s.shape[1] != c.shape[1]:
        raise ValueError("S component count must match C columns.")
    if data.shape[1] != s.shape[0]:
        raise ValueError("Data wavelength dimension must match S rows.")
    c_out = c.copy()
    use_broadening = g is not None and np.asarray(g).shape == c.shape
    if use_broadening:
        g = as_2d_float(g, "g")
    for i in range(data.shape[0]):
        s_work = s.copy()
        if use_broadening:
            for k in range(s.shape[1]):
                s_work[:, k] = convolve_spectrum(s[:, k], float(g[i, k]))
        b = data[i, :]
        if close_c and w_close_c != 0:
            s_aug = np.vstack([s_work, np.full((1, s_work.shape[1]), w_close_c)])
            b_aug = np.concatenate([b, np.array([w_close_c], dtype=float)])
        else:
            s_aug, b_aug = s_work, b
        c_out[i, :] = safe_solve(s_aug, b_aug, nonneg=nonneg_c)
    if null_c is not None:
        null_c = as_2d_float(null_c, "null_c")
        if null_c.shape == c_out.shape:
            c_out = c_out * null_c
    return c_out


def solve_s(
    c: np.ndarray,
    data: np.ndarray,
    s: np.ndarray,
    x_s: np.ndarray,
    nonneg_s: Union[bool, Sequence[bool]] = True,
    uni_s: bool = False,
    s0: Optional[np.ndarray] = None,
    norm_s: bool = True,
    smooth: float = 0.0,
    sum_s: bool = False,
    hard_s0: bool = True,
    w_hard_s0: float = 1.0,
    norm_mode: str = "intensity",
    g: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve for S given C and data, column-wise."""
    c, data, s = as_2d_float(c, "c"), as_2d_float(data, "data"), as_2d_float(s, "s")
    _ = as_1d_float(x_s, "x_s")
    if data.shape[0] != c.shape[0]:
        raise ValueError("C row count must match data rows.")
    if s.shape[1] != c.shape[1]:
        raise ValueError("S components must match C columns.")
    if data.shape[1] != s.shape[0]:
        raise ValueError("Data wavelength dimension must match S rows.")
    c_work, data_work, s_work = c.copy(), data.copy(), s.copy()
    nonneg_vec = normalize_nonneg_vector(nonneg_s, c_work.shape[1])
    c0 = None
    if s0 is not None:
        s0 = as_2d_float(s0, "s0")
        if s0.shape[0] != s.shape[0]:
            raise ValueError("s0 must have same wavelength length as S.")
        n_s0 = s0.shape[1]
        if hard_s0:
            if c_work.shape[1] == n_s0:
                return s0.copy()
            c0, c_work, s_work, nonneg_vec = (
                c_work[:, :n_s0],
                c_work[:, n_s0:],
                s_work[:, n_s0:],
                nonneg_vec[n_s0:],
            )
            data_work = data_work - c0 @ s0.T
            data_work[~np.isfinite(data_work)] = 0.0
        else:
            c_aug = np.vstack([c_work, np.zeros((n_s0, c_work.shape[1]))])
            for i in range(n_s0):
                c_aug[c_work.shape[0] + i, i] = w_hard_s0
            c_work, data_work = c_aug, np.vstack([data_work, w_hard_s0 * s0.T])
    if bool(np.all(nonneg_vec == nonneg_vec[0])):
        for j in range(data_work.shape[1]):
            s_work[j, :] = safe_solve(
                c_work, data_work[:, j], nonneg=bool(nonneg_vec[0])
            )
    else:
        idx_pos, idx_free = np.where(nonneg_vec)[0], np.where(~nonneg_vec)[0]
        for j in range(data_work.shape[1]):
            sol = np.zeros(c_work.shape[1], dtype=float)
            if idx_free.size:
                a_free = c_work[:, idx_free]
                sol[idx_free] = safe_lstsq(a_free, data_work[:, j])
                resid = data_work[:, j] - a_free @ sol[idx_free]
            else:
                resid = data_work[:, j]
            if idx_pos.size:
                sol[idx_pos] = safe_solve(c_work[:, idx_pos], resid, nonneg=True)
            s_work[j, :] = sol
    if uni_s:
        for i in range(s_work.shape[1]):
            s_work[:, i] = enforce_unimodal(s_work[:, i])
    if smooth and smooth > 0:
        for i in range(s_work.shape[1]):
            y = moving_average(s_work[:, i], float(smooth))
            s_work[:, i] = np.clip(y, 0.0, None) if nonneg_vec[i] else y
    if norm_s:
        for i in range(s_work.shape[1]):
            s_work[:, i] = normalize_column(s_work[:, i], bool(sum_s), norm_mode)
    if s0 is not None and hard_s0:
        s_work = np.hstack([s0, s_work])
        _ = c0
    return s_work


def solve_c_coupled(
    s: np.ndarray,
    data: np.ndarray,
    c: np.ndarray,
    nonneg_c: bool = True,
    null_c: Optional[np.ndarray] = None,
    close_c: bool = False,
    w_close_c: float = 0.0,
    n_fixed: int = 0,
) -> np.ndarray:
    """Solve for C then enforce correction/fixed pairwise coupling."""
    c_out = solve_c(s, data, c, nonneg_c, null_c, close_c, w_close_c, None)
    if n_fixed > 0:
        for i in range(n_fixed):
            corr_idx = n_fixed + i
            if corr_idx >= c_out.shape[1]:
                break
            c_fix, c_corr = c_out[:, i], c_out[:, corr_idx]
            norm_sq = float(np.sum(c_fix**2))
            c_out[:, corr_idx] = (
                0.0
                if norm_sq <= 1e-12
                else float(np.sum(c_corr * c_fix) / norm_sq) * c_fix
            )
    return c_out


def solve_s_coupled(
    c: np.ndarray,
    data: np.ndarray,
    s: np.ndarray,
    x_s: np.ndarray,
    nonneg_s: Union[bool, Sequence[bool]] = True,
    uni_s: bool = False,
    s0: Optional[np.ndarray] = None,
    norm_s: bool = True,
    smooth: float = 0.0,
    sum_s: bool = False,
    hard_s0: bool = True,
    w_hard_s0: float = 1.0,
    n_fixed: int = 0,
    lambda_corr: float = 0.0,
    norm_mode: str = "intensity",
) -> np.ndarray:
    """Solve for S with correction-spectra orthogonality/coupling constraints."""
    s_out = solve_s(
        c,
        data,
        s,
        x_s,
        nonneg_s,
        uni_s,
        s0,
        False,
        0.0,
        sum_s,
        hard_s0,
        w_hard_s0,
        norm_mode,
        None,
    )
    nonneg_vec = normalize_nonneg_vector(nonneg_s, s_out.shape[1])
    if n_fixed > 0:
        for i in range(n_fixed):
            corr_idx = n_fixed + i
            if corr_idx >= s_out.shape[1]:
                break
            s_fix, s_corr = s_out[:, i], s_out[:, corr_idx]
            norm_sq = float(np.sum(s_fix**2))
            if norm_sq > 1e-12:
                s_corr = s_corr - float(np.sum(s_corr * s_fix) / norm_sq) * s_fix
            s_corr = s_corr - float(np.mean(s_corr))
            if lambda_corr > 0:
                s_corr *= 1.0 / (1.0 + lambda_corr * np.sqrt(float(np.sum(s_corr**2))))
            s_out[:, corr_idx] = s_corr
    if smooth and smooth > 0:
        for i in range(s_out.shape[1]):
            if n_fixed > 0 and n_fixed <= i < 2 * n_fixed:
                continue
            smoothed = moving_average(s_out[:, i], float(smooth))
            s_out[:, i] = np.clip(smoothed, 0.0, None) if nonneg_vec[i] else smoothed
    if norm_s:
        if n_fixed > 0:
            for i in range(min(n_fixed, s_out.shape[1])):
                s_out[:, i] = normalize_column(s_out[:, i], bool(sum_s), norm_mode)
            for i in range(2 * n_fixed, s_out.shape[1]):
                s_out[:, i] = normalize_column(s_out[:, i], bool(sum_s), norm_mode)
        else:
            for i in range(s_out.shape[1]):
                s_out[:, i] = normalize_column(s_out[:, i], bool(sum_s), norm_mode)
    return s_out


def reconstruct_with_broadening(
    c: np.ndarray, s: np.ndarray, g: Optional[np.ndarray], broadening_vec: np.ndarray
) -> np.ndarray:
    """Reconstruct profiles, optionally convolving selected components."""
    c, s = as_2d_float(c, "c"), as_2d_float(s, "s")
    model = np.zeros((c.shape[0], s.shape[0]), dtype=float)
    if g is None or not np.any(broadening_vec):
        return c @ s.T
    g = as_2d_float(g, "g")
    for i in range(c.shape[0]):
        s_broad = s.copy()
        for k in range(s.shape[1]):
            if broadening_vec[k]:
                s_broad[:, k] = convolve_spectrum(s[:, k], float(g[i, k]))
        model[i, :] = c[i, :] @ s_broad.T
    return model
