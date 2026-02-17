"""Spectrokinetic transformers and numerical kernels.

This module ports core SK-Ana SVD + MCR-ALS behavior into sklearn-compatible
DataFrame transformers for use in xrd-analysis pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import TransformerMixin
from sklearn.decomposition import NMF
from sklearn.isotonic import IsotonicRegression


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _as_1d_float(arr: ArrayLike, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {out.shape}.")
    return out


def _as_2d_float(arr: ArrayLike, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {out.shape}.")
    return out


def _normalize_nonneg_vector(nonneg: Union[bool, Sequence[bool]], n_cols: int) -> np.ndarray:
    if isinstance(nonneg, (bool, np.bool_)):
        return np.full(n_cols, bool(nonneg), dtype=bool)
    vec = np.asarray(nonneg, dtype=bool)
    if vec.ndim != 1 or vec.size != n_cols:
        raise ValueError(
            f"nonneg_s must have length {n_cols}; got shape {vec.shape}."
        )
    return vec


def _normalize_bool_vector(flag: Union[bool, Sequence[bool]], n_cols: int) -> np.ndarray:
    if isinstance(flag, (bool, np.bool_)):
        return np.full(n_cols, bool(flag), dtype=bool)
    vec = np.asarray(flag, dtype=bool)
    if vec.ndim != 1 or vec.size != n_cols:
        raise ValueError(f"Boolean vector must have length {n_cols}, got {vec.shape}.")
    return vec


def _mask_to_bool(mask: Optional[ArrayLike], n: int) -> np.ndarray:
    if mask is None:
        return np.ones(n, dtype=bool)
    m = np.asarray(mask)
    if m.ndim != 1 or m.size != n:
        raise ValueError(f"Mask must have length {n}; got shape {m.shape}.")
    if m.dtype == bool:
        return m.copy()
    if np.issubdtype(m.dtype, np.floating):
        return ~np.isnan(m)
    return m.astype(bool)


def compute_lof(model: np.ndarray, data: np.ndarray) -> float:
    """Compute lack-of-fit percentage, matching SK-Ana convention."""
    model = _as_2d_float(model, "model")
    data = _as_2d_float(data, "data")
    denom = float(np.sum(data**2))
    if denom <= 0:
        return 100.0
    return float(100.0 * np.sqrt(np.sum((data - model) ** 2) / denom))


def convolve_spectrum(spectrum: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian convolution to a 1D spectrum."""
    spectrum = _as_1d_float(spectrum, "spectrum")
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


def _safe_lstsq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        sol = np.zeros(a.shape[1], dtype=float)
    return np.asarray(sol, dtype=float)


def _safe_solve(a: np.ndarray, b: np.ndarray, nonneg: bool) -> np.ndarray:
    if nonneg:
        try:
            sol, _ = nnls(a, b)
        except Exception:
            sol = np.zeros(a.shape[1], dtype=float)
    else:
        sol = _safe_lstsq(a, b)
    return np.asarray(sol, dtype=float)


def _moving_average(y: np.ndarray, span: float) -> np.ndarray:
    if span <= 0:
        return y.copy()
    n = y.size
    if n <= 2:
        return y.copy()
    w = max(3, int(round(span * n)))
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w <= 1:
        return y.copy()
    pad = w // 2
    y_pad = np.pad(y, (pad, pad), mode="reflect")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(y_pad, kernel, mode="valid")


def enforce_unimodal(y: np.ndarray) -> np.ndarray:
    """Project a vector onto a unimodal shape via isotonic segments."""
    y = _as_1d_float(y, "y")
    if y.size <= 2:
        return y.copy()

    mode_idx = int(np.nanargmax(y))
    left_x = np.arange(mode_idx + 1, dtype=float)
    right_x = np.arange(y.size - mode_idx, dtype=float)

    iso_inc = IsotonicRegression(increasing=True, out_of_bounds="clip")
    left = iso_inc.fit_transform(left_x, y[: mode_idx + 1])

    # Decreasing constraint on right by fitting increasing on reversed sequence.
    right_rev = iso_inc.fit_transform(right_x, y[mode_idx:][::-1])
    right = right_rev[::-1]

    out = y.copy()
    out[: mode_idx + 1] = left
    out[mode_idx:] = right
    out[mode_idx] = 0.5 * (left[-1] + right[0])
    return out


def _normalize_column(col: np.ndarray, sum_norm: bool, norm_mode: str) -> np.ndarray:
    if sum_norm or norm_mode == "l1":
        denom = float(np.sum(np.abs(col)))
    else:
        denom = float(np.max(np.abs(col)))
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
    data_row = _as_1d_float(data_row, "data_row")
    c_row = _as_1d_float(c_row, "c_row")
    s = _as_2d_float(s, "s")
    g_init = _as_1d_float(g_init, "g_init")

    n_components = s.shape[1]
    if c_row.size != n_components:
        raise ValueError("c_row length must match number of components.")
    if g_init.size != n_components:
        raise ValueError("g_init length must match number of components.")

    sigma_min = 0.0
    if sigma_max is None:
        sigma_max = 0.10 * s.shape[0]
    sigma_max = float(max(sigma_max, 1e-12))

    if broadening_vec is None:
        broadening_vec = np.ones(n_components, dtype=bool)
    else:
        broadening_vec = _normalize_bool_vector(broadening_vec, n_components)

    def eval_error(g_vec: np.ndarray) -> float:
        recon = np.zeros_like(data_row, dtype=float)
        for j in range(n_components):
            recon += c_row[j] * convolve_spectrum(s[:, j], float(g_vec[j]))
        return float(np.sum((data_row - recon) ** 2))

    g_opt = g_init.copy()

    for k in range(n_components):
        if not broadening_vec[k]:
            continue

        best_sigma = float(g_opt[k])
        best_error = np.inf

        grid_coarse = np.linspace(sigma_min, sigma_max, num=10)
        for sigma_test in grid_coarse:
            g_tmp = g_opt.copy()
            g_tmp[k] = sigma_test
            err = eval_error(g_tmp)
            if err < best_error:
                best_error = err
                best_sigma = float(sigma_test)

        range_medium = (sigma_max - sigma_min) * 0.1
        grid_medium = np.linspace(
            max(sigma_min, best_sigma - range_medium),
            min(sigma_max, best_sigma + range_medium),
            num=10,
        )
        for sigma_test in grid_medium:
            g_tmp = g_opt.copy()
            g_tmp[k] = sigma_test
            err = eval_error(g_tmp)
            if err < best_error:
                best_error = err
                best_sigma = float(sigma_test)

        range_fine = (sigma_max - sigma_min) * 0.02
        grid_fine = np.linspace(
            max(sigma_min, best_sigma - range_fine),
            min(sigma_max, best_sigma + range_fine),
            num=10,
        )
        for sigma_test in grid_fine:
            g_tmp = g_opt.copy()
            g_tmp[k] = sigma_test
            err = eval_error(g_tmp)
            if err < best_error:
                best_error = err
                best_sigma = float(sigma_test)

        g_opt[k] = best_sigma

    return g_opt


def solve_C(
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
    s = _as_2d_float(s, "s")
    data = _as_2d_float(data, "data")
    c = _as_2d_float(c, "c")

    if data.shape[0] != c.shape[0]:
        raise ValueError("C row count must match data row count.")
    if s.shape[1] != c.shape[1]:
        raise ValueError("S component count must match C columns.")
    if data.shape[1] != s.shape[0]:
        raise ValueError("Data wavelength dimension must match S rows.")

    c_out = c.copy()
    use_broadening = g is not None and np.asarray(g).shape == c.shape
    if use_broadening:
        g = _as_2d_float(g, "g")

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
            s_aug = s_work
            b_aug = b

        c_out[i, :] = _safe_solve(s_aug, b_aug, nonneg=nonneg_c)

    if null_c is not None:
        null_c = _as_2d_float(null_c, "null_c")
        if null_c.shape == c_out.shape:
            c_out = c_out * null_c

    return c_out


def solve_S(
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
    c = _as_2d_float(c, "c")
    data = _as_2d_float(data, "data")
    s = _as_2d_float(s, "s")
    _ = _as_1d_float(x_s, "x_s")

    if data.shape[0] != c.shape[0]:
        raise ValueError("C row count must match data rows.")
    if s.shape[1] != c.shape[1]:
        raise ValueError("S components must match C columns.")
    if data.shape[1] != s.shape[0]:
        raise ValueError("Data wavelength dimension must match S rows.")

    c_work = c.copy()
    data_work = data.copy()
    s_work = s.copy()
    nonneg_vec = _normalize_nonneg_vector(nonneg_s, c_work.shape[1])

    c0 = None
    if s0 is not None:
        s0 = _as_2d_float(s0, "s0")
        if s0.shape[0] != s.shape[0]:
            raise ValueError("s0 must have same wavelength length as S.")
        n_s0 = s0.shape[1]

        if hard_s0:
            if c_work.shape[1] == n_s0:
                return s0.copy()

            c0 = c_work[:, :n_s0]
            c_work = c_work[:, n_s0:]
            s_work = s_work[:, n_s0:]
            nonneg_vec = nonneg_vec[n_s0:]

            contrib = c0 @ s0.T
            data_work = data_work - contrib
            data_work[~np.isfinite(data_work)] = 0.0
        else:
            c_aug = np.vstack([c_work, np.zeros((n_s0, c_work.shape[1]))])
            for i in range(n_s0):
                c_aug[c_work.shape[0] + i, i] = w_hard_s0
            data_aug = np.vstack([data_work, w_hard_s0 * s0.T])
            c_work = c_aug
            data_work = data_aug

    all_same = bool(np.all(nonneg_vec == nonneg_vec[0]))
    use_broadening = g is not None and np.asarray(g).shape[0] == c_work.shape[0]

    if all_same:
        for j in range(data_work.shape[1]):
            if use_broadening:
                # Keep SK-Ana behavior path for broadening-enabled S updates.
                a = c_work
                b = data_work[:, j]
            else:
                a = c_work
                b = data_work[:, j]
            s_work[j, :] = _safe_solve(a, b, nonneg=bool(nonneg_vec[0]))
    else:
        idx_pos = np.where(nonneg_vec)[0]
        idx_free = np.where(~nonneg_vec)[0]
        for j in range(data_work.shape[1]):
            sol = np.zeros(c_work.shape[1], dtype=float)

            if idx_free.size > 0:
                a_free = c_work[:, idx_free]
                sol[idx_free] = _safe_lstsq(a_free, data_work[:, j])
                resid = data_work[:, j] - a_free @ sol[idx_free]
            else:
                resid = data_work[:, j]

            if idx_pos.size > 0:
                a_pos = c_work[:, idx_pos]
                sol[idx_pos] = _safe_solve(a_pos, resid, nonneg=True)

            s_work[j, :] = sol

    if uni_s:
        for i in range(s_work.shape[1]):
            s_work[:, i] = enforce_unimodal(s_work[:, i])

    if smooth and smooth > 0:
        for i in range(s_work.shape[1]):
            y = _moving_average(s_work[:, i], float(smooth))
            if nonneg_vec[i]:
                y = np.clip(y, 0.0, None)
            s_work[:, i] = y

    if norm_s:
        for i in range(s_work.shape[1]):
            s_work[:, i] = _normalize_column(
                s_work[:, i], sum_norm=bool(sum_s), norm_mode=norm_mode
            )

    if s0 is not None and hard_s0:
        s_work = np.hstack([s0, s_work])
        _ = c0  # kept for parity with R flow

    return s_work


def solve_C_coupled(
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
    c_out = solve_C(
        s=s,
        data=data,
        c=c,
        nonneg_c=nonneg_c,
        null_c=null_c,
        close_c=close_c,
        w_close_c=w_close_c,
        g=None,
    )

    if n_fixed > 0:
        for i in range(n_fixed):
            corr_idx = n_fixed + i
            if corr_idx >= c_out.shape[1]:
                break
            c_fix = c_out[:, i]
            c_corr = c_out[:, corr_idx]
            norm_sq = float(np.sum(c_fix**2))
            if norm_sq <= 1e-12:
                c_out[:, corr_idx] = 0.0
            else:
                alpha = float(np.sum(c_corr * c_fix) / norm_sq)
                c_out[:, corr_idx] = alpha * c_fix

    return c_out


def solve_S_coupled(
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
    s_out = solve_S(
        c=c,
        data=data,
        s=s,
        x_s=x_s,
        nonneg_s=nonneg_s,
        uni_s=uni_s,
        s0=s0,
        norm_s=False,
        smooth=0.0,
        sum_s=sum_s,
        hard_s0=hard_s0,
        w_hard_s0=w_hard_s0,
        norm_mode=norm_mode,
        g=None,
    )

    nonneg_vec = _normalize_nonneg_vector(nonneg_s, s_out.shape[1])

    if n_fixed > 0:
        for i in range(n_fixed):
            corr_idx = n_fixed + i
            if corr_idx >= s_out.shape[1]:
                break

            s_fix = s_out[:, i]
            s_corr = s_out[:, corr_idx]

            norm_sq = float(np.sum(s_fix**2))
            if norm_sq > 1e-12:
                proj = float(np.sum(s_corr * s_fix) / norm_sq)
                s_corr = s_corr - proj * s_fix

            s_corr = s_corr - float(np.mean(s_corr))

            if lambda_corr > 0:
                penalty = 1.0 / (1.0 + lambda_corr * np.sqrt(float(np.sum(s_corr**2))))
                s_corr = s_corr * penalty

            s_out[:, corr_idx] = s_corr

    if smooth and smooth > 0:
        for i in range(s_out.shape[1]):
            if n_fixed > 0 and n_fixed <= i < (2 * n_fixed):
                continue
            sm = _moving_average(s_out[:, i], float(smooth))
            if nonneg_vec[i]:
                sm = np.clip(sm, 0.0, None)
            s_out[:, i] = sm

    if norm_s:
        if n_fixed > 0:
            # Normalize fixed spectra.
            for i in range(min(n_fixed, s_out.shape[1])):
                s_out[:, i] = _normalize_column(
                    s_out[:, i], sum_norm=bool(sum_s), norm_mode=norm_mode
                )

            # Keep correction spectra as-is; normalize free spectra.
            start_free = 2 * n_fixed
            if s_out.shape[1] > start_free:
                for i in range(start_free, s_out.shape[1]):
                    s_out[:, i] = _normalize_column(
                        s_out[:, i], sum_norm=bool(sum_s), norm_mode=norm_mode
                    )
        else:
            for i in range(s_out.shape[1]):
                s_out[:, i] = _normalize_column(
                    s_out[:, i], sum_norm=bool(sum_s), norm_mode=norm_mode
                )

    return s_out


def _reconstruct_with_broadening(
    c: np.ndarray,
    s: np.ndarray,
    g: Optional[np.ndarray],
    broadening_vec: np.ndarray,
) -> np.ndarray:
    c = _as_2d_float(c, "c")
    s = _as_2d_float(s, "s")
    model = np.zeros((c.shape[0], s.shape[0]), dtype=float)

    if g is None or not np.any(broadening_vec):
        return c @ s.T

    g = _as_2d_float(g, "g")
    for i in range(c.shape[0]):
        s_broad = s.copy()
        for k in range(s.shape[1]):
            if broadening_vec[k]:
                s_broad[:, k] = convolve_spectrum(s[:, k], float(g[i, k]))
        model[i, :] = c[i, :] @ s_broad.T
    return model


@dataclass
class ALSConfig:
    n_components: int = 2
    n_start: Optional[int] = None
    maxiter: int = 100
    thresh: float = 1e-3
    init_method: str = "svd"
    opt_s_first: bool = True
    nonneg_c: bool = True
    nonneg_s: Union[bool, Sequence[bool]] = True
    uni_s: bool = False
    norm_s: bool = True
    sum_norm: bool = False
    norm_mode: str = "intensity"
    smooth: float = 0.0
    close_c: bool = False
    w_close_c: float = 0.0
    hard_s0: bool = True
    w_hard_s0: float = 1.0
    broadening: Union[bool, Sequence[bool]] = False
    broadening_max_pct: float = 10.0
    correction_spectra: bool = False
    correction_lambda: float = 0.0
    random_state: int = 42


@dataclass
class ALSResult:
    c: np.ndarray
    s: np.ndarray
    model: np.ndarray
    resid: np.ndarray
    rss: float
    iter: int
    lof: float
    converged: bool
    msg: str
    g: Optional[np.ndarray] = None
    broadening_vec: Optional[np.ndarray] = None
    n_fixed: int = 0
    lambda_corr: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


def run_als_iteration(
    c: np.ndarray,
    psi: np.ndarray,
    s: np.ndarray,
    x_c: np.ndarray,
    x_s: np.ndarray,
    config: ALSConfig,
    null_c: Optional[np.ndarray] = None,
    s0: Optional[np.ndarray] = None,
    g: Optional[np.ndarray] = None,
    n_fixed: int = 0,
) -> ALSResult:
    """Main ALS iterative loop (standard, broadening, or coupled mode)."""
    c = _as_2d_float(c, "c")
    psi = _as_2d_float(psi, "psi")
    s = _as_2d_float(s, "s")
    _ = _as_1d_float(x_c, "x_c")
    _ = _as_1d_float(x_s, "x_s")

    n_components = c.shape[1]
    if n_components != s.shape[1]:
        raise ValueError("C and S must have identical component count.")

    broadening_vec = _normalize_bool_vector(config.broadening, n_components)
    broadening_enabled = bool(np.any(broadening_vec))

    if config.correction_spectra and broadening_enabled:
        raise ValueError(
            "Correction-spectra and broadening modes are mutually exclusive in v1."
        )

    if broadening_enabled:
        if g is None:
            sigma_init = 0.001 * s.shape[0]
            g = np.zeros_like(c)
            for k in range(n_components):
                if broadening_vec[k]:
                    g[:, k] = sigma_init
        else:
            g = _as_2d_float(g, "g")
            if g.shape != c.shape:
                raise ValueError("G must have same shape as C when broadening is enabled.")

    model = _reconstruct_with_broadening(c, s, g, broadening_vec)
    resid = psi - model

    denom = max(float(np.sum(psi**2)), 1e-30)
    oldrss = float(np.sum(resid**2) / denom)
    rd = 1e20

    b = 1 if config.opt_s_first else 0
    iter_idx = 0
    min_steps = 3 if broadening_enabled else 2

    while iter_idx < config.maxiter and (iter_idx < min_steps or abs(rd) > config.thresh):
        iter_idx += 1

        if config.correction_spectra and n_fixed > 0:
            if iter_idx % 2 == b:
                s = solve_S_coupled(
                    c=c,
                    data=psi,
                    s=s,
                    x_s=x_s,
                    nonneg_s=config.nonneg_s,
                    uni_s=config.uni_s,
                    s0=s0,
                    norm_s=config.norm_s,
                    smooth=config.smooth,
                    sum_s=config.sum_norm,
                    hard_s0=config.hard_s0,
                    w_hard_s0=config.w_hard_s0,
                    n_fixed=n_fixed,
                    lambda_corr=config.correction_lambda,
                    norm_mode=config.norm_mode,
                )
            else:
                c = solve_C_coupled(
                    s=s,
                    data=psi,
                    c=c,
                    nonneg_c=config.nonneg_c,
                    null_c=null_c,
                    close_c=config.close_c,
                    w_close_c=config.w_close_c,
                    n_fixed=n_fixed,
                )
        elif broadening_enabled:
            step = iter_idx % 3
            if step == (b % 3):
                s = solve_S(
                    c=c,
                    data=psi,
                    s=s,
                    x_s=x_s,
                    nonneg_s=config.nonneg_s,
                    uni_s=config.uni_s,
                    s0=s0,
                    norm_s=config.norm_s,
                    smooth=config.smooth,
                    sum_s=config.sum_norm,
                    hard_s0=config.hard_s0,
                    w_hard_s0=config.w_hard_s0,
                    norm_mode=config.norm_mode,
                    g=g,
                )
            elif step == ((b + 1) % 3):
                c = solve_C(
                    s=s,
                    data=psi,
                    c=c,
                    nonneg_c=config.nonneg_c,
                    null_c=null_c,
                    close_c=config.close_c,
                    w_close_c=config.w_close_c,
                    g=g,
                )
            else:
                sigma_max = (config.broadening_max_pct / 100.0) * s.shape[0]
                for i in range(psi.shape[0]):
                    g_opt = optimize_broadening_single(
                        data_row=psi[i, :],
                        c_row=c[i, :],
                        s=s,
                        g_init=g[i, :],
                        sigma_max=sigma_max,
                        broadening_vec=broadening_vec,
                    )
                    for k in range(n_components):
                        if broadening_vec[k]:
                            g[i, k] = g_opt[k]
        else:
            if iter_idx % 2 == b:
                s = solve_S(
                    c=c,
                    data=psi,
                    s=s,
                    x_s=x_s,
                    nonneg_s=config.nonneg_s,
                    uni_s=config.uni_s,
                    s0=s0,
                    norm_s=config.norm_s,
                    smooth=config.smooth,
                    sum_s=config.sum_norm,
                    hard_s0=config.hard_s0,
                    w_hard_s0=config.w_hard_s0,
                    norm_mode=config.norm_mode,
                    g=None,
                )
            else:
                c = solve_C(
                    s=s,
                    data=psi,
                    c=c,
                    nonneg_c=config.nonneg_c,
                    null_c=null_c,
                    close_c=config.close_c,
                    w_close_c=config.w_close_c,
                    g=None,
                )

        model = _reconstruct_with_broadening(c, s, g, broadening_vec)
        resid = psi - model

        rss = float(np.sum(resid**2) / denom)
        if oldrss <= 0:
            rd = 0.0
        else:
            rd = (oldrss - rss) / oldrss
        oldrss = rss

        if not np.isfinite(rd):
            raise RuntimeError("ALS diverged: non-finite relative difference.")

    lof = compute_lof(model, psi)
    converged = bool(abs(rd) <= config.thresh)
    msg = (
        f"Dimension: {s.shape[1]}, |RD|={abs(rd):.3g}, "
        f"threshold={config.thresh:g}, LOF={lof:.3g}%"
    )

    return ALSResult(
        c=c,
        s=s,
        model=model,
        resid=resid,
        rss=float(oldrss),
        iter=iter_idx,
        lof=float(lof),
        converged=converged,
        msg=msg,
        g=g,
        broadening_vec=broadening_vec,
        n_fixed=int(n_fixed),
        lambda_corr=float(config.correction_lambda),
    )


class SpectroSVDTransformer(TransformerMixin):
    """SVD transformer for matrix-per-row spectrokinetic data."""

    def __init__(
        self,
        matrix_col: str = "spectro_matrix",
        delay_col: str = "delay_axis",
        wavelength_col: str = "wavelength_axis",
        delay_mask_col: Optional[str] = None,
        wavelength_mask_col: Optional[str] = None,
        max_rank: int = 10,
        model_rank: Optional[int] = None,
    ):
        self.matrix_col = matrix_col
        self.delay_col = delay_col
        self.wavelength_col = wavelength_col
        self.delay_mask_col = delay_mask_col
        self.wavelength_mask_col = wavelength_mask_col
        self.max_rank = int(max_rank)
        self.model_rank = model_rank

    def fit(self, x: pd.DataFrame, y=None):
        _ = x
        _ = y
        return self

    def _extract_masked_matrix(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mat = _as_2d_float(row[self.matrix_col], self.matrix_col)
        delay = _as_1d_float(row[self.delay_col], self.delay_col)
        wav = _as_1d_float(row[self.wavelength_col], self.wavelength_col)

        if mat.shape != (delay.size, wav.size):
            raise ValueError(
                f"Matrix shape {mat.shape} must equal (len(delay), len(wavelength)) "
                f"= ({delay.size}, {wav.size})."
            )

        dmask = _mask_to_bool(
            row[self.delay_mask_col] if self.delay_mask_col and self.delay_mask_col in row else None,
            delay.size,
        )
        wmask = _mask_to_bool(
            row[self.wavelength_mask_col]
            if self.wavelength_mask_col and self.wavelength_mask_col in row
            else None,
            wav.size,
        )

        return mat[np.ix_(dmask, wmask)], delay[dmask], wav[wmask]

    @staticmethod
    def _recommended_rank(svals: np.ndarray) -> int:
        if svals.size == 0:
            return 1
        if svals.size < 4:
            return int(min(1, svals.size))
        tail_start = max(1, int(np.floor(2 * svals.size / 3)))
        baseline = float(np.mean(svals[tail_start:]))
        if baseline <= 0:
            return 1
        idx = np.where(svals > 1.5 * baseline)[0]
        if idx.size == 0:
            return 1
        return int(idx[-1] + 1)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()

        for col in [
            "svd_u",
            "svd_s",
            "svd_vt",
            "svd_lof_curve",
            "svd_sd_resid_curve",
            "svd_model_rank_matrix",
            "svd_residual_rank_matrix",
            "svd_recommended_rank",
        ]:
            out[col] = None

        for idx, row in out.iterrows():
            mat, _, _ = self._extract_masked_matrix(row)
            n_rank = min(self.max_rank, min(mat.shape))
            u, svals, vt = np.linalg.svd(mat, full_matrices=False)

            u = u[:, :n_rank]
            svals = svals[:n_rank]
            vt = vt[:n_rank, :]

            lof_curve: List[float] = []
            sd_curve: List[float] = []

            model = np.zeros_like(mat)
            for r in range(n_rank):
                model = model + svals[r] * np.outer(u[:, r], vt[r, :])
                resid = mat - model
                lof_curve.append(compute_lof(model, mat))
                sd_curve.append(float(np.std(resid)))

            rec_rank = self._recommended_rank(svals)
            rank_for_model = int(self.model_rank) if self.model_rank is not None else rec_rank
            rank_for_model = max(1, min(rank_for_model, n_rank))

            model_rank = np.zeros_like(mat)
            for r in range(rank_for_model):
                model_rank += svals[r] * np.outer(u[:, r], vt[r, :])

            out.at[idx, "svd_u"] = u
            out.at[idx, "svd_s"] = svals
            out.at[idx, "svd_vt"] = vt
            out.at[idx, "svd_lof_curve"] = lof_curve
            out.at[idx, "svd_sd_resid_curve"] = sd_curve
            out.at[idx, "svd_model_rank_matrix"] = model_rank
            out.at[idx, "svd_residual_rank_matrix"] = mat - model_rank
            out.at[idx, "svd_recommended_rank"] = rec_rank

        return out


class MCRALSTransformer(TransformerMixin):
    """MCR-ALS transformer for matrix-per-row spectrokinetic data."""

    def __init__(
        self,
        matrix_col: str = "spectro_matrix",
        delay_col: str = "delay_axis",
        wavelength_col: str = "wavelength_axis",
        delay_mask_col: Optional[str] = None,
        wavelength_mask_col: Optional[str] = None,
        decomposition_mode: str = "row",
        group_col: Optional[str] = None,
        group_strategy: str = "tile_delay",
        allow_group_wavelength_interpolation: bool = False,
        n_components: int = 2,
        n_start: Optional[int] = None,
        init_method: str = "svd",
        maxiter: int = 100,
        thresh: float = 1e-3,
        opt_s_first: bool = True,
        nonneg_c: bool = True,
        nonneg_s: Union[bool, Sequence[bool]] = True,
        uni_s: bool = False,
        norm_s: bool = True,
        sum_norm: bool = False,
        norm_mode: str = "intensity",
        smooth: float = 0.0,
        close_c: bool = False,
        w_close_c: float = 0.0,
        presence_mask_col: Optional[str] = None,
        fixed_spectra: Optional[np.ndarray] = None,
        fixed_spectra_col: Optional[str] = None,
        fixed_wavelength_axis: Optional[np.ndarray] = None,
        interpolate_fixed: bool = False,
        hard_s0: bool = True,
        w_hard_s0: float = 1.0,
        correction_spectra: bool = False,
        correction_lambda: float = 0.0,
        broadening: Union[bool, Sequence[bool]] = False,
        broadening_max_pct: float = 10.0,
        random_state: int = 42,
        restart_result: Optional[ALSResult] = None,
    ):
        self.matrix_col = matrix_col
        self.delay_col = delay_col
        self.wavelength_col = wavelength_col
        self.delay_mask_col = delay_mask_col
        self.wavelength_mask_col = wavelength_mask_col

        self.decomposition_mode = decomposition_mode
        self.group_col = group_col
        self.group_strategy = group_strategy
        self.allow_group_wavelength_interpolation = allow_group_wavelength_interpolation

        self.n_components = int(n_components)
        self.n_start = n_start
        self.init_method = init_method
        self.maxiter = int(maxiter)
        self.thresh = float(thresh)
        self.opt_s_first = bool(opt_s_first)

        self.nonneg_c = bool(nonneg_c)
        self.nonneg_s = nonneg_s
        self.uni_s = bool(uni_s)
        self.norm_s = bool(norm_s)
        self.sum_norm = bool(sum_norm)
        self.norm_mode = norm_mode
        self.smooth = float(smooth)

        self.close_c = bool(close_c)
        self.w_close_c = float(w_close_c)
        self.presence_mask_col = presence_mask_col

        self.fixed_spectra = fixed_spectra
        self.fixed_spectra_col = fixed_spectra_col
        self.fixed_wavelength_axis = fixed_wavelength_axis
        self.interpolate_fixed = bool(interpolate_fixed)
        self.hard_s0 = bool(hard_s0)
        self.w_hard_s0 = float(w_hard_s0)

        self.correction_spectra = bool(correction_spectra)
        self.correction_lambda = float(correction_lambda)

        self.broadening = broadening
        self.broadening_max_pct = float(broadening_max_pct)

        self.random_state = int(random_state)
        self.restart_result = restart_result
        self._last_result: Optional[ALSResult] = restart_result

    def fit(self, x: pd.DataFrame, y=None):
        _ = x
        _ = y
        return self

    def _extract_masked_matrix(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mat = _as_2d_float(row[self.matrix_col], self.matrix_col)
        delay = _as_1d_float(row[self.delay_col], self.delay_col)
        wav = _as_1d_float(row[self.wavelength_col], self.wavelength_col)

        if mat.shape != (delay.size, wav.size):
            raise ValueError(
                f"Matrix shape {mat.shape} must equal (len(delay), len(wavelength)) "
                f"= ({delay.size}, {wav.size})."
            )

        dmask = _mask_to_bool(
            row[self.delay_mask_col] if self.delay_mask_col and self.delay_mask_col in row else None,
            delay.size,
        )
        wmask = _mask_to_bool(
            row[self.wavelength_mask_col]
            if self.wavelength_mask_col and self.wavelength_mask_col in row
            else None,
            wav.size,
        )

        return mat[np.ix_(dmask, wmask)], delay[dmask], wav[wmask]

    def _extract_presence_mask(self, row: pd.Series, n_delay: int, n_components: int) -> Optional[np.ndarray]:
        if self.presence_mask_col is None or self.presence_mask_col not in row:
            return None
        mask = row[self.presence_mask_col]
        if mask is None:
            return None
        arr = np.asarray(mask, dtype=float)
        if arr.ndim == 1:
            if arr.size != n_components:
                raise ValueError(
                    f"Presence mask vector must have length {n_components}, got {arr.size}."
                )
            arr = np.tile(arr[None, :], (n_delay, 1))
        if arr.shape != (n_delay, n_components):
            raise ValueError(
                f"Presence mask must have shape {(n_delay, n_components)}, got {arr.shape}."
            )
        return arr

    def _extract_fixed_spectra(self, row: pd.Series, wav: np.ndarray) -> Optional[np.ndarray]:
        s0 = None
        src_wav = None

        if self.fixed_spectra is not None:
            s0 = np.asarray(self.fixed_spectra, dtype=float)
            if self.fixed_wavelength_axis is not None:
                src_wav = _as_1d_float(self.fixed_wavelength_axis, "fixed_wavelength_axis")
        elif self.fixed_spectra_col is not None and self.fixed_spectra_col in row:
            value = row[self.fixed_spectra_col]
            if isinstance(value, dict):
                s0 = np.asarray(value.get("spectra"), dtype=float)
                wav_dict = value.get("wavelength")
                if wav_dict is not None:
                    src_wav = _as_1d_float(wav_dict, "fixed_spectra wavelength")
            elif value is not None:
                s0 = np.asarray(value, dtype=float)

        if s0 is None:
            return None

        if s0.ndim == 1:
            s0 = s0.reshape(-1, 1)
        if s0.ndim != 2:
            raise ValueError("Fixed spectra must be 1D or 2D.")

        if s0.shape[0] == wav.size:
            return s0

        if not self.interpolate_fixed:
            raise ValueError(
                "Fixed spectra wavelength length mismatch. "
                "Enable interpolate_fixed=True to interpolate onto data wavelength grid."
            )

        if src_wav is None:
            raise ValueError(
                "fixed_wavelength_axis (or per-row wavelength) is required for interpolation."
            )
        if src_wav.size != s0.shape[0]:
            raise ValueError("fixed_wavelength_axis length must match fixed spectra rows.")

        s0_interp = np.zeros((wav.size, s0.shape[1]), dtype=float)
        for j in range(s0.shape[1]):
            s0_interp[:, j] = np.interp(wav, src_wav, s0[:, j])
        return s0_interp

    def _initialize_cs(
        self,
        mat: np.ndarray,
        n_start: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        method = self.init_method.lower()

        if method in {"svd", "seq"}:
            u, _, vt = np.linalg.svd(mat, full_matrices=False)
            c_init = np.abs(u[:, :n_start])
            s_init = np.abs(vt[:n_start, :].T)
            return c_init, s_init

        if method == "pca":
            mat_centered = mat - np.mean(mat, axis=0, keepdims=True)
            u, d, vt = np.linalg.svd(mat_centered, full_matrices=False)
            c_init = np.abs(u[:, :n_start])
            s_init = np.abs(vt[:n_start, :].T)
            for i in range(n_start):
                denom = float(np.max(np.abs(s_init[:, i])))
                if denom > 0:
                    s_init[:, i] = s_init[:, i] * float(d[i]) / denom
            return c_init, s_init

        if method == "nmf":
            u, d, vt = np.linalg.svd(mat, full_matrices=False)
            fmat = np.zeros_like(mat)
            for i in range(min(n_start, d.size)):
                fmat += d[i] * np.outer(u[:, i], vt[i, :])
            nmf = NMF(
                n_components=n_start,
                init="nndsvda",
                random_state=rng.randint(0, 1_000_000),
                max_iter=500,
            )
            w = nmf.fit_transform(np.abs(fmat))
            h = nmf.components_
            return w, h.T

        if method == "restart":
            ref = self._last_result
            if ref is None:
                raise ValueError("init_method='restart' requires restart_result or previous run.")
            if ref.c.shape[0] != mat.shape[0] or ref.s.shape[0] != mat.shape[1]:
                raise ValueError("restart_result shape is incompatible with current matrix.")
            if ref.c.shape[1] < n_start or ref.s.shape[1] < n_start:
                raise ValueError("restart_result has fewer components than requested n_start.")
            return ref.c[:, :n_start].copy(), ref.s[:, :n_start].copy()

        raise ValueError(f"Unsupported init_method '{self.init_method}'.")

    def _build_config(self, n_components: int) -> ALSConfig:
        return ALSConfig(
            n_components=n_components,
            n_start=self.n_start,
            maxiter=self.maxiter,
            thresh=self.thresh,
            init_method=self.init_method,
            opt_s_first=self.opt_s_first,
            nonneg_c=self.nonneg_c,
            nonneg_s=self.nonneg_s,
            uni_s=self.uni_s,
            norm_s=self.norm_s,
            sum_norm=self.sum_norm,
            norm_mode=self.norm_mode,
            smooth=self.smooth,
            close_c=self.close_c,
            w_close_c=self.w_close_c,
            hard_s0=self.hard_s0,
            w_hard_s0=self.w_hard_s0,
            broadening=self.broadening,
            broadening_max_pct=self.broadening_max_pct,
            correction_spectra=self.correction_spectra,
            correction_lambda=self.correction_lambda,
            random_state=self.random_state,
        )

    def _run_single(self,
        mat: np.ndarray,
        delay: np.ndarray,
        wav: np.ndarray,
        row: pd.Series,
    ) -> ALSResult:
        rng = np.random.RandomState(self.random_state)
        n_target = int(self.n_components)
        if n_target < 1:
            raise ValueError("n_components must be >= 1.")

        n_start = self.n_start
        if n_start is None:
            n_start = 2 if self.init_method.lower() == "seq" and n_target > 1 else n_target
        n_start = int(max(1, min(n_start, n_target)))

        s0 = self._extract_fixed_spectra(row, wav)
        n_fixed = 0 if s0 is None else int(s0.shape[1])

        if self.correction_spectra and n_fixed > 0 and n_target < 2 * n_fixed:
            raise ValueError(
                f"n_components={n_target} must be >= 2*n_fixed={2 * n_fixed} for correction mode."
            )

        c_init, s_init = self._initialize_cs(mat, n_start=n_start, rng=rng)

        result: Optional[ALSResult] = None
        c_cur = c_init
        s_cur = s_init
        g_cur = None

        for n in range(n_start, n_target + 1):
            if n > c_cur.shape[1]:
                c_cur = np.hstack([c_cur, np.ones((c_cur.shape[0], 1), dtype=float)])
                s_cur = np.hstack([s_cur, np.ones((s_cur.shape[0], 1), dtype=float)])

            cfg = self._build_config(n_components=n)

            nonneg_s_eff: Union[bool, Sequence[bool]]
            if isinstance(self.nonneg_s, (bool, np.bool_)):
                nonneg_s_eff = bool(self.nonneg_s)
            else:
                vec = np.asarray(self.nonneg_s, dtype=bool)
                if vec.size < n:
                    raise ValueError(
                        f"nonneg_s vector length {vec.size} is smaller than required components {n}."
                    )
                nonneg_s_eff = vec[:n]
            cfg.nonneg_s = nonneg_s_eff

            if isinstance(self.broadening, (bool, np.bool_)):
                cfg.broadening = bool(self.broadening)
            else:
                bvec = np.asarray(self.broadening, dtype=bool)
                if bvec.size < n:
                    raise ValueError(
                        f"broadening vector length {bvec.size} is smaller than required components {n}."
                    )
                cfg.broadening = bvec[:n]

            null_c = self._extract_presence_mask(row, n_delay=mat.shape[0], n_components=n)

            result = run_als_iteration(
                c=c_cur[:, :n],
                psi=mat,
                s=s_cur[:, :n],
                x_c=delay,
                x_s=wav,
                config=cfg,
                null_c=null_c,
                s0=s0,
                g=g_cur[:, :n] if g_cur is not None and g_cur.shape[1] >= n else None,
                n_fixed=n_fixed,
            )

            c_cur = result.c
            s_cur = result.s
            g_cur = result.g

        if result is None:
            raise RuntimeError("ALS run produced no result.")

        self._last_result = result
        return result

    def _align_group_to_wavelength(
        self,
        mat: np.ndarray,
        wav: np.ndarray,
        base_wav: np.ndarray,
    ) -> np.ndarray:
        if np.array_equal(wav, base_wav):
            return mat
        if not self.allow_group_wavelength_interpolation:
            raise ValueError(
                "Group mode requires identical wavelength grids unless "
                "allow_group_wavelength_interpolation=True."
            )
        aligned = np.zeros((mat.shape[0], base_wav.size), dtype=float)
        for i in range(mat.shape[0]):
            aligned[i, :] = np.interp(base_wav, wav, mat[i, :])
        return aligned

    def _run_group(self, gdf: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        prepared: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, pd.Series]] = []
        for ridx, row in gdf.iterrows():
            mat, delay, wav = self._extract_masked_matrix(row)
            prepared.append((ridx, mat, delay, wav, row))

        if not prepared:
            return {}

        out_map: Dict[int, Dict[str, Any]] = {}

        if self.group_strategy == "mean":
            base_wav = prepared[0][3]
            base_delay = prepared[0][2]
            mats = []
            for _, mat, delay, wav, _ in prepared:
                if delay.size != base_delay.size or not np.allclose(delay, base_delay):
                    raise ValueError("group_strategy='mean' requires identical delay axes.")
                mats.append(self._align_group_to_wavelength(mat, wav, base_wav))
            mean_mat = np.mean(np.stack(mats, axis=0), axis=0)
            ref_row = prepared[0][4]
            result = self._run_single(mean_mat, base_delay, base_wav, ref_row)

            for ridx, _, _, _, _ in prepared:
                out_map[ridx] = {
                    "als_C": result.c,
                    "als_S": result.s,
                    "als_model": result.model,
                    "als_residual": result.resid,
                    "als_lof_pct": result.lof,
                    "als_rss": result.rss,
                    "als_iter": result.iter,
                    "als_converged": result.converged,
                    "als_G": result.g,
                    "als_broadening_enabled": bool(
                        result.broadening_vec is not None and np.any(result.broadening_vec)
                    ),
                    "als_meta": {
                        "decomposition_mode": "group",
                        "group_strategy": "mean",
                        "group_size": len(prepared),
                        "n_components": result.s.shape[1],
                        "n_fixed": result.n_fixed,
                        "init_method": self.init_method,
                        "constraints": {
                            "nonneg_c": self.nonneg_c,
                            "nonneg_s": (
                                np.asarray(self.nonneg_s, dtype=bool).tolist()
                                if not isinstance(self.nonneg_s, (bool, np.bool_))
                                else bool(self.nonneg_s)
                            ),
                            "uni_s": self.uni_s,
                            "norm_s": self.norm_s,
                            "sum_norm": self.sum_norm,
                            "norm_mode": self.norm_mode,
                            "smooth": self.smooth,
                            "close_c": self.close_c,
                            "w_close_c": self.w_close_c,
                        },
                        "correction_spectra": self.correction_spectra,
                        "broadening": self.broadening,
                    },
                }
            return out_map

        if self.group_strategy != "tile_delay":
            raise ValueError(
                f"Unsupported group_strategy '{self.group_strategy}'. "
                "Expected 'tile_delay' or 'mean'."
            )

        base_wav = prepared[0][3]
        mats: List[np.ndarray] = []
        delays: List[np.ndarray] = []
        slices: List[Tuple[int, int, int, pd.Series]] = []

        start = 0
        for ridx, mat, delay, wav, row in prepared:
            mat_aligned = self._align_group_to_wavelength(mat, wav, base_wav)
            mats.append(mat_aligned)
            delays.append(delay)
            end = start + mat_aligned.shape[0]
            slices.append((ridx, start, end, row))
            start = end

        tile_mat = np.vstack(mats)
        tile_delay = np.concatenate(delays)
        ref_row = prepared[0][4]
        result = self._run_single(tile_mat, tile_delay, base_wav, ref_row)

        for ridx, s0, s1, _row in slices:
            c_slice = result.c[s0:s1, :]
            model_slice = result.model[s0:s1, :]
            resid_slice = result.resid[s0:s1, :]
            g_slice = result.g[s0:s1, :] if result.g is not None else None

            out_map[ridx] = {
                "als_C": c_slice,
                "als_S": result.s,
                "als_model": model_slice,
                "als_residual": resid_slice,
                "als_lof_pct": result.lof,
                "als_rss": result.rss,
                "als_iter": result.iter,
                "als_converged": result.converged,
                "als_G": g_slice,
                "als_broadening_enabled": bool(
                    result.broadening_vec is not None and np.any(result.broadening_vec)
                ),
                "als_meta": {
                    "decomposition_mode": "group",
                    "group_strategy": "tile_delay",
                    "group_size": len(prepared),
                    "n_components": result.s.shape[1],
                    "n_fixed": result.n_fixed,
                    "init_method": self.init_method,
                    "constraints": {
                        "nonneg_c": self.nonneg_c,
                        "nonneg_s": (
                            np.asarray(self.nonneg_s, dtype=bool).tolist()
                            if not isinstance(self.nonneg_s, (bool, np.bool_))
                            else bool(self.nonneg_s)
                        ),
                        "uni_s": self.uni_s,
                        "norm_s": self.norm_s,
                        "sum_norm": self.sum_norm,
                        "norm_mode": self.norm_mode,
                        "smooth": self.smooth,
                        "close_c": self.close_c,
                        "w_close_c": self.w_close_c,
                    },
                    "correction_spectra": self.correction_spectra,
                    "broadening": self.broadening,
                    "tile_slice": (s0, s1),
                },
            }

        return out_map

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()

        for col in [
            "als_C",
            "als_S",
            "als_model",
            "als_residual",
            "als_lof_pct",
            "als_rss",
            "als_iter",
            "als_converged",
            "als_G",
            "als_broadening_enabled",
            "als_meta",
        ]:
            out[col] = None

        if self.decomposition_mode == "row":
            for idx, row in out.iterrows():
                mat, delay, wav = self._extract_masked_matrix(row)
                result = self._run_single(mat, delay, wav, row)

                out.at[idx, "als_C"] = result.c
                out.at[idx, "als_S"] = result.s
                out.at[idx, "als_model"] = result.model
                out.at[idx, "als_residual"] = result.resid
                out.at[idx, "als_lof_pct"] = result.lof
                out.at[idx, "als_rss"] = result.rss
                out.at[idx, "als_iter"] = result.iter
                out.at[idx, "als_converged"] = result.converged
                out.at[idx, "als_G"] = result.g
                out.at[idx, "als_broadening_enabled"] = bool(
                    result.broadening_vec is not None and np.any(result.broadening_vec)
                )
                out.at[idx, "als_meta"] = {
                    "decomposition_mode": "row",
                    "n_components": result.s.shape[1],
                    "n_fixed": result.n_fixed,
                    "init_method": self.init_method,
                    "constraints": {
                        "nonneg_c": self.nonneg_c,
                        "nonneg_s": (
                            np.asarray(self.nonneg_s, dtype=bool).tolist()
                            if not isinstance(self.nonneg_s, (bool, np.bool_))
                            else bool(self.nonneg_s)
                        ),
                        "uni_s": self.uni_s,
                        "norm_s": self.norm_s,
                        "sum_norm": self.sum_norm,
                        "norm_mode": self.norm_mode,
                        "smooth": self.smooth,
                        "close_c": self.close_c,
                        "w_close_c": self.w_close_c,
                    },
                    "correction_spectra": self.correction_spectra,
                    "broadening": self.broadening,
                }

            return out

        if self.decomposition_mode != "group":
            raise ValueError(
                f"Unsupported decomposition_mode '{self.decomposition_mode}'. "
                "Expected 'row' or 'group'."
            )

        if self.group_col is None or self.group_col not in out.columns:
            raise ValueError("group_col must be provided and present when decomposition_mode='group'.")

        for _g, gdf in out.groupby(self.group_col, sort=False):
            mapping = self._run_group(gdf)
            for ridx, payload in mapping.items():
                for col, val in payload.items():
                    out.at[ridx, col] = val

        return out


__all__ = [
    "ALSConfig",
    "ALSResult",
    "SpectroSVDTransformer",
    "MCRALSTransformer",
    "compute_lof",
    "convolve_spectrum",
    "optimize_broadening_single",
    "solve_C",
    "solve_S",
    "solve_C_coupled",
    "solve_S_coupled",
    "run_als_iteration",
    "enforce_unimodal",
]
