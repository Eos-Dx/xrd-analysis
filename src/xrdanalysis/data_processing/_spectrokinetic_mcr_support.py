"""Private stateless input preparation for spectrokinetic transformers."""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from ._spectrokinetic_math import as_1d_float, as_2d_float, mask_to_bool


class _RestartResult(Protocol):
    """Structural restart state needed by initialization."""

    c: np.ndarray
    s: np.ndarray


def extract_masked_matrix(
    row: pd.Series,
    matrix_col: str,
    delay_col: str,
    wavelength_col: str,
    delay_mask_col: Optional[str],
    wavelength_mask_col: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a matrix row and apply its optional delay/wavelength masks."""
    mat = as_2d_float(row[matrix_col], matrix_col)
    delay = as_1d_float(row[delay_col], delay_col)
    wav = as_1d_float(row[wavelength_col], wavelength_col)
    if mat.shape != (delay.size, wav.size):
        raise ValueError(
            f"Matrix shape {mat.shape} must equal (len(delay), len(wavelength)) "
            f"= ({delay.size}, {wav.size})."
        )
    dmask = mask_to_bool(
        row[delay_mask_col] if delay_mask_col and delay_mask_col in row else None,
        delay.size,
    )
    wmask = mask_to_bool(
        (
            row[wavelength_mask_col]
            if wavelength_mask_col and wavelength_mask_col in row
            else None
        ),
        wav.size,
    )
    return mat[np.ix_(dmask, wmask)], delay[dmask], wav[wmask]


def extract_presence_mask(
    row: pd.Series,
    presence_mask_col: Optional[str],
    n_delay: int,
    n_components: int,
) -> Optional[np.ndarray]:
    """Return a validated per-delay component-presence mask when supplied."""
    if presence_mask_col is None or presence_mask_col not in row:
        return None
    mask = row[presence_mask_col]
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


def extract_fixed_spectra(
    row: pd.Series,
    wav: np.ndarray,
    fixed_spectra: Optional[np.ndarray],
    fixed_spectra_col: Optional[str],
    fixed_wavelength_axis: Optional[np.ndarray],
    interpolate_fixed: bool,
) -> Optional[np.ndarray]:
    """Return fixed spectra, interpolating only when explicitly enabled."""
    s0 = None
    src_wav = None
    if fixed_spectra is not None:
        s0 = np.asarray(fixed_spectra, dtype=float)
        if fixed_wavelength_axis is not None:
            src_wav = as_1d_float(fixed_wavelength_axis, "fixed_wavelength_axis")
    elif fixed_spectra_col is not None and fixed_spectra_col in row:
        value = row[fixed_spectra_col]
        if isinstance(value, dict):
            s0 = np.asarray(value.get("spectra"), dtype=float)
            wav_dict = value.get("wavelength")
            if wav_dict is not None:
                src_wav = as_1d_float(wav_dict, "fixed_spectra wavelength")
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
    if not interpolate_fixed:
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


def initialize_cs(
    mat: np.ndarray,
    n_start: int,
    init_method: str,
    rng: np.random.RandomState,
    last_result: Optional[_RestartResult],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create initial C/S matrices using SVD, sequential, PCA, NMF, or restart mode."""
    method = init_method.lower()
    if method in {"svd", "seq"}:
        u, _, vt = np.linalg.svd(mat, full_matrices=False)
        return np.abs(u[:, :n_start]), np.abs(vt[:n_start, :].T)
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
        return nmf.fit_transform(np.abs(fmat)), nmf.components_.T
    if method == "restart":
        if last_result is None:
            raise ValueError(
                "init_method='restart' requires restart_result or previous run."
            )
        if (
            last_result.c.shape[0] != mat.shape[0]
            or last_result.s.shape[0] != mat.shape[1]
        ):
            raise ValueError(
                "restart_result shape is incompatible with current matrix."
            )
        if last_result.c.shape[1] < n_start or last_result.s.shape[1] < n_start:
            raise ValueError(
                "restart_result has fewer components than requested n_start."
            )
        return last_result.c[:, :n_start].copy(), last_result.s[:, :n_start].copy()
    raise ValueError(f"Unsupported init_method '{init_method}'.")


def align_group_to_wavelength(
    mat: np.ndarray,
    wav: np.ndarray,
    base_wav: np.ndarray,
    allow_interpolation: bool,
) -> np.ndarray:
    """Align a group matrix to its reference wavelength grid when allowed."""
    if np.array_equal(wav, base_wav):
        return mat
    if not allow_interpolation:
        raise ValueError(
            "Group mode requires identical wavelength grids unless "
            "allow_group_wavelength_interpolation=True."
        )
    aligned = np.zeros((mat.shape[0], base_wav.size), dtype=float)
    for i in range(mat.shape[0]):
        aligned[i, :] = np.interp(base_wav, wav, mat[i, :])
    return aligned
