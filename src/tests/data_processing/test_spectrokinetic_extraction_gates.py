"""Deterministic numerical and public-API gates for spectrokinetic extraction."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from xrdanalysis.data_processing.spectrokinetic_transformers import (
    MCRALSTransformer,
    SpectroSVDTransformer,
    compute_lof,
    convolve_spectrum,
    enforce_unimodal,
    optimize_broadening_single,
    run_als_iteration,
    solve_C,
    solve_C_coupled,
    solve_S,
    solve_S_coupled,
)


def _frame(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "spectro_matrix": [matrix],
            "delay_axis": [np.arange(matrix.shape[0], dtype=float)],
            "wavelength_axis": [np.arange(matrix.shape[1], dtype=float)],
        }
    )


def test_small_svd_and_unconstrained_als_numerics_are_frozen():
    """Freeze low-dimensional SVD and ALS values before moving numerical kernels."""
    matrix = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    frame = _frame(matrix)

    svd = SpectroSVDTransformer(max_rank=4, model_rank=2).transform(frame).iloc[0]
    np.testing.assert_allclose(
        svd["svd_s"],
        [12.318132363107893, 1.414213562373095, 0.513434596569075, 0.0],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        svd["svd_lof_curve"],
        [12.123860463131521, 4.13737388440467, 0.0, 0.0],
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        svd["svd_model_rank_matrix"],
        [
            [1.01494403969435, 2.016684386063207, 2.992017345781067, 3.993757692149925],
            [
                2.029888079388701,
                4.033368772126411,
                5.984034691562133,
                7.987515384299845,
            ],
            [
                0.827871809241621,
                -0.192173819510088,
                1.091945675751544,
                0.071900046999836,
            ],
            [
                -0.258192286137569,
                0.711739270734868,
                0.137918513627316,
                1.107850070499753,
            ],
        ],
        rtol=0,
        atol=1e-12,
    )

    als = (
        MCRALSTransformer(
            n_components=2,
            init_method="svd",
            maxiter=2,
            thresh=1e-12,
            nonneg_c=False,
            nonneg_s=False,
            norm_s=False,
            random_state=0,
        )
        .transform(frame)
        .iloc[0]
    )
    assert als["als_iter"] == 2
    assert not als["als_converged"]
    assert als["als_lof_pct"] == pytest.approx(8.0906719784237, abs=1e-10)
    assert als["als_rss"] == pytest.approx(0.006545897306245046, abs=1e-14)
    np.testing.assert_allclose(
        als["als_model"],
        [
            [
                1.001378768545577,
                2.014303301535294,
                2.988281605062651,
                4.001206138052367,
            ],
            [
                2.002757537091154,
                4.028606603070587,
                5.976563210125302,
                8.002412276104733,
            ],
            [
                1.041290363529852,
                0.428344932848942,
                0.649065981014876,
                0.036120550333965,
            ],
            [
                -0.061935545294779,
                0.357482600726587,
                0.526401028477686,
                0.945819174499052,
            ],
        ],
        rtol=0,
        atol=1e-12,
    )


def _broadened_frame() -> pd.DataFrame:
    x_axis = np.arange(31, dtype=float)
    spectra = np.column_stack(
        [
            np.exp(-((x_axis - 9.0) ** 2) / (2.0 * 1.3**2)),
            np.exp(-((x_axis - 22.0) ** 2) / (2.0 * 1.7**2)),
        ]
    )
    concentrations = np.array([[1.0, 0.4], [0.7, 0.9], [1.2, 0.3], [0.5, 1.1]])
    sigmas = np.array([3.0, 4.0, 5.0, 3.0])
    matrix = np.array(
        [
            concentration[0] * convolve_spectrum(spectra[:, 0], sigma)
            + concentration[1] * spectra[:, 1]
            for concentration, sigma in zip(concentrations, sigmas, strict=True)
        ]
    )
    return _frame(matrix)


def test_broadening_mask_reconstruction_cap_and_exclusion_are_end_to_end():
    """Exercise the three-step ALS broadening path through the public transformer."""
    frame = _broadened_frame()
    common = {
        "n_components": 2,
        "init_method": "svd",
        "maxiter": 18,
        "thresh": 1e-7,
        "norm_s": True,
        "random_state": 0,
        "broadening_max_pct": 25.0,
    }
    without_broadening = (
        MCRALSTransformer(broadening=False, **common).transform(frame).iloc[0]
    )
    with_broadening = (
        MCRALSTransformer(broadening=[True, False], **common).transform(frame).iloc[0]
    )

    c_est = np.asarray(with_broadening["als_C"], dtype=float)
    s_est = np.asarray(with_broadening["als_S"], dtype=float)
    g_est = np.asarray(with_broadening["als_G"], dtype=float)
    cap = 0.25 * s_est.shape[0]
    expected_model = np.vstack(
        [
            c_est[row, 0] * convolve_spectrum(s_est[:, 0], g_est[row, 0])
            + c_est[row, 1] * s_est[:, 1]
            for row in range(c_est.shape[0])
        ]
    )

    assert bool(with_broadening["als_broadening_enabled"])
    assert g_est.shape == c_est.shape
    assert np.all((0.0 <= g_est[:, 0]) & (g_est[:, 0] <= cap))
    assert np.any(g_est[:, 0] > 0.0)
    np.testing.assert_array_equal(g_est[:, 1], np.zeros(c_est.shape[0]))
    np.testing.assert_allclose(with_broadening["als_model"], expected_model)
    assert with_broadening["als_rss"] <= without_broadening["als_rss"]

    with pytest.raises(ValueError, match="mutually exclusive"):
        MCRALSTransformer(
            broadening=True,
            correction_spectra=True,
            **common,
        ).transform(frame)


def test_correction_mode_is_end_to_end_coupled_and_orthogonal():
    """Protect transformer-level fixed/correction pair coupling invariants."""
    x_axis = np.linspace(0.0, 1.0, 29)
    fixed = np.cos(2.0 * np.pi * x_axis)
    fixed -= fixed.mean()
    correction = np.sin(2.0 * np.pi * x_axis)
    correction -= correction.mean()
    correction -= fixed * np.dot(correction, fixed) / np.dot(fixed, fixed)
    correction -= correction.mean()
    c_fixed = np.array([0.5, 0.8, 1.1, 0.6, 1.3, 0.9])
    c_correction = 0.35 * c_fixed
    matrix = (
        c_fixed[:, None] * fixed[None, :] + c_correction[:, None] * correction[None, :]
    )
    frame = pd.DataFrame(
        {
            "spectro_matrix": [matrix],
            "delay_axis": [np.arange(c_fixed.size, dtype=float)],
            "wavelength_axis": [x_axis],
        }
    )

    result = (
        MCRALSTransformer(
            n_components=2,
            init_method="svd",
            maxiter=6,
            thresh=1e-10,
            fixed_spectra=fixed[:, None],
            hard_s0=True,
            nonneg_c=True,
            nonneg_s=[False, False],
            norm_s=False,
            correction_spectra=True,
            correction_lambda=0.05,
            random_state=0,
        )
        .transform(frame)
        .iloc[0]
    )

    c_est = np.asarray(result["als_C"], dtype=float)
    s_est = np.asarray(result["als_S"], dtype=float)
    ratio = c_est[:, 1] / c_est[:, 0]
    correction_cosine = np.dot(s_est[:, 0], s_est[:, 1]) / (
        np.linalg.norm(s_est[:, 0]) * np.linalg.norm(s_est[:, 1])
    )

    assert result["als_meta"]["correction_spectra"] is True
    np.testing.assert_allclose(s_est[:, 0], fixed, rtol=0, atol=1e-12)
    assert np.std(ratio) < 1e-12
    assert abs(float(np.mean(s_est[:, 1]))) < 1e-12
    assert abs(float(correction_cosine)) < 1e-12
    np.testing.assert_allclose(result["als_model"], c_est @ s_est.T)
    assert result["als_lof_pct"] < 1e-10


@pytest.mark.parametrize(
    "function",
    [
        compute_lof,
        convolve_spectrum,
        enforce_unimodal,
        optimize_broadening_single,
        solve_C,
        solve_S,
        solve_C_coupled,
        solve_S_coupled,
        run_als_iteration,
    ],
)
def test_public_numerical_functions_keep_legacy_joblib_module_identity(
    function, tmp_path
):
    """Extraction may move kernels internally, not direct import/pickle identities."""
    assert (
        function.__module__ == "xrdanalysis.data_processing.spectrokinetic_transformers"
    )
    path = tmp_path / f"{function.__name__}.joblib"
    joblib.dump(function, path)
    assert joblib.load(path) is function
