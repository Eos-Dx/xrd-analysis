"""Deterministic input and initialization contracts for MCR support.

These tests protect extractions without exposing new public API.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from xrdanalysis.data_processing import spectrokinetic_transformers as sk

_MATRIX_AXIS_MESSAGE = "".join(
    [
        "Matrix shape (2, 4) must equal ",
        "(len(delay), len(wavelength)) = (3, 4).",
    ]
)


def _masked_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.arange(12, dtype=float).reshape(3, 4)
    delay = np.array([10.0, 20.0, 30.0])
    wavelength = np.array([400.0, 500.0, 600.0, 700.0])
    return (
        pd.DataFrame(
            {
                "spectro_matrix": [matrix],
                "delay_axis": [delay],
                "wavelength_axis": [wavelength],
                "delay_mask": [np.array([False, True, True])],
                # Floating masks retain every non-NaN position, including 0.0.
                "wavelength_mask": [np.array([np.nan, 1.0, np.nan, 0.0])],
            }
        ),
        matrix,
        delay,
        wavelength,
    )


def test_svd_and_mcr_share_masked_axis_contract_without_mutating_source():
    frame, matrix, delay, wavelength = _masked_frame()
    matrix_before, delay_before, wavelength_before = (
        matrix.copy(),
        delay.copy(),
        wavelength.copy(),
    )
    expected_matrix = np.array([[5.0, 7.0], [9.0, 11.0]])

    svd_out = (
        sk.SpectroSVDTransformer(
            delay_mask_col="delay_mask",
            wavelength_mask_col="wavelength_mask",
            max_rank=2,
            model_rank=2,
        )
        .transform(frame)
        .iloc[0]
    )
    mcr_out = (
        sk.MCRALSTransformer(
            delay_mask_col="delay_mask",
            wavelength_mask_col="wavelength_mask",
            n_components=1,
            maxiter=0,
        )
        .transform(frame)
        .iloc[0]
    )
    assert svd_out["svd_u"].shape == (2, 2)
    assert svd_out["svd_vt"].shape == (2, 2)
    assert svd_out["svd_model_rank_matrix"].shape == (2, 2)
    assert mcr_out["als_C"].shape == (2, 1)
    assert mcr_out["als_S"].shape == (2, 1)
    assert mcr_out["als_model"].shape == (2, 2)
    svd_model = svd_out["svd_model_rank_matrix"]
    np.testing.assert_allclose(svd_model, expected_matrix)
    np.testing.assert_allclose(
        mcr_out["als_model"] + mcr_out["als_residual"], expected_matrix
    )

    np.testing.assert_array_equal(matrix, matrix_before)
    np.testing.assert_array_equal(delay, delay_before)
    np.testing.assert_array_equal(wavelength, wavelength_before)
    assert list(frame.columns) == [
        "spectro_matrix",
        "delay_axis",
        "wavelength_axis",
        "delay_mask",
        "wavelength_mask",
    ]


@pytest.mark.parametrize(
    "transformer_type", [sk.SpectroSVDTransformer, sk.MCRALSTransformer]
)
@pytest.mark.parametrize(
    ("frame_update", "message"),
    [
        (
            {"delay_mask": [np.array([True])]},
            "Mask must have length 3; got shape (1,).",
        ),
        (
            {"wavelength_mask": [np.ones((1, 4))]},
            "Mask must have length 4; got shape (1, 4).",
        ),
        (
            {"spectro_matrix": [np.ones((2, 4))]},
            _MATRIX_AXIS_MESSAGE,
        ),
    ],
)
def test_shared_mask_input_failures_keep_their_messages(
    transformer_type, frame_update, message
):
    frame, _matrix, _delay, _wavelength = _masked_frame()
    for column, value in frame_update.items():
        frame[column] = value

    with pytest.raises(ValueError, match=re.escape(message)):
        transformer_type(
            delay_mask_col="delay_mask", wavelength_mask_col="wavelength_mask"
        ).transform(frame)


def test_fixed_spectra_config_precedence_and_one_dimensional_shape():
    wavelength = np.array([400.0, 500.0, 600.0, 700.0])
    configured = np.array([0.1, 0.3, 0.6, 1.0])
    row_value = {
        "spectra": np.full(wavelength.size, 99.0),
        "wavelength": wavelength,
    }
    frame = pd.DataFrame(
        {
            "spectro_matrix": [np.outer(np.array([1.0, 2.0]), configured)],
            "delay_axis": [np.array([0.0, 1.0])],
            "wavelength_axis": [wavelength],
            "fixed": [row_value],
        }
    )
    result = (
        sk.MCRALSTransformer(
            fixed_spectra=configured,
            fixed_spectra_col="fixed",
            interpolate_fixed=True,
            n_components=1,
            hard_s0=True,
            maxiter=1,
        )
        .transform(frame)
        .iloc[0]
    )

    assert result["als_S"].shape == (wavelength.size, 1)
    np.testing.assert_array_equal(result["als_S"][:, 0], configured)


def test_fixed_spectra_interpolation_uses_masked_wavelength_grid():
    frame = pd.DataFrame(
        {
            "spectro_matrix": [np.arange(15, dtype=float).reshape(3, 5)],
            "delay_axis": [np.array([0.0, 1.0, 2.0])],
            "wavelength_axis": [np.array([0.0, 1.0, 2.0, 3.0, 4.0])],
            "wavelength_mask": [np.array([0.0, np.nan, 1.0, np.nan, 1.0])],
        }
    )
    result = (
        sk.MCRALSTransformer(
            wavelength_mask_col="wavelength_mask",
            fixed_spectra=np.array([10.0, 30.0]),
            fixed_wavelength_axis=np.array([0.0, 4.0]),
            interpolate_fixed=True,
            n_components=1,
            hard_s0=True,
            maxiter=1,
        )
        .transform(frame)
        .iloc[0]
    )

    assert result["als_S"].shape == (3, 1)
    np.testing.assert_array_equal(result["als_S"], [[10.0], [20.0], [30.0]])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"fixed_spectra": np.array([1.0, 2.0])},
            "Fixed spectra wavelength length mismatch. "
            "Enable interpolate_fixed=True to interpolate onto "
            "data wavelength grid.",
        ),
        (
            {
                "fixed_spectra": np.array([1.0, 2.0]),
                "interpolate_fixed": True,
            },
            "fixed_wavelength_axis (or per-row wavelength) "
            "is required for interpolation.",
        ),
        (
            {
                "fixed_spectra": np.array([1.0, 2.0]),
                "fixed_wavelength_axis": np.array([0.0, 1.0, 2.0]),
                "interpolate_fixed": True,
            },
            "fixed_wavelength_axis length must match " "fixed spectra rows.",
        ),
    ],
)
def test_fixed_spectra_failure_messages(kwargs, message):
    transformer = sk.MCRALSTransformer(**kwargs)
    frame = pd.DataFrame(
        {
            "spectro_matrix": [np.ones((3, 4))],
            "delay_axis": [np.arange(3.0)],
            "wavelength_axis": [np.arange(4.0)],
        }
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        transformer.transform(frame)


@pytest.mark.parametrize(
    ("init_method", "expected_c", "expected_s"),
    [
        (
            "svd",
            [
                [0.2705980500730985, 0.6532814824381882],
                [0.6532814824381882, 0.2705980500730984],
                [0.2705980500730985, 0.6532814824381883],
                [0.6532814824381882, 0.2705980500730985],
            ],
            [
                [0.6380711874576985, 0.3047378541243652],
                [0.4309644062711510, 0.9023689270621824],
                [0.6380711874576986, 0.3047378541243648],
            ],
        ),
        (
            "pca",
            [
                [0.4999999999999995, 0.4999999999999997],
                [0.4999999999999998, 0.5000000000000001],
                [0.4999999999999998, 0.4999999999999998],
                [0.4999999999999998, 0.4999999999999998],
            ],
            [
                [2.9999999999999996, 1.4142135623730943],
                [1.4999999999999990, 0.0],
                [2.9999999999999980, 1.4142135623730950],
            ],
        ),
    ],
)
def test_svd_and_pca_initialization_are_frozen_at_zero_iterations(
    init_method, expected_c, expected_s
):
    matrix = np.array(
        [
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 3.0],
            [0.0, 2.0, 1.0],
            [3.0, 1.0, 2.0],
        ]
    )
    frame = pd.DataFrame(
        {
            "spectro_matrix": [matrix],
            "delay_axis": [np.arange(matrix.shape[0], dtype=float)],
            "wavelength_axis": [np.arange(matrix.shape[1], dtype=float)],
        }
    )

    result = (
        sk.MCRALSTransformer(
            n_components=2, init_method=init_method, maxiter=0, random_state=13
        )
        .transform(frame)
        .iloc[0]
    )

    assert result["als_iter"] == 0
    assert not result["als_converged"]
    actual_c, actual_s = result["als_C"], result["als_S"]
    np.testing.assert_allclose(actual_c, expected_c, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual_s, expected_s, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        result["als_model"],
        np.asarray(expected_c) @ np.asarray(expected_s).T,
        rtol=0,
        atol=1e-12,
    )


def test_nmf_initialization_is_repeatable_at_zero_iterations():
    matrix = np.array(
        [
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 3.0],
            [0.0, 2.0, 1.0],
            [3.0, 1.0, 2.0],
        ]
    )
    frame = pd.DataFrame(
        {
            "spectro_matrix": [matrix],
            "delay_axis": [np.arange(matrix.shape[0], dtype=float)],
            "wavelength_axis": [np.arange(matrix.shape[1], dtype=float)],
        }
    )
    kwargs = {
        "n_components": 2,
        "init_method": "nmf",
        "maxiter": 0,
        "random_state": 17,
    }

    first = sk.MCRALSTransformer(**kwargs).transform(frame).iloc[0]
    second = sk.MCRALSTransformer(**kwargs).transform(frame).iloc[0]

    assert first["als_iter"] == second["als_iter"] == 0
    for column in ("als_C", "als_S", "als_model", "als_residual"):
        one, two = first[column], second[column]
        np.testing.assert_allclose(one, two, rtol=0, atol=1e-12)
    assert np.all(first["als_C"] >= 0.0)
    assert np.all(first["als_S"] >= 0.0)
