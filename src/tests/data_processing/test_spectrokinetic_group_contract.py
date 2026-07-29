"""Contract tests for grouped MCR-ALS output assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xrdanalysis.data_processing.spectrokinetic_transformers import MCRALSTransformer


def _matrix(delay: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    concentrations = np.column_stack(
        [
            0.3 + 0.8 * delay,
            1.1 - 0.5 * delay,
        ]
    )
    spectra = np.column_stack(
        [
            np.exp(-((wavelength - 340.0) ** 2) / (2.0 * 24.0**2)),
            np.exp(-((wavelength - 435.0) ** 2) / (2.0 * 31.0**2)),
        ]
    )
    return concentrations @ spectra.T


def _interpolate_rows(
    matrix: np.ndarray, source_wavelength: np.ndarray, target_wavelength: np.ndarray
) -> np.ndarray:
    return np.vstack(
        [np.interp(target_wavelength, source_wavelength, row) for row in matrix]
    )


def test_group_mean_rejects_mismatched_delay_axes():
    wavelength = np.linspace(300.0, 500.0, 20)
    delay_a = np.linspace(0.0, 1.0, 5)
    delay_b = np.linspace(0.0, 1.0, 6)
    frame = pd.DataFrame(
        {
            "group": ["g", "g"],
            "spectro_matrix": [
                _matrix(delay_a, wavelength),
                _matrix(delay_b, wavelength),
            ],
            "delay_axis": [delay_a, delay_b],
            "wavelength_axis": [wavelength, wavelength],
        }
    )

    transformer = MCRALSTransformer(
        decomposition_mode="group",
        group_col="group",
        group_strategy="mean",
        n_components=2,
        maxiter=4,
        random_state=13,
    )
    with pytest.raises(ValueError, match="identical delay axes"):
        transformer.transform(frame)


def test_group_tile_delay_matches_manual_interpolation_and_slices_global_result():
    base_wavelength = np.linspace(300.0, 500.0, 24)
    shifted_wavelength = base_wavelength + 0.5
    delay_a = np.linspace(0.0, 1.0, 4)
    delay_b = np.linspace(1.2, 2.0, 5)
    matrix_a = _matrix(delay_a, base_wavelength)
    matrix_b = _matrix(delay_b, shifted_wavelength)
    matrix_b_aligned = _interpolate_rows(matrix_b, shifted_wavelength, base_wavelength)
    fixed_first = np.exp(-((base_wavelength - 340.0) ** 2) / (2.0 * 24.0**2))
    fixed_second = fixed_first[::-1]
    grouped = pd.DataFrame(
        {
            "group": ["g", "g"],
            "spectro_matrix": [matrix_a, matrix_b],
            "delay_axis": [delay_a, delay_b],
            "wavelength_axis": [base_wavelength, shifted_wavelength],
            "fixed": [fixed_first, fixed_second],
        }
    )
    common = {
        "n_components": 2,
        "fixed_spectra_col": "fixed",
        "hard_s0": True,
        "broadening": [False, True],
        "broadening_max_pct": 20.0,
        "maxiter": 5,
        "thresh": 1e-6,
        "random_state": 14,
    }
    expected_matrix = np.vstack([matrix_a, matrix_b_aligned])
    expected_delay = np.concatenate([delay_a, delay_b])
    reference = (
        MCRALSTransformer(**common)
        .transform(
            pd.DataFrame(
                {
                    "spectro_matrix": [expected_matrix],
                    "delay_axis": [expected_delay],
                    "wavelength_axis": [base_wavelength],
                    "fixed": [fixed_first],
                }
            )
        )
        .iloc[0]
    )
    grouped_output = MCRALSTransformer(
        **common,
        decomposition_mode="group",
        group_col="group",
        group_strategy="tile_delay",
        allow_group_wavelength_interpolation=True,
    ).transform(grouped)

    for row_index, start, end in (
        (0, 0, delay_a.size),
        (1, delay_a.size, expected_delay.size),
    ):
        row = grouped_output.iloc[row_index]
        np.testing.assert_array_equal(row["als_C"], reference["als_C"][start:end, :])
        np.testing.assert_array_equal(
            row["als_model"], reference["als_model"][start:end, :]
        )
        np.testing.assert_array_equal(
            row["als_residual"], reference["als_residual"][start:end, :]
        )
        np.testing.assert_array_equal(row["als_G"], reference["als_G"][start:end, :])
        np.testing.assert_array_equal(row["als_S"], reference["als_S"])
        assert row["als_lof_pct"] == reference["als_lof_pct"]
        assert row["als_rss"] == reference["als_rss"]
        assert row["als_iter"] == reference["als_iter"]
        assert row["als_converged"] == reference["als_converged"]
        assert row["als_broadening_enabled"] is True
        assert row["als_meta"] == {
            "decomposition_mode": "group",
            "group_strategy": "tile_delay",
            "group_size": 2,
            "n_components": 2,
            "n_fixed": 1,
            "init_method": "svd",
            "constraints": {
                "nonneg_c": True,
                "nonneg_s": True,
                "uni_s": False,
                "norm_s": True,
                "sum_norm": False,
                "norm_mode": "intensity",
                "smooth": 0.0,
                "close_c": False,
                "w_close_c": 0.0,
            },
            "correction_spectra": False,
            "broadening": [False, True],
            "tile_slice": (start, end),
        }

    np.testing.assert_array_equal(grouped_output.iloc[0]["als_S"][:, 0], fixed_first)
    assert not np.array_equal(grouped_output.iloc[0]["als_S"][:, 0], fixed_second)


def test_group_mean_matches_manual_interpolation_reference_and_metadata():
    base_wavelength = np.linspace(300.0, 500.0, 22)
    shifted_wavelength = base_wavelength + 0.4
    delay = np.linspace(0.0, 1.0, 6)
    matrix_a = _matrix(delay, base_wavelength)
    matrix_b = _matrix(delay, shifted_wavelength)
    mean_matrix = np.mean(
        np.stack(
            [
                matrix_a,
                _interpolate_rows(matrix_b, shifted_wavelength, base_wavelength),
            ],
            axis=0,
        ),
        axis=0,
    )
    common = {"n_components": 2, "maxiter": 7, "thresh": 1e-6, "random_state": 15}
    reference = (
        MCRALSTransformer(**common)
        .transform(
            pd.DataFrame(
                {
                    "spectro_matrix": [mean_matrix],
                    "delay_axis": [delay],
                    "wavelength_axis": [base_wavelength],
                }
            )
        )
        .iloc[0]
    )
    grouped = MCRALSTransformer(
        **common,
        decomposition_mode="group",
        group_col="group",
        group_strategy="mean",
        allow_group_wavelength_interpolation=True,
    ).transform(
        pd.DataFrame(
            {
                "group": ["g", "g"],
                "spectro_matrix": [matrix_a, matrix_b],
                "delay_axis": [delay, delay],
                "wavelength_axis": [base_wavelength, shifted_wavelength],
            }
        )
    )

    expected_meta = {
        "decomposition_mode": "group",
        "group_strategy": "mean",
        "group_size": 2,
        "n_components": 2,
        "n_fixed": 0,
        "init_method": "svd",
        "constraints": {
            "nonneg_c": True,
            "nonneg_s": True,
            "uni_s": False,
            "norm_s": True,
            "sum_norm": False,
            "norm_mode": "intensity",
            "smooth": 0.0,
            "close_c": False,
            "w_close_c": 0.0,
        },
        "correction_spectra": False,
        "broadening": False,
    }
    for _, row in grouped.iterrows():
        for column in ("als_C", "als_S", "als_model", "als_residual"):
            np.testing.assert_array_equal(row[column], reference[column])
        assert row["als_lof_pct"] == reference["als_lof_pct"]
        assert row["als_rss"] == reference["als_rss"]
        assert row["als_iter"] == reference["als_iter"]
        assert row["als_converged"] == reference["als_converged"]
        assert row["als_G"] is None
        assert row["als_broadening_enabled"] is False
        assert row["als_meta"] == expected_meta


def test_row_metadata_keys_and_values_are_exact():
    wavelength = np.linspace(300.0, 500.0, 18)
    delay = np.linspace(0.0, 1.0, 5)
    output = MCRALSTransformer(
        n_components=2,
        nonneg_s=[True, False],
        maxiter=5,
        random_state=16,
    ).transform(
        pd.DataFrame(
            {
                "spectro_matrix": [_matrix(delay, wavelength)],
                "delay_axis": [delay],
                "wavelength_axis": [wavelength],
            }
        )
    )

    meta = output.iloc[0]["als_meta"]
    assert meta == {
        "decomposition_mode": "row",
        "n_components": 2,
        "n_fixed": 0,
        "init_method": "svd",
        "constraints": {
            "nonneg_c": True,
            "nonneg_s": [True, False],
            "uni_s": False,
            "norm_s": True,
            "sum_norm": False,
            "norm_mode": "intensity",
            "smooth": 0.0,
            "close_c": False,
            "w_close_c": 0.0,
        },
        "correction_spectra": False,
        "broadening": False,
    }
    assert output.iloc[0]["als_broadening_enabled"] is False
