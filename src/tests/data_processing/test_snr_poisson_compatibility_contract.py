"""Compatibility contract for native and legacy-regridded Poisson SNR."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone

from xrdanalysis.data_processing.transformers import SNRTransformer


def _native_poisson_reference(
    intensity: np.ndarray, sigma: np.ndarray
) -> tuple[float, float, float]:
    """Independent RMS Poisson reference on native aligned samples."""
    valid = np.isfinite(intensity) & np.isfinite(sigma) & (sigma > 0)
    assert np.count_nonzero(valid) >= 2
    snr_q = np.abs(intensity[valid]) / (sigma[valid] + 1e-12)
    snr_linear = float(np.sqrt(np.mean(np.square(snr_q))))
    noise_std = float(np.sqrt(np.mean(np.square(sigma[valid]))))
    snr_db = float(20.0 * np.log10(snr_linear + 1e-12))
    return noise_std, snr_linear, snr_db


def _legacy_regridded_poisson_reference(
    q: np.ndarray, intensity: np.ndarray, sigma: np.ndarray, n_points: int
) -> tuple[float, float, float]:
    """Independent reference for the historical common-grid Poisson metric."""
    uniform_q = np.linspace(q.min(), q.max(), n_points)
    return _native_poisson_reference(
        np.interp(uniform_q, q, intensity),
        np.interp(uniform_q, q, sigma),
    )


@pytest.fixture
def aligned_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Profile with q-dependent intensity and sigma for spacing-sensitive checks."""
    q = 0.1 + 3.9 * np.linspace(0.0, 1.0, 9) ** 1.7
    intensity = 2.0 + np.sin(2.0 * q) + 0.05 * np.cos(9.0 * q)
    sigma = 0.2 + 0.02 * np.cos(3.0 * q)
    return q, intensity, sigma


def _profile_frame(
    q: np.ndarray, intensity: np.ndarray, sigma: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "q_range": [q],
            "radial_profile_data": [intensity],
            "radial_profile_sigma": [sigma],
        }
    )


@pytest.mark.parametrize(
    "q_variant", ["uniform", "nonuniform", "descending", "unsorted", "nan_q"]
)
def test_default_poisson_uses_native_samples_independent_of_q_values(
    aligned_profile, q_variant
):
    """Default Poisson metrics must match preprocessing's native-pair formula."""
    q, intensity, sigma = aligned_profile
    if q_variant == "uniform":
        q_input = np.linspace(q.min(), q.max(), len(q))
    elif q_variant == "nonuniform":
        q_input = q
    elif q_variant == "descending":
        q_input, intensity, sigma = q[::-1], intensity[::-1], sigma[::-1]
    elif q_variant == "unsorted":
        permutation = np.array([4, 1, 7, 0, 8, 3, 6, 2, 5])
        q_input, intensity, sigma = (
            q[permutation],
            intensity[permutation],
            sigma[permutation],
        )
    else:
        q_input = q.copy()
        q_input[[1, 6]] = np.nan

    expected = _native_poisson_reference(intensity, sigma)
    result = SNRTransformer(snr_method="poisson").transform(
        _profile_frame(q_input, intensity, sigma)
    )

    assert result.at[0, "snr_method_used"] == "poisson"
    np.testing.assert_allclose(
        result.loc[0, ["noise_std", "snr_linear", "snr_db"]].to_numpy(dtype=float),
        expected,
        rtol=0,
        atol=1e-12,
    )


def test_native_poisson_preserves_analysis_output_schema_and_smoothed_outputs(
    aligned_profile,
):
    """Native metrics retain xrd-analysis's documented output-schema superset."""
    q, intensity, sigma = aligned_profile
    result = SNRTransformer(snr_method="poisson").transform(
        _profile_frame(q, intensity, sigma)
    )

    assert {
        "noise_std",
        "snr_linear",
        "snr_db",
        "snr",
        "snr_method_used",
        "radial_profile_data_snr",
        "radial_profile_residual",
    } <= set(result.columns)
    assert result.at[0, "snr"] == result.at[0, "snr_db"]
    assert result.at[0, "radial_profile_data_snr"].shape == intensity.shape
    assert result.at[0, "radial_profile_residual"].shape == intensity.shape


@pytest.mark.parametrize("sigma", [None, np.zeros(9), np.full(9, np.nan)])
def test_poisson_keeps_tolerant_missing_and_invalid_sigma_contract(
    aligned_profile, sigma
):
    """Malformed per-row sigma remains an annotated, non-fatal result."""
    q, intensity, _ = aligned_profile
    frame = pd.DataFrame({"q_range": [q], "radial_profile_data": [intensity]})
    if sigma is not None:
        frame["radial_profile_sigma"] = [sigma]

    result = SNRTransformer(snr_method="poisson").transform(frame)

    assert result.at[0, "snr_method_used"] == "poisson_missing_sigma"
    assert np.isnan(result.loc[0, ["noise_std", "snr_linear", "snr_db", "snr"]]).all()


def test_residual_and_auto_modes_remain_available(aligned_profile):
    """Preprocessing parity must not remove analysis-only residual and auto modes."""
    q, intensity, sigma = aligned_profile
    frame = _profile_frame(q, intensity, sigma)

    residual = SNRTransformer(snr_method="residual").transform(frame)
    auto = SNRTransformer(snr_method="auto").transform(frame)

    assert residual.at[0, "snr_method_used"] == "residual"
    assert np.isfinite(residual.at[0, "snr_db"])
    assert auto.at[0, "snr_method_used"] == "poisson"


def test_regrid_poisson_opt_in_preserves_legacy_common_grid_metric(aligned_profile):
    """Explicit opt-in retains the historical interpolation-weighted Poisson value."""
    q, intensity, sigma = aligned_profile
    expected = _legacy_regridded_poisson_reference(q, intensity, sigma, n_points=7)

    result = SNRTransformer(
        snr_method="poisson", regrid_poisson=True, n_points=7
    ).transform(_profile_frame(q, intensity, sigma))

    np.testing.assert_allclose(
        result.loc[0, ["noise_std", "snr_linear", "snr_db"]].to_numpy(dtype=float),
        expected,
        rtol=0,
        atol=1e-12,
    )


def test_joblib_instance_without_regrid_attribute_uses_legacy_regridding(
    aligned_profile, tmp_path
):
    """Old serialized transformers lack the new opt-in flag and keep old results."""
    q, intensity, sigma = aligned_profile
    expected = _legacy_regridded_poisson_reference(q, intensity, sigma, n_points=7)
    legacy = SNRTransformer(snr_method="poisson", n_points=7)
    legacy.__dict__.pop("regrid_poisson", None)
    path = tmp_path / "legacy_snr_transformer.joblib"
    joblib.dump(legacy, path)

    restored = joblib.load(path)
    assert restored.regrid_poisson is True
    assert restored.get_params(deep=False)["regrid_poisson"] is True
    assert clone(restored).regrid_poisson is True
    result = restored.transform(_profile_frame(q, intensity, sigma))

    np.testing.assert_allclose(
        result.loc[0, ["noise_std", "snr_linear", "snr_db"]].to_numpy(dtype=float),
        expected,
        rtol=0,
        atol=1e-12,
    )


def test_snr_transformer_is_sklearn_cloneable_base_estimator():
    """The public transformer must expose sklearn's standard estimator contract."""
    transformer = SNRTransformer(
        snr_method="poisson",
        regrid_poisson=True,
        n_points=7,
        save_smoothed=False,
    )

    assert isinstance(transformer, BaseEstimator)
    cloned = clone(transformer)
    assert type(cloned) is SNRTransformer
    assert cloned.get_params(deep=False)["regrid_poisson"] is True
