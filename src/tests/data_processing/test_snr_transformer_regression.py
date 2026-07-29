"""Numerical regression coverage for SNRTransformer helper extraction."""

import numpy as np
import pandas as pd
import pytest

from xrdanalysis.data_processing.transformers import SNRTransformer


@pytest.fixture
def profile_frame():
    q = np.linspace(0.0, 4.0, 9)
    intensity = np.array([2.0, 3.0, 2.5, 4.0, 3.0, 5.0, 4.5, 6.0, 5.0])
    return pd.DataFrame({"q_range": [q], "radial_profile_data": [intensity]})


def test_residual_mode_writes_expected_columns_and_original_sampling(profile_frame):
    transformer = SNRTransformer(enforce_common_q=False, window_frac=0.5)

    result = transformer.fit_transform(profile_frame)

    expected_columns = {
        "noise_std",
        "snr_linear",
        "snr_db",
        "snr",
        "snr_method_used",
        "radial_profile_data_snr",
        "radial_profile_residual",
    }
    assert expected_columns <= set(result.columns)
    assert result.at[0, "snr_method_used"] == "residual"
    assert np.isfinite(result.at[0, "noise_std"])
    assert np.isfinite(result.at[0, "snr_linear"])
    assert result.at[0, "snr"] == result.at[0, "snr_db"]

    smoothed = result.at[0, "radial_profile_data_snr"]
    residual = result.at[0, "radial_profile_residual"]
    source = profile_frame.at[0, "radial_profile_data"]
    assert smoothed.shape == source.shape
    assert residual.shape == source.shape
    np.testing.assert_allclose(residual, source - smoothed)


def test_poisson_mode_interpolates_to_common_grid_and_uses_sigma_rms():
    q = np.array([0.0, 0.3, 1.1, 2.0, 4.0])
    intensity = np.array([4.0, 5.0, 8.0, 10.0, 14.0])
    sigma = np.array([1.0, 2.0, 2.0, 4.0, 7.0])
    frame = pd.DataFrame(
        {
            "q_range": [q],
            "radial_profile_data": [intensity],
            "radial_profile_sigma": [sigma],
        }
    )
    transformer = SNRTransformer(
        snr_method="poisson", enforce_common_q=True, n_points=7, save_smoothed=True
    )

    result = transformer.transform(frame)

    q_uniform = np.linspace(q.min(), q.max(), 7)
    intensity_uniform = np.interp(q_uniform, q, intensity)
    sigma_uniform = np.interp(q_uniform, q, sigma)
    expected_linear = np.sqrt(np.mean((intensity_uniform / sigma_uniform) ** 2))
    assert result.at[0, "snr_method_used"] == "poisson"
    assert result.at[0, "snr_linear"] == pytest.approx(expected_linear)
    assert result.at[0, "snr_db"] == pytest.approx(20.0 * np.log10(expected_linear))
    assert result.at[0, "radial_profile_data_snr"].shape == intensity.shape
    assert result.at[0, "radial_profile_residual"].shape == intensity.shape


@pytest.mark.parametrize("sigma", [None, np.zeros(9)])
def test_auto_mode_falls_back_to_residual_when_sigma_is_missing_or_invalid(
    profile_frame, sigma
):
    if sigma is not None:
        profile_frame["radial_profile_sigma"] = [sigma]

    result = SNRTransformer(snr_method="auto").transform(profile_frame)

    assert result.at[0, "snr_method_used"] == "residual"
    assert np.isfinite(result.at[0, "snr_linear"])


def test_normalization_uses_median_when_surface_is_zero():
    transformer = SNRTransformer()
    q = np.array([0.0, 1.0, 2.0])
    intensity = np.array([2.0, -2.0, 2.0])

    normalized, details = transformer._normalize_by_surface(q, intensity)

    assert details == {"norm": "median", "scale": 2.0, "area": 0.0}
    np.testing.assert_allclose(normalized, [1.0, -1.0, 1.0])


def test_smoothing_fallback_and_short_profile_identity():
    transformer = SNRTransformer(window_frac=0.5)
    transformer._savgol = None

    smoothed, details = transformer._smooth(np.arange(7, dtype=float))
    short_smoothed, short_details = transformer._smooth(np.arange(4, dtype=float))

    assert details["method"] == "movavg"
    assert smoothed.shape == (7,)
    assert short_details["method"] == "identity"
    np.testing.assert_array_equal(short_smoothed, np.arange(4, dtype=float))


def test_custom_output_columns_and_save_smoothed_false(profile_frame):
    transformer = SNRTransformer(
        snr_col="quality_snr",
        smoothed_col="profile_smooth",
        residual_col="profile_residual",
        save_smoothed=False,
    )

    result = transformer.transform(profile_frame)

    assert "quality_snr" in result.columns
    assert "profile_smooth" not in result.columns
    assert "profile_residual" not in result.columns
