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
    assert result.at[0, "noise_std"] == pytest.approx(0.04358961135842314)
    assert result.at[0, "snr_linear"] == pytest.approx(3.16753063998172)
    assert result.at[0, "snr_db"] == pytest.approx(5.007208245690945)
    assert result.at[0, "snr"] == result.at[0, "snr_db"]

    smoothed = result.at[0, "radial_profile_data_snr"]
    residual = result.at[0, "radial_profile_residual"]
    source = profile_frame.at[0, "radial_profile_data"]
    assert smoothed.shape == source.shape
    assert residual.shape == source.shape
    np.testing.assert_allclose(
        smoothed,
        [
            2.01428571,
            2.74285714,
            3.18571429,
            3.14285714,
            3.94285714,
            4.14285714,
            5.27142857,
            5.48571429,
            5.12857143,
        ],
    )
    np.testing.assert_allclose(
        residual,
        [
            -0.01428571,
            0.25714286,
            -0.68571429,
            0.85714286,
            -0.94285714,
            0.85714286,
            -0.77142857,
            0.51428571,
            -0.12857143,
        ],
        atol=1e-8,
    )
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
    assert result.at[0, "noise_std"] == pytest.approx(
        np.sqrt(np.mean(sigma_uniform**2))
    )
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


def test_residual_mode_moving_average_fallback_has_frozen_metrics(profile_frame):
    transformer = SNRTransformer(enforce_common_q=False, window_frac=0.5)
    transformer._savgol = None

    result = transformer.transform(profile_frame)

    assert result.at[0, "noise_std"] == pytest.approx(0.03346325566315745)
    assert result.at[0, "snr_linear"] == pytest.approx(3.8250000000000006)
    assert result.at[0, "snr_db"] == pytest.approx(5.8263143948963645)
    np.testing.assert_allclose(
        result.at[0, "radial_profile_data_snr"],
        [2.6, 2.9, 2.9, 3.5, 3.8, 4.5, 4.7, 5.3, 5.2],
    )
    np.testing.assert_allclose(
        result.at[0, "radial_profile_residual"],
        [-0.6, 0.1, -0.4, 0.5, -0.8, 0.5, -0.2, 0.7, -0.2],
    )


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


@pytest.mark.parametrize("sigma", [None, np.zeros(9), np.full(9, np.nan)])
def test_poisson_mode_marks_missing_or_invalid_sigma_without_metrics(
    profile_frame, sigma
):
    if sigma is not None:
        profile_frame["radial_profile_sigma"] = [sigma]

    result = SNRTransformer(snr_method="poisson").transform(profile_frame)

    assert result.at[0, "snr_method_used"] == "poisson_missing_sigma"
    for column in ("noise_std", "snr_linear", "snr_db", "snr"):
        assert np.isnan(result.at[0, column])


@pytest.mark.parametrize("enforce_common_q", [False, True])
def test_poisson_mode_truncates_a_longer_sigma_profile(enforce_common_q):
    q = np.linspace(0.0, 5.0, 6)
    intensity = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    sigma = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, np.nan, np.inf])
    frame = pd.DataFrame(
        {
            "q_range": [q],
            "radial_profile_data": [intensity],
            "radial_profile_sigma": [sigma],
        }
    )

    result = SNRTransformer(
        snr_method="poisson", enforce_common_q=enforce_common_q
    ).transform(frame)

    expected_linear = np.sqrt(np.mean((intensity / sigma[: len(intensity)]) ** 2))
    assert result.at[0, "snr_method_used"] == "poisson"
    assert result.at[0, "snr_linear"] == pytest.approx(expected_linear)


def test_common_grid_poisson_discards_nonfinite_sigma_samples():
    q = np.linspace(0.0, 5.0, 6)
    intensity = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    sigma = np.array([np.nan, 1.0, 2.0, np.inf, 3.0, 4.0])
    frame = pd.DataFrame(
        {
            "q_range": [q],
            "radial_profile_data": [intensity],
            "radial_profile_sigma": [sigma],
        }
    )

    result = SNRTransformer(snr_method="poisson", n_points=6).transform(frame)

    sigma_uniform = np.interp(
        np.linspace(q.min(), q.max(), 6), q[[1, 2, 4, 5]], sigma[[1, 2, 4, 5]]
    )
    expected_linear = np.sqrt(np.mean((intensity / sigma_uniform) ** 2))
    assert result.at[0, "snr_method_used"] == "poisson"
    assert result.at[0, "snr_linear"] == pytest.approx(expected_linear)


def test_constant_residual_profile_keeps_machine_precision_behavior():
    frame = pd.DataFrame(
        {
            "q_range": [np.linspace(0.0, 6.0, 7)],
            "radial_profile_data": [np.full(7, 3.0)],
        }
    )

    result = SNRTransformer(snr_method="residual").transform(frame)

    assert result.at[0, "snr_method_used"] == "residual"
    assert result.at[0, "noise_std"] == pytest.approx(1.0560327403778455e-16)
    assert result.at[0, "snr_linear"] == pytest.approx(1.0592105263157896)
    assert result.at[0, "snr_db"] == pytest.approx(0.24982288087077215)
    np.testing.assert_allclose(result.at[0, "radial_profile_data_snr"], np.full(7, 3.0))


def test_short_and_nonfinite_profiles_keep_initialized_outputs():
    frame = pd.DataFrame(
        {
            "q_range": [np.array([0.0]), np.array([np.nan, np.nan])],
            "radial_profile_data": [np.array([1.0]), np.array([2.0, 3.0])],
        }
    )

    result = SNRTransformer().transform(frame)

    for index in result.index:
        assert result.at[index, "snr_method_used"] is None
        for column in ("noise_std", "snr_linear", "snr_db", "snr"):
            assert np.isnan(result.at[index, column])
        assert result.at[index, "radial_profile_data_snr"] is None
        assert result.at[index, "radial_profile_residual"] is None


@pytest.mark.parametrize(
    ("snr_method", "include_sigma", "expected_method", "expected_metrics"),
    [
        (
            "residual",
            False,
            "residual",
            (0.028297074270656763, 1.2555248618784522, 0.9882531702482028),
        ),
        (
            "poisson",
            True,
            "poisson",
            (2.0873770280289223, 1.811383755891411, 5.160209375257253),
        ),
        (
            "auto",
            True,
            "poisson",
            (2.0873770280289223, 1.811383755891411, 5.160209375257253),
        ),
    ],
)
@pytest.mark.parametrize("enforce_common_q", [False, True])
def test_stable_metric_golden_values_across_snr_modes(
    snr_method, include_sigma, expected_method, expected_metrics, enforce_common_q
):
    q = np.arange(7, dtype=float)
    intensity = np.array([2.0, 3.0, 2.5, 4.0, 3.0, 5.0, 4.5])
    frame_data = {"q_range": [q], "radial_profile_data": [intensity]}
    if include_sigma:
        frame_data["radial_profile_sigma"] = [
            np.array([1.0, 2.0, 1.5, 2.0, 2.5, 2.0, 3.0])
        ]
    transformer = SNRTransformer(
        snr_method=snr_method, enforce_common_q=enforce_common_q, window_frac=0.5
    )
    transformer._savgol = None

    result = transformer.transform(pd.DataFrame(frame_data))

    assert result.at[0, "snr_method_used"] == expected_method
    for column, expected in zip(
        ("noise_std", "snr_linear", "snr_db"), expected_metrics, strict=True
    ):
        assert result.at[0, column] == pytest.approx(expected)
