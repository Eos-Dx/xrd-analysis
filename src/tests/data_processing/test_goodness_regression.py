"""Regression contracts for goodness scoring and filtering transformers."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from xrdanalysis.data_processing.transformers import GoodnessFilter, GoodnessTransformer


@pytest.fixture
def frozen_pattern() -> np.ndarray:
    """Deterministic pattern with zero and NaN samples for FFT regression."""
    rows, columns = np.indices((8, 10))
    pattern = (
        5.0
        + 0.7 * np.sin(2 * np.pi * rows / 3.0)
        + 0.4 * np.cos(2 * np.pi * columns / 4.0)
    )
    pattern[1, 4] = 0.0
    pattern[6, 7] = np.nan
    return pattern


def test_goodness_fft_score_and_deviation_matrix_are_frozen(frozen_pattern):
    source = pd.DataFrame({"polar_data": [frozen_pattern]})
    transformer = GoodnessTransformer(
        skip_bins=2,
        hf_cutoff_fraction=0.25,
        save_dev=True,
    )

    result = transformer.transform(source)
    deviation = result.at[0, "data_diff"]

    assert result.at[0, "goodness"] == pytest.approx(66.42918647987604)
    assert deviation.shape == frozen_pattern.shape
    assert np.isnan(deviation[:, :2]).all()
    assert np.isnan(deviation[1, 4])
    assert np.isnan(deviation[6, 7])
    np.testing.assert_allclose(
        deviation[0, 2:],
        [
            -1.62063373,
            -1.49291861,
            0.0,
            -1.49291861,
            -1.62063373,
            -1.70256158,
            -1.38386241,
            -1.49291861,
        ],
        atol=1e-8,
    )


def test_goodness_keeps_zero_nan_input_and_does_not_mutate_source(frozen_pattern):
    constant = np.ones((8, 10))
    source = pd.DataFrame({"polar_data": [constant, frozen_pattern]})
    constant_before = constant.copy()
    frozen_before = frozen_pattern.copy()

    result = GoodnessTransformer(column="polar_data", skip_bins=2).transform(source)

    assert result.at[0, "goodness"] == 0.0
    assert np.isfinite(result.at[1, "goodness"])
    np.testing.assert_equal(constant, constant_before)
    np.testing.assert_equal(frozen_pattern, frozen_before)


def test_goodness_cutoff_is_monotonic(frozen_pattern):
    source = pd.DataFrame({"polar_data": [frozen_pattern]})

    low = GoodnessTransformer(skip_bins=2, hf_cutoff_fraction=0.0).transform(source)
    high = GoodnessTransformer(skip_bins=2, hf_cutoff_fraction=0.45).transform(source)

    assert low.at[0, "goodness"] >= high.at[0, "goodness"]


def test_goodness_falls_back_to_radial_profile_with_warning(capsys, frozen_pattern):
    source = pd.DataFrame({"radial_profile_data": [frozen_pattern]})

    result = GoodnessTransformer(column="not_present", skip_bins=2).transform(source)

    assert "goodness" in result
    assert capsys.readouterr().out == (
        "Warning: Column 'not_present' not found. "
        "Using 'radial_profile_data' instead.\n"
    )


def test_goodness_prefers_polar_data_before_radial_profile(capsys, frozen_pattern):
    source = pd.DataFrame(
        {
            "polar_data": [np.ones((8, 10))],
            "radial_profile_data": [frozen_pattern],
        }
    )

    result = GoodnessTransformer(column="not_present", skip_bins=2).transform(source)

    assert result.at[0, "goodness"] == 0.0
    assert capsys.readouterr().out == (
        "Warning: Column 'not_present' not found. Using 'polar_data' instead.\n"
    )


def test_goodness_requires_a_configured_or_legacy_profile_column():
    with pytest.raises(
        KeyError,
        match="Neither 'not_present', 'polar_data', nor 'radial_profile_data' found",
    ):
        GoodnessTransformer(column="not_present").transform(pd.DataFrame({"id": [1]}))


def test_goodness_rejects_non_2d_profiles():
    source = pd.DataFrame({"polar_data": [np.array([1.0, 2.0, 3.0])]})

    with pytest.raises(ValueError, match="Column 'polar_data' must contain 2D arrays"):
        GoodnessTransformer(skip_bins=0).transform(source)


def test_goodness_saved_deviations_follow_input_order(frozen_pattern):
    constant = np.ones((8, 10))
    source = pd.DataFrame({"polar_data": [constant, frozen_pattern]}, index=[20, 10])

    result = GoodnessTransformer(skip_bins=2, save_dev=True).transform(source)

    assert list(result.index) == [20, 10]
    first, second = result["data_diff"].tolist()
    assert np.isnan(first[:, :2]).all()
    assert np.allclose(first[:, 2:], 0.0)
    assert np.isnan(second[1, 4])
    assert result.at[20, "goodness"] == 0.0
    assert result.at[10, "goodness"] > 0.0


def test_goodness_empty_frame_preserves_legacy_output_schema():
    source = pd.DataFrame({"polar_data": pd.Series(dtype=object)})

    result = GoodnessTransformer(save_dev=True).transform(source)

    assert result["goodness"].dtype == object
    assert result["data_diff"].dtype == float


@pytest.mark.parametrize(
    ("rule", "expected_index"),
    [
        (">", [20]),
        (">=", [10, 20]),
        ("<", [30]),
        ("<=", [10, 30]),
    ],
)
def test_goodness_filter_operators_thresholds_and_index_order(rule, expected_index):
    source = pd.DataFrame(
        {
            "type_measurement": ["SAXS", "WAXS", "UNKNOWN"],
            "goodness": [10.0, 11.0, 8.0],
        },
        index=[10, 20, 30],
    )
    transformer = GoodnessFilter(
        thresholds={"SAXS": 10.0, "WAXS": 10.0},
        default_threshold=9.0,
        rule=rule,
    )

    result = transformer.transform(source)

    assert list(result.index) == expected_index
    assert list(result["type_measurement"]) == [
        source.at[index, "type_measurement"] for index in expected_index
    ]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            pd.DataFrame({"type_measurement": ["SAXS"]}),
            "Goodness column 'goodness' not found",
        ),
        (pd.DataFrame({"goodness": [1.0]}), "Type column 'type_measurement' not found"),
    ],
)
def test_goodness_filter_requires_configured_columns(source, message):
    with pytest.raises(KeyError, match=message):
        GoodnessFilter().transform(source)


def test_goodness_filter_verbose_summary_is_stable(capsys):
    source = pd.DataFrame(
        {
            "type_measurement": ["SAXS", "WAXS"],
            "goodness": [12.0, 5.0],
        }
    )

    result = GoodnessFilter(
        thresholds={"SAXS": 10.0, "WAXS": 10.0}, rule=">", verbose=True
    ).transform(source)

    assert list(result.index) == [0]
    assert capsys.readouterr().out == (
        "GoodnessFilter: 2 -> 1 rows (1 removed, rule: '>')\n"
        "  SAXS: 1 -> 1 rows (goodness > 10.0)\n"
        "  WAXS: 1 -> 0 rows (goodness > 10.0)\n"
    )


@pytest.mark.parametrize(
    ("instance", "qualified_name"),
    [
        (
            GoodnessTransformer(),
            "xrdanalysis.data_processing.transformers.GoodnessTransformer",
        ),
        (GoodnessFilter(), "xrdanalysis.data_processing.transformers.GoodnessFilter"),
    ],
)
def test_goodness_classes_current_joblib_module_path_smoke(
    tmp_path, instance, qualified_name
):
    path = tmp_path / "goodness.joblib"
    joblib.dump(instance, path)
    restored = joblib.load(path)

    assert type(restored) is type(instance)
    assert (
        f"{type(restored).__module__}.{type(restored).__qualname__}" == qualified_name
    )
