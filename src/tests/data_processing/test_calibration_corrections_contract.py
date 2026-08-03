"""Contracts for calibration-profile computation and application."""

import inspect
import pickle

import numpy as np
import pandas as pd
import pytest

from xrdanalysis import data_processing
from xrdanalysis.data_processing import calibration_corrections as calibration


def _polar_map():
    return np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])


def test_public_functions_keep_facade_identity_signature_and_pickle_path():
    expected_parameters = {
        "compute_calib_correction_profiles": [
            "df_calib",
            "id_col",
            "calib_type_col",
            "type_col",
            "q_col",
            "polar_col",
            "out_profile_col",
            "targets",
            "inplace",
            "modify",
            "window_size",
        ],
        "correct_with_calib_profiles": [
            "df",
            "df_calib",
            "df_calib_id_col",
            "df_calib_type_col",
            "df_calib_profile_col",
            "df_id_lookup_col",
            "polar_col",
            "radial_out_col",
            "strict",
            "inplace",
            "keep",
            "raw_polar_col",
            "raw_radial_col",
        ],
    }

    for name, parameters in expected_parameters.items():
        function = getattr(calibration, name)
        assert getattr(data_processing, name) is function
        assert function.__module__ == calibration.__name__
        assert list(inspect.signature(function).parameters) == parameters
        assert pickle.loads(pickle.dumps(function)) is function


def test_private_helper_imports_preserve_numerical_and_error_contracts():
    polar = np.array([[1.0, 2.0, 9.0], [1.0, 4.0, 3.0], [1.0, 0.0, 6.0]])

    assert calibration._closest_index([0.5, 1.0, 2.0], 1.2) == 1
    with pytest.raises(ValueError, match="1D non-empty"):
        calibration._closest_index([], 1.0)

    np.testing.assert_allclose(
        calibration._per_angle_correction_from_column(polar, 1),
        [1.5, 0.75, 1.0],
    )
    with pytest.raises(IndexError, match="out of bounds"):
        calibration._per_angle_correction_from_column(polar, 3)

    expected = np.mean(
        [
            calibration._per_angle_correction_from_column(polar, index)
            for index in range(3)
        ],
        axis=0,
    )
    np.testing.assert_allclose(
        calibration._per_angle_correction_from_multi_columns(
            polar, center_col_idx=1, window_size=2
        ),
        expected,
    )


def test_apply_profile_accepts_angle_or_q_orientation_and_rejects_bad_shapes():
    polar = _polar_map()
    np.testing.assert_allclose(
        calibration._apply_profile_to_polar(polar, np.array([2.0, 0.5])),
        np.array([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]]),
    )
    np.testing.assert_allclose(
        calibration._apply_profile_to_polar(polar, np.array([1.0, 2.0, 3.0])),
        np.array([[1.0, 4.0, 9.0], [2.0, 8.0, 18.0]]),
    )
    with pytest.raises(ValueError, match="polar_data must be 2D"):
        calibration._apply_profile_to_polar(np.array([1.0, 2.0]), [1.0, 2.0])
    with pytest.raises(ValueError, match="does not match"):
        calibration._apply_profile_to_polar(polar, [1.0, 2.0, 3.0, 4.0])


def test_compute_profiles_selects_agbh_and_can_leave_polar_data_unchanged():
    agbh_polar = _polar_map()
    other_polar = _polar_map() * 10.0
    frame = pd.DataFrame(
        {
            "id": ["agbh", "water"],
            "calib_calibrationType": ["AGBH", "WATER"],
            "type_measurement": ["SAXS", "SAXS"],
            "q_range": [np.array([1.0, 2.0, 3.0])] * 2,
            "polar_data": [agbh_polar, other_polar],
        }
    )

    result = calibration.compute_calib_correction_profiles(
        frame, modify=False, window_size=0
    )

    assert result is not frame
    np.testing.assert_allclose(result.at[0, "correction_profile"], [1.5, 0.75])
    assert result.at[1, "correction_profile"] is None
    np.testing.assert_array_equal(result.at[0, "polar_data"], agbh_polar)
    assert "polar_data_raw" not in result


def test_compute_profiles_modify_and_inplace_contracts():
    original = _polar_map()
    frame = pd.DataFrame(
        {
            "id": ["agbh"],
            "calib_calibrationType": ["agbh"],
            "type_measurement": ["SAXS"],
            "q_range": [np.array([1.0, 2.0, 3.0])],
            "polar_data": [original.copy()],
        }
    )

    result = calibration.compute_calib_correction_profiles(
        frame, inplace=True, modify=True, window_size=0
    )

    assert result is frame
    np.testing.assert_array_equal(result.at[0, "polar_data_raw"], original)
    np.testing.assert_allclose(
        result.at[0, "polar_data"],
        np.array([[1.5, 3.0, 4.5], [1.5, 3.0, 4.5]]),
    )


def test_correct_profiles_updates_polar_and_radial_with_optional_backups():
    original = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 8.0]])
    frame = pd.DataFrame(
        {
            "calib_name": ["calib-1"],
            "polar_data": [original.copy()],
            "radial_profile_data": [np.array([2.5, 2.0, 8.0])],
        }
    )
    calibrations = pd.DataFrame(
        {
            "id": ["calib-1", "ignored"],
            "calib_calibrationType": ["AGBH", "WATER"],
            "correction_profile": [np.array([2.0, 0.5]), np.array([100.0, 100.0])],
        }
    )

    result = calibration.correct_with_calib_profiles(frame, calibrations, keep=True)

    assert result is not frame
    expected_polar = np.array([[2.0, 4.0, 0.0], [2.0, 0.0, 4.0]])
    np.testing.assert_allclose(result.at[0, "polar_data"], expected_polar)
    np.testing.assert_allclose(result.at[0, "radial_profile_data"], [2.0, 4.0, 4.0])
    np.testing.assert_array_equal(result.at[0, "polar_data_raw"], original)
    np.testing.assert_array_equal(
        result.at[0, "radial_profile_data_raw"], [2.5, 2.0, 8.0]
    )
    np.testing.assert_array_equal(frame.at[0, "polar_data"], original)


def test_correct_profiles_missing_lookup_is_tolerant_or_strict():
    frame = pd.DataFrame({"calib_name": ["missing"], "polar_data": [_polar_map()]})
    calibrations = pd.DataFrame(
        {
            "id": ["calib-1"],
            "calib_calibrationType": ["AGBH"],
            "correction_profile": [np.array([1.0, 1.0])],
        }
    )

    tolerant = calibration.correct_with_calib_profiles(frame, calibrations)
    np.testing.assert_array_equal(tolerant.at[0, "polar_data"], _polar_map())
    with pytest.raises(KeyError, match="missing"):
        calibration.correct_with_calib_profiles(frame, calibrations, strict=True)


def test_correct_profiles_propagates_profile_dimension_errors():
    frame = pd.DataFrame({"calib_name": ["calib-1"], "polar_data": [_polar_map()]})
    calibrations = pd.DataFrame(
        {
            "id": ["calib-1"],
            "calib_calibrationType": ["AGBH"],
            "correction_profile": [np.ones(4)],
        }
    )

    with pytest.raises(ValueError, match="does not match"):
        calibration.correct_with_calib_profiles(frame, calibrations)
