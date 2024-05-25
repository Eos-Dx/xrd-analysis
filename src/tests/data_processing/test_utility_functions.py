"""Tests for utility functions"""

import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Assuming functions are imported from the module `your_module`
from xrdanalysis.data_processing.utility_functions import (
    MLCluster,
    create_mask,
    generate_poni,
    get_center,
    interpolate_cluster,
    is_all_none,
    is_nan_pair,
    normalize_scale_cluster,
    remove_outliers_by_cluster,
)


# Test for get_center
def test_get_center():
    """Tests for get center function"""
    data = np.zeros((10, 10))
    data[5, 5] = 10  # Artificial center
    center = get_center(data, threshold=0.5)
    assert center == (5.0, 5.0)

    data[5, 5] = 0  # Zero center
    center = get_center(data, threshold=0.5)
    assert np.isnan(center[0]) and np.isnan(center[1])


# Test for generate_poni
def test_generate_poni(tmp_path):
    """Test for poni generation"""
    df = pd.DataFrame(
        {
            "calibration_measurement_id": [1, 2, 3],
            "ponifile": ["content1", "content2", "content3"],
        }
    )
    path = tmp_path / "poni_files"
    result = generate_poni(df, str(path))
    assert os.path.exists(result)
    for i in range(1, 4):
        assert os.path.isfile(os.path.join(result, f"{i}.poni"))


# Test for create_mask
def test_create_mask():
    """Test for mask creation"""
    faulty_pixels = [(0, 0), (1, 1)]
    mask = create_mask(faulty_pixels)
    assert mask[0, 0] == 1
    assert mask[1, 1] == 1
    assert mask[2, 2] == 0

    mask = create_mask(None)
    assert mask is None


# Test for is_all_none
def test_is_all_none():
    """Tests to check for all None"""
    assert is_all_none([None, None, None]) is True
    assert is_all_none([None, 1, None]) is False


# Test for is_nan_pair
def test_is_nan_pair():
    """Tests to check is nan pair"""
    assert is_nan_pair((np.NaN, np.NaN)) is True
    assert is_nan_pair((np.nan, 1)) is False
    assert is_nan_pair((1, np.nan)) is False
    assert is_nan_pair((1, 1)) is False
    assert is_nan_pair(np.nan) is False


# Test for interpolate_cluster
def test_interpolate_cluster():
    """Test to check cluster interpolation"""
    df = pd.DataFrame(
        {"q_cluster_label": [0, 0, 1, 1], "q_range_max": [10, 20, 30, 40]}
    )
    cluster_label = 0
    perc_min = 0.1
    perc_max = 0.9
    azimuth = MagicMock()
    result = interpolate_cluster(
        df, cluster_label, perc_min, perc_max, azimuth
    )

    args, _ = azimuth.transform.call_args

    assert args[0]["interpolation_q_range"][0][0] == 1
    assert args[0]["interpolation_q_range"][0][1] == 9
    assert isinstance(result, MLCluster)
    assert result.q_cluster == cluster_label


# Test for normalize_scale_cluster
def test_normalize_scale_cluster():
    """Tests to check cluster scaling and normalization results"""
    df = pd.DataFrame(
        {"radial_profile_data": [np.array([1, 2, 3]), np.array([4, 5, 6])]}
    )
    cluster = MLCluster(df=df, q_cluster=0, q_range=None)
    normalize_scale_cluster(cluster)
    assert "radial_profile_data_norm" in cluster.df.columns
    assert "radial_profile_data_norm_scaled" in cluster.df.columns


# Test for remove_outliers_by_cluster
def test_remove_outliers_by_cluster_no_outliers():
    """Test to check if no outlier case is possible"""
    df = pd.DataFrame(
        {
            "q_cluster_label": [0, 0, 1, 1, 2, 2, 3, 3],
            "q_range_max": [10, 12, 15, 16, 20, 22, 25, 26],
        }
    )
    z_score_threshold = 2.0
    direction = "both"
    num_clusters = 4

    filtered_df = remove_outliers_by_cluster(
        df, z_score_threshold, direction, num_clusters
    )
    assert len(filtered_df) == len(df)


def test_remove_outliers_by_cluster_positive_direction():
    """Test to check if positive direction outliers are removed"""

    df = pd.DataFrame(
        {
            "q_cluster_label": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "q_range_max": [
                10,
                12,
                50,
                15,
                16,
                50,
                20,
                21,
                50,
                25,
                24,
                50,
            ],  # 50 is an outlier
        }
    )
    z_score_threshold = 1
    direction = "positive"
    num_clusters = 4

    filtered_df = remove_outliers_by_cluster(
        df, z_score_threshold, direction, num_clusters
    )

    print(filtered_df)
    assert len(filtered_df) == len(df) - 4
    assert 50 not in filtered_df["q_range_max"].values


def test_remove_outliers_by_cluster_negative_direction():
    """Test to check if negative direction outliers are removed"""

    df = pd.DataFrame(
        {
            "q_cluster_label": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "q_range_max": [
                10,
                12,
                -50,
                15,
                16,
                -50,
                20,
                21,
                -50,
                25,
                24,
                -50,
            ],  # -50 is an outlier
        }
    )
    z_score_threshold = 1
    direction = "negative"
    num_clusters = 4

    filtered_df = remove_outliers_by_cluster(
        df, z_score_threshold, direction, num_clusters
    )

    print(filtered_df)
    assert len(filtered_df) == len(df) - 4
    assert -50 not in filtered_df["q_range_max"].values


def test_remove_outliers_by_cluster_both_directions():
    """Test to check if both directions outliers are removed"""

    df = pd.DataFrame(
        {
            "q_cluster_label": [
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
            ],
            "q_range_max": [
                80,
                10,
                12,
                -80,
                80,
                15,
                16,
                -80,
                80,
                20,
                21,
                -80,
                80,
                25,
                24,
                -80,
            ],  # -80 is an outlier, 80 is an outlier
        }
    )
    z_score_threshold = 1
    direction = "both"
    num_clusters = 4

    filtered_df = remove_outliers_by_cluster(
        df, z_score_threshold, direction, num_clusters
    )

    assert len(filtered_df) == len(df) - 8
    assert -80 not in filtered_df["q_range_max"].values
    assert 80 not in filtered_df["q_range_max"].values


def test_remove_outliers_by_cluster_invalid_direction():
    """Test to check if invalid input is handled"""

    df = pd.DataFrame(
        {
            "q_cluster_label": [0, 0, 1, 1, 2, 2, 3, 3],
            "q_range_max": [10, 12, 15, 16, 20, 22, 25, 26],
        }
    )
    z_score_threshold = 1.0
    direction = "invalid"
    num_clusters = 4

    with pytest.raises(
        ValueError,
        match="Invalid direction. Use 'both', 'positive', or 'negative'.",
    ):
        remove_outliers_by_cluster(
            df, z_score_threshold, direction, num_clusters
        )
