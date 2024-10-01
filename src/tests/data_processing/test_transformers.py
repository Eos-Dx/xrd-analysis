"""Test file for transformers"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from xrdanalysis.data_processing.containers import (
    MLClusterContainer,
    ModelScale,
)
from xrdanalysis.data_processing.transformers import (
    AzimuthalIntegration,
    Clusterization,
    DataPreparation,
    InterpolatorClusters,
    NormScalerClusters,
)
from xrdanalysis.data_processing.utility_functions import is_all_none


@pytest.fixture()
def sample_dataframe():
    """Dataframe to be used in tests"""
    return pd.DataFrame(
        {
            "q_cluster_label": [0, 0, 1, 1, 2, 2, 3, 3],
            "q_range_max": [10, 12, 15, 16, 20, 22, 25, 50],
            "q_range": [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [90, 100, 110, 120],
                [91, 101, 111, 121],
                [130, 140, 150, 160],
                [131, 141, 151, 161],
                [230, 240, 250, 260],
                [231, 241, 251, 261],
            ],
            "measurement_data": [np.random.rand(256, 256) for _ in range(8)],
            "center": [(128, 128)] * 8,
            "wavelength": [0.154] * 8,
            "pixel_size": [0.1] * 8,
            "interpolation_q_range": [(1, 5)] * 8,
            "calibration_measurement_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "ponifile": [
                "file1.poni",
                "file2.poni",
                "file3.poni",
                "file4.poni",
                "file5.poni",
                "file6.poni",
                "file7.poni",
                "file8.poni",
            ],
            "center_col": [1, 2, np.nan, 4, 5, 6, 7, 8],
            "center_row": [1, np.nan, 3, 4, 5, 6, 7, 8],
            "calculated_distance": [1, 2, 3, 4, np.nan, 6, 7, 8],
            "study_name": [
                "Study A",
                "Study B",
                "Study C",
                "Study D",
                "Study E",
                "Study F",
                "Study G",
                "Study H",
            ],
            "study_id": [101, 102, 103, 104, 105, 106, 107, 108],
            "cancer_tissue": [
                "Tissue A",
                "Tissue B",
                "Tissue C",
                "Tissue D",
                "Tissue E",
                "Tissue F",
                "Tissue G",
                "Tissue H",
            ],
            "cancer_diagnosis": [
                "Diagnosis A",
                "Diagnosis B",
                "Diagnosis C",
                "Diagnosis D",
                "Diagnosis E",
                "Diagnosis F",
                "Diagnosis G",
                "Diagnosis H",
            ],
            "patient_id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "calibration_manual_distance": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
            ],
        }
    )


@patch(
    (
        "xrdanalysis.data_processing."
        "transformers.perform_azimuthal_integration"
    )
)
def test_azimuthal_integration_1d(
    mock_perform_azimuthal_integration, sample_dataframe
):
    """Test for 1D azimuthal integration through transformer"""
    mock_perform_azimuthal_integration.side_effect = lambda *args, **kwargs: (
        [1, 2, 3],
        [4, 5, 6],
        0.1,
    )

    transformer = AzimuthalIntegration(
        faulty_pixels=[(1, 1), (2, 2)],
        integration_mode="1D",
        transformation_mode="dataframe",
    )
    transformed_df = transformer.transform(sample_dataframe)

    assert "q_range" in transformed_df.columns
    assert "radial_profile_data" in transformed_df.columns
    assert transformed_df.iloc[0]["calculated_distance"] == 0.1


@patch(
    (
        "xrdanalysis.data_processing."
        "transformers.perform_azimuthal_integration"
    )
)
def test_azimuthal_integration_2d(
    mock_perform_azimuthal_integration, sample_dataframe
):
    """Test for 2D azimuthal integration through transformer"""

    mock_perform_azimuthal_integration.side_effect = lambda *args, **kwargs: (
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        0.1,
    )

    transformer = AzimuthalIntegration(
        faulty_pixels=[(1, 1), (2, 2)],
        integration_mode="2D",
        transformation_mode="dataframe",
    )
    transformed_df = transformer.transform(sample_dataframe)

    assert "q_range" in transformed_df.columns
    assert "radial_profile_data" in transformed_df.columns
    assert "azimuthal_positions" in transformed_df.columns
    assert transformed_df.iloc[0]["calculated_distance"] == 0.1


@patch(
    (
        "xrdanalysis.data_processing."
        "transformers.perform_azimuthal_integration"
    )
)
def test_azimuthal_integration_dataframe_mode(
    mock_perform_azimuthal_integration, sample_dataframe
):
    """Test for dataframe transformation mode in transformer"""

    mock_perform_azimuthal_integration.side_effect = lambda *args, **kwargs: (
        [1, 2, 3],
        [4, 5, 6],
        0.1,
    )

    transformer = AzimuthalIntegration(
        faulty_pixels=[(1, 1), (2, 2)],
        integration_mode="1D",
        transformation_mode="dataframe",
    )

    transformed_df = transformer.transform(sample_dataframe)

    assert isinstance(transformed_df, pd.DataFrame)
    assert "q_range" in transformed_df.columns
    assert "radial_profile_data" in transformed_df.columns
    assert set(transformed_df.columns).issuperset(sample_dataframe.columns)
    assert transformed_df.iloc[0]["calculated_distance"] == 0.1


@patch(
    (
        "xrdanalysis.data_processing."
        "transformers.perform_azimuthal_integration"
    )
)
def test_azimuthal_integration_pipeline_mode(
    mock_perform_azimuthal_integration, sample_dataframe
):
    """Test for pipeline transformation mode in transformer"""
    mock_perform_azimuthal_integration.side_effect = lambda *args, **kwargs: (
        [1, 2, 3],
        [4, 5, 6],
        0.1,
    )

    transformer = AzimuthalIntegration(
        faulty_pixels=[(1, 1), (2, 2)],
        integration_mode="1D",
        transformation_mode="pipeline",
    )
    transformed_data = transformer.transform(sample_dataframe)

    assert isinstance(transformed_data, pd.DataFrame)
    assert (
        transformed_data.shape[1] == 3
    )  # Assuming radial_profile_data contains 3 points as per the side_effect


@patch("xrdanalysis.data_processing.transformers.generate_poni")
@patch(
    (
        "xrdanalysis.data_processing."
        "transformers.perform_azimuthal_integration"
    )
)
def test_azimuthal_integration_poni_mode(
    mock_perform_azimuthal_integration, mock_generate_poni, sample_dataframe
):
    """Test for poni file calibration mode in transformer"""
    mock_perform_azimuthal_integration.side_effect = lambda *args, **kwargs: (
        [1, 2, 3],
        [4, 5, 6],
        0.1,
    )
    mock_generate_poni.return_value = "test_poni_directory"

    transformer = AzimuthalIntegration(
        faulty_pixels=[(1, 1), (2, 2)],
        integration_mode="1D",
        calibration_mode="poni",
    )
    transformed_df = transformer.transform(sample_dataframe)

    mock_generate_poni.assert_called_once()
    assert isinstance(transformed_df, pd.DataFrame)
    assert "q_range" in transformed_df.columns
    assert "radial_profile_data" in transformed_df.columns
    assert transformed_df.iloc[0]["calculated_distance"] == 0.1


def test_data_preparation_transformer(sample_dataframe):
    """Data preparation transformer test"""
    transformer = DataPreparation()
    df = sample_dataframe.copy()

    # Add necessary columns for testing
    transformed_df = transformer.transform(df)

    assert "center" in transformed_df.columns
    assert not transformed_df["center"].isnull().any()
    assert not transformed_df["measurement_data"].apply(is_all_none).any()


def test_clusterization_transform(sample_dataframe):
    """Clusterization transformer test"""
    # Initialize the transformer
    clusterizer = Clusterization(
        n_clusters=3, z_score_threshold=3, direction="both"
    )

    # Transform the sample DataFrame
    result_df = clusterizer.transform(sample_dataframe)

    # Check if the output is a DataFrame
    assert isinstance(result_df, pd.DataFrame)

    # Check if the output DataFrame has the same number of rows as the
    # input DataFrame
    assert len(result_df) == len(sample_dataframe)

    # Check if the output DataFrame contains the expected columns
    expected_columns = ["q_range_min", "q_range_max", "q_cluster_label"]
    assert all(col in result_df.columns for col in expected_columns)

    # Check if the cluster labels are assigned correctly
    unique_cluster_labels = result_df["q_cluster_label"].unique()
    assert len(unique_cluster_labels) == 3  # Expecting 3 clusters


class TestInterpolatorClusters(unittest.TestCase):

    def setUp(self):
        """Test class setup"""
        self.perc_min = 0.1
        self.perc_max = 0.9
        self.resolution = 100
        self.faulty_pixel_array = [(10, 10), (20, 20)]
        self.model_names = ["model1", "model2"]
        self.data = pd.DataFrame(
            {
                "q_cluster_label": [1, 1, 2, 2],
                "q_range_max": [10, 12, 15, 16],
                "measurement_data": [
                    np.random.rand(256, 256) for _ in range(4)
                ],
                "center": [(128, 128)] * 4,
                "wavelength": [0.154] * 4,
                "pixel_size": [0.1] * 4,
                "calculated_distance": [1, 2, 3, 4],
            }
        )

        self.interpolator = InterpolatorClusters(
            perc_min=self.perc_min,
            perc_max=self.perc_max,
            resolution=self.resolution,
            faulty_pixel_array=self.faulty_pixel_array,
            model_names=self.model_names,
        )

    def test_initialization(self):
        """Initialization test"""
        self.assertEqual(self.interpolator.perc_min, self.perc_min)
        self.assertEqual(self.interpolator.perc_max, self.perc_max)
        self.assertEqual(self.interpolator.q_resolution, self.resolution)
        self.assertEqual(
            self.interpolator.faulty_pixel_array, self.faulty_pixel_array
        )
        self.assertEqual(self.interpolator.model_names, self.model_names)

    @patch(
        "xrdanalysis.data_processing.transformers.interpolate_cluster",
        return_value="mock_cluster",
    )
    @patch(
        "xrdanalysis.data_processing.transformers.AzimuthalIntegration",
        return_value="mock_azimuthal_integration",
    )
    @patch("xrdanalysis.data_processing.transformers.MLClusterContainer")
    def test_transform(self, MockMLClusterContainer, *args):
        """Transform test"""
        MockMLClusterContainer.side_effect = lambda name, clusters: {
            name: clusters
        }
        expected_output = {
            "model1": {"model1": {1: "mock_cluster", 2: "mock_cluster"}},
            "model2": {"model2": {1: "mock_cluster", 2: "mock_cluster"}},
        }
        result = self.interpolator.transform(self.data)
        self.assertEqual(result, expected_output)


class TestNormScalerClusters(unittest.TestCase):

    def setUp(self):
        """Test class setup"""
        self.models = {"model1": ModelScale(), "model2": ModelScale()}
        self.do_fit = True
        self.scaler = NormScalerClusters(
            modelscales=self.models, do_fit=self.do_fit
        )
        self.containers = {
            "model1": MagicMock(spec=MLClusterContainer),
            "model2": MagicMock(spec=MLClusterContainer),
        }
        self.containers["model1"].clusters = {
            "1": "mock_cluster1",
            "2": "mock_cluster2",
        }
        self.containers["model2"].clusters = {
            "1": "mock_cluster3",
            "2": "mock_cluster4",
        }

    def test_initialization(self):
        """Initialization test"""
        self.assertEqual(self.scaler.modelscales, self.models)
        self.assertEqual(self.scaler.do_fit, self.do_fit)

    @patch("xrdanalysis.data_processing.transformers.normalize_scale_cluster")
    def test_transform(self, mock_normalize_scale_cluster):
        """Transform test"""
        result = self.scaler.transform(self.containers)
        self.assertEqual(result, self.containers)
        self.assertEqual(mock_normalize_scale_cluster.call_count, 4)
        mock_normalize_scale_cluster.assert_any_call(
            "mock_cluster1",
            normt=self.scaler.modelscales["model1"].normt,
            norm=self.scaler.modelscales["model1"].norm,
            do_fit=self.do_fit,
        )
        mock_normalize_scale_cluster.assert_any_call(
            "mock_cluster2",
            normt=self.scaler.modelscales["model1"].normt,
            norm=self.scaler.modelscales["model1"].norm,
            do_fit=self.do_fit,
        )
        mock_normalize_scale_cluster.assert_any_call(
            "mock_cluster3",
            normt=self.scaler.modelscales["model2"].normt,
            norm=self.scaler.modelscales["model2"].norm,
            do_fit=self.do_fit,
        )
        mock_normalize_scale_cluster.assert_any_call(
            "mock_cluster4",
            normt=self.scaler.modelscales["model2"].normt,
            norm=self.scaler.modelscales["model2"].norm,
            do_fit=self.do_fit,
        )
