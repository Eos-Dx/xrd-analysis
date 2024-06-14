"""
The transformer classes are stored here
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans

from xrdanalysis.data_processing.azimuthal_integration import (
    perform_azimuthal_integration,
)
from xrdanalysis.data_processing.containers import MLClusterContainer
from xrdanalysis.data_processing.utility_functions import (
    create_mask,
    generate_poni,
    interpolate_cluster,
    normalize_scale_cluster,
    remove_outliers_by_cluster,
)


@dataclass
class AzimuthalIntegration(TransformerMixin):
    """
    Transformer class for azimuthal integration to be used in
    sklearn pipeline.

    Args:
        faulty_pixels (Tuple[int]): A tuple containing the
            coordinates of faulty pixels.
        npt (int): The number of points for azimuthal integration.
            Defaults to 256.
        integration_mode (str): The integration mode, either "1D" or "2D".
            Defaults to "1D".
        transformation_mode (str): The transformation mode, either 'dataframe'
            returns a dataframe for further analysis or 'pipeline' to use in
            sklearn pipeline.
            Defaults to 'dataframe'.
        calibration_mode (str): Mode of calibration, 'dataframe'
            is used when calibration values are columns in dataframe,
            'poni' is used when calibration is in poni file.
        poni_dir_path (str): Directory path containing where .poni files
            for the rows will be saved.
    """

    faulty_pixels: Tuple[int] = None
    npt: int = 256
    integration_mode: str = "1D"
    transformation_mode: str = "dataframe"
    calibration_mode: str = "dataframe"
    poni_dir_path: str = "data/poni"

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fit method for the transformer. Since this transformer does
        not learn from the data, the fit method does not perform
        any operations.

        Args:
            x (pandas.DataFrame): The data to fit.
            y: Ignored. Not used, present here for API consistency
                by convention.

        Returns:
            object: Returns the instance itself.
        """
        _ = x
        _ = y

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Applies azimuthal integration to each row of the
        DataFrame and adds the result as a new column.

        Args:
            X (pandas.DataFrame): The data to transform.
                Must contain 'measurement_data' and 'center' columns.

        Returns:
            pandas.DataFrame: Either a copy of the input DataFrame with an
                additional 'radial_profile_data' column containing the results
                of the azimuthal integration. 'q_range' column with q-ranges
                and optionally 'azimuthal_positions' column for angles in 2D
                integration or a dataframe with 'radial_profile_data' to be
                used in pipeline.
        """
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        x_copy = x.copy()

        # Mark the faulty pixels in the mask
        mask = create_mask(self.faulty_pixels)

        directory_path = None
        if self.calibration_mode == "poni":
            directory_path = generate_poni(x_copy, self.poni_dir_path)

        integration_results = x_copy.apply(
            lambda row: perform_azimuthal_integration(
                row,
                self.npt,
                mask,
                self.integration_mode,
                self.calibration_mode,
                poni_dir=directory_path,
            ),
            axis=1,
        )

        if self.integration_mode == "1D":
            # Extract q_range and profile arrays from the integration_results
            x_copy[["q_range", "radial_profile_data"]] = (
                integration_results.apply(lambda x: pd.Series([x[0], x[1]]))
            )
        elif self.integration_mode == "2D":
            x_copy[
                ["q_range", "radial_profile_data", "azimuthal_positions"]
            ] = integration_results.apply(
                lambda x: pd.Series([x[0], x[1], x[2]])
            )

        if self.transformation_mode == "pipeline":
            x_copy = np.asarray(x_copy["radial_profile_data"].values.tolist())

        return x_copy


COLUMNS_DEF = [
    "calibration_measurement_id",
    "study_name",
    "study_id",
    "cancer_tissue",
    "cancer_diagnosis",
    "patient_id",
    "wavelength",
    "pixel_size",
    "calibration_manual_distance",
    "calculated_distance",
    "measurement_data",
    "center",
    "ponifile",
]


class DataPreparation(TransformerMixin):
    """
    Transformer class to prepare raw DataFrame according to the standard.

    Methods:
        fit: Fits the transformer to the data. Since this transformer does
            not learn from the data, this method does not perform
            any operations.
        transform: Transforms the input DataFrame to adhere to the
            standard format.
    """

    columns = COLUMNS_DEF

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fit method for the transformer. Since this transformer does
        not learn from the data, the fit method does not perform
        any operations.

        Args:
            x (pandas.DataFrame): The data to fit.
            y: Ignored. Not used, present here for API consistency
                by convention.

        Returns:
            object: Returns the instance itself.
        """
        _ = x
        _ = y

        return self

    def transform(self, df: pd.DataFrame, no_poni=False) -> pd.DataFrame:
        """
        Transforms the input DataFrame to adhere to the standard format.

        Args:
            df (pandas.DataFrame): The raw DataFrame to be transformed.

        Returns:
            pandas.DataFrame: The transformed DataFrame with selected columns.
        """
        dfc = df.copy()

        if "center" in dfc.columns:
            dfc = dfc[~dfc["center"].isna()]

        if "calculated_distance" in dfc.columns:
            dfc = dfc[~dfc["calculated_distance"].isna()]

        if not no_poni:
            if "ponifile" in dfc.columns:
                dfc = dfc.dropna(subset="ponifile")
        else:
            if "ponifile" in self.columns:
                self.columns.remove("ponifile")

        dfc["measurement_data"] = dfc["measurement_data"].apply(
            lambda x: np.nan_to_num(x)
        )

        return dfc[self.columns]


class Clusterization(TransformerMixin):
    """
    Transformer class to perform clusterization and remove outliers
    from the DataFrame.

    Args:
        n_clusters (int): The number of clusters to use in K-Means clustering.
        z_score_threshold (float): The threshold for Z-score based outlier
            removal.
        direction (str): The direction of outlier removal, either "both",
            "positive", or "negative".

    Methods:
        fit: Fits the transformer to the data. Since this transformer does not
            learn from the data, this method does not perform any operations.
        transform: Transforms the input DataFrame by performing clusterization
                   and outlier removal.
    """

    def __init__(self, n_clusters, z_score_threshold, direction):
        """
        Initialize the Clusterization transformer with parameters.

        Args:
            n_clusters (int): The number of clusters to use in
                K-Means clustering.
            z_score_threshold (float): The threshold for Z-score
                based outlier removal.
            direction (str): The direction of outlier removal, either
                "both", "positive", or "negative".
        """
        self.n_clusters = n_clusters
        self.z_score_threshold = z_score_threshold
        self.direction = direction

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fit method for the transformer. Since this transformer does
        not learn from the data, the fit method does not perform
        any operations.

        Args:
            x (pandas.DataFrame): The data to fit.
            y: Ignored. Not used, present here for API consistency
                by convention.

        Returns:
            object: Returns the instance itself.
        """
        _ = x
        _ = y

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by performing clusterization
        and outlier removal.

        Args:
            df (pandas.DataFrame): The input DataFrame.

        Returns:
            pandas.DataFrame: The transformed DataFrame with outliers removed.
        """

        dfc = df.copy()

        dfc["q_range_min"] = dfc["q_range"].apply(lambda x: np.min(x))
        dfc["q_range_max"] = dfc["q_range"].apply(lambda x: np.max(x))

        num_clusters = self.n_clusters
        column_names = ["q_range_min", "q_range_max"]
        # Extract the q_range_min and q_range_max columns
        q_range_data = dfc[column_names].values
        # Perform K-Means clustering with the optimal number of clusters
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            algorithm="elkan",
            n_init="auto",
        )
        cluster_labels = kmeans.fit_predict(q_range_data)
        # Add the cluster labels to your DataFrame
        dfc["q_cluster_label"] = cluster_labels

        return remove_outliers_by_cluster(
            dfc,
            z_score_threshold=self.z_score_threshold,
            direction=self.direction,
            num_clusters=self.n_clusters,
        )


class InterpolatorClusters(TransformerMixin):
    """
    Transformer class for interpolating clusters of azimuthal integration data.

    Parameters:
        perc_min (float): The minimum percentage of the
            maximum q-range for interpolation.
        perc_max (float): The maximum percentage of the
            maximum q-range for interpolation.
        resolution (int): The resolution for interpolation.
        faulty_pixel_array (List): A list of faulty pixel coordinates.
        model_names (str): Names of the models.

    Methods:
        fit(x: pd.DataFrame) -> self:
            Fit the interpolator to the data.
        transform(df) -> Dict[str, MLClusterContainer]:
            Perform interpolation on the input data.
    """

    def __init__(
        self,
        perc_min: float,
        perc_max: float,
        resolution: int,
        faulty_pixel_array: List,
        model_names: str,
    ):
        self.perc_min = perc_min
        self.perc_max = perc_max
        self.q_resolution = resolution
        self.model_names = model_names
        self.faulty_pixel_array = faulty_pixel_array

    def fit(self, x: pd.DataFrame):
        """
        Fit method for the transformer. Since this transformer does
        not learn from the data, the fit method does not perform
        any operations.

        Args:
            x (pandas.DataFrame): The data to fit.
            y: Ignored. Not used, present here for API consistency
                by convention.

        Returns:
            object: Returns the instance itself.
        """
        self.x = x
        return self

    def transform(self, df) -> Dict[str, MLClusterContainer]:
        """
        Perform interpolation on the input data.

        Args:
            df (pd.DataFrame): Input DataFrame containing data to
                be interpolated.

        Returns:
            dict: Dictionary containing interpolated clusters.
        """
        dfc = df.copy()
        clusters_global = {}
        clusters = {}
        for cluster_label in dfc["q_cluster_label"].unique():
            azimuth = AzimuthalIntegration(
                faulty_pixels=self.faulty_pixel_array, npt=self.q_resolution
            )
            clusters[cluster_label] = interpolate_cluster(
                dfc, cluster_label, self.perc_min, self.perc_max, azimuth
            )

        for model_name in self.model_names:
            clusters_global[model_name] = MLClusterContainer(
                model_name, deepcopy(clusters)
            )
        return clusters_global


class NormScalerClusters(TransformerMixin):
    """
    Transformer class for normalizing and scaling clusters of azimuthal
    integration data.

    Args:
        model_names (List[str]): Names of the models.
        do_fit (bool): Whether to fit the scaler. Defaults to True.

    Methods:
        fit(x: pd.DataFrame) -> self:
            Fit the scaler to the data.
        transform(containers: Dict[str, MLClusterContainer])
        -> Dict[str, MLClusterContainer]:
            Perform normalization and scaling on the input data.
    """

    def __init__(self, model_names: List[str], do_fit=True):
        self.model_names = model_names
        self.do_fit = do_fit

    def fit(self, x):
        """
        Fit method for the transformer. Since this transformer does
        not learn from the data, the fit method does not perform
        any operations.

        Args:
            x (pandas.DataFrame): The data to fit.
            y: Ignored. Not used, present here for API consistency
                by convention.

        Returns:
            object: Returns the instance itself.
        """
        self.x = x
        return self

    def transform(
        self, containers: Dict[str, MLClusterContainer]
    ) -> Dict[str, MLClusterContainer]:
        """
        Perform normalization and scaling on the input data.

        Args:
            containers (dict): Dictionary containing MLClusterContainers.

        Returns:
            dict: Dictionary containing normalized and scaled clusters.
        """
        for container in containers.values():
            for cluster in container.clusters.values():
                normalize_scale_cluster(cluster, do_fit=self.do_fit)
        return containers
