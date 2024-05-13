"""
The transformer classes are stored here
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans

from xrdanalysis.data.azimuthal_integration import (
    perform_azimuthal_integration,
)
from xrdanalysis.data.containers import MLClusterContainer
from xrdanalysis.data.utility_functions import (
    create_mask,
    generate_poni,
    get_center,
    interpolate_cluster,
    is_all_none,
    is_nan_pair,
    normalize_scale_cluster,
)


@dataclass
class AzimuthalIntegration(TransformerMixin):
    """
    Transformer class for azimuthal integration to be used in
    sklearn pipeline.

    Attributes:
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
        Fit method for the transformer. Since this transformer does not learn
        from the data, the fit method does not perform any operations.

        Parameters:
        - x : pandas.DataFrame
            The data to fit.
        - y : Ignored
            Not used, present here for API consistency by convention.

        Returns:
        - self : object
            Returns the instance itself.
        """
        _ = x
        _ = y

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Applies azimuthal integration to each row of the DataFrame and adds
        the result as a new column.

        Parameters:
        - X : pandas.DataFrame
            The data to transform. Must contain 'measurement_data' and 'center'
            columns.

        Returns:
        - X_copy : pandas.DataFrame
            A copy of the input DataFrame with an additional 'profile' column
            containing the results of the azimuthal integration.
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
        not learn from the data, this method does not perform any operations.
        transform: Transforms the input DataFrame to adhere to the
        standard format.
    """

    columns = COLUMNS_DEF

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fit method for the transformer. Since this transformer does not learn
        from the data, the fit method does not perform any operations.

        Parameters:
        - x : pandas.DataFrame
            The data to fit.
        - y : Ignored
            Not used, present here for API consistency by convention.

        Returns:
        - self : object
            Returns the instance itself.
        """
        _ = x
        _ = y

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame to adhere to the standard format.

        Parameters:
        - df: pandas.DataFrame
            The raw DataFrame to be transformed.

        Returns:
        - dfc: pandas.DataFrame
            The transformed DataFrame with selected columns.
        """
        dfc = df.copy()
        if "center_col" in dfc.columns:
            dfc = dfc.dropna(subset="center_col")
            no_center_col = False
        else:
            no_center_col = True

        if "center_row" in dfc.columns:
            dfc = dfc.dropna(subset="center_row")
            no_center_row = False
        else:
            no_center_row = True

        if "calculated_distance" in dfc.columns:
            dfc = dfc = dfc.dropna(subset="calculated_distance")

        if "ponifile" in dfc.columns:
            dfc = dfc = dfc.dropna(subset="ponifile")

        # Apply this function to the 'measurement_data' column and
        # filter the DataFrame
        dfc = dfc[~dfc["measurement_data"].apply(is_all_none)]

        dfc["measurement_data"] = dfc["measurement_data"].apply(
            lambda x: np.nan_to_num(x)
        )

        if "center" not in dfc.columns:
            dfc["center"] = dfc["measurement_data"].apply(get_center)

        # Apply the function and filter out the rows where 'center' is
        # (np.NaN, np.NaN)
        dfc = dfc[~dfc["center"].apply(is_nan_pair)]

        if not no_center_col:
            self.columns.append("center_col")

        if not no_center_row:
            self.columns.append("center_row")

        return dfc[self.columns]


class Clusterization(TransformerMixin):
    """
    Transformer class to perform clusterization and remove outliers
    from the DataFrame.

    Attributes:
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

        Parameters:
        - n_clusters (int): The number of clusters to use in K-Means
            clustering.
        - z_score_threshold (float): The threshold for Z-score based outlier
            removal.
        - direction (str): The direction of outlier removal, either "both",
                           "positive", or "negative".
        """
        self.n_clusters = n_clusters
        self.z_score_threshold = z_score_threshold
        self.direction = direction

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fit method for the transformer. Since this transformer does not learn
        from the data, the fit method does not perform any operations.

        Parameters:
        - x : pandas.DataFrame
            The data to fit.
        - y : Ignored
            Not used, present here for API consistency by convention.

        Returns:
        - self : object
            Returns the instance itself.
        """
        _ = x
        _ = y

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by performing clusterization
        and outlier removal.

        Parameters:
        - df : pandas.DataFrame
            The input DataFrame.

        Returns:
        - df_filtered : pandas.DataFrame
            The transformed DataFrame with outliers removed.
        """

        def remove_outliers_by_cluster(
            dataframe, z_score_threshold, direction="negative", num_clusters=4
        ):
            # Create an empty DataFrame to store non-outlier values
            filtered_data = pd.DataFrame(columns=dataframe.columns)

            for cluster in range(num_clusters):
                cluster_indices = dataframe[
                    dataframe["q_cluster_label"] == cluster
                ].index

                # Extract q_range_max values for the current cluster
                q_range_max_values = dataframe.loc[
                    cluster_indices, "q_range_max"
                ]

                # Calculate Z-scores for q_range_max values
                z_scores = zscore(q_range_max_values)

                # Determine indices of non-outliers based on the
                # chosen direction
                if direction == "both":
                    non_outlier_indices = np.abs(z_scores) < z_score_threshold
                elif direction == "positive":
                    non_outlier_indices = z_scores < z_score_threshold
                elif direction == "negative":
                    non_outlier_indices = z_scores > -z_score_threshold
                else:
                    raise ValueError(
                        "Invalid direction. Use 'both', 'positive', \
                                      or 'negative'."
                    )

                # Add non-outlier values to the filtered_data DataFrame
                filtered_data = pd.concat(
                    [
                        filtered_data,
                        dataframe.loc[cluster_indices[non_outlier_indices]],
                    ]
                )

            return filtered_data

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
        Fit the interpolator to the data.

        Parameters:
            x (pd.DataFrame): Input DataFrame containing data
            to be interpolated.

        Returns:
            self
        """
        self.x = x
        return self

    def transform(self, df) -> Dict[str, MLClusterContainer]:
        """
        Perform interpolation on the input data.

        Parameters:
            df : DataFrame
                Input DataFrame containing data to be interpolated.

        Returns:
            Dict[str, MLClusterContainer]
                Dictionary containing interpolated clusters.
        """
        dfc = df.copy()
        clusters_global = {}
        clusters = {}
        for cluster_label in dfc["q_cluster_label"].unique():
            azimuth = AzimuthalIntegration(
                faulty_pixels=self.faulty_pixels_array, npt=self.q_resolution
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
    Transformer class for normalizing and scaling clusters of
    azimuthal integration data.

    Parameters:
        model_names (List[str]): Names of the models.
        do_fit (bool): Whether to fit the scaler. Defaults to True.

    Methods:
        fit(x) -> self:
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
        Fit the scaler to the data.

        Parameters:
            x : Input data

        Returns:
            self
        """
        self.x = x
        return self

    def transform(
        self, containers: Dict[str, MLClusterContainer]
    ) -> Dict[str, MLClusterContainer]:
        """
        Perform normalization and scaling on the input data.

        Parameters:
            containers (Dict[str, MLClusterContainer]): Dictionary
            containing MLClusterContainers.

        Returns:
            Dict[str, MLClusterContainer]: Dictionary containing normalized
            and scaled clusters.
        """
        for container in containers.values():
            for cluster in container.clusters.values():
                normalize_scale_cluster(cluster, do_fit=self.do_fit)
        return containers
