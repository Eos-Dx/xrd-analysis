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
from sklearn.preprocessing import Normalizer, StandardScaler

from xrdanalysis.data_processing.azimuthal_integration import (
    perform_azimuthal_integration,
)
from xrdanalysis.data_processing.containers import MLClusterContainer, ModelScale, Limits, Rule
from xrdanalysis.data_processing.utility_functions import (
    create_mask,
    generate_poni,
    get_center,
    interpolate_cluster,
    is_all_none,
    is_nan_pair,
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
            x_copy[["q_range", "radial_profile_data", "calculated_distance"]] = (
                integration_results.apply(lambda x: pd.Series([x[0], x[1], x[2]]))
            )
        elif self.integration_mode == "2D":
            x_copy[
                ["q_range", "radial_profile_data", "azimuthal_positions", "calculated_distance"]
            ] = integration_results.apply(
                lambda x: pd.Series([x[0], x[1], x[2], x[3]])
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

    def __init__(self, columns=COLUMNS_DEF,
                 limits: Limits = None,
                 cleaning_rules: List[Rule] = None):
        self.columns = columns
        self.limits = limits
        self.cleaning_rules = cleaning_rules

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

        if not no_poni:
            if "ponifile" in dfc.columns:
                dfc = dfc.dropna(subset="ponifile")
        else:
            if "ponifile" in self.columns:
                self.columns.remove("ponifile")
            if "calculated_distance" in dfc.columns:
                dfc = dfc[~dfc["calculated_distance"].isna()]

        if 'age' in dfc.columns:
            dfc['age'] = df['age'].fillna(-1)

        if "measurement_data" in dfc.columns:
            dfc["measurement_data"] = dfc["measurement_data"].apply(
                lambda x: np.nan_to_num(x)
            )

        if 'calculated_distance' in dfc.columns:
            dfc['type_measurement'] = dfc['calculated_distance'].apply(lambda d: 'WAXS' if d < 0.05 else 'SAXS')

        if self.limits:
            limits_waxs = (self.limits.q_min_waxs, self.limits.q_max_waxs)
            limits_saxs = (self.limits.q_min_saxs, self.limits.q_max_saxs)
            if 'type_measurement' not in dfc.columns:
                dfc['type_measurement'] = dfc['calibration_manual_distance'].apply(lambda d: 'WAXS' if d < 50 else 'SAXS')
            dfc['interpolation_q_range'] = dfc['type_measurement'].apply(lambda x:
                                                                         limits_waxs if x == 'WAXS' else limits_saxs)
        if self.cleaning_rules:
            dfc = self._clean(dfc)
        return dfc[self.columns]

    def _clean(self, df: pd.DataFrame):
        def clean(row, cleaning_rules: List[Rule]):
            res = []
            for r in cleaning_rules:
                r: Rule = r
                ret = False
                idx = np.argmin(np.abs(row['q_range'] - r.q_value))
                intensity = row['radial_profile_data'][idx]
                if r.lower is not None and r.upper is not None:
                    ret = (intensity > r.lower) and (intensity < r.upper)
                elif r.lower is not None:
                    ret = intensity > r.lower
                elif r.upper is not None:
                    ret = intensity < r.upper
                res.append(ret)
            return all(res)

        if self.cleaning_rules:
            return df[df.apply(clean, axis=1, cleaning_rules=self.cleaning_rules)].copy()
        else:
            print('No cleaning was done, thera are no cleaning_rules')
            return df


class NormScaler(TransformerMixin):
    """
    Does normalization and scaling of the dataframe
    """
    def __init__(self, scalers: Dict[str, StandardScaler] = None, name='Scaler'):
        self._name = name
        if scalers:
            self.scalers = scalers
        else:
            self.scalers = {}

    def fit(self, df: pd.DataFrame, y=None):
        print(f'NormScaler {self._name}: is doing fit.')
        dfc = df.copy()
        norm = Normalizer('l1')
        dfc["radial_profile_data_norm"] = dfc["radial_profile_data"].apply(
            lambda x: norm.transform([x])[0])

        df_saxs = dfc[dfc['type_measurement'] == 'SAXS'].copy()
        df_waxs = dfc[dfc['type_measurement'] == 'WAXS'].copy()

        if not df_saxs.empty:
            scaler_saxs = StandardScaler()
            matrix_2d_saxs = np.vstack(df_saxs["radial_profile_data_norm"].values)
            scaler_saxs.fit(matrix_2d_saxs)
            self.scalers['SAXS'] = scaler_saxs

        # Apply the scaler for WAXS data
        if not df_waxs.empty:
            scaler_waxs = StandardScaler()
            matrix_2d_waxs = np.vstack(df_waxs["radial_profile_data_norm"].values)
            scaler_waxs.fit(matrix_2d_waxs)
            self.scalers['WAXS'] = scaler_waxs

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f'NormScaler {self._name}: is doing transform.')
        if not self.scalers:
            self.fit(df)
        dfc = df.copy()
        raw_index = dfc.index
        norm = Normalizer('l1')
        dfc["radial_profile_data_norm"] = dfc["radial_profile_data"].apply(
            lambda x: norm.transform([x])[0])

        df_saxs = dfc[dfc['type_measurement'] == 'SAXS'].copy()
        df_waxs = dfc[dfc['type_measurement'] == 'WAXS'].copy()

        # Apply the scaler for SAXS data
        if not df_saxs.empty:
            matrix_2d_saxs = np.vstack(df_saxs["radial_profile_data_norm"].values)
            scaled_data_saxs = self.scalers['SAXS'].transform(matrix_2d_saxs)
            df_saxs["radial_profile_data_norm_scaled"] = [arr for arr in scaled_data_saxs]

        # Apply the scaler for WAXS data
        if not df_waxs.empty:
            matrix_2d_waxs = np.vstack(df_waxs["radial_profile_data_norm"].values)
            scaled_data_waxs = self.scalers['WAXS'].transform(matrix_2d_waxs)
            df_waxs["radial_profile_data_norm_scaled"] = [arr for arr in scaled_data_waxs]

        # Combine the processed DataFrames back into one
        dfc_processed = pd.concat([df_saxs, df_waxs])
        return dfc_processed.loc[raw_index]


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
                faulty_pixels=self.faulty_pixel_array, npt=self.q_resolution, calibration_mode='poni'
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
        modelscales (List[str]): Names of the models.
        do_fit (bool): Whether to fit the scaler. Defaults to True.

    Methods:
        fit(x: pd.DataFrame) -> self:
            Fit the scaler to the data.
        transform(containers: Dict[str, MLClusterContainer])
        -> Dict[str, MLClusterContainer]:
            Perform normalization and scaling on the input data.
    """

    def __init__(self, modelscales: Dict[str, ModelScale], do_fit=True):
        self.modelscales = modelscales
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
        for name, container in containers.items():
            for cluster in container.clusters.values():

                normalize_scale_cluster(cluster, normt=self.modelscales[name].normt,
                                        norm=self.modelscales[name].norm,
                                        do_fit=self.do_fit)
        return containers
