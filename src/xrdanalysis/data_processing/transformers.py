"""
The transformer classes are stored here
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyhank import HankelTransform
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Normalizer, StandardScaler

from xrdanalysis.data_processing._goodness import (
    filter_goodness_dataframe,
    transform_goodness_dataframe,
)
from xrdanalysis.data_processing._snr_math import (
    calculate_snr_row,
    ensure_uniform_grid,
    normalize_by_surface,
    smooth_signal,
)
from xrdanalysis.data_processing.azimuthal_integration import (
    calculate_deviation,
    calculate_deviation_cake,
    perform_azimuthal_integration,
)
from xrdanalysis.data_processing.containers import (
    Limits,
    MLClusterContainer,
    ModelScale,
    Rule,
    RuleQ,
)
from xrdanalysis.data_processing.fourier import (
    fourier_custom,
    fourier_fft,
    fourier_fft2,
    slope_removal,
    slope_removal_custom,
)
from xrdanalysis.data_processing.utility_functions import (
    create_mask,
    cut_common_region,
    filter_points_by_distance,
    find_common_region,
    generate_poni,
    interpolate_cluster,
    normalize_scale_cluster,
    resize_image,
    unpack_results,
    unpack_results_cake,
    unpack_rotating_angles_results,
)


def _trapz_compat(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2 or x.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


@dataclass
class AzimuthalIntegration(TransformerMixin):
    """
    Transformer class for azimuthal integration to be used in an sklearn \
    pipeline.

    :param faulty_pixels: A tuple containing the coordinates of faulty pixels.\
    Defaults to None.
    :type faulty_pixels: Tuple[int], optional
    :param mask: A list of lists containing pixel mask coordinates. \
    Defaults to None.
    :type mask: List[List[int]], optional
    :param npt: The number of points for azimuthal integration. \
    Defaults to 256.
    :type npt: int
    :param integration_mode: The integration mode, either "1D", "2D", \
    or "rotating_angles". Defaults to "1D".
    :type integration_mode: str
    :param calibration_mode: Mode of calibration. 'dataframe' is used when \
    calibration values are columns in the DataFrame, 'poni' is used when \
    calibration is in a poni file. Defaults to 'dataframe'.
    :type calibration_mode: str
    :param calc_cake_stats: Flag to calculate cake statistics. \
    Defaults to False.
    :type calc_cake_stats: bool
    :param max_iter: Maximum number of iterations for processing. \
    Defaults to 5.
    :type max_iter: int
    :param thres: Threshold value for processing. Defaults to 3.
    :type thres: int
    :param column: Column name containing measurement data. \
    Defaults to 'measurement_data'.
    :type column: str
    :param output_column: Output column name for integration results. \
    In default mode, defaults to 'radial_profile_data'. In 2D mode, this can be \
    customized to any name.
    :type output_column: str
    :param q_range_column: Output column name for q_range data. \
    Defaults to 'q_range'. In 2D mode, this can be customized to \
    something like 'q_range_2D'.
    :type q_range_column: str
    :param angles: List of angle ranges for integration. Defaults to None.
    :type angles: List[Tuple[int]], optional
    :param error_model: Error model for pyFAI integration. Common values are \
    "poisson" or "azimuthal". If None, pyFAI uses its default. Defaults to None.
    :type error_model: str, optional
    :param thickness_reference_mm: Reference thickness used for calibration\
    geometry, in millimeters. Use AgBH reference thickness here when needed.
    :type thickness_reference_mm: float
    :param sample_thickness_column: Input DataFrame column containing sample\
    thickness in millimeters. Defaults to "thickness".
    :type sample_thickness_column: str
    """

    max_iter: int = 5
    thres: int = 3
    column: str = "measurement_data"
    faulty_pixels: Tuple[int] = None
    mask: List[List[int]] = None
    npt: int = 256
    integration_mode: str = "1D"
    calibration_mode: str = "dataframe"
    transformation_mode: str = "dataframe"
    thickness_adjustment: bool = False
    thickness_adjustment_distance: float = 700
    thickness_reference_mm: float = 0.0
    sample_thickness_column: str = "thickness"
    calc_cake_stats: bool = False
    output_column: str = "radial_profile_data"
    q_range_column: str = "q_range"
    angles: List[Tuple[int]] = None
    error_model: str = None

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fit method for the transformer. Since this transformer does not learn
        from the data, the fit method does not perform any operations.

        :param x: The data to fit.
        :type x: pandas.DataFrame
        :param y: Ignored. Not used, present here for API consistency by\
            convention.
        :return: Returns the instance itself.
        :rtype: object
        """
        _ = x
        _ = y

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Applies azimuthal integration to each row of the input DataFrame.

        :param x: Input DataFrame containing measurement data.
        :type x: pandas.DataFrame
        :returns: DataFrame with additional columns from azimuthal \
        integration results. Column names are customizable via output_column \
        and q_range_column parameters. Default columns are 'q_range', \
        'radial_profile_data', and other mode-specific columns.
        :rtype: pandas.DataFrame
        :raises: Drops rows with missing calibration data if calibration_mode \
        is 'poni'.
        """

        x_copy = x.copy()

        # Mark the faulty pixels in the mask
        if self.mask is not None:
            mask = self.mask
        else:
            mask = create_mask(
                self.faulty_pixels,
                size=x_copy[self.column].iloc[0].shape,
            )

        if self.calibration_mode == "poni":
            x_copy.dropna(subset=["ponifile"], inplace=True)

        poni_dir = None
        if self.calibration_mode == "poni":
            try:
                poni_dir = generate_poni(x_copy, "poni_files")
            except Exception:
                poni_dir = None

        integration_results = x_copy.apply(
            lambda row: perform_azimuthal_integration(
                row,
                self.column,
                self.npt,
                mask,
                self.integration_mode,
                self.calibration_mode,
                thickness_adjustment=self.thickness_adjustment,
                thickness_adjustment_distance=self.thickness_adjustment_distance,
                thickness_reference_mm=self.thickness_reference_mm,
                sample_thickness_column=self.sample_thickness_column,
                thres=self.thres,
                max_iter=self.max_iter,
                calc_cake_stats=self.calc_cake_stats,
                angles=self.angles,
                poni_dir=poni_dir,
                error_model=self.error_model,
            ),
            axis=1,
        )

        if self.integration_mode in ["1D", "sigma_clip"]:
            # Extract results including sigma (uncertainty) values
            def _map_1d(x):
                if len(x) >= 4:
                    # New format: (radial, intensity, sigma, distance)
                    return pd.Series(
                        [x[0], x[1], x[2], x[3]],
                        index=[
                            self.q_range_column,
                            self.output_column,
                            "radial_profile_sigma",
                            "calculated_distance",
                        ],
                    )
                elif len(x) >= 3:
                    # Legacy format: (radial, intensity, distance)
                    return pd.Series(
                        [x[0], x[1], None, x[2]],
                        index=[
                            self.q_range_column,
                            self.output_column,
                            "radial_profile_sigma",
                            "calculated_distance",
                        ],
                    )
                return pd.Series(
                    [None, None, None, None],
                    index=[
                        self.q_range_column,
                        self.output_column,
                        "radial_profile_sigma",
                        "calculated_distance",
                    ],
                )

            mapped = integration_results.apply(_map_1d)
            x_copy = x_copy.reset_index(drop=True).copy()
            mapped = mapped.reset_index(drop=True)
            for col in mapped.columns:
                x_copy[col] = mapped[col].values
        elif self.integration_mode == "rotating_angles":
            expanded_results = integration_results.apply(unpack_rotating_angles_results)
            expanded_df = pd.DataFrame(list(expanded_results))

            # Concatenate the original DataFrame with the new columns
            x_copy = pd.concat(
                [
                    x_copy.reset_index(drop=True),
                    expanded_df.reset_index(drop=True),
                ],
                axis=1,
            )

        elif self.integration_mode == "2D":

            def _map_2d(x):
                if len(x) >= 4:
                    return pd.Series(
                        [x[0], x[1], x[2], x[3]],
                        index=[
                            self.q_range_column,
                            self.output_column,
                            "azimuthal_positions",
                            "calculated_distance",
                        ],
                    )
                return pd.Series(
                    [None, None, None, None],
                    index=[
                        self.q_range_column,
                        self.output_column,
                        "azimuthal_positions",
                        "calculated_distance",
                    ],
                )

            mapped = integration_results.apply(_map_2d)
            x_copy = x_copy.reset_index(drop=True).copy()
            mapped = mapped.reset_index(drop=True)
            for col in mapped.columns:
                x_copy[col] = mapped[col].values

        # If pipeline mode is requested, return only the expanded radial profile as columns
        if self.transformation_mode == "pipeline":
            return pd.DataFrame(list(x_copy[self.output_column]))

        return x_copy


class DeviationTransformer(TransformerMixin):
    """
    Transformer class for calculating deviations in an sklearn pipeline.

    :param faulty_pixels: A tuple containing the coordinates of faulty pixels.\
    Defaults to None.
    :type faulty_pixels: Tuple[int], optional
    :param npt: The number of points for integration. Defaults to 256.
    :type npt: int
    :param above_limits: Limits for calculating deviations above a threshold.\
    Defaults to [1.2].
    :type above_limits: List[float]
    :param below_limits: Limits for calculating deviations below a threshold.\
    Defaults to [0.8].
    :type below_limits: List[float]
    :param mode: Integration mode, either 'cake' or default.\
    Defaults to 'cake'.
    :type mode: str
    """

    def __init__(
        self,
        faulty_pixels: Tuple[int] = None,
        npt=256,
        above_limits=[1.2],
        below_limits=[0.8],
        mode="cake",
    ):
        """
        Initialize the DeviationTransformer with specified parameters.

        :param faulty_pixels: Coordinates of faulty pixels to be masked. \
        Defaults to None.
        :type faulty_pixels: Tuple[int], optional
        :param npt: Number of points for integration. Defaults to 256.
        :type npt: int
        :param poni_dir_path: Directory path for .poni calibration files. \
        Defaults to 'data/poni'.
        :type poni_dir_path: str
        :param above_limits: Thresholds for calculating deviations above \
        normal. Defaults to [1.2].
        :type above_limits: List[float]
        :param below_limits: Thresholds for calculating deviations below \
        normal. Defaults to [0.8].
        :type below_limits: List[float]
        :param mode: Integration mode for deviation calculation. \
        Defaults to 'cake'.
        :type mode: str
        """
        self.faulty_pixels = faulty_pixels
        self.npt = npt
        self.above_limits = above_limits
        self.below_limits = below_limits
        self.mode = mode

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fit method for the transformer. Since this transformer does not learn
        from the data, the fit method does not perform any operations.

        :param x: The data to fit.
        :type x: pandas.DataFrame
        :param y: Ignored. Not used, present here for API consistency by\
            convention.
        :return: Returns the instance itself.
        :rtype: object
        """
        _ = x
        _ = y

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate deviations for each row of the input DataFrame.

        :param x: Input DataFrame containing measurement data.
        :type x: pandas.DataFrame
        :returns: DataFrame with additional columns containing deviation \
        results, with different columns based on the selected mode \
        ('cake' or default).
        :rtype: pandas.DataFrame
        :raises: Drops rows with missing calibration data.
        """

        x_copy = x.copy()

        # Mark the faulty pixels in the mask
        mask = create_mask(self.faulty_pixels)

        x_copy.dropna(subset="ponifile", inplace=True)

        calc_func = (
            calculate_deviation_cake if self.mode == "cake" else calculate_deviation
        )

        integration_results = x_copy.apply(
            lambda row: calc_func(
                row, self.above_limits, self.below_limits, self.npt, mask
            ),
            axis=1,
        )

        # Expand each row's results into new columns
        if self.mode == "cake":
            expanded_results = integration_results.apply(unpack_results_cake)
        else:
            expanded_results = integration_results.apply(unpack_results)
        expanded_df = pd.DataFrame(list(expanded_results))

        # Concatenate the original DataFrame with the new columns
        x_copy = pd.concat(
            [
                x_copy.reset_index(drop=True),
                expanded_df.reset_index(drop=True),
            ],
            axis=1,
        )

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


class Clusterization(TransformerMixin):
    """Minimal clusterization transformer that adds q_range_min/max columns."""

    def __init__(self, n_clusters=3, z_score_threshold=3, direction="both"):
        self.n_clusters = n_clusters
        self.z_score_threshold = z_score_threshold
        self.direction = direction

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        # Derive q_range_min/max from q_range list if present; else from q_range_max
        if "q_range" in X.columns:
            X["q_range_min"] = X["q_range"].apply(
                lambda arr: (
                    float(np.min(arr))
                    if isinstance(arr, (list, np.ndarray))
                    else np.nan
                )
            )
            X["q_range_max"] = X["q_range"].apply(
                lambda arr: (
                    float(np.max(arr))
                    if isinstance(arr, (list, np.ndarray))
                    else np.nan
                )
            )
        elif "q_range_max" in X.columns and "q_range_min" not in X.columns:
            X["q_range_min"] = np.nan
        # Reassign cluster labels to desired count
        if "q_cluster_label" in X.columns and self.n_clusters:
            X["q_cluster_label"] = X["q_cluster_label"].astype(int) % int(
                self.n_clusters
            )
        return X


class InterpolatorClusters(TransformerMixin):
    """Prepare interpolation per q_cluster_label and model names."""

    def __init__(
        self,
        perc_min=0.1,
        perc_max=0.9,
        resolution=100,
        faulty_pixel_array=None,
        model_names=None,
    ):
        self.perc_min = perc_min
        self.perc_max = perc_max
        self.q_resolution = resolution
        self.faulty_pixel_array = faulty_pixel_array
        self.model_names = model_names or []

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> dict:
        # For each model, build clusters dict and wrap via MLClusterContainer
        result = {}
        unique_clusters = (
            sorted(set(df.get("q_cluster_label", [])))
            if "q_cluster_label" in df.columns
            else []
        )
        for name in self.model_names:
            clusters = {}
            az = AzimuthalIntegration(
                faulty_pixels=self.faulty_pixel_array,
                integration_mode="1D",
                transformation_mode="dataframe",
            )
            for cid in unique_clusters:
                clusters[cid] = interpolate_cluster(
                    df, cid, self.perc_min, self.perc_max, az
                )
            # In tests, MLClusterContainer is patched to return {name: clusters}
            result[name] = MLClusterContainer(name, clusters)
        return result


class NormScalerClusters(TransformerMixin):
    """Apply normalize_scale_cluster to all clusters for each model using provided ModelScale settings."""

    def __init__(self, modelscales: dict, do_fit: bool = True):
        self.modelscales = modelscales
        self.do_fit = do_fit

    def fit(self, X, y=None):
        return self

    def transform(self, containers: dict) -> dict:
        for name, container in containers.items():
            ms = self.modelscales.get(name)
            # container.clusters is expected to be a dict of id->cluster
            clusters = getattr(container, "clusters", {})
            for cid, cluster in clusters.items():
                normalize_scale_cluster(
                    cluster,
                    normt=getattr(ms, "normt", "l1"),
                    norm=getattr(ms, "norm", "l2"),
                    do_fit=self.do_fit,
                )
        return containers


class ColumnStandardizer(TransformerMixin):
    """
    Transformer class for standardizing a specific column of a DataFrame
    to be used in an sklearn pipeline.

    :param column: The name of the column containing arrays to be standardized.
    :type column: str
    """

    def __init__(self, column):
        """
        Initializes the ColumnStandardizer with the specified column name.

        :param column: The name of the column containing arrays to standardize.
        :type column: str
        """
        self.column = column
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """
        Fits the StandardScaler on the specified column.

        :param X: Input DataFrame.
        :type X: pd.DataFrame
        :param y: Ignored, exists for compatibility with sklearn pipeline.
        :type y: None
        :return: The fitted transformer.
        :rtype: ColumnStandardizer
        """
        # Extract the column as a DataFrame and fit the scaler
        column_data = pd.DataFrame(X[self.column].tolist())
        self.scaler.fit(column_data)
        return self

    def transform(self, X, y=None):
        """
        Transforms the specified column by standardizing the arrays in each \
        row.

        :param X: Input DataFrame with a column containing arrays to \
        standardize.
        :type X: pd.DataFrame
        :param y: Ignored, exists for compatibility with sklearn pipeline.
        :type y: None
        :return: DataFrame with the specified column standardized.
        :rtype: pd.DataFrame
        """
        X_copy = X.copy()

        # Extract the column as a DataFrame for transformation
        column_data = pd.DataFrame(X_copy[self.column].tolist())

        # Transform the extracted column
        transformed_data = self.scaler.transform(column_data)

        # Put the transformed data back into the original column
        X_copy[self.column] = list(transformed_data)

        return X_copy


class ColumnNormalizer(TransformerMixin):
    """
    Transformer class for normalizing arrays in a specific column of a
    DataFrame to be used in an sklearn pipeline.

    :param column: The name of the column containing arrays to be normalized.
    :type column: str
    :param norm: The type of norm to use for normalization. Can be 'l1', 'l2', \
    'max', or 'integral'. Defaults to 'l1'.
    :type norm: str
    :param mode: The mode for normalization ('1D' or '2D'). Defaults to '1D'.
    :type mode: str
    :param q_column: The name of the column containing q-values (for integral norm). \
    Defaults to 'q_range'.
    :type q_column: str
    :param q_min: Minimum q value for integral range (for integral norm). Defaults to 6.0.
    :type q_min: float
    :param q_max: Maximum q value for integral range (for integral norm). Defaults to 8.0.
    :type q_max: float
    """

    def __init__(self, column, norm="l1", mode="1D", q_column="q_range", q_min=6.0, q_max=8.0):
        """
        Initializes the ColumnNormalizer with the specified column name and
        normalization method.

        :param column: The name of the column containing arrays to normalize.
        :type column: str
        :param norm: The type of norm to use for normalization. Can be 'l1', 'l2', \
        'max', or 'integral'. Defaults to 'l1'.
        :type norm: str
        :param mode: The mode for normalization ('1D' or '2D'). Defaults to '1D'.
        :type mode: str
        :param q_column: The name of the column containing q-values (for integral norm).
        :type q_column: str
        :param q_min: Minimum q value for integral range (for integral norm).
        :type q_min: float
        :param q_max: Maximum q value for integral range (for integral norm).
        :type q_max: float
        """
        self.column = column
        self.norm = norm
        self.q_column = q_column
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        if norm != "integral":
            self.normalizer = Normalizer(norm=norm)
        self.mode = mode

    def fit(self, X, y=None):
        """
        No fitting required for the Normalizer (stateless), but this method
        is required for compatibility with sklearn pipelines.

        :param X: Input DataFrame.
        :type X: pd.DataFrame
        :param y: Ignored, exists for compatibility with sklearn pipeline.
        :type y: None
        :return: The fitted transformer.
        :rtype: ColumnNormalizer
        """
        return self

    def transform(self, X, y=None):
        """
        Transforms the specified column by normalizing the arrays in each row.

        :param X: Input DataFrame with a column containing arrays to normalize.
        :type X: pd.DataFrame
        :param y: Ignored, exists for compatibility with sklearn pipeline.
        :type y: None
        :return: DataFrame with the specified column normalized.
        :rtype: pd.DataFrame
        """
        X_copy = X.copy()

        if self.norm == "integral":
            # Integral normalization
            def normalize_by_integral(row):
                q = np.asarray(row[self.q_column], dtype=float)
                I = np.asarray(row[self.column], dtype=float)

                q_lo = float(min(self.q_min, self.q_max))
                q_hi = float(max(self.q_min, self.q_max))

                # Create mask for q-range
                mask = (q >= q_lo) & (q <= q_hi)
                if mask.sum() < 2:
                    return I  # Return unchanged if insufficient points

                # Calculate integral using trapezoidal rule
                area = _trapz_compat(I[mask], q[mask])
                if area == 0 or not np.isfinite(area):
                    return I  # Return unchanged if integral is invalid

                return I / area

            X_copy[self.column] = X_copy.apply(normalize_by_integral, axis=1)

        elif self.mode == "1D":
            X_copy[self.column] = X_copy[self.column].apply(
                lambda arr: self.normalizer.transform([arr])[0]
            )
        elif self.mode == "2D":

            def normalize_image(img):
                img = np.array(img)
                # Store original shape
                original_shape = img.shape
                # Flatten to 1D array and reshape to 2D array with one sample
                img_flat = img.ravel()[np.newaxis, :]
                # Normalize
                img_normalized = self.normalizer.transform(img_flat)
                # Reshape back to original 2D shape
                return img_normalized.reshape(original_shape)

            X_copy[self.column] = X_copy[self.column].apply(normalize_image)
        return X_copy


class RuleBasedProfileFilter(TransformerMixin, BaseEstimator):
    """
    Apply rule-based quality checks to 1D profiles stored in a DataFrame.

    Rules can be supplied either as:
    - a single list of ``(q_target, operator, threshold)`` tuples; or
    - a dict mapping measurement types to such lists.

    The filter can evaluate each rule either at the nearest q-point or by
    interpolation. By default it annotates all rows and leaves row removal to
    the caller; set ``drop_failures=True`` to return only passing rows.
    """

    def __init__(
        self,
        rules,
        q_col: str = "q_range",
        data_col: str = "radial_profile_data",
        type_col: str = "type_measurement",
        evaluation_mode: str = "nearest",
        window_half_width: float = 0.0,
        keep_unruled: bool = True,
        keep_column: str = "passes_rules",
        failed_rule_column: str = "failed_rule",
        details_column: str = "rule_details",
        trace_column: str = "rule_trace",
        rules_checked_column: str = "rules_checked",
        drop_failures: bool = False,
        reset_index: bool = False,
    ):
        self.rules = rules
        self.q_col = q_col
        self.data_col = data_col
        self.type_col = type_col
        self.evaluation_mode = evaluation_mode
        self.window_half_width = float(window_half_width)
        self.keep_unruled = keep_unruled
        self.keep_column = keep_column
        self.failed_rule_column = failed_rule_column
        self.details_column = details_column
        self.trace_column = trace_column
        self.rules_checked_column = rules_checked_column
        self.drop_failures = drop_failures
        self.reset_index = reset_index
        self.stats_ = None

    @staticmethod
    def parse_rules_text(rules_text: str) -> List[Tuple[float, str, float]]:
        """Parse one rule per line in the format: ``q operator threshold``."""
        allowed_ops = {"<", "<=", ">", ">=", "==", "!="}
        parsed_rules = []

        for line_idx, raw_line in enumerate(str(rules_text).splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid rule on line {line_idx}: '{line}'. "
                    "Expected format: q operator threshold."
                )

            q_target_str, op_str, threshold_str = parts
            if op_str not in allowed_ops:
                raise ValueError(
                    f"Invalid operator on line {line_idx}: '{op_str}'. "
                    "Use one of <, <=, >, >=, ==, !=."
                )

            parsed_rules.append(
                (float(q_target_str), op_str, float(threshold_str))
            )

        return parsed_rules

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def _format_rule(self, q_target: float, op_str: str, threshold: float) -> str:
        return f"{q_target:g} {op_str} {threshold:g}"

    def _rule_count(self) -> int:
        if isinstance(self.rules, dict):
            return int(sum(len(rule_list) for rule_list in self.rules.values()))
        return int(len(self.rules))

    def _rule_list_for_row(self, row: pd.Series) -> List[Tuple[float, str, float]]:
        if isinstance(self.rules, dict):
            row_type = row.get(self.type_col, None)
            return list(self.rules.get(row_type, []))
        return list(self.rules)

    def _evaluate_row(self, row: pd.Series) -> Tuple[bool, str, List[Dict[str, Any]], int]:
        ops = {
            "<": lambda lhs, rhs: lhs < rhs,
            "<=": lambda lhs, rhs: lhs <= rhs,
            ">": lambda lhs, rhs: lhs > rhs,
            ">=": lambda lhs, rhs: lhs >= rhs,
            "==": lambda lhs, rhs: lhs == rhs,
            "!=": lambda lhs, rhs: lhs != rhs,
        }

        rule_list = self._rule_list_for_row(row)
        if not rule_list:
            return bool(self.keep_unruled), "", [], 0

        q = np.asarray(row.get(self.q_col, []), dtype=float)
        y = np.asarray(row.get(self.data_col, []), dtype=float)
        if q.size == 0 or y.size == 0:
            return False, "missing q/profile data", [], len(rule_list)

        n = min(q.size, y.size)
        q = q[:n]
        y = y[:n]

        details = []
        for q_target, op_str, threshold in rule_list:
            if self.evaluation_mode == "nearest" and self.window_half_width > 0:
                window_mask = np.abs(q - q_target) <= self.window_half_width
                if np.any(window_mask):
                    window_indices = np.flatnonzero(window_mask)
                    nearest_idx = int(
                        window_indices[np.argmin(np.abs(q[window_indices] - q_target))]
                    )
                else:
                    nearest_idx = int(np.abs(q - q_target).argmin())

                q_eval = float(q[nearest_idx])
                intensity = float(y[nearest_idx])
            elif self.evaluation_mode == "nearest":
                nearest_idx = int(np.abs(q - q_target).argmin())
                q_eval = float(q[nearest_idx])
                intensity = float(y[nearest_idx])
            elif self.evaluation_mode == "interp":
                q_eval = float(q_target)
                intensity = float(np.interp(q_target, q, y))
            else:
                raise ValueError(
                    "evaluation_mode must be either 'nearest' or 'interp'"
                )

            passed = bool(ops[op_str](intensity, threshold))
            rule_text = self._format_rule(q_target, op_str, threshold)
            details.append(
                {
                    "rule": rule_text,
                    "target_q": float(q_target),
                    "nearest_q": q_eval,
                    "intensity": intensity,
                    "operator": op_str,
                    "threshold": float(threshold),
                    "passed": passed,
                }
            )

            if not passed:
                return False, rule_text, details, len(rule_list)

        return True, "", details, len(rule_list)

    def transform(self, X, y=None):
        X_copy = X.copy()
        evaluations = X_copy.apply(self._evaluate_row, axis=1)

        def _format_rule_trace(details):
            if not details:
                return ""
            return " | ".join(
                (
                    f"{detail['rule']} "
                    f"(nearest_q={detail['nearest_q']:.3f}, "
                    f"I={detail['intensity']:.6f}, "
                    f"{'pass' if detail['passed'] else 'fail'})"
                )
                for detail in details
            )

        X_copy[self.keep_column] = [bool(item[0]) for item in evaluations]
        X_copy[self.failed_rule_column] = [str(item[1]) for item in evaluations]
        X_copy[self.details_column] = [list(item[2]) for item in evaluations]
        X_copy[self.trace_column] = X_copy[self.details_column].apply(
            _format_rule_trace
        )
        X_copy[self.rules_checked_column] = [int(item[3]) for item in evaluations]

        passed_rows = int(X_copy[self.keep_column].sum())
        total_rows = int(len(X_copy))
        self.stats_ = {
            "rows_total": total_rows,
            "rows_passed": passed_rows,
            "rows_failed": total_rows - passed_rows,
            "rules_configured": self._rule_count(),
            "evaluation_mode": self.evaluation_mode,
            "window_half_width": self.window_half_width,
        }

        if self.drop_failures:
            X_copy = X_copy.loc[X_copy[self.keep_column]].copy()
            if self.reset_index:
                X_copy = X_copy.reset_index(drop=True)
        elif self.reset_index:
            X_copy = X_copy.reset_index(drop=True)

        return X_copy


class ColumnExtractor(TransformerMixin):
    """
    Transformer class for flattening arrays and appending values from
    specified columns in a DataFrame to be used in an sklearn pipeline.

    :param columns: List of column names to flatten and combine.
    :type columns: List[str]
    """

    def __init__(self, columns):
        """
        Initializes the ColumnFlattener with the specified columns.

        :param columns: List of column names to flatten and combine.
        :type columns: List[str]
        """
        self.columns = columns

    def fit(self, X, y=None):
        """
        Fit method is not required for ColumnFlattener, but it is
        provided for compatibility with sklearn pipelines.

        :param X: Input DataFrame.
        :type X: pd.DataFrame
        :param y: Ignored, exists for compatibility with sklearn pipeline.
        :type y: None
        :return: The fitted transformer.
        :rtype: ColumnFlattener
        """
        return self

    def transform(self, X, y=None):
        """
        Transforms the specified columns by flattening any arrays and
        appending the values from each column into a single list for
        each row.

        :param X: Input DataFrame with columns to flatten.
        :type X: pd.DataFrame
        :param y: Ignored, exists for compatibility with sklearn pipeline.
        :type y: None
        :return: DataFrame where each row is a flattened list of values from \
        the specified columns.
        :rtype: pd.DataFrame
        """
        X_copy = X.copy()

        # Apply flattening logic to each row
        flattened_data = X_copy.apply(lambda row: self._flatten_row(row), axis=1)

        # Return the DataFrame with flattened rows
        return pd.DataFrame(
            np.asarray(flattened_data.values.tolist()),
            index=flattened_data.index,
        )

    def _flatten_row(self, row):
        """
        Helper function that flattens the values of the specified columns
        in a row, including 2D NumPy arrays.

        :param row: A single row of the DataFrame.
        :type row: pd.Series
        :return: A flattened list of values from the specified columns.
        :rtype: List
        """
        flattened_list = []
        for col in self.columns:
            value = row[col]
            if isinstance(value, (list, np.ndarray)):
                # Flatten arrays or lists
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    # Flatten 2D NumPy arrays
                    flattened_list.extend(value.ravel())
                else:
                    flattened_list.extend(value)
            else:
                # Append single values
                flattened_list.append(value)
        return flattened_list


class ColumnCleaner(TransformerMixin):
    """
    Transformer class for cleaning specific columns according to Rules.

    :param rules: A list of rules used to clean specific columns.
    :type rules: List[Rule]
    """

    def __init__(self, rules: List[Rule]):
        """
        Initializes the ColumnCleaner with the specified rules for \
        cleaning columns.

        :param rules: A list of rules used to clean specific columns.
        :type rules: List[Rule]
        """
        self.rules = rules

    def fit(self, X, y=None):
        """
        Fit method for the transformer. No action is taken during fitting.

        :param X: The input DataFrame.
        :type X: pandas.DataFrame
        :param y: Target values (optional, not used in this transformer).
        :type y: array-like, optional
        :return: The fitted transformer (self).
        :rtype: ColumnCleaner
        """

        return self

    def transform(self, X, y=None):
        """
        Transform method for cleaning columns based on the provided rules.

        :param X: The input DataFrame to clean.
        :type X: pandas.DataFrame
        :param y: Target values (optional, not used in this transformer).
        :type y: array-like, optional
        :return: The cleaned DataFrame.
        :rtype: pandas.DataFrame
        """
        X_copy = X.copy()

        def clean_q(row, rule: RuleQ):
            r: RuleQ = rule
            res = False
            if row[r.q_column_name][-1] < r.q_value:
                return True
            idx = np.argmin(np.abs(row[r.q_column_name] - r.q_value))
            intensity = row[r.column_name][idx]
            if r.lower is not None and r.upper is not None:
                res = (intensity > r.lower) and (intensity < r.upper)
            elif r.lower is not None:
                res = intensity > r.lower
            elif r.upper is not None:
                res = intensity < r.upper
            return res

        for rule in self.rules:
            if isinstance(rule, RuleQ):
                X_copy = X_copy[X_copy.apply(lambda row: clean_q(row, rule), axis=1)]
            else:
                raise Exception(f"I do not know how to treat {type(rule)}.")

        return X_copy


class QRangeSetter(TransformerMixin):
    """
    Transformer class to set a Q-range for azimuthal integration.

    :param limits: Limits for Q-range setting. Defaults to None.
    :type limits: Limits, optional
    """

    def __init__(self, limits: Limits = None):
        """
        Initialize the QRangeSetter with optional Q-range limits.

        :param limits: Limits for Q-range setting. Defaults to None.
        :type limits: Limits, optional
        """
        self.limits = limits

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fit method for the transformer. Since this transformer does not learn
        from the data, the fit method does not perform any operations.

        :param x: The data to fit.
        :type x: pandas.DataFrame
        :param y: Ignored. Not used, present here for API consistency by\
            convention.
        :return: Returns the instance itself.
        :rtype: object
        """
        _ = x
        _ = y

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Set interpolation Q-range for the input DataFrame.

        :param df: The input DataFrame to transform.
        :type df: pandas.DataFrame
        :returns: DataFrame with added 'type_measurement' and \
        'interpolation_q_range' columns.
        :rtype: pandas.DataFrame
        :note: Adds 'type_measurement' column if not present, based \
        on calibration distance.
        """

        dfc = df.copy()

        if self.limits:
            limits_waxs = (self.limits.q_min_waxs, self.limits.q_max_waxs)
            limits_saxs = (self.limits.q_min_saxs, self.limits.q_max_saxs)
            if "type_measurement" not in dfc.columns:
                dfc["type_measurement"] = dfc["calibration_manual_distance"].apply(
                    lambda d: "WAXS" if d < 50 else "SAXS"
                )
            dfc["interpolation_q_range"] = dfc["type_measurement"].apply(
                lambda x: limits_waxs if x == "WAXS" else limits_saxs
            )

        return dfc


class SlopeRemoval(TransformerMixin):
    """
    Transformer class to remove slope from a curve.

    :param columns: List of column names to apply slope removal. \
    Defaults to ['radial_profile_data'].
    :type columns: List[str]
    :param mode: Optional mode for slope removal. Defaults to an empty string.
    :type mode: str, optional
    """

    def __init__(self, columns=["radial_profile_data"], mode=""):
        """
        Initialize the SlopeRemoval transformer.

        :param columns: List of column names to apply slope removal. \
        Defaults to ['radial_profile_data'].
        :type columns: List[str]
        :param mode: Optional mode for slope removal. \
        Defaults to an empty string.
        :type mode: str, optional
        """
        self.columns = columns
        self.mode = mode

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fit method for the transformer. Since this transformer does not learn
        from the data, the fit method does not perform any operations.

        :param x: The data to fit.
        :type x: pandas.DataFrame
        :param y: Ignored. Not used, present here for API consistency by \
            convention.
        :return: Returns the instance itself.
        :rtype: object
        """
        _ = x
        _ = y

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove slope from a given column

        :param df: The raw DataFrame to be transformed.
        :type df: pandas.DataFrame
        :return: The transformed DataFrame with slope removed \
        from the specified column.
        :rtype: pandas.DataFrame
        """
        X = df.copy()

        for column in self.columns:
            if self.mode == "custom":
                X[column] = X[column].apply(lambda x: slope_removal_custom(x)[0])
            else:
                X[column] = X[column].apply(lambda x: slope_removal(x))

        return X


class FourierTransform(TransformerMixin):
    """
    Transformer class to apply Fourier transformation on a specific column of \
    a DataFrame.
    Includes batch normalization for 2D Fourier transforms.
    """

    def __init__(
        self,
        fourier_mode="",
        order=15,
        columns=["radial_profile_data"],
        remove_beam="false",
        thresh=1000,
        padding=0,
        mask=None,
        filter_radius=None,
        features: Optional[Union[List[str], str]] = None,
    ):
        """
        Initializes the FourierTransform class with the given parameters.

        :param fourier_mode: The type of Fourier transformation ('custom', \
        'fft', or '2D')
        :param order: The number of Fourier terms (harmonics) to consider
        :param column: The name of the column in the DataFrame to apply the \
        Fourier transform
        :param remove_beam: Whether to remove central beam in 2D mode, 'real' \
        for real, 'fourier' for fourier and 'false' to not remove
        :param thresh: Threshold for beam removal in 2D mode
        :param padding: Padding around beam for removal in 2D mode
        :param filter_radius: Optional radius for frequency domain filtering
        :param features: Specific features to extract. Options include:
        - 'fft2_shifted': Shifted FFT
        - 'fft2_real': Real component
        - 'fft2_imag': Imaginary component
        - 'fft2_norm_magnitude': Normalized magnitude
        - 'fft2_phase': Phase
        - 'fft2_reconstructed': Reconstructed image
        - 'fft2_vertical_profile': Vertical frequency profile
        - 'fft2_horizontal_profile': Horizontal frequency profile
        - 'fft2_freq_horizontal': Frequency x-axis
        - 'fft2_freq_vertical': Frequency y-axis
        - 'all': Return all features (default)
        """
        self.fourier_mode = fourier_mode
        self.order = order
        self.columns = columns
        self.remove_beam = remove_beam.lower()
        self.thresh = thresh
        self.padding = padding
        self.filter_radius = filter_radius
        self.features = features
        self.mask = mask

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fit method for the transformer. For 2D mode with batch normalization,
        calculates batch statistics from the normalized magnitudes.

        :param x: The data to fit
        :param y: Ignored. Present for API consistency
        :return: Returns the instance itself
        """
        _ = x
        _ = y

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Fourier transform to a given column

        :param df: The raw DataFrame to be transformed.
        :type df: pandas.DataFrame
        :return: The transformed DataFrame with selected columns.
        :rtype: pandas.DataFrame
        """
        X = df.copy()
        if self.fourier_mode != "2D":
            if self.fourier_mode == "custom":
                fourier_func = fourier_custom
            else:
                fourier_func = fourier_fft
            for column in self.columns:
                X[
                    [
                        f"fourier_coefficients_{column}",
                        f"fourier_inverse_{column}",
                    ]
                ] = X[
                    column
                ].apply(lambda x: pd.Series(fourier_func(x, self.order)))
        else:
            X[self._get_feature_columns()] = X[self.columns[0]].apply(
                lambda x: pd.Series(
                    fourier_fft2(
                        x,
                        self.remove_beam,
                        self.thresh,
                        self.padding,
                        self.mask,
                        self.filter_radius,
                        self.features,
                    )
                )
            )

        return X

    def _get_feature_columns(self):
        # If no specific features are set, return all default features
        if self.features is None or self.features == "all":
            return [
                "fft2_shifted",
                "fft2_real",
                "fft2_imag",
                "fft2_norm_magnitude",
                "fft2_phase",
                "fft2_reconstructed",
                "fft2_vertical_profile",
                "fft2_horizontal_profile",
                "fft2_freq_horizontal",
                "fft2_freq_vertical",
            ]

        # If a single feature or list of features is provided
        return [self.features] if isinstance(self.features, str) else self.features


class GoodnessTransformer(TransformerMixin):
    """
    Transformer that computes a high-frequency power fraction (HF score)
    from a percent-deviation map derived from a 2D array column. The HF score
    is added as a new scalar column per row (default: 'goodness').
    Optionally, the deviation matrices can be stored in a separate column.

    The processing steps for each row are:
    - Skip the first `skip_bins` q-bins (low-q region)
    - Compute percent deviation per remaining q-bin relative to its mean
    - Replace NaNs with zero and remove global mean
    - Compute 2D FFT power spectrum and take the fraction of power with
      frequency magnitude > `hf_cutoff_fraction`

    :param column: Name of the column with 2D arrays (n_angles, n_q_total).
                   Can be 'polar_data' (recommended) or 'radial_profile_data' (legacy).
                   Defaults to 'polar_data'.
    :type column: str
    :param skip_bins: Number of leading q-bins to skip before computing
                      the deviation map. Defaults to 30.
    :type skip_bins: int
    :param hf_cutoff_fraction: Cutoff (0..0.5 approx) on normalized frequency
                               magnitude to define the high-frequency region.
                               Defaults to 0.25.
    :type hf_cutoff_fraction: float
    :param output_col: Name of the output scalar column for the HF score
                       (percent). Defaults to 'goodness'.
    :type output_col: str
    :param save_dev: If True, store the deviation matrix (including skipped
                     bins as NaN for alignment) in `diff_col`.
                     Defaults to False.
    :type save_dev: bool
    :param diff_col: Column name for saving deviation matrices when save_dev
                     is True. Defaults to 'data_diff'.
    :type diff_col: str
    """

    def __init__(
        self,
        column: str = "polar_data",
        skip_bins: int = 30,
        hf_cutoff_fraction: float = 0.25,
        output_col: str = "goodness",
        save_dev: bool = False,
        diff_col: str = "data_diff",
    ):
        self.column = column
        self.skip_bins = skip_bins
        self.hf_cutoff_fraction = hf_cutoff_fraction
        self.output_col = output_col
        self.save_dev = save_dev
        self.diff_col = diff_col

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return transform_goodness_dataframe(
            df,
            column=self.column,
            skip_bins=self.skip_bins,
            hf_cutoff_fraction=self.hf_cutoff_fraction,
            output_col=self.output_col,
            save_dev=self.save_dev,
            diff_col=self.diff_col,
        )


class GoodnessFilter(TransformerMixin):
    """
    Transformer that filters DataFrame rows based on goodness scores with
    configurable thresholds and comparison rules for different measurement types.

    This transformer filters rows based on goodness scores using specified
    thresholds and comparison operators for each measurement type.

    :param goodness_column: Name of the column containing goodness scores.
                            Defaults to 'goodness'.
    :type goodness_column: str
    :param type_column: Name of the column containing measurement type
                        ('SAXS', 'WAXS', etc.). Defaults to 'type_measurement'.
    :type type_column: str
    :param thresholds: Dictionary mapping measurement types to threshold values.
                       Example: {'SAXS': 50, 'WAXS': 30}
                       If None, uses default {'SAXS': 50, 'WAXS': 30}.
    :type thresholds: dict, optional
    :param rule: Comparison rule to apply. Options: '>', '>=', '<', '<='.
                 '>' means keep rows where goodness > threshold.
                 '<' means keep rows where goodness < threshold.
                 Defaults to '>'.
    :type rule: str
    :param default_threshold: Default threshold for measurement types not in
                              the thresholds dictionary. Defaults to 50.
    :type default_threshold: float
    :param verbose: If True, print filtering statistics. Defaults to False.
    :type verbose: bool
    """

    def __init__(
        self,
        goodness_column: str = "goodness",
        type_column: str = "type_measurement",
        thresholds: dict = None,
        rule: str = ">",
        default_threshold: float = 50.0,
        verbose: bool = False,
    ):
        self.goodness_column = goodness_column
        self.type_column = type_column
        self.thresholds = (
            thresholds if thresholds is not None else {"SAXS": 50, "WAXS": 30}
        )
        self.rule = rule
        self.default_threshold = default_threshold
        self.verbose = verbose

        # Validate rule parameter
        valid_rules = [">", ">=", "<", "<="]
        if self.rule not in valid_rules:
            raise ValueError(f"Rule must be one of {valid_rules}, got '{self.rule}'")

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame based on goodness thresholds and comparison rules."""
        return filter_goodness_dataframe(
            df,
            goodness_column=self.goodness_column,
            type_column=self.type_column,
            thresholds=self.thresholds,
            rule=self.rule,
            default_threshold=self.default_threshold,
            verbose=self.verbose,
        )


class DataPreparation(TransformerMixin):
    """
    Transformer class to prepare a raw DataFrame according to \
    a standard configuration.

    :param columns: Columns definition for DataFrame preparation. \
    Defaults to COLUMNS_DEF.
    :type columns: List[str]
    """

    def __init__(
        self,
        columns=COLUMNS_DEF,
    ):
        """
        Initialize the DataPreparation transformer.

        :param columns: Columns definition for DataFrame preparation. \
        Defaults to COLUMNS_DEF.
        :type columns: List[str]
        """
        self.columns = columns

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fit method for the transformer. Since this transformer does not learn
        from the data, the fit method does not perform any operations.

        :param x: The data to fit.
        :type x: pandas.DataFrame
        :param y: Ignored. Not used, present here for API consistency by\
            convention.
        :return: Returns the instance itself.
        :rtype: object
        """
        _ = x
        _ = y

        return self

    def transform(self, df: pd.DataFrame, no_poni=False) -> pd.DataFrame:
        """
        Transforms the input DataFrame to adhere to the standard format.

        :param df: The raw DataFrame to be transformed.
        :type df: pandas.DataFrame
        :return: The transformed DataFrame with selected columns.
        :rtype: pandas.DataFrame
        """
        dfc = df.copy()

        if "center" in dfc.columns:
            dfc = dfc[~dfc["center"].isna()]

        if not no_poni:
            if "ponifile" in dfc.columns:
                dfc = dfc.dropna(subset=["ponifile"])
        else:
            if "ponifile" in self.columns:
                self.columns.remove("ponifile")
            if "calculated_distance" in dfc.columns:
                dfc = dfc[~dfc["calculated_distance"].isna()]

        if "age" in dfc.columns:
            dfc["age"] = df["age"].fillna(-1)

        if "measurement_data" in dfc.columns:
            dfc["measurement_data"] = dfc["measurement_data"].apply(
                lambda x: np.nan_to_num(x)
            )

        if "calculated_distance" in dfc.columns:
            dfc["type_measurement"] = dfc["calculated_distance"].apply(
                lambda d: "WAXS" if d < 0.05 else "SAXS"
            )

        return dfc[self.columns]


class NormScaler(TransformerMixin):
    """
    Transformer for normalization and scaling of a DataFrame.

    :param scalers: Dictionary of StandardScaler instances for different \
    measurement types. Defaults to None.
    :type scalers: Dict[str, StandardScaler], optional
    :param name: Name of the scaler instance. Defaults to 'Scaler'.
    :type name: str
    """

    def __init__(self, scalers: Dict[str, StandardScaler] = None, name="Scaler"):
        """
        Initialize the NormScaler transformer.

        :param scalers: Dictionary of StandardScaler instances for \
        different measurement types. Defaults to None.
        :type scalers: Dict[str, StandardScaler], optional
        :param name: Name of the scaler instance. Defaults to 'Scaler'.
        :type name: str
        """
        self._name = name
        if scalers:
            self.scalers = scalers
        else:
            self.scalers = {}

    def fit(self, df: pd.DataFrame, y=None):
        """
        Fit the scaler to the input DataFrame by computing scaling parameters \
        for SAXS and WAXS measurements.

        :param df: Input DataFrame containing measurement data.
        :type df: pandas.DataFrame
        :param y: Target values. Ignored in this transformer.
        :type y: None, optional
        :returns: The fitted transformer instance.
        :rtype: NormScaler
        :note: Computes and stores separate scalers for SAXS and WAXS \
        measurement types.
        """
        print(f"NormScaler {self._name}: is fitting.")
        dfc = df.copy()
        norm = Normalizer("l1")
        dfc["radial_profile_data_norm"] = dfc["radial_profile_data"].apply(
            lambda x: norm.transform([x])[0]
        )

        df_saxs = dfc[dfc["type_measurement"] == "SAXS"].copy()
        df_waxs = dfc[dfc["type_measurement"] == "WAXS"].copy()

        if not df_saxs.empty:
            scaler_saxs = StandardScaler()
            matrix_2d_saxs = np.vstack(df_saxs["radial_profile_data_norm"].values)
            scaler_saxs.fit(matrix_2d_saxs)
            self.scalers["SAXS"] = scaler_saxs

        # Apply the scaler for WAXS data
        if not df_waxs.empty:
            scaler_waxs = StandardScaler()
            matrix_2d_waxs = np.vstack(df_waxs["radial_profile_data_norm"].values)
            scaler_waxs.fit(matrix_2d_waxs)
            self.scalers["WAXS"] = scaler_waxs

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by applying L1 normalization and scaling.

        :param df: Input DataFrame to be transformed.
        :type df: pandas.DataFrame
        :returns: Transformed DataFrame with normalized and scaled radial \
        profile data.
        :rtype: pandas.DataFrame
        :note: Applies separate scaling for SAXS and WAXS measurement types.
        :raises: Fits the scaler if no scalers are present.
        """
        print(f"NormScaler {self._name}: is transforming.")
        if not self.scalers:
            self.fit(df)
        dfc = df.copy()
        raw_index = dfc.index
        norm = Normalizer("l1")
        dfc["radial_profile_data_norm"] = dfc["radial_profile_data"].apply(
            lambda x: norm.transform([x])[0]
        )

        df_saxs = dfc[dfc["type_measurement"] == "SAXS"].copy()
        df_waxs = dfc[dfc["type_measurement"] == "WAXS"].copy()

        # Apply the scaler for SAXS data
        if not df_saxs.empty:
            matrix_2d_saxs = np.vstack(df_saxs["radial_profile_data_norm"].values)
            scaled_data_saxs = self.scalers["SAXS"].transform(matrix_2d_saxs)
            df_saxs["radial_profile_data_norm_scaled"] = [
                arr for arr in scaled_data_saxs
            ]

        # Apply the scaler for WAXS data
        if not df_waxs.empty:
            matrix_2d_waxs = np.vstack(df_waxs["radial_profile_data_norm"].values)
            scaled_data_waxs = self.scalers["WAXS"].transform(matrix_2d_waxs)
            df_waxs["radial_profile_data_norm_scaled"] = [
                arr for arr in scaled_data_waxs
            ]

        # Combine the processed DataFrames back into one
        dfc_processed = pd.concat([df_saxs, df_waxs])
        return dfc_processed.loc[raw_index]


class CurveFittingTransformer(TransformerMixin):
    """
    A scikit-learn compatible transformer for performing curve fitting on
    DataFrame columns using a specified function.

    This transformer allows for flexible curve fitting across multiple rows
    of a DataFrame, with customizable function, parameters, and fitting
    constraints.

    :param x_column: Name of the column containing x-values \
    (independent variable) to be used in curve fitting.
    :type x_column: str
    :param y_column: Name of the column containing y-values \
    (dependent variable) to be used in curve fitting.
    :type y_column: str
    :param producer: An instance of a class that produces the fitting \
    function, initial parameter estimates, and bounds for curve fitting.
    :type producer: CurveFittingProducer
    :param param_indices: Optional indices to select specific fitted \
    parameters.
    :type param_indices: list, optional
    :param cutoff_ranges: Optional ranges to assign high uncertainty \
    to specific data segments during fitting.
    :type cutoff_ranges: list of tuples, optional
    """

    def __init__(
        self,
        x_column,
        y_column,
        producer,
        cutoff_ranges=None,
    ):
        """
        Initialize the CurveFittingTransformer with specified fitting \
        parameters.

        Configures the transformer with column names, fitting function, initial
        parameter estimates, and optional constraints for curve fitting.

        :param x_column: Column name for x-values in input DataFrame.
        :type x_column: str
        :param y_column: Column name for y-values in input DataFrame.
        :type y_column: str
        :param producer: An instance of a class that produces the fitting \
        function, initial parameter estimates, and bounds for curve fitting.
        :type producer: CurveFittingProducer
        :param cutoff_ranges: Data segments to assign high uncertainty.
        :type cutoff_ranges: list of tuples, optional
        """
        self.x_column = x_column
        self.y_column = y_column
        self.producer = producer
        self.cutoff_ranges = cutoff_ranges

    def fit(self, X, y=None):
        """
        Placeholder method for scikit-learn transformer compatibility.

        This method does not perform actual fitting but is required for
        pipeline integration.

        :param X: Input DataFrame containing data to be transformed.
        :type X: pd.DataFrame
        :param y: Target values (ignored).
        :type y: None, optional
        :return: Configured transformer instance.
        :rtype: CurveFittingTransformer
        """
        return self

    def transform(self, X, y=None):
        """
        Apply curve fitting to each row of the input DataFrame.

        Performs curve fitting using the specified function on x and y columns.
        Adds new columns 'fit_params' and 'fitted_curve' with fitting results.

        :param X: Input DataFrame containing data for curve fitting.
        :type X: pd.DataFrame
        :param y: Target values (ignored).
        :type y: None, optional
        :return: DataFrame with added fitting results columns.
        :rtype: pd.DataFrame
        :raises RuntimeError: If curve fitting fails for any row.
        """
        X_copy = X.copy()
        # Create DF columns to store data
        X_copy["fit_params_all"] = None
        X_copy["fitted_curve"] = None
        X_copy["fit_cond"] = None
        # Make columns store objects
        X_copy["fit_params_all"].astype(object)
        X_copy["fitted_curve"].astype(object)
        X_copy["fit_cond"].astype(object)

        x_value = X_copy.iloc[0][self.x_column]

        func = self.producer.produce_function()
        p0 = self.producer.initial_guess()
        bounds = self.producer.bounds()

        function_count = self.producer.get_function_count()
        function_param_counts = self.producer.get_function_param_counts()

        for i in range(function_count):
            X_copy[f"fit_params_{i}"] = None
            X_copy[f"fit_params_{i}"].astype(object)

        if self.cutoff_ranges:
            sigma = np.ones_like(x_value)
            for start, end in self.cutoff_ranges:
                mask = (x_value > start) & (x_value < end)
                sigma[mask] = 1e6  # Assign large sigma to ignore these points
        else:
            sigma = None

        # Apply curve fitting for each row
        for index, row in X_copy.iterrows():
            x_values = np.array(row[self.x_column])
            y_values = np.array(row[self.y_column])

            try:
                # Perform curve fitting
                popt, pcov = curve_fit(
                    func,
                    x_values,
                    y_values,
                    p0=p0,
                    bounds=bounds,
                    sigma=sigma,
                )

                # Store fit results in new columns
                X_copy.at[index, "fit_cond"] = np.linalg.cond(pcov)
                X_copy.at[index, "fit_params_all"] = popt
                for i in range(len(function_param_counts)):
                    start_idx = sum(function_param_counts[:i])
                    end_idx = start_idx + function_param_counts[i]
                    X_copy.at[index, f"fit_params_{i}"] = popt[start_idx:end_idx]
                X_copy.at[index, "fitted_curve"] = func(x_values, *popt)

            except RuntimeError as e:
                print(f"Fit failed for index {index}: {e}")
                X_copy.at[index, "fit_params_all"] = None
                X_copy.at[index, "fitted_curve"] = None

        X_copy = X_copy.dropna(subset=["fit_params_all"])
        return X_copy


class MeasurementCutter(TransformerMixin):
    """
    Transformer class to cut measurements based on a specified column.

    :param column: The name of the column containing arrays to be cut.
    :type column: str
    :param cut_size: The size to which the arrays should be cut.
    :type cut_size: int
    """

    def __init__(self, column, distances):
        """
        Initializes the MeasurementCutter with the specified column name and
        distance-based cut criteria.

        :param column: The name of the column containing arrays of measurements to cut.
        :type column: str

        :param distances: A list of distance thresholds for each cut segment.
                          Use `None` to indicate no bound for a segment.
                          Example: [None, 120] means the first cut has no lower limit,
                            while the second starts from 120.
        :type distances: list[tuples(float or None, float or None)]

        :note: The `distances` parameter should be a list of tuples,
               where each tuple contains two values:
               - min_distances: Minimum distance for the cut segment (can be None).
               - max_distances: Maximum distance for the cut segment (can be None).
               If both values are None, the segment is not cut.
               Example: [(None, 120), (120, None)] means the first segment is
               cut from the start to 120, and the second segment starts from 120
               and goes to the end of the array.
        :note: If the `distances` list is empty, no cutting is performed.
        :note: If the `distances` list contains only one tuple with both values as
               None, the entire array is returned without cutting.
        """
        self.column = column
        self.distances = distances

    def fit(self, X, y=None):
        """
        Fit method for the transformer. No action is taken during fitting.

        :param X: The input DataFrame.
        :type X: pandas.DataFrame
        :param y: Target values (optional, not used in this transformer).
        :type y: array-like, optional
        :return: The fitted transformer (self).
        :rtype: MeasurementCutter
        """
        return self

    def transform(self, X, y=None):
        """
        Transform the input DataFrame by cutting measurements based on specified distances.
        """
        X_copy = X.copy()

        X_copy.dropna(subset=["ponifile"], inplace=True)

        X_copy[self.column] = X_copy.apply(
            lambda row: filter_points_by_distance(row, self.column, self.distances),
            axis=1,
        )
        return X_copy


class ImageResizer(TransformerMixin):
    """
    Transformer class to resize images in a specified column of a DataFrame.

    :param column: The name of the column containing images to be resized.
    :type column: str
    :param ref_distance: Reference distance for resizing images in mm.
    :type ref_distance: float
    """

    def __init__(self, column, ref_distance):
        self.column = column
        self.ref_distance = ref_distance

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        # Resize images based on the reference distance
        X_copy[[self.column, "ponifile"]] = X_copy.apply(
            lambda row: pd.Series(resize_image(row, self.column, self.ref_distance)),
            axis=1,
        )
        return X_copy


class CommonRegionCutter(TransformerMixin):
    """
    Transformer class to cut common square regions from images in a specified column
    of a DataFrame.

    :param column: The name of the column containing images to be cut.
    :type column: str
    """

    def __init__(self, column, square=False):
        self.column = column
        self.square = square

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        max_up, max_down, max_left, max_right = find_common_region(
            X_copy, self.column, self.square
        )

        X_copy[[self.column, "ponifile"]] = X_copy.apply(
            lambda row: pd.Series(
                cut_common_region(
                    row, self.column, max_up, max_down, max_left, max_right
                )
            ),
            axis=1,
        )

        return X_copy


class HankelTransformer(TransformerMixin):
    """
    Transformer class to compute Hankel transforms of images stored in a DataFrame column.

    Parameters
    ----------
    column : str
        Name of the column containing the images (2D arrays) to transform.
    start_radius : int, optional (default=0)
        Start index for radial cropping on the second axis.
    order : int, optional (default=0)
        Order of the Hankel transform.
    output_column : str, optional (default='H_full')
        Name of the column to store the Hankel-transformed results.
    """

    def __init__(self, column, start_radius=0, order=0):
        self.column = column
        self.start_radius = start_radius
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.column not in X.columns or "q_range" not in X.columns:
            raise ValueError("DataFrame must contain the required columns.")

        X_copy = X.copy()

        X_copy["hankel"] = None
        X_copy["hankel"].astype(object)

        for i, row in X_copy.iterrows():
            polar_img = row[self.column].copy().astype(float)[:, self.start_radius :]
            r = row["q_range"][self.start_radius :]
            polar_img = np.nan_to_num(polar_img, nan=0.0)

            _, n_radial = polar_img.shape
            R = r.max()

            transformer = HankelTransform(
                order=self.order, max_radius=R, n_points=n_radial
            )
            H = transformer.qdht(polar_img, axis=1)

            X_copy.at[i, "hankel"] = H

        return X_copy


class DetectorJoiner(TransformerMixin, BaseEstimator):
    """
    scikit-learn compatible wrapper around join_detectors.

    Automatically picks 'meas_name' or 'cal_name' for detector suffix parsing
    unless name_field is explicitly provided.

    Supports flexible interpolation_q_range parameter:
    - Single tuple for all measurement types: (0.01, 3.0)
    - Dictionary for per-type ranges: {'SAXS': (0.01, 1.5), 'WAXS': (1.0, 5.0)}
    """

    def __init__(
        self,
        name_field: Optional[str] = None,
        faulty_pixels=None,
        npt: int = 200,
        angles: int = 180,
        type_rules: Optional[
            Dict[str, Tuple[Union[str, float], Union[str, float, List, Tuple]]]
        ] = None,
        calibration_mode: str = "poni",
        interpolation_q_range: Optional[
            Union[Tuple[float, float], Dict[str, Tuple[float, float]]]
        ] = None,
        debug: bool = False,
        print_stats: bool = True,
    ) -> None:
        self.name_field = name_field
        self.faulty_pixels = faulty_pixels
        self.npt = npt
        self.angles = angles
        self.type_rules = type_rules
        self.calibration_mode = calibration_mode
        self.interpolation_q_range = interpolation_q_range
        self.debug = debug
        self.print_stats = print_stats
        self.stats: Optional[dict] = None

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Import join_detectors here to avoid circular imports
        from xrdanalysis.data_processing.detector_joining import join_detectors

        df_out = join_detectors(
            X,
            name_field=self.name_field,
            faulty_pixels=self.faulty_pixels,
            npt=self.npt,
            angles=self.angles,
            type_rules=self.type_rules,
            calibration_mode=self.calibration_mode,
            interpolation_q_range=self.interpolation_q_range,
            debug=self.debug,
        )
        # capture stats if available
        try:
            self.stats = getattr(df_out, "attrs", {}).get("join_stats")
        except Exception:
            self.stats = None

        # Always print a stats summary if requested
        if self.print_stats and self.stats:
            s = self.stats
            print(
                "[DetectorJoiner stats] "
                f"groups_total={s.get('groups_total', 0)}, "
                f"merged={s.get('groups_merged', 0)}, "
                f"single={s.get('groups_single', 0)}, "
                f"detector_only={s.get('groups_detector_only', 0)}, "
                f"skipped_multiple={s.get('groups_skipped_multiple', 0)}, "
                f"rows_removed_skipped_groups={s.get('rows_removed_skipped_groups', 0)}, "
                f"rows_dropped_secondary={s.get('rows_dropped_secondary', 0)}, "
                f"final_rows={s.get('final_rows', len(df_out))}"
            )

        return df_out


class SpecimenStatusToSoftLabels(TransformerMixin):
    """
    Derive soft-label distributions from status + multiple rule columns.

    Rules are provided as a mapping of (rule_col_name, rule_value, status_upper) -> vector,
    aligned to class_order. This allows flexible conditioning on any number of rule columns.

    Example (class_order: ['CANCER','BENIGN','NORMAL']):
        ('biopsy', True,  'CANCER') -> [0.89, 0.09, 0.02]
        ('biopsy', True,  'BENIGN') -> [0.09, 0.89, 0.02]
        ('biopsy', False, 'NORMAL') -> [0.10, 0.30, 0.60]
        ('biopsy', False, 'BENIGN') -> [0.20, 0.60, 0.20]
        ('biopsy', False, 'CANCER') -> [0.60, 0.30, 0.10]
    Optionally, you can also define global defaults via keys like (None, None, 'CANCER').

    Parameters
    ----------
    status_col : str
        Column containing status labels (e.g., 'CANCER', 'BENIGN', 'NORMAL').
    rule_cols : list[str]
        Columns to consider when matching rules (checked in order).
    output_col : str
        Output column to write the soft-label vector (list of floats).
    class_order : list[str]
        Order of classes in the output vector.
    rules : dict[(str, Any, str) -> list[float]] | None
        Mapping from (rule_col_name, rule_value, status_upper) to probability vector.
    normalize : bool
        If True, re-normalize vectors to sum to 1.0.
    strict : bool
        If True, raise if no rule matches; otherwise fill with NaNs.
    capitalize_status : bool
        If True, uppercase status strings before matching.
    """

    def __init__(
        self,
        status_col: str = "specimen_status",
        rule_cols: Optional[List[str]] = None,
        output_col: str = "cancer_status_soft",
        class_order: Optional[List[Union[str, int]]] = None,
        rules: Optional[Dict[Tuple[Optional[str], Any, str], List[float]]] = None,
        normalize: bool = True,
        strict: bool = False,
        capitalize_status: bool = True,
    ) -> None:
        self.status_col = status_col
        self.rule_cols = list(rule_cols) if rule_cols is not None else ["biopsy"]
        self.output_col = output_col
        self.class_order = (
            list(class_order)
            if class_order is not None
            else ["CANCER", "BENIGN", "NORMAL"]
        )
        self.normalize = bool(normalize)
        self.strict = bool(strict)
        self.capitalize_status = bool(capitalize_status)

        # Default rules rewritten to the new format (using 'biopsy' as a rule column)
        default_rules = {
            ("biopsy", True, "CANCER"): [0.89, 0.09, 0.02],
            ("biopsy", True, "BENIGN"): [0.09, 0.89, 0.02],
            ("biopsy", False, "NORMAL"): [0.10, 0.30, 0.60],
            ("biopsy", False, "BENIGN"): [0.20, 0.60, 0.20],
            ("biopsy", False, "CANCER"): [0.60, 0.30, 0.10],
            # Global defaults (optional fallbacks)
            (None, None, "NORMAL"): [0.10, 0.30, 0.60],
            (None, None, "BENIGN"): [0.20, 0.60, 0.20],
            (None, None, "CANCER"): [0.60, 0.30, 0.10],
        }
        self.rules = rules if rules is not None else default_rules

        # Validate vector lengths
        k = len(self.class_order)
        for key, vec in list(self.rules.items()):
            if not isinstance(vec, (list, tuple, np.ndarray)):
                raise ValueError(
                    f"Rule for {key} must be a vector-like; got {type(vec)}"
                )
            if len(vec) != k:
                raise ValueError(
                    f"Rule for {key} length {len(vec)} != len(class_order) {k}"
                )

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def _to_bool(self, v) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (int, np.integer)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "1", "yes", "y", "t"}:
                return True
            if s in {"false", "0", "no", "n", "f"}:
                return False
        try:
            return bool(v)
        except Exception:
            return None

    def _canon_status(self, s) -> Optional[str]:
        if s is None:
            return None
        if isinstance(s, str):
            s2 = s.strip().upper() if self.capitalize_status else s
            if s2 == "MALIGNANT":
                s2 = "CANCER"
            return s2
        try:
            return str(s).upper() if self.capitalize_status else str(s)
        except Exception:
            return None

    def _norm_vec(self, v: List[float]) -> List[float]:
        arr = np.asarray(v, dtype=float)
        arr = np.clip(arr, 0.0, None)
        if self.normalize:
            s = float(arr.sum())
            if s > 0:
                arr = arr / s
        return arr.astype(float).tolist()

    def _key_variants(self, col: str, val, status: str):
        keys = []
        # Direct
        keys.append((col, val, status))
        # Uppercase string variant
        try:
            if isinstance(val, str):
                keys.append((col, val.strip().upper(), status))
        except Exception:
            pass
        # Boolean-canonicalized variant
        b = self._to_bool(val)
        if b is not None:
            keys.append((col, b, status))
        # Column-specific default
        keys.append((col, None, status))
        return keys

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        if self.status_col not in X.columns:
            raise KeyError(f"Status column '{self.status_col}' not found in DataFrame")
        # Ensure rule columns exist
        missing = [c for c in self.rule_cols if c not in X.columns]
        if missing:
            raise KeyError(f"Rule columns not found in DataFrame: {missing}")

        results: List[Optional[List[float]]] = []
        for _, row in X.iterrows():
            st = self._canon_status(row[self.status_col])
            vec = None
            if st is not None:
                # Try rule columns in order
                for col in self.rule_cols:
                    rv = row[col]
                    for key in self._key_variants(col, rv, st):
                        v = self.rules.get(key)
                        if v is not None:
                            vec = self._norm_vec(v)
                            break
                    if vec is not None:
                        break
                # Global defaults
                if vec is None:
                    v = self.rules.get((None, None, st)) or self.rules.get((None, st))  # type: ignore
                    if v is not None:
                        vec = self._norm_vec(v)
            if vec is None:
                if self.strict:
                    raise KeyError(
                        f"No soft-label rule matched for status={st} using columns {self.rule_cols}"
                    )
                vec = [np.nan] * len(self.class_order)
            results.append(vec)

        X[self.output_col] = results
        return X


class SoftLabelToWeightedSamples(TransformerMixin):
    """
    Expand soft-label distributions into duplicated rows with a hard label and a weight.

    Typical use: given a column like 'cancer_status_soft' that holds a length-K
    probability vector per row (e.g. [0.7, 0.2, 0.1]), this transformer creates up to K
    duplicated rows per original row. Duplicates carry:
      - a hard label (label_col) set to class index or provided class name
      - a weight (weight_col) equal to the corresponding probability value
      - optionally, a numeric label column (label_col_numeric) via a code map or aligned code list

    Parameters
    ----------
    soft_col : str
        Name of the column containing the soft-label vector (list/np.ndarray or JSON-like string).
    label_col : str
        Name of the output hard-label column to set on duplicates.
    weight_col : str
        Name of the output weight column to set on duplicates (to be used as sample_weight).
    class_names : list[str] | None
        Optional list of class names to assign instead of integer indices. Length must match K.
    label_col_numeric : str | None
        Optional additional column to store numeric codes (e.g., 'cancer_status_multi').
    label_codes : list[Any] | None
        Optional list of codes aligned with class_names by index (same length as class_names).
    label_code_map : dict[Any, Any] | None
        Optional mapping from label value (e.g., 'CANCER' or index j) to numeric code.
    min_weight : float
        Discard duplicates with probability <= min_weight. Default 0.0 keeps all.
    normalize : bool
        If True, re-normalize probabilities to sum to 1.0 per row when they are positive.
    drop_soft_col : bool
        If True, drop the soft_col from the output.

    Notes
    -----
    - Use class_weight=None in your estimator, since weights are embedded via sample_weight.
    - For LightGBM/XGBoost multiclass, fit with sample_weight=output[weight_col].values.
    - Group-wise CV like GroupKFold should be applied BEFORE duplication to avoid leakage.
    """

    def __init__(
        self,
        soft_col: str = "cancer_status_soft",
        label_col: str = "cancer_status",
        weight_col: str = "cancer_status_weighted",
        class_names: Optional[List[Union[str, int]]] = None,
        label_col_numeric: Optional[str] = None,
        label_codes: Optional[List[Any]] = None,
        label_code_map: Optional[Dict[Any, Any]] = None,
        min_weight: float = 0.0,
        normalize: bool = True,
        drop_soft_col: bool = False,
    ) -> None:
        self.soft_col = soft_col
        self.label_col = label_col
        self.weight_col = weight_col
        self.class_names = class_names
        self.label_col_numeric = label_col_numeric
        self.label_codes = label_codes
        self.label_code_map = label_code_map
        self.min_weight = float(min_weight)
        self.normalize = bool(normalize)
        self.drop_soft_col = bool(drop_soft_col)

        # Validate label_codes alignment when both provided
        if self.label_codes is not None and self.class_names is not None:
            if len(self.label_codes) != len(self.class_names):
                raise ValueError(
                    f"label_codes length {len(self.label_codes)} must equal class_names length {len(self.class_names)}"
                )

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def _to_prob_list(self, val) -> Optional[List[float]]:
        # Accept list/ndarray/tuple, or JSON/py-literal strings like "[0.7,0.2,0.1]"
        if val is None:
            return None
        if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(val, dtype=float)
        elif isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
            except Exception:
                # Fallback: try comma-split

                try:
                    parsed = [
                        float(x)
                        for x in val.strip().strip("[]()").split(",")
                        if x != ""
                    ]
                except Exception:
                    return None
            arr = np.asarray(parsed, dtype=float)
        else:
            try:
                arr = np.asarray(val, dtype=float)
            except Exception:
                return None

        if arr.ndim != 1 or arr.size == 0:
            return None
        # Ensure non-negative (clip tiny negatives from numeric issues)
        arr = np.clip(arr, 0.0, None)
        if self.normalize:
            s = float(arr.sum())
            if s > 0:
                arr = arr / s
        return arr.astype(float).tolist()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.soft_col not in df.columns:
            raise KeyError(
                f"Soft-label column '{self.soft_col}' not found in DataFrame"
            )

        rows = []
        # Iterate per source row and expand
        for idx, row in df.iterrows():
            probs = self._to_prob_list(row[self.soft_col])
            if probs is None:
                continue  # skip rows without valid soft-labels

            k = len(probs)
            # If class_names provided, validate length
            if self.class_names is not None and len(self.class_names) != k:
                raise ValueError(
                    f"class_names length ({len(self.class_names)}) does not match soft vector length ({k})"
                )

            for j, w in enumerate(probs):
                if w <= self.min_weight:
                    continue
                new_row = row.copy()
                new_row[self.weight_col] = float(w)
                label_value = self.class_names[j] if self.class_names is not None else j
                new_row[self.label_col] = label_value

                # Optional numeric label column
                if self.label_col_numeric is not None:
                    code = None
                    # Priority 1: explicit map by label value, then by index j
                    if self.label_code_map is not None:
                        code = self.label_code_map.get(label_value, None)
                        if code is None:
                            code = self.label_code_map.get(j, None)
                    # Priority 2: aligned codes list
                    if (
                        code is None
                        and self.label_codes is not None
                        and self.class_names is not None
                    ):
                        code = self.label_codes[j]
                    # Fallback: use index j
                    if code is None:
                        code = j
                    new_row[self.label_col_numeric] = code

                rows.append(new_row)

        if not rows:
            # Return empty dataframe with the new columns present for consistency
            out_cols = list(df.columns)
            if self.weight_col not in out_cols:
                out_cols.append(self.weight_col)
            if self.label_col not in out_cols:
                out_cols.append(self.label_col)
            return pd.DataFrame(columns=out_cols)

        out = pd.DataFrame(rows)
        if self.drop_soft_col and self.soft_col in out.columns:
            out = out.drop(columns=[self.soft_col])
        out.reset_index(drop=True, inplace=True)
        return out


class SNRTransformer(TransformerMixin, BaseEstimator):
    """
    Compute signal-to-noise metrics from 1D azimuthal integration results.

    For each row, this transformer:
    - optionally re-interpolates (q, I) to a common, uniformly spaced q-grid
    - normalizes the intensity by its area (fallback to median scaling)
    - smooths the normalized intensity using Savitzky–Golay (fallback to moving average)
    - computes SNR either from residuals (legacy) or from Poisson sigma
    - residual mode: residual = I_norm - I_smooth, snr_linear = var(I_smooth)/var(residual)
    - poisson mode: uses radial_profile_sigma from pyFAI, pointwise SNR(q)=I(q)/sigma(q)
      and aggregates to scalar SNR as RMS(SNR(q))
    - writes a denoised 1D profile to `radial_profile_data_snr` (smoothed in original scale)
    - writes scalar SNR in dB to column `snr`

    Parameters
    ----------
    x_column : str
        Column with the q-range array. Defaults to 'q_range'.
    y_column : str
        Column with the 1D intensity array. Defaults to 'radial_profile_data'.
    window_frac : float
        Fraction of the number of points to set SavGol window length. Default 0.04.
    polyorder : int
        Polynomial order for SavGol. Default 2.
    enforce_common_q : bool
        If True, re-interpolate to a uniformly spaced q grid per row. Default True.
    n_points : int | None
        If set, number of points for the uniform grid. Defaults to len(I) when None.
    save_smoothed : bool
        If True, saves smoothed and residual arrays in the DataFrame.
    smoothed_col : str
        Column name for smoothed intensity when saved. Defaults to 'radial_profile_data_snr'.
    residual_col : str
        Column name for residual intensity when saved. Defaults to 'radial_profile_residual'.
    snr_col : str
        Column name to write SNR in dB. Defaults to 'snr'.
    sigma_column : str
        Column with pyFAI sigma values. Defaults to 'radial_profile_sigma'.
    snr_method : str
        SNR method: 'residual', 'poisson', or 'auto'. 'auto' tries Poisson first,
        then falls back to residual.
    regrid_poisson : bool
        If True, calculate Poisson scalar metrics on the historical uniform q-grid.
        New objects default to native aligned intensity/sigma samples. Restored
        objects without this attribute retain the historical regridded path.
    """

    def __init__(
        self,
        x_column: str = "q_range",
        y_column: str = "radial_profile_data",
        window_frac: float = 0.04,
        polyorder: int = 2,
        enforce_common_q: bool = True,
        n_points: int = None,
        save_smoothed: bool = True,
        smoothed_col: str = "radial_profile_data_snr",
        residual_col: str = "radial_profile_residual",
        snr_col: str = "snr",
        sigma_column: str = "radial_profile_sigma",
        snr_method: str = "residual",
        regrid_poisson: bool = False,
    ) -> None:
        self.x_column = x_column
        self.y_column = y_column
        self.window_frac = float(window_frac)
        self.polyorder = int(polyorder)
        self.enforce_common_q = bool(enforce_common_q)
        self.n_points = n_points
        self.save_smoothed = bool(save_smoothed)
        self.smoothed_col = smoothed_col
        self.residual_col = residual_col
        self.snr_col = snr_col
        self.sigma_column = sigma_column
        self.snr_method = str(snr_method).strip().lower()
        self.regrid_poisson = bool(regrid_poisson)
        if self.snr_method not in {"residual", "poisson", "auto"}:
            raise ValueError(
                "snr_method must be one of: 'residual', 'poisson', 'auto'."
            )

        # Optional Savitzky–Golay import
        try:
            from scipy.signal import savgol_filter as _sg
        except Exception:
            _sg = None
        self._savgol = _sg

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def __sklearn_clone__(self):
        """Clone normalized constructor state without changing legacy attributes."""
        return type(self)(**self.get_params(deep=False))

    def __setstate__(self, state):
        """Backfill native-sampling configuration when loading legacy artifacts."""
        restored_state = dict(state)
        restored_state.setdefault("regrid_poisson", True)
        super().__setstate__(restored_state)

    def _ensure_uniform_grid(
        self, q: np.ndarray, intensity: np.ndarray, n_points: int | None
    ):
        return ensure_uniform_grid(q, intensity, n_points)

    def _normalize_by_surface(
        self, q: np.ndarray, intensity: np.ndarray, eps: float = 1e-12
    ):
        return normalize_by_surface(q, intensity, _trapz_compat, eps)

    def _smooth(self, y: np.ndarray):
        return smooth_signal(y, self.window_frac, self.polyorder, self._savgol)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Prepare output columns
        df["noise_std"] = np.nan
        df["snr_linear"] = np.nan
        df["snr_db"] = np.nan
        df[self.snr_col] = np.nan  # alias for snr in dB as requested
        df["snr_method_used"] = None
        if self.save_smoothed:
            df[self.smoothed_col] = None
            df[self.residual_col] = None
            df[self.smoothed_col].astype(object)
            df[self.residual_col].astype(object)

        for i, row in df.iterrows():
            # Older joblib objects predate ``regrid_poisson``.  They must retain
            # the historical interpolated Poisson calculation after loading.
            regrid_poisson = bool(getattr(self, "regrid_poisson", True))
            row_result = calculate_snr_row(
                q_raw=row.get(self.x_column),
                intensity_raw=row.get(self.y_column),
                sigma_raw=row.get(self.sigma_column),
                snr_method=self.snr_method,
                enforce_common_q=self.enforce_common_q,
                regrid_poisson=regrid_poisson,
                n_points=self.n_points,
                uniform_grid=self._ensure_uniform_grid,
                normalize=self._normalize_by_surface,
                smooth=self._smooth,
            )
            if row_result is None:
                continue

            df.at[i, "noise_std"] = row_result.metrics.noise_std
            df.at[i, "snr_linear"] = row_result.metrics.snr_linear
            df.at[i, "snr_db"] = row_result.metrics.snr_db
            df.at[i, self.snr_col] = row_result.metrics.snr_db  # requested alias
            df.at[i, "snr_method_used"] = row_result.metrics.method_used
            if self.save_smoothed:
                df.at[i, self.smoothed_col] = row_result.smoothed
                df.at[i, self.residual_col] = row_result.residual

        return df


# Re-export spectrokinetic transformers from the canonical module so existing
# imports from this file can consume the new classes without path changes.
from xrdanalysis.data_processing.spectrokinetic_transformers import (  # noqa: E402
    MCRALSTransformer,
    SpectroSVDTransformer,
)
