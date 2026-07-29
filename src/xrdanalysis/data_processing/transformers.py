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

    def __init__(
        self, column, norm="l1", mode="1D", q_column="q_range", q_min=6.0, q_max=8.0
    ):
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


# isort: off
from xrdanalysis.data_processing._transformer_compat import (  # noqa: E402
    rebind_class_functions as _rebind_class_functions,
)

# Re-export private implementations under established public paths.  The
# canonical module metadata preserves sklearn/joblib and historical pickle
# compatibility without expanding this compatibility façade again.
from xrdanalysis.data_processing._transformer_dataframe import (  # noqa: E402
    ColumnCleaner as _ColumnCleaner,
    ColumnExtractor as _ColumnExtractor,
    DataPreparation as _DataPreparation,
    MeasurementCutter as _MeasurementCutter,
    QRangeSetter as _QRangeSetter,
)
from xrdanalysis.data_processing._transformer_goodness import (  # noqa: E402
    GoodnessFilter as _GoodnessFilter,
    GoodnessTransformer as _GoodnessTransformer,
)
from xrdanalysis.data_processing._transformer_profile import (  # noqa: E402
    ColumnStandardizer as _ColumnStandardizer,
    RuleBasedProfileFilter as _RuleBasedProfileFilter,
)
from xrdanalysis.data_processing._transformer_signal import (  # noqa: E402
    CommonRegionCutter as _CommonRegionCutter,
    CurveFittingTransformer as _CurveFittingTransformer,
    FourierTransform as _FourierTransform,
    HankelTransformer as _HankelTransformer,
    ImageResizer as _ImageResizer,
    NormScaler as _NormScaler,
    SlopeRemoval as _SlopeRemoval,
)
from xrdanalysis.data_processing._transformer_soft_labels import (  # noqa: E402
    SoftLabelToWeightedSamples as _SoftLabelToWeightedSamples,
    SpecimenStatusToSoftLabels as _SpecimenStatusToSoftLabels,
)

# isort: on

_EXTRACTED_TRANSFORMERS = {
    "ColumnStandardizer": _ColumnStandardizer,
    "RuleBasedProfileFilter": _RuleBasedProfileFilter,
    "ColumnExtractor": _ColumnExtractor,
    "ColumnCleaner": _ColumnCleaner,
    "QRangeSetter": _QRangeSetter,
    "SlopeRemoval": _SlopeRemoval,
    "FourierTransform": _FourierTransform,
    "GoodnessTransformer": _GoodnessTransformer,
    "GoodnessFilter": _GoodnessFilter,
    "DataPreparation": _DataPreparation,
    "NormScaler": _NormScaler,
    "CurveFittingTransformer": _CurveFittingTransformer,
    "MeasurementCutter": _MeasurementCutter,
    "ImageResizer": _ImageResizer,
    "CommonRegionCutter": _CommonRegionCutter,
    "HankelTransformer": _HankelTransformer,
    "SpecimenStatusToSoftLabels": _SpecimenStatusToSoftLabels,
    "SoftLabelToWeightedSamples": _SoftLabelToWeightedSamples,
}
for _transformer_name, _transformer_type in _EXTRACTED_TRANSFORMERS.items():
    _rebind_class_functions(_transformer_type, globals())
    _transformer_type.__module__ = __name__
    globals()[_transformer_name] = _transformer_type

# ``DataPreparation`` historically captured this public mutable list as its
# constructor default. Keep that identity for callers that customize the
# legacy ``COLUMNS_DEF`` object before constructing a transformer.
_DataPreparation.__init__.__defaults__ = (COLUMNS_DEF,)

# Legacy aliases intentionally remain available from this module.
from xrdanalysis.data_processing.spectrokinetic_transformers import (  # noqa: E402
    MCRALSTransformer,
    SpectroSVDTransformer,
)
