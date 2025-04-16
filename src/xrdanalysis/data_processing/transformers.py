"""
The transformer classes are stored here
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Normalizer, StandardScaler

from xrdanalysis.data_processing.azimuthal_integration import (
    calculate_deviation,
    calculate_deviation_cake,
    perform_azimuthal_integration,
)
from xrdanalysis.data_processing.containers import Limits, Rule, RuleQ
from xrdanalysis.data_processing.fourier import (
    fourier_custom,
    fourier_fft,
    fourier_fft2,
    slope_removal,
    slope_removal_custom,
)
from xrdanalysis.data_processing.utility_functions import (
    create_mask,
    unpack_results,
    unpack_results_cake,
    unpack_rotating_angles_results,
)


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
    :param angles: List of angle ranges for integration. Defaults to None.
    :type angles: List[Tuple[int]], optional
    """

    max_iter: int = 5
    thres: int = 3
    column: str = "measurement_data"
    faulty_pixels: Tuple[int] = None
    mask: List[List[int]] = None
    npt: int = 256
    integration_mode: str = "1D"
    calibration_mode: str = "dataframe"
    thickness_adjustment: bool = False
    thickness_adjustment_distance: float = 700
    calc_cake_stats: bool = False
    angles: List[Tuple[int]] = None

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
        integration results, including 'q_range', 'radial_profile_data', \
        and other mode-specific columns.
        :rtype: pandas.DataFrame
        :raises: Drops rows with missing calibration data if calibration_mode \
        is 'poni'.
        """

        x_copy = x.copy()

        # Mark the faulty pixels in the mask
        if self.mask is not None:
            mask = self.mask
        else:
            mask = create_mask(self.faulty_pixels)

        if self.calibration_mode == "poni":
            x_copy.dropna(subset=["ponifile"], inplace=True)

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
                thres=self.thres,
                max_iter=self.max_iter,
                calc_cake_stats=self.calc_cake_stats,
                angles=self.angles,
            ),
            axis=1,
        )

        if self.integration_mode in ["1D", "sigma_clip"]:
            # Extract q_range and profile arrays from the integration_results
            x_copy[
                [
                    "q_range",
                    "radial_profile_data",
                    "radial_sem",
                    "radial_std",
                    "calculated_distance",
                    "center_x",
                    "center_y",
                ]
            ] = integration_results.apply(
                lambda x: pd.Series([x[0], x[1], x[2], x[3], x[4], x[5], x[6]])
            )
        elif self.integration_mode == "rotating_angles":
            expanded_results = integration_results.apply(
                unpack_rotating_angles_results
            )
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
            x_copy[
                [
                    "q_range",
                    "radial_profile_data",
                    "azimuthal_positions",
                    "calculated_distance",
                    "center_x",
                    "center_y",
                    "cake_col_mean",
                    "cake_col_variance",
                    "cake_col_std",
                    "cake_col_skew",
                    "cake_col_kurtosis",
                ]
            ] = integration_results.apply(
                lambda x: pd.Series(
                    [
                        x[0],
                        x[1],
                        x[2],
                        x[3],
                        x[4],
                        x[5],
                        x[6],
                        x[7],
                        x[8],
                        x[9],
                        x[10],
                    ]
                )
            )

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
            calculate_deviation_cake
            if self.mode == "cake"
            else calculate_deviation
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
    :param norm: The type of norm to use for normalization \
    ('l1', 'l2', or 'max'). Defaults to 'l1'.
    :type norm: str
    """

    def __init__(self, column, norm="l1", mode="1D"):
        """
        Initializes the ColumnNormalizer with the specified column name and
        normalization method.

        :param column: The name of the column containing arrays to normalize.
        :type column: str
        :param norm: The type of norm to use for normalization. Can be 'l1', \
        'l2', or 'max'. Defaults to 'l2'.
        :type norm: str
        """
        self.column = column
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
        if self.mode == "1D":
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
        flattened_data = X_copy.apply(
            lambda row: self._flatten_row(row), axis=1
        )

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
                X_copy = X_copy[
                    X_copy.apply(lambda row: clean_q(row, rule), axis=1)
                ]
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
                dfc["type_measurement"] = dfc[
                    "calibration_manual_distance"
                ].apply(lambda d: "WAXS" if d < 50 else "SAXS")
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
                X[column] = X[column].apply(
                    lambda x: slope_removal_custom(x)[0]
                )
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
                ] = X[column].apply(
                    lambda x: pd.Series(fourier_func(x, self.order))
                )
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
        return (
            [self.features]
            if isinstance(self.features, str)
            else self.features
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
                dfc = dfc.dropna(subset="ponifile")
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

    def __init__(
        self, scalers: Dict[str, StandardScaler] = None, name="Scaler"
    ):
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
            matrix_2d_saxs = np.vstack(
                df_saxs["radial_profile_data_norm"].values
            )
            scaler_saxs.fit(matrix_2d_saxs)
            self.scalers["SAXS"] = scaler_saxs

        # Apply the scaler for WAXS data
        if not df_waxs.empty:
            scaler_waxs = StandardScaler()
            matrix_2d_waxs = np.vstack(
                df_waxs["radial_profile_data_norm"].values
            )
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
            matrix_2d_saxs = np.vstack(
                df_saxs["radial_profile_data_norm"].values
            )
            scaled_data_saxs = self.scalers["SAXS"].transform(matrix_2d_saxs)
            df_saxs["radial_profile_data_norm_scaled"] = [
                arr for arr in scaled_data_saxs
            ]

        # Apply the scaler for WAXS data
        if not df_waxs.empty:
            matrix_2d_waxs = np.vstack(
                df_waxs["radial_profile_data_norm"].values
            )
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
    :param func: The mathematical function to fit. Must accept x as the first \
    argument, followed by parameters to be estimated.
    :type func: callable
    :param p0: Initial parameter guesses for the curve fitting algorithm.
    :type p0: list or array-like
    :param bounds: Parameter boundaries for constrained optimization. Defaults\
    to unconstrained (-∞, +∞) bounds.
    :type bounds: tuple, optional
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
        func,
        p0,
        bounds=(-np.inf, np.inf),
        param_indices=None,
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
        :param func: Callable function to be used for curve fitting.
        :type func: callable
        :param p0: Initial parameter guesses for curve fitting.
        :type p0: list
        :param bounds: Parameter boundaries for curve fitting. Defaults to \
        unconstrained bounds.
        :type bounds: tuple, optional
        :param param_indices: Indices of parameters to retain after fitting.
        :type param_indices: list, optional
        :param cutoff_ranges: Data segments to assign high uncertainty.
        :type cutoff_ranges: list of tuples, optional
        """
        self.x_column = x_column
        self.y_column = y_column
        self.func = func
        self.p0 = p0
        self.bounds = bounds
        self.param_indices = param_indices
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
        X_copy["fit_params"] = None
        X_copy["fitted_curve"] = None
        # Make columns store objects
        X_copy["fit_params"].astype(object)
        X_copy["fitted_curve"].astype(object)

        x_value = X_copy.iloc[0][self.x_column]

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
                popt, _ = curve_fit(
                    self.func,
                    x_values,
                    y_values,
                    p0=self.p0,
                    bounds=self.bounds,
                    sigma=sigma,
                )

                selected_params = (
                    popt
                    if self.param_indices is None
                    else popt[np.array(self.param_indices)]
                )

                # Store fit results in new columns
                X_copy.at[index, "fit_params"] = selected_params
                X_copy.at[index, "fitted_curve"] = self.func(x_values, *popt)

            except RuntimeError as e:
                print(f"Fit failed for index {index}: {e}")
                X_copy.at[index, "fit_params"] = None
                X_copy.at[index, "fitted_curve"] = None

        X_copy = X_copy.dropna(subset=["fit_params"])
        return X_copy
