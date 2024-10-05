"""
The transformer classes are stored here
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Normalizer, StandardScaler

from xrdanalysis.data_processing.azimuthal_integration import (
    perform_azimuthal_integration,
)
from xrdanalysis.data_processing.containers import Limits, Rule, RuleQ
from xrdanalysis.data_processing.utility_functions import (
    create_mask,
    generate_poni,
)


@dataclass
class AzimuthalIntegration(TransformerMixin):
    """
    Transformer class for azimuthal integration to be used in
    an sklearn pipeline.

    :param faulty_pixels: A tuple containing the coordinates of faulty pixels.
    :type faulty_pixels: Tuple[int]
    :param npt: The number of points for azimuthal integration. Defaults to\
        256.
    :type npt: int
    :param integration_mode: The integration mode, either "1D" or "2D".\
        Defaults to "1D".
    :type integration_mode: str
    :param transformation_mode: The transformation mode, either 'dataframe'\
        (returns a DataFrame for further analysis) or 'pipeline'\
        (to use in an sklearn pipeline). Defaults to 'dataframe'.
    :type transformation_mode: str
    :param calibration_mode: Mode of calibration. 'dataframe' is used when\
        calibration values are columns in the DataFrame, 'poni' is\
        used when calibration is in a poni file.
    :type calibration_mode: str
    :param poni_dir_path: Directory path where .poni files for the rows will\
        be saved.
    :type poni_dir_path: str
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
        Applies azimuthal integration to each row of the DataFrame and adds
        the result as a new column.

        :param X: The data to transform. Must contain 'measurement_data' and\
            'center' columns.
        :type X: pandas.DataFrame
        :return: A copy of the input DataFrame with an additional\
            'radial_profile_data' column containing the results of the\
            azimuthal integration, and optionally 'q_range' and\
            'azimuthal_positions' columns for 2D integration.
        :rtype: pandas.DataFrame
        """

        x_copy = x.copy()

        # Mark the faulty pixels in the mask
        mask = create_mask(self.faulty_pixels)

        directory_path = None
        if self.calibration_mode == "poni":
            directory_path = generate_poni(x_copy, self.poni_dir_path)
            x_copy.dropna(subset="ponifile", inplace=True)

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
            x_copy[
                ["q_range", "radial_profile_data", "calculated_distance"]
            ] = integration_results.apply(
                lambda x: pd.Series([x[0], x[1], x[2]])
            )
        elif self.integration_mode == "2D":
            x_copy[
                [
                    "q_range",
                    "radial_profile_data",
                    "azimuthal_positions",
                    "calculated_distance",
                ]
            ] = integration_results.apply(
                lambda x: pd.Series([x[0], x[1], x[2], x[3]])
            )

        if self.transformation_mode == "pipeline":
            x_copy = pd.DataFrame(
                np.asarray(x_copy["radial_profile_data"].values.tolist()),
                index=x_copy.index,
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

    def __init__(self, column, norm="l1"):
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
        X_copy[self.column] = X_copy[self.column].apply(
            lambda arr: self.normalizer.transform([arr])[0]
        )
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
        in a row.

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
                flattened_list.extend(value)
            else:
                # Append single values
                flattened_list.append(value)
        return flattened_list


class ColumnCleaner(TransformerMixin):
    """
    Transformer class for cleaning specific columns according to Rules
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


class DataPreparation(TransformerMixin):
    """
    Transformer class to prepare a raw DataFrame according to the standard.
    """

    def __init__(
        self,
        columns=COLUMNS_DEF,
        limits: Limits = None,
        cleaning_rules: List[Rule] = None,
    ):
        self.columns = columns
        self.limits = limits
        self.cleaning_rules = cleaning_rules

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

        if self.cleaning_rules:
            dfc = self._clean(dfc)
        return dfc[self.columns]

    def _clean(self, df: pd.DataFrame):
        def clean(row, cleaning_rules: List[Rule]):
            res = []
            for r in cleaning_rules:
                r: Rule = r
                ret = False
                idx = np.argmin(np.abs(row["q_range"] - r.q_value))
                intensity = row["radial_profile_data"][idx]
                if r.lower is not None and r.upper is not None:
                    ret = (intensity > r.lower) and (intensity < r.upper)
                elif r.lower is not None:
                    ret = intensity > r.lower
                elif r.upper is not None:
                    ret = intensity < r.upper
                res.append(ret)
            return all(res)

        if self.cleaning_rules:
            return df[
                df.apply(clean, axis=1, cleaning_rules=self.cleaning_rules)
            ].copy()
        else:
            print("No cleaning was done, thera are no cleaning_rules")
            return df


class NormScaler(TransformerMixin):
    """
    Does normalization and scaling of the dataframe
    """

    def __init__(
        self, scalers: Dict[str, StandardScaler] = None, name="Scaler"
    ):
        self._name = name
        if scalers:
            self.scalers = scalers
        else:
            self.scalers = {}

    def fit(self, df: pd.DataFrame, y=None):
        print(f"NormScaler {self._name}: is doing fit.")
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
        print(f"NormScaler {self._name}: is doing transform.")
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
