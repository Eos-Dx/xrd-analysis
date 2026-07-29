"""Private DataFrame-shape transformers re-exported by ``transformers``."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from xrdanalysis.data_processing.containers import Limits, Rule, RuleQ
from xrdanalysis.data_processing.utility_functions import filter_points_by_distance

_DEFAULT_COLUMNS = [
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


class ColumnExtractor(TransformerMixin):
    """Flatten selected scalar or array columns into an estimator matrix."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        flattened_data = X.copy().apply(lambda row: self._flatten_row(row), axis=1)
        return pd.DataFrame(
            np.asarray(flattened_data.values.tolist()), index=flattened_data.index
        )

    def _flatten_row(self, row):
        flattened_list = []
        for col in self.columns:
            value = row[col]
            if isinstance(value, (list, np.ndarray)):
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    flattened_list.extend(value.ravel())
                else:
                    flattened_list.extend(value)
            else:
                flattened_list.append(value)
        return flattened_list


class ColumnCleaner(TransformerMixin):
    """Filter rows according to configured :class:`RuleQ` instances."""

    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        def clean_q(row, rule: RuleQ):
            if row[rule.q_column_name][-1] < rule.q_value:
                return True
            index = np.argmin(np.abs(row[rule.q_column_name] - rule.q_value))
            intensity = row[rule.column_name][index]
            if rule.lower is not None and rule.upper is not None:
                return (intensity > rule.lower) and (intensity < rule.upper)
            if rule.lower is not None:
                return intensity > rule.lower
            if rule.upper is not None:
                return intensity < rule.upper
            return False

        for rule in self.rules:
            if not isinstance(rule, RuleQ):
                raise Exception(f"I do not know how to treat {type(rule)}.")
            X_copy = X_copy[X_copy.apply(lambda row: clean_q(row, rule), axis=1)]
        return X_copy


class QRangeSetter(TransformerMixin):
    """Set interpolation ranges from configured SAXS/WAXS limits."""

    def __init__(self, limits: Limits = None):
        self.limits = limits

    def fit(self, x: pd.DataFrame, y=None):
        _ = x
        _ = y
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dfc = df.copy()
        if self.limits:
            limits_waxs = (self.limits.q_min_waxs, self.limits.q_max_waxs)
            limits_saxs = (self.limits.q_min_saxs, self.limits.q_max_saxs)
            if "type_measurement" not in dfc.columns:
                dfc["type_measurement"] = dfc["calibration_manual_distance"].apply(
                    lambda distance: "WAXS" if distance < 50 else "SAXS"
                )
            dfc["interpolation_q_range"] = dfc["type_measurement"].apply(
                lambda measurement_type: (
                    limits_waxs if measurement_type == "WAXS" else limits_saxs
                )
            )
        return dfc


class DataPreparation(TransformerMixin):
    """Select and normalize the historical default measurement columns."""

    def __init__(self, columns=_DEFAULT_COLUMNS):
        self.columns = columns

    def fit(self, x: pd.DataFrame, y=None):
        _ = x
        _ = y
        return self

    def transform(self, df: pd.DataFrame, no_poni=False) -> pd.DataFrame:
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
            dfc["measurement_data"] = dfc["measurement_data"].apply(np.nan_to_num)
        if "calculated_distance" in dfc.columns:
            dfc["type_measurement"] = dfc["calculated_distance"].apply(
                lambda distance: "WAXS" if distance < 0.05 else "SAXS"
            )
        return dfc[self.columns]


class MeasurementCutter(TransformerMixin):
    """Cut array-valued measurements with the existing distance rules."""

    def __init__(self, column, distances):
        self.column = column
        self.distances = distances

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy.dropna(subset=["ponifile"], inplace=True)
        X_copy[self.column] = X_copy.apply(
            lambda row: filter_points_by_distance(row, self.column, self.distances),
            axis=1,
        )
        return X_copy
