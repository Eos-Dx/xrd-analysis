"""Private curve, Fourier, image, and Hankel transformers."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pyhank import HankelTransform
from scipy.optimize import curve_fit
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Normalizer, StandardScaler

from xrdanalysis.data_processing.fourier import (
    fourier_custom,
    fourier_fft,
    fourier_fft2,
    slope_removal,
    slope_removal_custom,
)
from xrdanalysis.data_processing.utility_functions import (
    cut_common_region,
    find_common_region,
    resize_image,
)


class SlopeRemoval(TransformerMixin):
    """Remove a linear slope from configured profile columns."""

    def __init__(self, columns=["radial_profile_data"], mode=""):
        self.columns = columns
        self.mode = mode

    def fit(self, x: pd.DataFrame, y=None):
        _ = x
        _ = y
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        for column in self.columns:
            if self.mode == "custom":
                X[column] = X[column].apply(
                    lambda value: slope_removal_custom(value)[0]
                )
            else:
                X[column] = X[column].apply(slope_removal)
        return X


class FourierTransform(TransformerMixin):
    """Apply the historical 1D or 2D Fourier feature transformation."""

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
        _ = x
        _ = y
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        if self.fourier_mode != "2D":
            fourier_func = (
                fourier_custom if self.fourier_mode == "custom" else fourier_fft
            )
            for column in self.columns:
                X[[f"fourier_coefficients_{column}", f"fourier_inverse_{column}"]] = X[
                    column
                ].apply(lambda value: pd.Series(fourier_func(value, self.order)))
        else:
            X[self._get_feature_columns()] = X[self.columns[0]].apply(
                lambda value: pd.Series(
                    fourier_fft2(
                        value,
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
        return [self.features] if isinstance(self.features, str) else self.features


class NormScaler(TransformerMixin):
    """Fit and apply separate L1-normalized profile scalers by measurement type."""

    def __init__(self, scalers: Dict[str, StandardScaler] = None, name="Scaler"):
        self._name = name
        self.scalers = scalers if scalers else {}

    def fit(self, df: pd.DataFrame, y=None):
        print(f"NormScaler {self._name}: is fitting.")
        dfc = df.copy()
        norm = Normalizer("l1")
        dfc["radial_profile_data_norm"] = dfc["radial_profile_data"].apply(
            lambda value: norm.transform([value])[0]
        )
        for measurement_type in ("SAXS", "WAXS"):
            subset = dfc[dfc["type_measurement"] == measurement_type].copy()
            if not subset.empty:
                scaler = StandardScaler()
                scaler.fit(np.vstack(subset["radial_profile_data_norm"].values))
                self.scalers[measurement_type] = scaler
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"NormScaler {self._name}: is transforming.")
        if not self.scalers:
            self.fit(df)
        dfc, raw_index = df.copy(), df.index
        norm = Normalizer("l1")
        dfc["radial_profile_data_norm"] = dfc["radial_profile_data"].apply(
            lambda value: norm.transform([value])[0]
        )
        processed = []
        for measurement_type in ("SAXS", "WAXS"):
            subset = dfc[dfc["type_measurement"] == measurement_type].copy()
            if not subset.empty:
                scaled = self.scalers[measurement_type].transform(
                    np.vstack(subset["radial_profile_data_norm"].values)
                )
                subset["radial_profile_data_norm_scaled"] = [value for value in scaled]
            processed.append(subset)
        return pd.concat(processed).loc[raw_index]


class CurveFittingTransformer(TransformerMixin):
    """Fit a producer-defined curve for every row of two array columns."""

    def __init__(self, x_column, y_column, producer, cutoff_ranges=None):
        self.x_column = x_column
        self.y_column = y_column
        self.producer = producer
        self.cutoff_ranges = cutoff_ranges

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy["fit_params_all"] = None
        X_copy["fitted_curve"] = None
        X_copy["fit_cond"] = None
        X_copy["fit_params_all"].astype(object)
        X_copy["fitted_curve"].astype(object)
        X_copy["fit_cond"].astype(object)
        x_value = X_copy.iloc[0][self.x_column]
        func = self.producer.produce_function()
        p0 = self.producer.initial_guess()
        bounds = self.producer.bounds()
        function_count = self.producer.get_function_count()
        function_param_counts = self.producer.get_function_param_counts()
        for index in range(function_count):
            X_copy[f"fit_params_{index}"] = None
            X_copy[f"fit_params_{index}"].astype(object)
        if self.cutoff_ranges:
            sigma = np.ones_like(x_value)
            for start, end in self.cutoff_ranges:
                sigma[(x_value > start) & (x_value < end)] = 1e6
        else:
            sigma = None
        for index, row in X_copy.iterrows():
            x_values, y_values = np.array(row[self.x_column]), np.array(
                row[self.y_column]
            )
            try:
                popt, pcov = curve_fit(
                    func, x_values, y_values, p0=p0, bounds=bounds, sigma=sigma
                )
                X_copy.at[index, "fit_cond"] = np.linalg.cond(pcov)
                X_copy.at[index, "fit_params_all"] = popt
                for group_index in range(len(function_param_counts)):
                    start_idx = sum(function_param_counts[:group_index])
                    end_idx = start_idx + function_param_counts[group_index]
                    X_copy.at[index, f"fit_params_{group_index}"] = popt[
                        start_idx:end_idx
                    ]
                X_copy.at[index, "fitted_curve"] = func(x_values, *popt)
            except RuntimeError as error:
                print(f"Fit failed for index {index}: {error}")
                X_copy.at[index, "fit_params_all"] = None
                X_copy.at[index, "fitted_curve"] = None
        return X_copy.dropna(subset=["fit_params_all"])


class ImageResizer(TransformerMixin):
    """Resize images while updating their PONI calibration text."""

    def __init__(self, column, ref_distance):
        self.column = column
        self.ref_distance = ref_distance

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy[[self.column, "ponifile"]] = X_copy.apply(
            lambda row: pd.Series(resize_image(row, self.column, self.ref_distance)),
            axis=1,
        )
        return X_copy


class CommonRegionCutter(TransformerMixin):
    """Cut common image regions while keeping PONI geometry aligned."""

    def __init__(self, column, square=False):
        self.column = column
        self.square = square

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        bounds = find_common_region(X_copy, self.column, self.square)
        X_copy[[self.column, "ponifile"]] = X_copy.apply(
            lambda row: pd.Series(cut_common_region(row, self.column, *bounds)), axis=1
        )
        return X_copy


class HankelTransformer(TransformerMixin):
    """Calculate a quasi-discrete Hankel transform for each polar image."""

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
        for index, row in X_copy.iterrows():
            polar_img = row[self.column].copy().astype(float)[:, self.start_radius :]
            radius = row["q_range"][self.start_radius :]
            polar_img = np.nan_to_num(polar_img, nan=0.0)
            _, n_radial = polar_img.shape
            transformer = HankelTransform(
                order=self.order, max_radius=radius.max(), n_points=n_radial
            )
            X_copy.at[index, "hankel"] = transformer.qdht(polar_img, axis=1)
        return X_copy
