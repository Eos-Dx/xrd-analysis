"""Interpolate class and transformer for use with scikit-learn pipeline.
"""
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from sklearn.base import OneToOneFeatureMixin
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

DEFAULT_RADIAL_PROFILE_DATA_COLUMN_NAME = "radial_profile_data"
DEFAULT_INTERPOLATED_RADIAL_PROFILE_DATA_COLUMN_NAME = "interpolated_radial_profile_data"
DEFAULT_Q_RANGE_COLUMN_NAME = "q_range"
DEFAULT_RESOLUTION = 256


def interpolate_radial_profile(
        radial_profile, q_start=0, q_end=None, resolution=DEFAULT_RESOLUTION):
    """Function to interpolate a radial intensity profile.

    Parameters
    ==========

    radial_profile : (n, 2) - array-like
        Radial intensity profile. First column are q-values, second column are
        intensity values (photon counts).

    q_start : number
        Starting q value.

    q_end : number
        Final q value.

    resolution: int
        Number of data points in resulting radial profile (first dimension)

    Notes
    =====

    Desired q range includes endpoints and number of points specified by
    resolution.
    """
    # Ensure q_end is given
    if type(q_end) == type(None):
        raise ValueError("``q_end`` must be a number.")

    # Set desired q range
    desired_q_range = np.linspace(q_start, q_end, endpoint=True, num=resolution)

    # Extract original q range
    orig_q_range = radial_profile[:, 0]
    # Extract original intensity values
    intensity_values = radial_profile[:, 1]

    # Instantiate interpolator class
    interpolator = interp1d(orig_q_range, intensity_values)
    # Interpolate radial profile
    interpolated_result = interpolator(desired_q_range)

    return interpolated_result


class Interpolator(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Adapted from scikit-learn transforms
    Interpolate intensity versus q so entire dataset q-values are equal.
    """

    def __init__(self, *,
            copy=True,
            q_start=0,
            q_end=None,
            resolution=DEFAULT_RESOLUTION,
            radial_profile_data_column_name : str = \
                    DEFAULT_RADIAL_PROFILE_DATA_COLUMN_NAME,
            interpolated_radial_profile_data_column_name : str = \
                    DEFAULT_INTERPOLATED_RADIAL_PROFILE_DATA_COLUMN_NAME,
            q_range_column_name : str = \
                   DEFAULT_Q_RANGE_COLUMN_NAME,
            ):

        """Parameters
        ----------
        copy : bool
            Creates copy of array if True (default = False).

        q_start : number

        q_end : number

        radial_profile_data_column_name : str
            The column name containing radial profile data
        """
        # Ensure q_end is given
        if type(q_end) == type(None):
            raise ValueError("``q_end`` must be a number.")

        self.copy = copy
        self.q_start = q_start
        self.q_end = q_end
        self.resolution = resolution
        self.radial_profile_data_column_name = radial_profile_data_column_name
        self.interpolated_radial_profile_data_column_name = \
                interpolated_radial_profile_data_column_name
        self.q_range_column_name = q_range_column_name

    def fit(self, X, y=None, sample_weight=None):
        """Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        return self

    def transform(self, X, copy=True):
        """Parameters
        ----------
        X : {array-like}, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        q_start = self.q_start
        q_end = self.q_end
        resolution = self.resolution
        radial_profile_data_column_name = self.radial_profile_data_column_name
        interpolated_radial_profile_data_column_name = \
                self.interpolated_radial_profile_data_column_name
        q_range_column_name = self.q_range_column_name

        if copy:
            X = X.copy()

        # Create new empty column
        X[interpolated_radial_profile_data_column_name] = np.nan

        # Set column data type to object
        X[interpolated_radial_profile_data_column_name] = \
                X[interpolated_radial_profile_data_column_name].astype(object)

        # Loop over all samples using batches
        for idx in X.index:

            radial_intensity = X.loc[idx, radial_profile_data_column_name]
            q_range = X.loc[idx, q_range_column_name]

            radial_profile = np.vstack([q_range, radial_intensity]).T

            interpolated_result = interpolate_radial_profile(
                    radial_profile,
                    q_start=q_start,
                    q_end=q_end,
                    resolution=resolution,
                    )

            X.at[idx, interpolated_radial_profile_data_column_name] = \
                    interpolated_result

        return X
