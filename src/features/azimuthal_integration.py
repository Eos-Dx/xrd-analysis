"""
This file includes functions and classes essential for azimuthal integration
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class AzimuthalIntegration(BaseEstimator, TransformerMixin):
    """Transformer class for azimuthal integration to be used in
    sklearn pipeline"""

    def fit(self, X, y=None):
        """
        Fit method for the transformer. Since this transformer does not learn
        from the data, the fit method does not perform any operations.

        Parameters:
        - X : pandas.DataFrame
            The data to fit.
        - y : Ignored
            Not used, present here for API consistency by convention.

        Returns:
        - self : object
            Returns the instance itself.
        """
        _ = X
        _ = y

        return self

    def transform(self, X):
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
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        X_copy = X.copy()
        X_copy["profile"] = X_copy.apply(azimuthal_integration_df, axis=1)
        return X_copy


def azimuthal_integration_df(row):
    """
    Performs azimuthal integration on a single row of a DataFrame.

    Parameters:
    - row : pandas.Series
        A row from a pandas DataFrame, expected to contain 'measurement_data'
        (a 2D array) and 'center' (a tuple of (x, y) representing the center of
        integration).

    Returns:
    - numpy.ndarray
        The azimuthal integration profile for the given row.
    """
    data = row["measurement_data"]
    center = row["center"]

    return azimuthal_integration(data, center)


def azimuthal_integration(data, center):
    """
    Calculates the azimuthal integration profile for a given 2D array
    and center.

    Parameters:
    - data : numpy.ndarray
        The 2D array of measurement data to integrate.
    - center : tuple
        A tuple of (x, y) coordinates representing the center point
        for integration.

    Returns:
    - azimuthal_integrated_values : numpy.ndarray
        An array of azimuthal integrated values.
    """

    # Calculate the distances of each pixel from the center
    data = np.nan_to_num(data)
    x_indices, y_indices = np.indices(data.shape)
    distances = np.sqrt(
        (x_indices - center[0]) ** 2 + ((y_indices - center[1]) ** 2)
    )

    max_distance = np.max(distances)
    max_distance_ceil = int(np.ceil(max_distance))

    # Define the number of bins (adjust as needed)
    num_bins = (
        max_distance_ceil + 1
        if max_distance < max_distance_ceil - 0.5
        else max_distance_ceil + 2
    )

    half_step_sequence = np.arange(0.5, num_bins - 0.5, 1)
    bins = np.hstack(([0], half_step_sequence))

    # Bin the pixel values based on their distances from the center
    binned_values, _ = np.histogram(distances, bins=bins, weights=data)
    bin_counts, _ = np.histogram(distances, bins=bins)

    # Perform azimuthal integration and normalize
    azimuthal_integrated_values = binned_values / bin_counts

    with np.errstate(divide="ignore", invalid="ignore"):
        azimuthal_integrated_values = np.divide(
            binned_values,
            bin_counts,
            out=np.zeros_like(binned_values, dtype=np.float64),
            where=bin_counts != 0,
        )

    return azimuthal_integrated_values
