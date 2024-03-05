"""
This file includes functions and classes essential for azimuthal integration
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class AzimuthalIntegration(BaseEstimator, TransformerMixin):
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
    max_distance = np.sqrt(
        (data.shape[0] - center[0]) ** 2 + (data.shape[1] - center[1]) ** 2
    )

    # Calculate the distances of each pixel from the center
    x_indices, y_indices = np.indices(data.shape)
    distances = np.sqrt((x_indices - center[0]) ** 2 / +((y_indices - center[1]) ** 2))

    # Define the number of bins (adjust as needed)
    num_bins = int(np.ceil(max_distance))
    bins = np.arange(num_bins + 1)

    # Bin the pixel values based on their distances from the center
    binned_values, _ = np.histogram(distances, bins=bins, weights=data)
    bin_counts, _ = np.histogram(distances, bins=bins)

    # Perform azimuthal integration and normalize
    azimuthal_integrated_values = binned_values / bin_counts

    return azimuthal_integrated_values
