"""
This file includes functions and classes essential for azimuthal integration
"""

from dataclasses import dataclass
from functools import wraps

import pandas as pd
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import Detector
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class AzimuthalIntegration(BaseEstimator, TransformerMixin):
    """Transformer class for azimuthal integration to be used in
    sklearn pipeline"""

    pixel_size: float

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

        detector = Detector(pixel1=self.pixel_size, pixel2=self.pixel_size)

        integration_results = X_copy.apply(
            lambda row: azimuthal_integration_row(row, detector), axis=1
        )

        # Extract q_range and profile arrays from the integration_results
        X_copy["q_range"] = integration_results.apply(lambda x: x[0])
        X_copy["profile"] = integration_results.apply(lambda x: x[1])
        return X_copy


def azimuthal_integration_row(row, detector=None):
    """
    Perform azimuthal integration on a single row of a DataFrame.

    Parameters:
    - row : pandas.Series
        A row from a pandas DataFrame, expected to contain 'measurement_data'
        (a 2D array), 'center' (a tuple of (x, y) representing the center of
        integration), 'wavelength' and 'calculated_distance'.
    - detector : pyFAI.detectors.Detector
        The detector used for integration.

    Returns:
    - q_range : numpy.ndarray
        The array of q values (momentum transfer) resulting
        from the integration.
    - I : numpy.ndarray
        The intensity values resulting from the integration.
    """

    if not detector:
        pixel_size = row["pixel_size"] * (10**-6)
        detector = Detector(pixel1=pixel_size, pixel2=pixel_size)

    data = row["measurement_data"]
    center = (row["center"][1], row["center"][0])
    sample_distance_mm = row["calculated_distance"] * (10**3)
    wavelength_m = row["wavelength"] * (10**-9)

    ai = initialize_azimuthal_integrator(
        detector, wavelength_m, sample_distance_mm, center
    )

    return azimuthal_integration(data, ai=ai)


def azimuthal_integration(data, ai=None):
    """
    Perform azimuthal integration on a 2D array of measurement data.

    Parameters:
    - data : numpy.ndarray
        The 2D array of measurement data to integrate.
    - ai : pyFAI.azimuthalIntegrator.AzimuthalIntegrator.

    Returns:
    - q_range : numpy.ndarray
        The array of q values (momentum transfer) resulting
        from the integration.
    - I : numpy.ndarray
        The intensity values resulting from the integration.
    """

    res = ai.integrate1d(data, 300)

    q_range = res[0]
    intensity = res[1]

    return q_range, intensity


def memoize_integrator(func):
    """
    Memoization decorator for initializing azimuthal integrator.

    Parameters:
    - func : function
        The function to be memoized.

    Returns:
    - memoized_integrator : function
        The memoized version of the function.
    """
    cache = {}

    @wraps(func)
    def memoized_integrator(detector, wavelength, sample_dist, center):
        key = (wavelength, sample_dist, tuple(center))
        if key not in cache:
            cache[key] = func(detector, wavelength, sample_dist, center)
        return cache[key]

    return memoized_integrator


@memoize_integrator
def initialize_azimuthal_integrator(detector, wavelength, sample_dist, center):
    """
    Initializes an azimuthal integrator with given parameters.

    Parameters:
    - detector : pyFAI.detectors.Detector
        The detector used for integration.
    - wavelength : float
        The wavelength of the incident X-ray beam.
    - sample_dist : float
        The sample-to-detector distance.
    - center : tuple
        A tuple of (x, y) coordinates representing the center point
        for integration.

    Returns:
    - ai : pyFAI.azimuthalIntegrator.AzimuthalIntegrator
        An instance of the PyFAI AzimuthalIntegrator.
    """
    ai = AzimuthalIntegrator(detector=detector)
    ai.wavelength = wavelength
    ai.setFit2D(sample_dist, center[0], center[1])
    return ai
