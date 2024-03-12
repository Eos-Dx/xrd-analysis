"""
This file includes functions and classes essential for azimuthal integration
"""

from dataclasses import dataclass
import time
import pandas as pd
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import Detector
from sklearn.base import TransformerMixin


@dataclass
class AzimuthalIntegration(TransformerMixin):
    """
    Transformer class for azimuthal integration to be used in
    sklearn pipeline
    """

    pixel_size: float
    npt: int = 256

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fit method for the transformer. Since this transformer does not learn
        from the data, the fit method does not perform any operations.

        Parameters:
        - x : pandas.DataFrame
            The data to fit.
        - y : Ignored
            Not used, present here for API consistency by convention.

        Returns:
        - self : object
            Returns the instance itself.
        """
        _ = x
        _ = y

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
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
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        x_copy = x.copy()

        # Creating a PyFAI AzimuthalIntegrator instance
        self.ai = pyFAI.AzimuthalIntegrator()

        # Assuming you have a detector
        detector = pyFAI.detectors.Detector(self.pixel_size, self.pixel_size)
        self.ai.detector = detector

        integration_results = x_copy.apply(
            lambda row: perform_azimuthal_integration(row, self.ai, self.npt), axis=1
        )

        # Extract q_range and profile arrays from the integration_results
        x_copy[["q_range", "radial_profile"]] = integration_results.apply(lambda x: pd.Series([x[0], x[1]]))

        return x_copy


def perform_azimuthal_integration(row: pd.Series, ai: AzimuthalIntegrator, npt=256):
    """
    Perform azimuthal integration on a single row of a DataFrame.

    Parameters:
    - row : pandas.Series
        A row from a pandas DataFrame, expected to contain 'measurement_data'
        (a 2D array), 'center' (a tuple of (x, y) representing the center of
        integration), 'wavelength' and 'calculated_distance'.
    - ai : pyFAI.azimuthalIntegrator.AzimuthalIntegrator
        The AzimuthalIntegrator used for integration.
    - npt: int
        number of points

    Returns:
    - q_range : numpy.ndarray
        The array of q values (momentum transfer) resulting
        from the integration.
    - I : numpy.ndarray
        The intensity values resulting from the integration.
    """
    pixel_size = row['pixel_size'] * 10**-6
    if ai.detector.pixel1 != pixel_size or ai.detector.pixel2 != pixel_size:
        raise ValueError('Pixel size are different in AI and df entry')
    data = row["measurement_data"]
    center = row["center"]
    center_column, center_row = center[1], center[0]
    sample_distance_mm = row["calculated_distance"] * (10 ** 3)
    wavelength_m = row["wavelength"] * (10 ** -9)

    ai.setFit2D(centerX=center_column,
                centerY=center_row,
                wavelength=wavelength_m,
                directDist=sample_distance_mm)
    q_range, profile = ai.integrate1d(data, npt=npt)
    return q_range, profile

