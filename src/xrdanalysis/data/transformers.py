"""
The transformer classes are stored here
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.base import TransformerMixin

from xrdanalysis.data.azimuthal_integration import (
    perform_azimuthal_integration,
)


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
        print("In transformer")

        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        x_copy = x.copy()

        integration_results = x_copy.apply(
            lambda row: perform_azimuthal_integration(row, self.npt), axis=1
        )

        # Extract q_range and profile arrays from the integration_results
        x_copy[["q_range", "radial_profile"]] = integration_results.apply(
            lambda x: pd.Series([x[0], x[1]])
        )

        return x_copy
