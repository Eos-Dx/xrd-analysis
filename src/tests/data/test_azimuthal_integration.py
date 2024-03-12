"""
Test for azimuthal integration
"""

import numpy as np
import pandas as pd
import pytest
import os
from time import time
from joblib import load
from xrdanalysis.data.transformers import AzimuthalIntegration


@pytest.fixture
def sample_data():
    """
    Fixture providing a sample DataFrame with measurement data.

    Returns:
        pd.DataFrame: DataFrame containing measurement data along with other related information.
    """
    file_path = os.path.join(os.path.dirname(__file__), 'test_integration_data.joblib')
    df = load(file_path)
    df = pd.concat([df] * 10, ignore_index=True)
    return df


def test_azimuthal_integration_transform(sample_data):
    """
    Test azimuthal_integration_transform method, to see whether Transformer
    works with DataFrames
    """
    df = sample_data
    transformer = AzimuthalIntegration(pixel_size=55 * 10**-6)
    a = time()
    transformed_df = transformer.transform(df)
    print(time() - a)

    q_range_array_count = (
        transformed_df["q_range"]
        .apply(lambda x: isinstance(x, np.ndarray))
        .sum()
    )
    profile_array_count = (
        transformed_df["radial_profile"]
        .apply(lambda x: isinstance(x, np.ndarray))
        .sum()
    )

    assert q_range_array_count == len(df)
    assert profile_array_count == len(df)

    assert "q_range" in transformed_df.columns
    assert "radial_profile" in transformed_df.columns
