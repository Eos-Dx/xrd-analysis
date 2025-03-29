import numpy as np
from xrdanalysis.data_processing.pipeline import MLPipeline
from xrdanalysis.data_processing.transformers import AzimuthalIntegration


def process_dataframe(df):
    """
    Process the input DataFrame using the azimuthal integration pipeline.
    """
    faulty_pixel_array = np.array([[146, 170]])
    N = 100  # number of points in radial profile
    azimuth = AzimuthalIntegration(faulty_pixels=faulty_pixel_array, calibration_mode='poni', npt=N)
    wrangling = [('azimuthal_integration', azimuth)]
    pipeline = MLPipeline(data_wrangling_steps=wrangling)
    dfm = pipeline.wrangle(df)
    return dfm
