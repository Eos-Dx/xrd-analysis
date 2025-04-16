import numpy as np
from xrdanalysis.data_processing.pipeline import MLPipeline
from xrdanalysis.data_processing.transformers import AzimuthalIntegration


def process_dataframe(df):
    """
    Process the input DataFrame using the azimuthal integration pipeline.
    """
    N = 100  # number of points in radial profile
    azimuth = AzimuthalIntegration(calibration_mode='poni', npt=N)
    wrangling = [('azimuthal_integration', azimuth)]
    pipeline = MLPipeline(data_wrangling_steps=wrangling)
    try:
        dfm = pipeline.wrangle(df)
        return dfm
    except Exception as e:
        print(e)



def process_dataframe_2D(df):
    """
    Process the input DataFrame using the azimuthal integration pipeline.
    """
    azimuth = AzimuthalIntegration(calibration_mode='poni', integration_mode='2D')
    wrangling = [('azimuthal_integration', azimuth)]
    pipeline = MLPipeline(data_wrangling_steps=wrangling)
    dfm = pipeline.wrangle(df)
    return dfm
