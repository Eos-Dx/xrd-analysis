"""
Feature extraction functions
"""
import numpy as np


def feature_image_intensity(image):
    """
    Compute the intensity of the imagew

    Parameters
    ----------

    image : ndarray
        The measurement image data

    Output
    ------

    image_intensity : number
        the total intensity

    """

    image_intensity = np.sum(image)

    return image_intensity

def dia_features_region_9A(data):
    """
    The function collects the features for the 9A region

    Input parameters:
    ----------------

    data - str
        the data picture

    Output:
    ------

    features - list
         features later used in analysis

    """

    features = 0

    return features

def dia_features_region_5A(data):
    """
    The function collects the features for the 9A region

    Input parameters:
    ----------------

    data - str
        the data picture

    Output:
    ------

    features - list
         features later used in analysis

    """

    features = 0

    return features
