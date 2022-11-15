import numpy as np


def dia_features_global(data):
    """
    The function collects the features for the total picture

    Input parameters:
    ----------------

    data - str
        the data picture

    Output:
    ------

    features - list
         features later used in analysis

    """

    total_brightness = np.sum(data)

    features = [total_brightness]

    return features

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
