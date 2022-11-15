import os
import glob
import argparse
import json
from datetime import datetime
import subprocess

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.filters import threshold_local
from skimage.transform import EuclideanTransform
from skimage.transform import warp
from skimage.transform import rotate

from eosdxanalysis.preprocessing.center_finding import find_center
from eosdxanalysis.preprocessing.center_finding import find_centroid
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.utils import get_angle
from eosdxanalysis.preprocessing.utils import find_maxima
from eosdxanalysis.preprocessing.denoising import find_hot_spots
from eosdxanalysis.preprocessing.denoising import filter_hot_spots
from eosdxanalysis.preprocessing.image_processing import crop_image
from eosdxanalysis.preprocessing.image_processing import quadrant_fold
from eosdxanalysis.preprocessing.beam_utils import beam_extent

from eosdxanalysis.simulations.utils import feature_pixel_location

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
