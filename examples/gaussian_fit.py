"""
Gaussian fit to keratin diffraction patterns

1 Run Gaussian decomposition on a data set, reducing each measurement to
  13 parameters
2 Run K-means on the Gaussian parameters (unsupervised learning)
3 Run logistic regression on the Gaussian parameters (supervised learning)

"""

import os
import glob
from collections import OrderedDict
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
from scipy.optimize import curve_fit

from scipy.signal import find_peaks
from scipy.signal import peak_widths

from skimage.feature import peak_local_max

from eosdxanalysis.models.utils import cart2pol
from eosdxanalysis.models.feature_engineering import EngineeredFeatures
from eosdxanalysis.models.curve_fitting import GaussianDecomposition
from eosdxanalysis.models.curve_fitting import gaussian_decomposition
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.simulations.utils import feature_pixel_location

t0 = time.time()

cmap="hot"

# 1 Run Gaussian decomposition on data set
# ----------------------------------------
run_gauss_decomp = True
if run_gauss_decomp:
    # Set the input path
    input_path = ""
    output_path = ""

    # Run Gaussian decomposition
    gaussian_decomposition(input_path, output_path)

# 2 Run K-means on Gaussian parameters
# ------------------------------------
run_kmeans = False
if run_kmeans:
    pass


# 3 Run logistic regression on Gaussian parameters
# ------------------------------------------------
run_logreg = False
if run_logreg:
    pass
