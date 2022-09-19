"""
Gaussian fit to keratin diffraction patterns

1 Run Gaussian decomposition on a data set, reducing each measurement to
  13 parameters
2 Run K-means on the Gaussian parameters (unsupervised learning)
3 Run logistic regression on the Gaussian parameters (supervised learning)

"""

import os
import glob
import argparse
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

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

from skimage.feature import peak_local_max

from eosdxanalysis.models.utils import cart2pol
from eosdxanalysis.models.feature_engineering import EngineeredFeatures
from eosdxanalysis.models.curve_fitting import GaussianDecomposition
from eosdxanalysis.models.curve_fitting import gaussian_decomposition
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.simulations.utils import feature_pixel_location


def main(
        input_path, output_path=None, params_init_method=None,
        run_gauss_decomp=False, run_pca=False, run_kmeans=False,
        run_logreg=False):
    t0 = time.time()

    cmap="hot"

    # Run Gaussian decomposition on data set
    # --------------------------------------
    if run_gauss_decomp:
        # Run Gaussian decomposition
        gaussian_decomposition(input_path, output_path, params_init_method)

    # Run Principal Component Analysis
    if run_pca:
        pass

    # Run K-means on Gaussian parameters
    # ----------------------------------
    if run_kmeans:
        pass

    # Run logistic regression on Gaussian parameters
    # ----------------------------------------------
    if run_logreg:
        # Set feature vector
        pass

if __name__ == '__main__':
    """
    Run Gaussian decomposition example
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", default=None, required=True,
            help="The path containing raw files to perform fitting on.")
    parser.add_argument(
            "--output_path", default=None, required=False,
            help="The output path to save results in.")
    parser.add_argument(
            "--params_init_method", default="ideal", required=False,
            help="The default method to initialize the parameters"
            " Options are: ``ideal`` and ``estimation``.")

    # Collect arguments
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    params_init_method = args.params_init_method

    main(input_path, output_path, params_init_method)
