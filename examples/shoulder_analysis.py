"""
Use the 2D Gaussian fit results to analyze 5 A meridional "shoulder"
"""
import os
import argparse
import glob
from collections import OrderedDict
from datetime import datetime
from time import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def main(fit_results_file, output_path=None):
    t0 = time()

    cmap="hot"

    # Load dataframe
    df = pd.read_csv(fit_results_file, index_col=0)

    # Set up figure
    fig = plt.figure(figsize=(8,8))

    # Plot normal
    normal_rows = df[df["Cancer"] == 0]
    Xnormal = normal_rows["peak_location_radius_5A"]
    Ynormal = normal_rows["peak_location_radius_5_4A"]
    plt.scatter(Xnormal, Ynormal, color="blue", label="Normal")

    # Plot cancer
    cancer_rows = df[df["Cancer"] == 1]
    Xcancer = cancer_rows["peak_location_radius_5A"]
    Ycancer = cancer_rows["peak_location_radius_5_4A"]
    plt.scatter(Xcancer, Ycancer, color="red", label="Cancer")

    # Plot settings
    plt.xlim([45, 70])
    plt.ylim([55, 80])

    plt.xlabel("Peak location radius 5 A")
    plt.ylabel("Peak location radius 5-4 A")

    plt.title("Gaussian Fitting Features Subspace")
    plt.legend()

    plt.show()

    print("Cancer:",cancer_rows.shape[0])
    print("Normal:",normal_rows.shape[0])

if __name__ == '__main__':
    """
    Run shoulder analysis on 2D Gaussian fit results
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_filepath", default=None, required=True,
            help="The file path containing 2D Gaussian fit results.")
    parser.add_argument(
            "--output_path", default=None, required=False,
            help="The output path to save results in.")

    # Collect arguments
    args = parser.parse_args()
    input_filepath = args.input_filepath
    output_path = args.output_path

    main(input_filepath, output_path)
