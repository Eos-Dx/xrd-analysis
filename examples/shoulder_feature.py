"""
Code to analyze the 5.1A shoulder feature
"""
import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

import time

from eosdxanalysis.models.feature_engineering import shoulder_analysis

def main(input_path, output_path=None):
    t0 = time.time()

    # 1 Run shoulder feature analysis on data set
    # ----------------------------------------
    run_shoulder_analysis = True
    if run_shoulder_analysis:
        # Run shoulder feature analysis
        shoulder_analysis(input_path, output_path)

    t1 = time.time()
    print("Time passed (s):", t1-t0)

if __name__ == '__main__':
    """
    Run shoulder analysis example
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", default=None, required=True,
            help="The path containing raw files to perform analysis on.")
    parser.add_argument(
            "--output_path", default=None, required=False,
            help="The output path to save results in.")
    parser.add_argument(
            "--params_init_method", default="ideal", required=False,
            help="For Gaussian fitting, the default method to initialize the"
            " parameters Options are: ``ideal`` and ``approximate``.")

    # Collect arguments
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    params_init_method = args.params_init_method

    main(input_path, output_path)
