"""
Analyze statistics of curve fitting fit parameters
"""
import os
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_histograms(
        input_filepath, output_path=None, visualize=False, save=False):
    """
    Plot histograms of parameters in csv file

    Parameters
    ----------

    input_filepath : str
        Full filepath to the csv file containing features to analyze

    output_path : str (optional)
        Full path to the output directory to save the plots in. If not
        specified and ``save`` is ``True``, then generate a timestamped
        output directory.

    visualize : bool
        Flag to display plots to screen if ``True``. Default ``False``.

    save : bool
        Flag to save plots to file(s) if ``True``. Default ``False``.

    """
    if visualize == False and save == False:
        raise Warning("Must specify at least ``visualize`` or ``save``. Nothing to do.")

    # Load dataframe
    df = pd.read_csv(input_filepath, index_col="Filename")
    # Get features
    feature_list = df.columns

    if save:
        # Set timestamp
        timestr = "%y%m%dT%H%M%S.%f"
        timestamp = datetime.utcnow().strftime(timestr)

        # Get the parent output path
        input_parentpath = os.path.dirname(input_filepath)

        # Set output dir with a timestamp if not specified
        if not output_path:
            output_dir = "feature_histograms_{}".format(timestamp)
            output_path = os.path.join(input_parentpath, output_dir)
            # Create the output dir
            os.makedirs(output_path, exist_ok=True)

    # Loop over features to create individual histograms
    for feature in feature_list:
        # Set up figure
        fig = plt.figure(feature, figsize=(8,6))

        feature_values = df[feature].values

        # Sanitize the values
        feature_values[feature_values <= 1e-5] = 0

        # Create auto histogram
        plt.hist(feature_values, bins='auto')

        # Set plot title
        plot_title = "{}: Mean: {:.2f}, Standard Deviation: {:.2f}".format(
                feature, feature_values.mean(), feature_values.std())
        plt.title(plot_title)

        # Plot all figure to screen
        if visualize:
            plt.show()

        # Save figure to file
        if save:
            data_filepath = "feature_{}_histogram.png".format(feature)
            output_filepath = os.path.join(output_path, data_filepath)
            plt.savefig(output_filepath)

        # Clear figure memory
        fig.clear()
        plt.close(fig)


if __name__ == '__main__':
    """
    Run analysis on curve_fitting results from a csv file using CLI. See
    ``plot_histogram`` documentation or run with ``--help``.

    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_filepath", default=None, required=True,
            help="The path containing the csv file to analyze.")
    parser.add_argument(
            "--output_path", default=None, required=False,
            help="The path to save plots to.")
    parser.add_argument(
            "--visualize", default=False, required=False,
            action="store_true",
            help="Display the results.")
    parser.add_argument(
            "--save", default=False, required=False,
            action="store_true",
            help="Save the plots.")

    # Collect arguments
    args = parser.parse_args()
    input_filepath = args.input_filepath
    output_path = args.output_path
    visualize = args.visualize
    save = args.save

    plot_feature_histograms(
            input_filepath, output_path=output_path, visualize=visualize,
            save=save)
