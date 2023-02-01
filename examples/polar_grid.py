"""
Example with polar grid
"""
import os

import numpy as np
import pandas as pd

import argparse
import json
import glob

from eosdxanalysis.preprocessing.utils import polar_meshgrid


def run_feature_extraction(input_path, meshgrid_params, output_filepath):
    """
    """
    # Get meshgrid parameters
    output_shape = meshgrid_params["output_shape"]
    r_count = meshgrid_params["r_count"]
    theta_count = meshgrid_params["theta_count"]
    rmin = meshgrid_params["rmin"]
    rmax = meshgrid_params["rmax"]
    quadrant_fold = meshgrid_params["quadrant_fold"]

    mesh = polar_meshgrid(
            output_shape=output_shape,
            r_count=r_count,
            theta_count=theta_count,
            rmin=rmin,
            rmax=rmax,
            quadrant_fold=quadrant_fold)

    # Get filepath list
    filepath_list = glob.glob(os.path.join(input_path, "*.txt"))
    # Sort files list
    filepath_list.sort()

    # Get list of features to extract
    num_features = r_count * theta_count
    feature_list = np.arange(1,num_features+1).tolist()
    feature_values = np.zeros(num_features)

    # Create dataframe to collect extracted features
    columns = ["Filename"] + feature_list
    df = pd.DataFrame(data={}, columns=columns)

    # Loop over files list
    for filepath in filepath_list:
        filename = os.path.basename(filepath)
        image = np.loadtxt(filepath, dtype=np.uint32)

        for idx in range(num_features):
            cell = (image == feature_list[idx])
            feature_values[idx] = image[cell].sum()

        # Add extracted features to dataframe
        df.loc[len(df.index)+1] = [filename] + feature_values.tolist()

    # Save dataframe to csv
    df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    """
    Run feature extraction on an image using a mesh grid
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", default=None, required=True,
            help="The path to data to extract features from")
    parser.add_argument(
            "--meshgrid_params_filepath", default=None, required=True,
            help="The meshgrid file in text format")
    parser.add_argument(
            "--output_filepath", default=None, required=True,
            help="The csv output file with features")

    args = parser.parse_args()

    input_path = args.input_path
    output_filepath = args.output_filepath

    # Get parameters from file or from JSON string commandline argument
    meshgrid_params_filepath = args.meshgrid_params_filepath
    with open(meshgrid_params_filepath,"r") as meshgrid_params_fp:
        meshgrid_params = json.loads(meshgrid_params_fp.read())

    run_feature_extraction(
        input_path=input_path,
        meshgrid_params=meshgrid_params,
        output_filepath=output_filepath,
        )

