"""
Code to train multiple k-means unsupervised clustering models on a dataset
"""
import os
import shutil
import argparse

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from joblib import dump

def run_kmeans(
        data_filepath, output_path=None, feature_list=None, cluster_count_min=2,
        cluster_count_max=2, image_source_path=None):
    """
    Runs k-means on a dataset of extracted features for between
    ``cluster_count_min`` and ``cluster_count_max`` number of clusters.

    Parameters
    ----------

    data_filepath : str
        Path to input csv file with extracted features data

    output_path : str
        Path to save k-means models and cluster image previews

    feature_list : str
        List of features to perform k-means analysis on. If blank,
        all columns except ``Filename`` are used.

    cluster_count_min : int
        Mniimum number of clusters to use for k-means

    cluster_count_max : int
        Maximum number of clusters to use for k-means

    image_source_path : str
        Path to image previews
    """
    # Load data into dataframe
    df = pd.read_csv(data_filepath, usecols=feature_list, index_col="Filename")

    # Set K-means output results path
    kmeans_results_filename = "kmeans_results.csv"

    # Set timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    # Check if output path is specified
    if not output_path:
        # Set output path to input file parent path
        output_dir = "kmeans_models_{}".format(timestamp)
        output_path = os.path.dirname(data_filepath)

    df_kmeans_path = (output_path, kmeans_results_filename)

    # Set up standard scaler
    scaler = StandardScaler()
    # Fit standard scaler to data
    scaler.fit(df)

    # Transform data using standard scaler
    X = scaler.transform(df)

    # Train K-means models for each cluster number
    for idx in range(cluster_min, cluster_max+1):
        kmeans = KMeans(idx, random_state=0).fit(X)
        # Save the labels in the dataframe
        df["kmeans_{}".format(idx)] = kmeans.labels_

        # Save the model to file
        model_filename = "kmeans_model_{}.joblib".format(timestamp)
        model_filepath = os.path.join(output_path, model_filename)
        shutil.makedirs(model_filepath, exist_ok=True)
        dump(pipe, model_filepath)

    # Save K-means results to file
    df.to_csv(df_kmeans_path, index=True)

    # Use K-means results to create cluster image preview folders
    # Loop over files to copy the file to individual K-means cluster folders
    if image_source_path:
        # Create the cluster image previews directory
        kmeans_cluster_dir = "kmeans_clusters"
        cluster_image_path = os.path.join(output_path, kmeans_cluster_dir)
        shutil.makedirs(cluster_image_path, exist_ok=True)

        for idx in df.index:
            filename = idx + ".png"

            image_model_dir = "kmeans_n{}".format(idx)
            image_model_path = os.path.join(
                    cluster_image_path, image_model_dir)
            shutil.makedirs(image_model_path, exist_ok=True)

            # Copy the file to the appropriate directory or directories
            # Loop over K-means models
            for jdx in range(cluster_min, cluster_max+1):
                # Get the cluster label
                cluster = df["kmeans_{}".format(jdx)][idx]
                # Set the cluster image path
                image_cluster_dir = "kmeans_n{}_c{}".format( jdx, cluster)
                image_cluster_path = os.path.join(
                        image_model_path, image_cluster_dir)
                # Create the cluster image path
                os.makedirs(image_cluster_path, exist_ok=True)
                # Copy the file from the image source path to the image cluster
                # path
                shutil.copy(
                        os.path.join(image_source_path, filename),
                        os.path.join(image_cluster_path, filename))


if __name__ == '__main__':
    """
    Run K-means on extracted features
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--data_filepath", default=None, required=False,
            help="The file path containing features to perform k-means"
            " clustering on.")
    parser.add_argument(
            "--output_path", default=None, required=False,
            help="The output path to save results in.")
    parser.add_argument(
            "--feature_list", default=None, required=False,
            help="List of features to perform k-means clustering on.")
    parser.add_argument(
            "--cluster_count_min", default=2, required=False,
            help="Minimum number of clusters to use for k-means.")
    parser.add_argument(
            "--cluster_count_max", default=2, required=False,
            help="List of features to perform k-means clustering on.")
    parser.add_argument(
            "--image_source_path", default=None, required=False,
            help="Path to image previews.")

    # Collect arguments
    args = parser.parse_args()

    data_filepath = args.data_filepath
    output_path = args.output_path
    feature_list_kwarg = args.feature_list
    cluster_count_min = args.cluster_count_min
    cluster_count_max = args.cluster_count_max

    feature_list = feature_list_kwarg.split(",") if feature_list_kwarg else []

    image_source_path = args.image_source_path

    run_kmeans(
            data_filepath, output_path=output_path, feature_list=feature_list,
            cluster_count_min=cluster_count_min,
            cluster_count_max=cluster_count_max,
            image_source_path=image_source_path,
            )
