"""
Code to train multiple k-means unsupervised clustering models on a dataset
"""
import os
import shutil
import argparse

from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from joblib import dump

def run_kmeans(
        data_filepath, db_filepath=None, output_path=None, feature_list=None,
        cluster_count_min=2, cluster_count_max=2, image_source_path=None,
        divide_by=None):
    """
    Runs k-means on a dataset of extracted features for between
    ``cluster_count_min`` and ``cluster_count_max`` number of clusters.

    Parameters
    ----------

    data_filepath : str
        Path to input csv file with extracted features data

    db_filepath : str
        Path to patients database csv file (optional)

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
    df = pd.read_csv(data_filepath, index_col="Filename")

    # Get list of features
    if feature_list is None:
        feature_list = df.columns.tolist()

    # Scale all features
    if divide_by:
        df = df.div(df[divide_by], axis="rows")
        df = df[feature_list]
        if divide_by in feature_list:
            # Drop divide_by column
            df = df.drop(columns=[divide_by])
    else:
        # Get features
        df = df[feature_list]

    # Set timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    # Set K-means output results path
    kmeans_results_filename = "kmeans_results_{}.csv".format(timestamp)

    # Check if output path is specified
    if not output_path:
        # Set output path to input file parent path
        output_dir = "kmeans_models_{}".format(timestamp)
        output_path = os.path.dirname(data_filepath)

    # Create the k-means results directory
    kmeans_results_dir = "kmeans_{}".format(timestamp)
    kmeans_results_path = os.path.join(output_path, kmeans_results_dir)
    os.makedirs(kmeans_results_path, exist_ok=True)

    # Fit the standard scaler
    scaler = StandardScaler()
    scaler.fit(df)
    # Create dataframe with transformed features
    transformed_features = scaler.transform(df)
    df_transformed = pd.DataFrame(
            data=transformed_features,
            columns=feature_list,
            index=df.index)

    # Train K-means models for each cluster number
    for cluster_count in range(cluster_count_min, cluster_count_max+1):
        kmeans = KMeans(cluster_count, random_state=0)
        # Fit k-means on transformed features
        kmeans.fit(df_transformed)

        # Save the labels in a new dataframe
        df_transformed["kmeans_{}".format(cluster_count)] = kmeans.labels_

        estimator = make_pipeline(scaler, kmeans)

        # Save the estimator to file
        estimator_filename = "estimator_scaler_kmeans_n{}_{}.joblib".format(
                cluster_count, timestamp)
        estimator_filepath = os.path.join(kmeans_results_path, estimator_filename)
        dump(estimator, estimator_filepath)

        # Save cluster centers to file
        clusters_filename = "kmeans_clusters_n{}_{}.csv".format(
                cluster_count, timestamp)
        clusters_filepath = os.path.join(
                kmeans_results_path, clusters_filename)
        # Create dataframe of cluster centers only
        df_clusters = pd.DataFrame(
                data=kmeans.cluster_centers_, columns=feature_list)
        df_clusters.to_csv(
                clusters_filepath)

    # Save the transformed data with k-means labels
    kmeans_results_filename = "kmeans_results_n{}_{}.csv".format(
                cluster_count, timestamp)
    kmeans_results_filepath = os.path.join(
            kmeans_results_path, kmeans_results_filename)
    df_transformed.to_csv(kmeans_results_filepath)

    # If patients database provided, save extended version
    if db_filepath:
        db = pd.read_csv(db_filepath, index_col="Barcode")
        extraction = df_transformed.index.str.extractall(
                "CR_([A-Z]{1}).*?([0-9]+)")
        extraction_series = extraction[0] + extraction[1].str.zfill(5)
        extraction_list = extraction_series.tolist()

        assert(len(extraction_list) == df_transformed.shape[0])
        df_transformed_ext = df_transformed.copy()
        df_transformed_ext["Barcode"] = extraction_list

        df_transformed_ext = pd.merge(
                df_transformed_ext, db, left_on="Barcode", right_index=True)
        # Save the transformed data with k-means labels and patient IDs
        kmeans_results_ext_filename = "kmeans_results_ext_n{}_{}.csv".format(
                    cluster_count, timestamp)
        kmeans_results_ext_filepath = os.path.join(
                kmeans_results_path, kmeans_results_ext_filename)
        df_transformed_ext.to_csv(kmeans_results_ext_filepath)

    # Use K-means results to create cluster image preview folders
    # Loop over files to copy the file to individual K-means cluster folders
    if image_source_path:

        # Create image cluster paths for all models
        # Loop over model numbers
        for cluster_count in range(cluster_count_min, cluster_count_max+1):
            # Create the models paths
            kmeans_model_dir = "kmeans_n{}".format(cluster_count)
            kmeans_model_path = os.path.join(
                    kmeans_results_path, kmeans_model_dir)
            # Loop over clusters
            for cluster_label in range(cluster_count):
                cluster_image_dir = "kmeans_n{}_c{}".format(cluster_count, cluster_label)
                # Create the paths
                cluster_image_path = os.path.join(
                        kmeans_model_path, cluster_image_dir)
                os.makedirs(cluster_image_path, exist_ok=True)

        for idx in df_transformed.index:
            filename = idx + ".png"

            # Copy the file to the appropriate directory or directories
            # Loop over K-means models
            for cluster_count in range(cluster_count_min, cluster_count_max+1):
                kmeans_model_dir = "kmeans_n{}".format(cluster_count)
                kmeans_model_path = os.path.join(
                        kmeans_results_path, kmeans_model_dir)

                # Get the cluster label
                cluster_label = df_transformed["kmeans_{}".format(cluster_count)][idx]
                # Set the cluster image path
                cluster_image_dir = "kmeans_n{}_c{}".format(
                        cluster_count, cluster_label)
                cluster_image_path = os.path.join(
                        kmeans_model_path, cluster_image_dir)

                # Copy the file from the image source path to the image cluster
                # path
                shutil.copy(
                        os.path.join(image_source_path, filename),
                        os.path.join(cluster_image_path, filename))


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
            "--db_filepath", type=str, default=None, required=False,
            help="The patients database")
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
    parser.add_argument(
            "--divide_by", default=None, required=False,
            help="Input feature to scale by. This feature is not used for clustering.")

    # Collect arguments
    args = parser.parse_args()

    data_filepath = args.data_filepath
    db_filepath = args.db_filepath
    output_path = args.output_path
    feature_list_kwarg = args.feature_list
    cluster_count_min = int(args.cluster_count_min)
    cluster_count_max = int(args.cluster_count_max)

    feature_list = feature_list_kwarg.split(",") if feature_list_kwarg else None

    image_source_path = args.image_source_path

    divide_by = args.divide_by

    run_kmeans(
            data_filepath, db_filepath=db_filepath, output_path=output_path,
            feature_list=feature_list, cluster_count_min=cluster_count_min,
            cluster_count_max=cluster_count_max,
            image_source_path=image_source_path, divide_by=divide_by,
            )
