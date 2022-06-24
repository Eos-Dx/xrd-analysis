"""
Implements K-means clustering analysis
See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
"""
import os
import argparse
import glob
import re

import numpy as np
import pandas as pd
import scipy as sp

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report


def read_data(input_dir, samplecsv_path, samplecsv, size=256*256, average=False):
    """
    Reads data into numpy array
    """

    # Get list of files from input directory
    fileslist = glob.glob(os.path.join(input_dir,"*.txt"))
    # Sort files list
    fileslist.sort()
    barcodes = [re.search("A[0-9]+",fname)[0] for fname in fileslist]

    # Get number of files
    file_num = len(fileslist)
    # Create numpy array to store data
    # First index is the file, second is the size of raw data
    data_array = np.zeros((file_num,size))

    # Read preprocessed measurments data from files into array
    for idx in range(file_num):
        data_array[idx,...] = np.loadtxt(fileslist[idx]).ravel()

    # Read cancer class labels from csv
    df = pd.read_csv(os.path.join(samplecsv_path, samplecsv), sep=",")

    # Now take subset of data from csv file
    df_subset = df[df["Barcode"].isin(barcodes)].copy()

    # Now add data to dataframe
    df_subset["PreprocessedData"] = list(data_array)

    if average:

        # Calculate average
        series_mean = df_subset.groupby("Patient")["PreprocessedData"].agg('mean').copy()
        df_mean = pd.DataFrame({'Patient':series_mean.index, 'PreprocessedData':series_mean.values})

        # Add back in Cancer labels
        # First, delete unnecessary data from df_subset
        df_subset.drop(["Barcode", "PreprocessedData"], axis=1, inplace=True)
        df_subset.drop_duplicates(inplace=True)

        # Now merge
        df_merge = df_subset.merge(df_mean)

        return df_merge

    else:
        return df_subset

def run_kmeans_analysis(df, clusters):
    X = np.array(df["PreprocessedData"].tolist(), dtype=np.float64)

    kmeans = KMeans(n_clusters=clusters, random_state=0, ).fit(X)

    print("Score: ", kmeans.score(X))

    cancer_labels = df["Cancer"].tolist()
    kmeans_labels = np.array(kmeans.labels_)

    if len(kmeans_labels) == 2 and len(cancer_labels) == 2:
        # Re-assign kmeans labels based on popularity (mode)
        kmeans_mode = sp.stats.mode(kmeans.labels_)[0]
        orig_mode = sp.stats.mode(cancer_labels)[0]

        if kmeans_mode != orig_mode:
            kmeans_labels[kmeans_labels == 0] = 2
            kmeans_labels[kmeans_labels == 1] = 0
            kmeans_labels[kmeans_labels == 0] = 1

        # Print classification report
        print(classification_report(
            cancer_labels,
            kmeans_labels,
            target_names=["Normal", "Cancer"]))

    # Store clusters in dataframe
    df.insert(3, "Cluster", list(kmeans_labels))

    df_ext = df.set_index("Patient").copy()

    del df_ext["PreprocessedData"]

    return df_ext

if __name__ == '__main__':
    print("Current directory: " + os.getcwd())

    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument("clusters", type=int, help="The number of clusters to create")
    parser.add_argument("average", type=int, help="Average data per patient")
    parser.add_argument("filesdir", help="The files directory to analyze")
    parser.add_argument("samplecsv_path", help="The path to the csv file with labels")
    parser.add_argument("samplecsv", help="The filename of the csv file with labels")
    parser.add_argument("outputdir", help="The output directory of the results")
    parser.add_argument("outputfile", help="The output filename of the results")

    args = parser.parse_args()

    # Set variables based on input arguments
    filesdir = args.filesdir
    samplecsv_path = args.samplecsv_path
    samplecsv = args.samplecsv
    outputdir = args.outputdir
    outputfile = args.outputfile
    clusters = args.clusters
    average = args.average

    # Read the data
    print("Reading input data...")
    df = read_data(filesdir, samplecsv_path, samplecsv, average=average)

    # Perform K-means
    print("Performing K-means clustering...")
    df_ext = run_kmeans_analysis(df, clusters)

    # Save results
    df_ext.to_csv(os.path.join(outputdir, outputfile))

    print("Done.")
