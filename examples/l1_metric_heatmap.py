"""
Code for creating a heatmap of samples collection using L1 distance
"""
import os
import argparse
import re
import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from eosdxanalysis.models.utils import l1_metric
from eosdxanalysis.models.utils import l1_metric_optimized
from eosdxanalysis.models.utils import cluster_corr

from eosdxanalysis.visualization.utils import heatmap
from eosdxanalysis.visualization.utils import annotate_heatmap

params = {
    "h": 256,
    "w": 256,
    "beam_rmax": 25,
    "rmin": 25,
    "rmax": 90,
    "eyes_rmin": 30,
    "eyes_rmax": 45,
    "eyes_blob_rmax": 20,
    "eyes_percentile": 99,
    "local_thresh_block_size": 21,
    "crop_style": "both"
}


def read_data(input_dir, samplecsv_path, samplecsv, size=256*256):
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
    df_subset = df[df["Barcode"].isin(barcodes)].copy().reset_index()
    del df_subset["index"]

    # Now add data
    df_subset["PreprocessedData"] = list(data_array)

    return df_subset

def generate_l1_matrix(df):
    """
    Generates a matrix of L1 distance of size data_array.shape[0] square
    """
    # Create an array to store the L1 distances
    data_array = df["PreprocessedData"].values
    barcodes = df["Barcode"].values.tolist()
    matrix_df = pd.DataFrame([], barcodes, barcodes)

    # Calculate the L1 distance for each pair
    # (only need to compute half since it is symmetric)
    for idx in range(data_array.shape[0]):
        for jdx in range(data_array.shape[0]):
            # Get the barcode and image data
            barcode1 = barcodes[idx]
            barcode2 = barcodes[jdx]
            image1 = data_array[idx].reshape((256,256)).astype(np.uint16)
            image2 = data_array[jdx].reshape((256,256)).astype(np.uint16)
            # Calculate the L1 distance (minimization optimization algorithm)
            # between the images
            distance = l1_metric_optimized(image1, image2, params)
            matrix_df[barcode1][barcode2] = distance
    return matrix_df

def plot_matrix(matrix_df, log_df, plotdir, plotname):
    fig = plt.figure(figsize=(8,8))
    fig.set_facecolor("white")
    ax1 = fig.add_subplot(111)

    plt.title("Heatmap of L1 metric (inverse) correlation matrix")

    matrix = matrix_df.to_numpy(dtype=np.float64)
    # matrix = matrix_df
    # ax1.imshow(matrix,cmap='magma')

    # Get barcodes
    barcodes = matrix_df.head()

    im, cbar = heatmap(matrix, barcodes, barcodes, ax=ax1,
                       cmap="magma", cbarlabel="L1 Distance")
    # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

    # Color the axis labels according to cancer label
    # Get cancer labels
    cancer_labels = log_df["Cancer"]

    color_dict = {
            0: 'b',
            1: 'r',
            }
    for xtick, ytick, cancer_label in zip(
                        ax1.get_xticklabels(), ax1.get_yticklabels(), cancer_labels):
        xtick.set_color(color_dict[cancer_label])
        ytick.set_color(color_dict[cancer_label])

    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, plotname))


if __name__ == '__main__':
    print("Current directory: " + os.getcwd())

    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument("filesdir", help="The files directory to analyze")
    parser.add_argument("plotdir", help="The directory to save plot")
    parser.add_argument("plotname", help="The output file name to save plot")
    parser.add_argument("samplecsv_path", help="The path to the csv file with labels")
    parser.add_argument("samplecsv", help="The filename of the csv file with labels")

    args = parser.parse_args()

    # Set variables based on input arguments
    filesdir = args.filesdir
    plotdir = args.plotdir
    plotname = args.plotname
    samplecsv_path = args.samplecsv_path
    samplecsv = args.samplecsv

    # Read the data
    print("Reading input data...")
    df1 = read_data(filesdir, samplecsv_path, samplecsv)
    # Generate the L1 matrix
    print("Generating L1 heatmap...")
    l1_df = generate_l1_matrix(df1)
    # print("Clustering...")
    # l1_df_clustered_np = cluster_corr(l1_df.to_numpy(dtype=np.float64))
    # Plot the L1 matrix
    print("Plotting the L1 matrix...")
    plot_matrix(l1_df, df1, plotdir, plotname)
    print("Saving data to file...")
    l1_df.to_csv(os.path.join(plotdir, "l1_heatmap.csv"))
    # l1_df_clustered.to_csv(os.path.join(plotdir, "l1_heatmap_clustered.csv"))
    print("Done.")
