"""
Code for creating a heatmap of samples collection using L1 distance
"""
import os
import argparse
import re
import glob

import numpy as np
import pandas as pd

from models.utils import l1_metric
from models.utils import l1_metric_normalized
import matplotlib.pyplot as plt

def sort_data(fileslist, samplecsv_path, samplecsv):
    """
    Use pandas DataFrame to sort data by class and patient
    """
    # Import csv as dataframe
    csv_file = os.path.join(samplecsv_path, samplecsv)
    df1 = pd.read_csv(csv_file, sep=",")

    # Get the basename of each file
    fnames = [os.path.basename(fname) for fname in fileslist]
    # Get the barcode from each file
    barcodes = [re.search("A[0-9]+",fname)[0] for fname in fnames]
    # Labels
    labels = df1["Cancer"].values.tolist()
    # Patients
    patients = df1["Patient"].values.tolist()

    # Create an array of the barcodes to match with fileslist
    dataframe2_arr = list(zip(barcodes, fileslist, patients, labels))

    df2 = pd.DataFrame(data=dataframe2_arr,
                       columns=["Barcode","Filename", "Patient", "Label"])

    # Sort by Label, then Patient, then Barcode, and reset index
    df2_sorted = df2.sort_values(by=["Label", "Patient", "Barcode"]).reset_index()

    return df2_sorted

def read_data(input_dir, samplecsv_path, samplecsv, size=256*256):
    """
    Reads data into numpy array
    """

    # Get list of files from input directory
    fileslist = glob.glob(os.path.join(input_dir,"*.txt"))
    fileslist.sort()
    # Sort data
    # df = sort_data(fileslist, samplecsv_path, samplecsv)
    # fileslist = df["Filename"].values.tolist()

    # Get number of files
    file_num = len(fileslist)
    # Create numpy array to store data
    # First index is the file, second is the size of raw data
    data_array = np.zeros((file_num,size))

    # Read files into array
    for idx in range(file_num):
        data_array[idx,...] = np.loadtxt(fileslist[idx]).ravel()

    return data_array

def generate_l1_matrix(data_array):
    """
    Generates a matrix of L1 distance of size data_array.shape[0] square
    """
    # Create an array to store the L1 distances
    l1_matrix = np.zeros((data_array.shape[0], data_array.shape[0]))
    # Calculate the L1 distance for each pair
    # (only need to compute half since it is symmetric)
    for idx in range(data_array.shape[0]):
        for jdx in range(data_array.shape[0]):
            l1_matrix[idx,jdx] = l1_metric_normalized(data_array[idx], data_array[jdx])

    return l1_matrix

def plot_matrix(matrix, plotdir, plotname):
    fig = plt.figure(figsize=(8,8))
    fig.set_facecolor("white")
    ax1 = fig.add_subplot(111)
    plt.title("Heatmap of L1 metric")
    ax1.imshow(matrix, cmap='magma')
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
    data_array = read_data(filesdir, samplecsv_path, samplecsv)
    # Generate the L1 matrix
    print("Generating L1 heatmap...")
    l1_matrix = generate_l1_matrix(data_array)
    # Plot the L1 matrix
    print("Plotting the L1 matrix...")
    plot_matrix(l1_matrix, plotdir, plotname)
    print("Done.")
