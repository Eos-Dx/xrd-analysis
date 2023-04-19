"""
Plot normalized peak intensity versus q
"""
import os
import argparse
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.signal import find_peaks_cwt


def plot_peak_intensity_versus_q(
        data_list_filepath, cancer_df_filepath, output_filepath=None):

    # Open cancer dataframe
    cancer_df = pd.read_csv(cancer_df_filepath, index_col="Filename")

    plt.close("all")

    # Plot
    plot_title = "Azimuthal Profiles"
    # plot_title = filename
    fig = plt.figure(plot_title, figsize=(12,8))

    with open(data_list_filepath) as infile:
        filepath_list = infile.read().splitlines()

    q_peak_list = []
    peak_intensity_list = []

    for filepath in filepath_list:
        filename = os.path.basename(filepath)
        radial_data = np.loadtxt(filepath)
        q_range = radial_data[:,0]
        mean_intensity_profile = radial_data[:,1]
        max_intensity = np.nanmax(mean_intensity_profile[q_range > 5])
        mean_intensity_profile /= max_intensity

        # Plot mean intensity profile
        plt.scatter(q_range, mean_intensity_profile, s=10)
        
        # Find peaks with continuous wavelet transform
        peak_indices = find_peaks_cwt(
                mean_intensity_profile, range(7,20), min_snr=1)

        if peak_indices.size != 0:
            # Plot peaks
            plt.scatter(
                    q_range[peak_indices], mean_intensity_profile[peak_indices],
                    s=200, marker="+")
        elif peak_indices.size == 0:
            peak_indices = []

        q_peak_list.append(q_range[peak_indices])
        peak_intensity_list.append(mean_intensity_profile[peak_indices])

    plt.xlabel(r"q $\mathrm{nm^{-1}}$")
    plt.ylabel("Intensity [photon count]")
        
    plt.title(plot_title)
    plt.show()

    # Save peak location and normalized intensity data
    peak_data = [q_peak_list, peak_intensity_list]
    with open(output_filepath, "wb") as outfile:
        pickle.dump(peak_data, outfile)

    # Plot peaks
    plot_title = "Normalized Peak Intensity Versus Position - Colored by Cancer Status"
    fig = plt.figure(plot_title, figsize=(12,8))
    cancer_seen = 0
    non_cancer_seen = 0

    for idx in range(len(q_peak_list)):
        filename = os.path.basename(os.path.abspath(filepath_list[idx]))
        cancer_status = cancer_df.loc[filename]["Cancer"]
        color = "red" if cancer_status else "blue"
        label = ""
        if cancer_status and cancer_seen == 0:
            label = "cancer"
            cancer_seen = 1
        elif ~cancer_status and non_cancer_seen == 0:
            label = "non-cancer"
            non_cancer_seen = 1

        plt.scatter(q_peak_list[idx], peak_intensity_list[idx], label=label, color=color)

    plt.xlabel(r"q $\mathrm{nm^{-1}}$")
    plt.ylabel("Intensity [photon count]")

    plt.title(plot_title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    """
    Plot peak intensity versus q
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--data_list_filepath", type=str, required=True,
            help="The path to data to extract features from")
    parser.add_argument(
            "--cancer_df_filepath", type=str, required=True,
            help="Cancer dataframe file path.")
    parser.add_argument(
            "--output_filepath", type=str, required=False,
            help="File path to save data.")

    args = parser.parse_args()

    data_list_filepath = os.path.abspath(args.data_list_filepath)
    cancer_df_filepath = os.path.abspath(args.cancer_df_filepath)
    output_filepath = os.path.abspath(args.output_filepath)

    plot_peak_intensity_versus_q(
        data_list_filepath,
        cancer_df_filepath,
        output_filepath,
        )
