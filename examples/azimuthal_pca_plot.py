"""
Perform PCA on radial intensity profiles
"""
import os
import glob
import re

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline



def run_pca_plot(
        input_path=None, input_dataframe_filepath=None, annotate=False,
        distance_mm="29", scaling=None):

    ##############
    # Load data
    ##############

    if input_dataframe_filepath:
        input_path = os.path.basename(input_dataframe_filepath)
        input_df = pd.read_csv(input_dataframe_filepath, index_col="Filename")
        filepath_list = input_df["Filepath"].tolist()

    elif input_path:
        filepath_list = glob.glob(os.path.join(
            input_path, "radial_dist_{}mm*.txt".format(distance_mm)
            ))
        filepath_list.sort()

    # Get the size from the first file
    shape_orig = np.loadtxt(filepath_list[0]).shape
    dataset_size = len(filepath_list)
    radial_profile_size = shape_orig[0]

    X_unscaled = np.zeros((dataset_size, radial_profile_size))
    q_ranges = np.zeros_like(X_unscaled)
    y = np.zeros(dataset_size)

    start_index = 0
    end_index = 1e6
    for idx in range(dataset_size):
        filepath = filepath_list[idx]
        filename = os.path.basename(filepath)
        # Set diagnosis based on filename
        if input_dataframe_filepath:
            diagnosis = 1 if "Cancer" in filepath else 0
        else:
            diagnosis = 0 if ("SC" or  "LC") in filename else 1
        y[idx] = diagnosis
        # Get data
        radial_data = np.loadtxt(filepath)
        mean_intensity_profile = radial_data[:,1]
        q_ranges[idx,:] = radial_data[:,0]

        # Clip nan
        finite_indices = np.where(np.isfinite(mean_intensity_profile))
        start_index_sub = finite_indices[0][0]
        end_index_sub = finite_indices[0][-1]
        start_index = start_index_sub if start_index_sub > start_index \
                else start_index
        end_index = end_index_sub if end_index_sub < end_index \
                else end_index

        X_unscaled[idx,:] = mean_intensity_profile

    # Set final array length after removing nans
    array_len = end_index - start_index

    q_range = q_ranges[0, start_index:end_index]
    q_max = q_range[-1]
    q_min = q_range[0]
    uniform_q_range = np.linspace(q_min, q_max, num=array_len)

    # Rescale each sample to the same q-range
    for idx in range(X_unscaled.shape[0]):
        sample_q_range = q_ranges[idx]
        sample_values = X_unscaled[idx,:]
        sample_interp = interp1d(sample_q_range, sample_values)
        X_unscaled[idx,:array_len] = sample_interp(uniform_q_range)

    X_unscaled = X_unscaled[:, :array_len]

    if scaling == "max":
        X_max = np.max(X_unscaled, axis=1).reshape(-1,1)
        X = X_unscaled/X_max
    elif scaling == "sum":
        X_sum = np.sum(X_unscaled, axis=1).reshape(-1,1)
        X = X_unscaled/X_sum
    else:
        X = X_unscaled

    ##########
    # 3-D PCA
    ##########

    n_components = 3
    pca = PCA(n_components=n_components)

    estimator_3d = make_pipeline(StandardScaler(), pca)
    estimator_3d.fit(X)

    pca_3d = estimator_3d["pca"]
    print("Explained variance ratios:")
    print(pca_3d.explained_variance_ratio_)
    print("Total explained variance:",
            np.sum(pca_3d.explained_variance_ratio_))

    # Print principal components
    pca_3d_components = pca_3d.components_
    for idx in range(n_components):
        # print(dict(zip(feature_list, pca_components[idx,:])))
        print("PC{}".format(idx))
        for jdx in range(array_len):
            print("{},{:.2f},{}".format(
                jdx, uniform_q_range[jdx], pca_3d_components[idx,jdx]))

    ###########
    # 3-D Plot
    ###########

    plot_title = "3D PCA on {} features at {} mm colored by diagnosis".format(
            X.shape[1], distance_mm)
    aspect = (12,8)

    fig, ax = plt.subplots(figsize=aspect, num=plot_title, subplot_kw={"projection": "3d"})

    colors = {
            1: "red",
            0: "blue",
            }

    pc_a = 0
    pc_b = 1
    pc_c = 2

    X_pca = estimator_3d.transform(X)

    for diagnosis in colors.keys():
        X_pca_diagnosis = X_pca[y == diagnosis, :]
        label = "cancer" if diagnosis == 1 else "non-cancer"
        ax.scatter(
                X_pca_diagnosis[:,pc_a],
                X_pca_diagnosis[:,pc_b],
                X_pca_diagnosis[:,pc_c],
                c=colors[diagnosis], label=label, s=10)

    if annotate:
        # Annotate data points with filenames
        for i, filepath in enumerate(filepath_list):
            filename = os.path.basename(filepath).format(distance_mm)
            annotation = re.sub(
                    "radial_dist_[0-9]{1,3}mm_SLA1_",
                    "",
                    filename).replace(".txt", "")

            ax.text(
                X_pca[i,pc_a], X_pca[i,pc_b], X_pca[i,pc_c],
                annotation,
                fontsize=10)

    ax.set_xlabel("PC{}".format(pc_a))
    ax.set_ylabel("PC{}".format(pc_b))
    ax.set_zlabel("PC{}".format(pc_c))
    # ax.set_zlim([-1, 1])

    ax.set_title(plot_title)
    ax.legend()

    fig.tight_layout()

    plt.show()


    ##########
    # 2-D PCA
    ##########

    n_components = 3
    pca = PCA(n_components=n_components)

    estimator_2d = make_pipeline(StandardScaler(), pca)
    estimator_2d.fit(X)

    pca_2d = estimator_2d["pca"]
    print("Explained variance ratios:")
    print(pca_2d.explained_variance_ratio_)
    print("Total explained variance:",
            np.sum(pca_2d.explained_variance_ratio_))

    # Print principal components
    pca_2d_components = pca_2d.components_
    for idx in range(n_components):
        # print(dict(zip(feature_list, pca_components[idx,:])))
        print("PC{}".format(idx))
        for jdx in range(array_len):
            print("{},{:.2f},{}".format(
                jdx, uniform_q_range[jdx], pca_2d_components[idx,jdx]))

    # Plot principal components
    for idx in range(n_components):
        plot_title = "PC{} Loading Plot".format(idx)
        fig = plt.figure(plot_title)
        plt.scatter(q_range, pca_2d_components[idx,:], s=10)

        plt.xlabel(r"q $\mathrm{{nm}^{-1}}$")
        plt.ylabel("Loading")

        plt.title(plot_title)
    plt.show()


    ###########
    # 2-D Plot
    ###########

    plot_title = "2D PCA on {} features at {} mm colored by diagnosis".format(
            X.shape[1], distance_mm)
    aspect = (12,8)

    fig, ax = plt.subplots(figsize=aspect, num=plot_title)

    colors = {
            1: "red",
            0: "blue",
            }

    pc_a = 0
    pc_b = 1

    X_pca = estimator_2d.transform(X)

    for diagnosis in colors.keys():
        X_pca_diagnosis = X_pca[y == diagnosis, :]
        label = "cancer" if diagnosis == 1 else "non-cancer"
        ax.scatter(
                X_pca_diagnosis[:,pc_a],
                X_pca_diagnosis[:,pc_b],
                c=colors[diagnosis], label=label, s=10)

    if annotate:
        # Annotate data points with filenames
        for i, filepath in enumerate(filepath_list):
            filename = os.path.basename(filepath).format(distance_mm)
            annotation = re.sub(
                    "radial_dist_[0-9]{1,3}mm_SLA1_",
                    "",
                    filename).replace(".txt", "")
            ax.text(
                X_pca[i,pc_a], X_pca[i,pc_b],
                annotation,
                fontsize=10)



    ax.set_xlabel("PC{}".format(pc_a))
    ax.set_ylabel("PC{}".format(pc_b))
    # ax.set_zlim([-1, 1])

    ax.set_title(plot_title)
    ax.legend()

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    """
    Run PCA and K-means on PCA-reduced data and generate plots
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", default=None, required=False,
            help="The path to training data files")
    parser.add_argument(
            "--input_dataframe_filepath", default=None, required=False,
            help="The path to training data files")
    parser.add_argument(
            "--annotate", action="store_true",
            help="Annotate plots with file names.")
    parser.add_argument(
            "--distance_mm", type=int,
            help="Approximate sample distance in mm (integer).")
    parser.add_argument(
            "--scaling", type=str,
            help="Scaling method to use: \"max\" (default), \"min\", or \"mean\"")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path) if args.input_path \
            else None
    input_dataframe_filepath = args.input_dataframe_filepath
    annotate = args.annotate
    distance_mm = args.distance_mm
    scaling = args.scaling

    run_pca_plot(
            input_path=input_path,
            input_dataframe_filepath=input_dataframe_filepath,
            annotate=annotate,
            distance_mm=distance_mm,
            scaling=scaling,
            )
