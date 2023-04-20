"""
Perform PCA on radial intensity profiles
"""
import os
import glob

import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline



def run_pca_plot(input_path, annotate=False):

    ##############
    # Load data
    ##############

    filepath_list = glob.glob(os.path.join(input_path, "radial_dist_12mm*.txt"))
    filepath_list.sort()

    array_len = 225
    dataset_size = len(filepath_list)


    X = np.zeros((dataset_size, array_len))
    y = np.zeros(dataset_size)

    for idx in range(dataset_size):
        filepath = filepath_list[idx]
        filename = os.path.basename(filepath)
        # Set diagnosis based on filename
        diagnosis = 0 if "SC" in filename else 1
        y[idx] = diagnosis
        # Get data
        radial_data = np.loadtxt(filepath)
        mean_intensity_profile = radial_data[17:-120,1]
        # Scaling
        max_intensity = np.nanmax(mean_intensity_profile)
        mean_intensity_profile /= max_intensity
    #    total_intensity = np.nansum(mean_intensity_profile)
    #    mean_intensity_profile /= total_intensity
        X[idx,:] = mean_intensity_profile


    ##########
    # 3-D PCA
    ##########

    n_components = 3
    pca = PCA(n_components=n_components)

    estimator_3d = make_pipeline(StandardScaler(), pca)
    estimator_3d.fit(X)


    ###########
    # 3-D Plot
    ###########

    plot_title = "3D PCA on {} features colored by diagnosis".format(X.shape[1])
    aspect = (16,9)

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
        label = "tumor" if diagnosis == 1 else "control"
        ax.scatter(
                X_pca_diagnosis[:,pc_a],
                X_pca_diagnosis[:,pc_b],
                X_pca_diagnosis[:,pc_c],
                c=colors[diagnosis], label=label, s=10)

    if annotate:
        # Annotate data points with filenames
        for i, filepath in enumerate(filepath_list):
            filename = os.path.basename(filepath).replace(
                    "radial_dist_12mm_SLA1_","").replace(
                            ".txt", "")
            ax.text(
                X_pca[i,pc_a], X_pca[i,pc_b], X_pca[i,pc_c],
                filename,
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

    n_components = 2
    pca = PCA(n_components=n_components)

    estimator_2d = make_pipeline(StandardScaler(), pca)
    estimator_2d.fit(X)


    ###########
    # 2-D Plot
    ###########

    plot_title = "2D PCA on {} features colored by diagnosis".format(X.shape[1])
    aspect = (16,9)

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
        label = "tumor" if diagnosis == 1 else "control"
        ax.scatter(
                X_pca_diagnosis[:,pc_a],
                X_pca_diagnosis[:,pc_b],
                c=colors[diagnosis], label=label, s=10)

    if annotate:
        # Annotate data points with filenames
        for i, filepath in enumerate(filepath_list):
            filename = os.path.basename(filepath).replace(
                    "radial_dist_12mm_SLA1_","").replace(
                            ".txt", "")
            ax.text(
                X_pca[i,pc_a], X_pca[i,pc_b],
                filename,
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
            "--input_path", default=None, required=True,
            help="The path to training data files")
    parser.add_argument(
            "--annotate", action="store_true",
            help="Annotate plots with file names.")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path) if args.input_path \
            else None
    annotate = args.annotate

    run_pca_plot(input_path, annotate=annotate)
