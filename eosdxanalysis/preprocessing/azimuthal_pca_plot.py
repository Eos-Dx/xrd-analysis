"""
Perform PCA on radial intensity profiles
"""
import os
import glob
import re

from datetime import datetime

from joblib import dump
from joblib import load

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
        distance_mm="29", scaling=None, output_path=None,
        patient_dataframe_filepath=None, estimator_filepath=None,
        q_min=None, q_max=None):

    ##############
    # Load data
    ##############

    if input_dataframe_filepath:
        input_path = os.path.basename(input_dataframe_filepath)
        input_df = pd.read_csv(input_dataframe_filepath, index_col="Filename")
        filepath_list = input_df["Filepath"].tolist()

    elif input_path:
        filepath_list = glob.glob(os.path.join(
            input_path, "radial_*.txt"
            ))
        filepath_list.sort()

    # Associate file with patient with diagnosis
    filename_list = [os.path.basename(filepath) for filepath in filepath_list]
    assert(len(filename_list) == len(filepath_list))
    df = pd.DataFrame(data={}, index=filepath_list)
    df["Filename"] = filename_list
    df["Sample_ID"] = df["Filename"].astype(str).str.replace(
            "radial_", "").str.replace(
                    "\.txt.*", "", regex=True)

    # Get patient diagnoses
    df_patients = pd.read_csv(patient_dataframe_filepath, index_col="Patient_ID")
    df_ext = pd.merge(df, df_patients, left_on="Sample_ID", right_on="Sample_ID")
    df_ext.index = df["Filename"]

    # Get the size from the first file
    shape_orig = np.loadtxt(filepath_list[0]).shape
    dataset_size = len(filepath_list)
    radial_profile_size = shape_orig[0]

    X_unscaled = np.zeros((dataset_size, radial_profile_size))

    # Get diagnoses
    y = df_ext["Diagnosis"].replace("non_cancer", 0).replace("cancer", 1).values

    # Set q-range
    array_len=256
    q_range = np.linspace(q_min, q_max, num=array_len)

    # TODO: Refactor this code
    # Read mean radial intensity data from files
    for idx in range(dataset_size):
        filepath = filepath_list[idx]
        filename = os.path.basename(filepath)

        # Get data
        radial_data = np.loadtxt(filepath)
        mean_intensity_profile = radial_data[:,1]

        # Perform quality control on q-range
        sample_q_range = radial_data[:,0]
        if any([sample_q_range[0] > q_min, sample_q_range[-1] < q_max]):
            message = "File ``{}`` q-range is too small, {}-{}," + \
                    " should be {}-{}.".format(
                filename_list[idx],
                sample_q_range[0],
                sample_q_range[-1],
                q_range[0],
                q_range[-1])
            raise ValueError(message)

        # Store data in array
        X_unscaled[idx,:] = mean_intensity_profile

    # Rescale each sample to the same q-range
    for idx in range(X_unscaled.shape[0]):
        sample_values = X_unscaled[idx,:]
        sample_interp = interp1d(sample_q_range, sample_values)
        interp_profile = sample_interp(q_range)
        X_unscaled[idx,:array_len] = interp_profile

    X_unscaled = X_unscaled[:, :array_len]

    if scaling == "sum":
        X_sum = np.sum(X_unscaled, axis=1).reshape(-1,1)
        X = X_unscaled/X_sum
    elif scaling == "max":
        X_max = np.max(X_unscaled, axis=1).reshape(-1,1)
        X = X_unscaled/X_max
    else:
        X = X_unscaled

    ##########
    # 3-D PCA
    ##########

    n_components = 3
    pca = PCA(n_components=n_components)


    if not estimator_filepath:
        estimator_3d = make_pipeline(StandardScaler(), pca)
    else:
        estimator_3d = load(estimator_filepath)

    estimator_3d.fit(X)

    X_pca = estimator_3d.transform(X)

    # Save
    if not output_path:
        # Set output path to parent directory of input path
        output_path = os.path.join(
                os.path.dirname(input_path),
                "preprocessed_data")

    # Create timestamped data output directory
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    data_dir = "pca_{}_scaling_results_{}".format(scaling, timestamp)
    pca_output_path = os.path.join(output_path, data_dir)

    os.makedirs(pca_output_path, exist_ok=True)

    # Save dataframe
    pca_data_output_filename = "pca_{}mm_{}_scaling_{}.csv".format(
            distance_mm, scaling, timestamp)
    filename_list = [os.path.basename(filepath) \
            for filepath in filepath_list]
    columns = {"PC0", "PC1", "PC2"}
    df = pd.DataFrame(data=X_pca, columns=columns)
    df["Filename"] = filename_list
    df = df.set_index("Filename")
    # Set cancer label
    df["Cancer"] = ~(
            df.index.str.contains("SC") | df.index.str.contains("LC"))
    pca_data_output_filepath = os.path.join(
            pca_output_path, pca_data_output_filename)
    df.to_csv(pca_data_output_filepath)

    # Save model
    model_output_filename = "estimator_3d_{}.joblib".format(timestamp)
    model_output_filepath = os.path.join(
            pca_output_path, model_output_filename)
    dump(estimator_3d, model_output_filepath)

    # Save q range
    q_range_output_filename = "q_range_{}.txt".format(timestamp)
    q_range_output_filepath = os.path.join(
            pca_output_path, q_range_output_filename)
    print(q_range_output_filepath)
    np.savetxt(q_range_output_filepath, q_range)

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
                jdx, q_range[jdx], pca_3d_components[idx,jdx]))

    ###########
    # 3-D Plot
    ###########

    plot_title = "3D PCA on {} features at {} mm colored by diagnosis".format(
            X.shape[1], distance_mm)
    if scaling:
        plot_title += " scaled by {}".format(scaling)
    aspect = (12,8)

    fig, ax = plt.subplots(figsize=aspect, num=plot_title, subplot_kw={"projection": "3d"})

    colors = {
            1: "red",
            0: "blue",
            }

    pc_a = 0
    pc_b = 1
    pc_c = 2




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


    if not estimator_filepath:
        estimator_2d = make_pipeline(StandardScaler(), pca)
    else:
        estimator_2d = load(estimator_filepath)

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
                jdx, q_range[jdx], pca_2d_components[idx,jdx]))

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
    if scaling:
        plot_title += " scaled by {}".format(scaling)
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
        for i, filepath in enumerate(filepath_list[:]):
            filename = os.path.basename(filepath).format(distance_mm)
            annotation = filename.replace(
                    "radial_",
                    "",
                    ).replace(".txt", "")
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
            help="Scaling method to use: \"sum\" (default), or \"max\"")
    parser.add_argument(
            "--output_path", type=str,
            help="Path to save PCA-transformed data and model.")
    parser.add_argument(
            "--patient_dataframe_filepath", type=str,
            help="Path to save PCA-transformed data and model.")
    parser.add_argument(
            "--estimator_filepath", type=str,
            help="Path to PCA estimator.")
    parser.add_argument(
            "--q_range", type=str, required=True,
            help="Specify q-range as ``min-max`` string.")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path) if args.input_path \
            else None
    input_dataframe_filepath = args.input_dataframe_filepath
    annotate = args.annotate
    distance_mm = args.distance_mm
    scaling = args.scaling
    output_path = args.output_path
    estimator_filepath = args.estimator_filepath
    patient_dataframe_filepath = args.patient_dataframe_filepath
    q_range_min_max = args.q_range
    if q_range_min_max:
        q_min, q_max = q_range_min_max.split("-")

    run_pca_plot(
            input_path=input_path,
            input_dataframe_filepath=input_dataframe_filepath,
            annotate=annotate,
            distance_mm=distance_mm,
            scaling=scaling,
            output_path=output_path,
            patient_dataframe_filepath=patient_dataframe_filepath,
            estimator_filepath=estimator_filepath,
            q_min=int(q_min),
            q_max=int(q_max),
            )
