"""
Example for plotting PCA-reduced K-means clusters
"""
import os
import argparse

from datetime import datetime

import numpy as np
import pandas as pd

from joblib import dump

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt


def run_pca_plot(
        training_filepath=None, blind_filepath=None, db_filepath=None,
        output_path=None, scale_by=None, feature_list=None,
        n_components=2):
    """
    """
    # Set empty blind dataframe
    df_blind = pd.DataFrame()

    # Set aspect ratio for figure size
    aspect = (16,9)

    # Set timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    # Load training data into dataframe
    df_train = pd.read_csv(training_filepath, index_col="Filename")
    df_list = [df_train]
    if blind_filepath:
        # Load training data into dataframe
        df_blind = pd.read_csv(blind_filepath, index_col="Filename")
        df_list.append(df_blind)
    # Load patients database
    db = pd.read_csv(db_filepath, index_col="Barcode")

    # Get list of features
    if feature_list is None:
        feature_list = df_train.columns.tolist()

    # Scale all features
    if scale_by:
        df_train = df_train.div(df_train[scale_by], axis="rows")
        df_train = df_train[feature_list]
        if scale_by in feature_list:
            # Drop scale_by column
            df_train = df_train.drop(columns=[scale_by])
    else:
        # Get features
        df_train = df_train[feature_list]
    if not df_blind.empty:
        # Scale all features
        if scale_by:
            df_blind = df_blind.div(df_blind[scale_by], axis="rows")
            df_blind = df_blind[feature_list]
            if scale_by in feature_list:
                # Drop scale_by column
                df_blind = df_blind.drop(columns=[scale_by])
        else:
            # Get features
            df_blind = df_blind[feature_list]

    used_feature_list = df_train.columns

    # Set title based on feature scaling
    if scale_by:
        title = "PCA-Reduced Xena Dataset 2-D Subspace Projection, divide by {}".format(scale_by)
    else:
        title = "PCA-Reduced Xena Dataset 2-D Subspace Projection"

    # Set PCA components
    pca = PCA(n_components=n_components)

    # Create pipeline including standard scaling
    # Run PCA fit on training data
    estimator = make_pipeline(StandardScaler(), pca).fit(df_train.values)

    # Save PCA estimator to file
    pca_class_filename = "pca_class_{}.joblib".format(timestamp)
    pca_class_filepath = os.path.join(output_path, pca_class_filename)
    dump(estimator, pca_class_filepath)

    print("Explained variance ratios:")
    print(estimator['pca'].explained_variance_ratio_)
    print(
            "Total explained variance:",
            np.sum(estimator['pca'].explained_variance_ratio_))

    # Print first two principal components
    pca_components = estimator['pca'].components_
    for idx in range(n_components):
        # print(dict(zip(feature_list, pca_components[idx,:])))
        print("PC{}".format(idx))
        for jdx in range(len(used_feature_list)):
            print("{},{}".format(used_feature_list[jdx], pca_components[idx,jdx]))

    # Transform data using PCA
    X_train_pca = estimator.transform(df_train.values)
    if not df_blind.empty:
        X_blind_pca = estimator.transform(df_blind.values)
        X_pca = np.vstack([X_train_pca, X_blind_pca])
    else:
        X_pca = X_train_pca

    if not df_blind.empty:
        df_all = pd.concat([df_train, df_blind])
    else:
        df_all = df_train

    # Set offsets
    x_label_offset = 0.01
    y_label_offset = 0.01

    ###################################
    # 3D PCA subspace projection plot #
    ###################################

    if False:

        # Show 3D surface maps
        plot_title = "3D {}".format(title)
        fig, ax = plt.subplots(figsize=aspect, num=plot_title, subplot_kw={"projection": "3d"})
        ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2])

        # ax.view_init(30, +60+180)

        # ax.set_title("2D Sinusoid - 3D Surface Plot")
        ax.set_xlabel("PC0")
        ax.set_ylabel("PC1")
        ax.set_zlabel("PC2")
        # ax.set_zlim([-1, 1])

        fig.tight_layout()

        plt.show()


    ################################
    # PCA subspace projection plot #
    ################################

    if False:

        # Plot PCA-reduced dataset with file labels
        plot_title = title
        fig, ax = plt.subplots(figsize=aspect, num=plot_title, tight_layout=True)
        fig.suptitle(title)
        plt.scatter(X_pca[:,0], X_pca[:,1])


        filename_list = df.index.to_list()

        # Annotate data points with filenames
        for i, filename in enumerate(filename_list):
            ax.annotate(
                filename.replace("CR_","").replace(".txt",""),
                (X_pca[i,0], X_pca[i,1]),
                xytext=(X_pca[i,0]+x_label_offset, X_pca[i,1]+y_label_offset))

        # Label plot axes and title
        plt.xlabel("PC0")
        plt.ylabel("PC1")

        plt.show()

    #################
    #  Subsets plot
    #################

    if False:

        # Collect data subsets for plotting
        series_dict = {
                "AT_series": df_AT,
                "A_series": df_A,
                "B_series": df_B,
                }

        # Plot all data subsets
        plot_title="{}, color by subset".format(title)
        fig, ax = plt.subplots(figsize=aspect, num=plot_title, tight_layout=True)
        fig.suptitle(plot_title)

        colors = {
                "AT_series": "#17becf",
                "A_series": "#bcbd22",
                "B_series": "#7f7f7f",
                }

        for series_name, df_sub in series_dict.items():
            X = estimator.transform(df_sub.values)
            plt.scatter(
                    X[:,0], X[:,1], label=series_name)
                    # X[:,0], X[:,1], label=series_name, c=colors[series_name])


        plt.xlabel("PC0")
        plt.ylabel("PC1")

        plt.legend()
        plt.show()

    ###################
    #  3D Subsets plot
    ###################

    if False:

        # Collect data subsets for plotting
        series_dict = {
                "AT_series": df_AT,
                "A_series": df_A,
                "B_series": df_B,
                }

        # Plot all data subsets
        plot_title="3D {}, color by subset".format(title)
        fig, ax = plt.subplots(figsize=aspect, num=plot_title, subplot_kw={"projection": "3d"})

        colors = {
                "AT_series": "#17becf",
                "A_series": "#bcbd22",
                "B_series": "#7f7f7f",
                }

        for series_name, df_sub in series_dict.items():
            X = estimator.transform(df_sub.values)
            ax.scatter(
                    X[:,0], X[:,1], X[:,2], label=series_name)
                    # X[:,0], X[:,1], label=series_name, c=colors[series_name])

        ax.set_xlabel("PC0")
        ax.set_ylabel("PC1")
        ax.set_zlabel("PC2")

        fig.tight_layout()

        plt.show()

    #################
    # Patients plot #
    #################

    if False:

        # Plot all data highlighting patients
        plot_title="{}, color by patient".format(title)
        fig, ax = plt.subplots(figsize=aspect, num=plot_title, tight_layout=True)
        fig.suptitle(plot_title)

        if True:
            count = 0
            for barcode in df_patients.index.str.strip():
                if count > 4:
                    break
                X_plot = estimator.transform(df[df.index.str.contains(barcode)])
                plt.scatter(X_plot[:,0], X_plot[:,1], label=barcode)
                count += 1

        plt.xlabel("PC0")
        plt.ylabel("PC1")

        plt.legend()
        plt.show()


    ##################
    # Diagnosis plot #
    ##################

    if True:

        # Plot all data highlighting patient diagnosis
        plot_title="{}, color by diagnosis".format(title)
        fig, ax = plt.subplots(figsize=aspect, num=plot_title, tight_layout=True)
        fig.suptitle(plot_title)

        # Add a Barcode column to the dataframe
        # Extract the first letter and all numbers in the filename before the subindex
        # E.g., from filename AB12345-01.txt -> A12345 is extracted
        # Note: Issue if the Barcode format changes
        extraction = df_all.index.str.extractall("CR_([A-Z]{1}).*?([0-9]+)")
        extraction_series = extraction[0] + extraction[1].str.zfill(5)
        extraction_list = extraction_series.tolist()

        assert(len(extraction_list) == df_all.shape[0])
        df_ext = df_all.copy()
        df_ext["Barcode"] = extraction_list

        df_ext = pd.merge(df_ext, db, left_on="Barcode", right_index=True)

        colors = {
                "cancer": "red",
                "healthy": "blue",
                "blind": "green",
                }

        # Loop over series
        for diagnosis in df_ext["Diagnosis"].dropna().unique():
            df_diagnosis = df_ext[df_ext["Diagnosis"] == diagnosis]
            X_plot = estimator.transform(df_diagnosis[used_feature_list])
            plt.scatter(
                    X_plot[:,0], X_plot[:,1], label=diagnosis, c=colors[diagnosis])


        # Annotate data points with filenames
        for i, filename in enumerate(df_ext.index):
            ax.annotate(
                filename.replace("CR_","").replace(".txt",""),
                (X_pca[i,0], X_pca[i,1]),
                xytext=(X_pca[i,0]+x_label_offset, X_pca[i,1]+y_label_offset))

        plt.xlabel("PC0")
        plt.ylabel("PC1")

        plt.legend()
        plt.show()


    #####################
    # 3D Diagnosis plot #
    #####################

    if True:

        # Plot all data highlighting patient diagnosis
        plot_title="3D {}, color by diagnosis".format(title)
        fig, ax = plt.subplots(figsize=aspect, num=plot_title, subplot_kw={"projection": "3d"})

        # Add a Barcode column to the dataframe
        # Extract the first letter and all numbers in the filename before the subindex
        # E.g., from filename AB12345-01.txt -> A12345 is extracted
        # Note: Issue if the Barcode format changes
        extraction = df_all.index.str.extractall("CR_([A-Z]{1}).*?([0-9]+)")
        extraction_series = extraction[0] + extraction[1].str.zfill(5)
        extraction_list = extraction_series.tolist()

        assert(len(extraction_list) == df_all.shape[0])
        df_ext = df_all.copy()
        df_ext["Barcode"] = extraction_list

        df_ext = pd.merge(df_ext, db, left_on="Barcode", right_index=True)

        colors = {
                "cancer": "red",
                "healthy": "blue",
                "blind": "green",
                }

        markers = {
                "cancer": "^",
                "healthy": "o",
                # "blind": "+",
                }

        # Loop over series
        for diagnosis in df_ext["Diagnosis"].dropna().unique():
            df_diagnosis = df_ext[df_ext["Diagnosis"] == diagnosis]
            X_plot = estimator.transform(df_diagnosis[used_feature_list])
            ax.scatter(
                    X_plot[:,0], X_plot[:,1], X_plot[:,2], label=diagnosis, c=colors[diagnosis])

        if False:
            # Annotate data points with filenames
            for i, filename in enumerate(df_ext.index):
                label = filename.replace("CR_","").replace(".txt","")
                ax.text(X_pca[i,0], X_pca[i,1], X_pca[i,2], label)

        ax.set_xlabel("PC0")
        ax.set_ylabel("PC1")
        ax.set_zlabel("PC2")

        # Set axis limits
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_zlim([-4, 4])

        ax.set_proj_type('persp')

        fig.tight_layout()

        plt.show()



    #################
    # K-means plots #
    #################

    if True:

        # df_pca = data + kmeans cluster labels
        df_pca = pd.DataFrame(data=X_train_pca, index=df_train.index)
        # df_pca.to_csv(kmeans_filepath)

        cluster_count_min = 20
        cluster_count_max = 20

        # Run K-means on pca-reduced features
        for idx in range(cluster_count_min, cluster_count_max+1):

            cluster_count = idx

            if False:
                title = "K-Means on PCA-reduced Xena Dataset with {} clusters".format(idx)
                fig, ax = plt.subplots(figsize=aspect, num=title, tight_layout=True)
                fig.suptitle(title)

            # Run k-means
            reduced_data = X_train_pca
            kmeans = KMeans(cluster_count)
            kmeans.fit(reduced_data)

            df_pca["kmeans_{}".format(idx)] = kmeans.labels_

            if False:

                # Plot decision boundaries
                # Step size of the mesh. Decrease to increase the quality of the VQ.
                h = 0.01  # point in the mesh [x_min, x_max]x[y_min, y_max].

                # Plot the decision boundary. For that, we will assign a color to each
                x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
                y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

                # Obtain labels for each point in mesh. Use last trained model.
                Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
                # Put the result into a color plot
                Z = Z.reshape(xx.shape)

                plt.imshow(
                    Z,
                    interpolation="nearest",
                    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                    cmap=plt.cm.Paired,
                    aspect="auto",
                    origin="lower",
                )

                # Plot the data
                plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=10)

                # Plot the centroids
                centroids = kmeans.cluster_centers_
                print("Centroids:")
                print(centroids)
                plt.scatter(
                        centroids[:, 0],
                        centroids[:, 1],
                        marker="x",
                        s=200,
                        linewidths=3,
                        color="w",
                        zorder=10000,
                        )

                plt.xlabel("PC0")
                plt.ylabel("PC1")

            # Show K-means plots
            plt.show()


        extraction = df_pca.index.str.extractall("CR_([A-Z]{1}).*?([0-9]+)")
        extraction_series = extraction[0] + extraction[1].str.zfill(5)
        extraction_list = extraction_series.tolist()

        assert(len(extraction_list) == df_pca.shape[0])

        df_pca_ext = df_pca.copy()
        df_pca_ext["Barcode"] = extraction_list
        df_pca_ext = pd.merge(df_pca_ext, db, left_on="Barcode", right_index=True)
        df_pca_ext = df_pca_ext.rename(columns={0: "PC0", 1: "PC1", 2: "PC2"})

        # Save dataframe
        kmeans_pca_filename = "kmeans_pca_model_{}.joblib".format(timestamp)
        kmeans_pca_filepath = os.path.join(output_path, kmeans_pca_filename)
        df_pca_ext.to_csv(kmeans_pca_filepath )


if __name__ == '__main__':
    """
    Run PCA and K-means on PCA-reduced data and generate plots
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--training_filepath", default=None, required=True,
            help="The csv input file containing training data features")
    parser.add_argument(
            "--blind_filepath", default=None, required=False,
            help="The csv input file containing blind data features")
    parser.add_argument(
            "--db_filepath", default=None, required=True,
            help="The csv input file containing patient data")
    parser.add_argument(
            "--output_path", default=None, required=True,
            help="The output path to save PCA and K-means data")
    parser.add_argument(
            "--scale_by", default=None, required=False,
            help="The name of the feature to scale by")
    parser.add_argument(
            "--feature_list", default=None, required=False,
            help="A list of features to analyze")
    parser.add_argument(
            "--n_components", type=int, default=2, required=False,
            help="Number of PCA components")

    args = parser.parse_args()

    training_filepath = args.training_filepath
    blind_filepath = args.blind_filepath
    db_filepath = args.db_filepath
    output_path = args.output_path
    scale_by = args.scale_by
    feature_list = args.feature_list
    n_components = args.n_components

    run_pca_plot(
        training_filepath=training_filepath, blind_filepath=blind_filepath,
        db_filepath=db_filepath, output_path=output_path, scale_by=scale_by,
        feature_list=feature_list, n_components=n_components)
