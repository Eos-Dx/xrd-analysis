"""
Example for plotting PCA-reduced K-means clusters
"""
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

# Set aspect ratio for figure size
aspect = (16,9)

# Load data into dataframe
df_path = "extracted_features.csv"
df = pd.read_csv(df_path, index_col="Filename")

# Load patients database
db_path = "patients_database.csv"
db = pd.read_csv(db_path, index_col="Barcode")

# Set kmeans csv output filepath
kmeans_filepath="kmeans_pca.csv"

# Specify feature list
feature_list = [
        # 'total_intensity',
        # 'total_flux',
        # 'bright_pixel_count',
        # 'max_intensity',
        # 'annulus_intensity_9A',
        # 'annulus_intensity_5A',
        'annulus_intensity_inner',
        'annulus_intensity_4A',
        'sector_intensity_equator_pair_9A',
        'sector_intensity_meridian_pair_9A',
        'sector_intensity_meridian_pair_5A',
        ]

if True:
    # Divide
    # divide_by = "cropped_intensity"
    divide_by = "max_intensity"
    # divide_by = "total_flux"
    df = df.div(df[divide_by], axis="rows")
    title = "PCA-Reduced Xena Dataset 2-D Subspace Projection, divide by {}".format(divide_by)
else:
    title = "PCA-Reduced Xena Dataset 2-D Subspace Projection"

# Set features to use
df = df[feature_list]

df_AT = df[df.index.str.match(r"(CR_AT)")]
df_B = df[df.index.str.match(r"(CR_B)")]
df_A = df[df.index.str.match(r"(CR_A0|CR_AA)")]

# Exclude B-series
df_train = df[~df.index.isin(df_B.index)]
# df_train = df

# Set PCA to 2 components
n_components = 3
pca = PCA(n_components=n_components)

# Create pipeline including standard scaling
estimator = make_pipeline(StandardScaler(), pca).fit(df_train.values)
# kmeans = KMeans(init=estimator['pca'].components_, n_clusters=cluster_count)


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
    for jdx in range(len(feature_list)):
        print("{},{}".format(feature_list[jdx], pca_components[idx,jdx]))

# Transform data using PCA
X_pca = estimator.transform(df.values)
X_train_pca = estimator.transform(df_train.values)

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
    extraction = df_train.index.str.extractall("CR_([A-Z]{1}).*?([0-9]+)")
    extraction_series = extraction[0] + extraction[1].str.zfill(5)
    extraction_list = extraction_series.tolist()

    assert(len(extraction_list) == df_train.shape[0])
    df_ext = df_train.copy()
    df_ext["Barcode"] = extraction_list

    df_ext = pd.merge(df_ext, db, left_on="Barcode", right_index=True)

    colors = {
            "cancer": "red",
            "healthy": "blue",
            # "blind": "green",
            }

    # Loop over series
    for diagnosis in df_ext["Diagnosis"].dropna().unique():
        df_diagnosis = df_ext[df_ext["Diagnosis"] == diagnosis]
        X_plot = estimator.transform(df_diagnosis[feature_list])
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
    extraction = df_train.index.str.extractall("CR_([A-Z]{1}).*?([0-9]+)")
    extraction_series = extraction[0] + extraction[1].str.zfill(5)
    extraction_list = extraction_series.tolist()

    assert(len(extraction_list) == df_train.shape[0])
    df_ext = df_train.copy()
    df_ext["Barcode"] = extraction_list

    df_ext = pd.merge(df_ext, db, left_on="Barcode", right_index=True)

    colors = {
            "cancer": "red",
            "healthy": "blue",
            # "blind": "green",
            }

    # Loop over series
    for diagnosis in df_ext["Diagnosis"].dropna().unique():
        df_diagnosis = df_ext[df_ext["Diagnosis"] == diagnosis]
        X_plot = estimator.transform(df_diagnosis[feature_list])
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
    ax.set_xlim([-5, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-0.5, 1.5])

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

        if True:
            title = "K-Means on PCA-reduced Xena Dataset with {} clusters".format(idx)
            fig, ax = plt.subplots(figsize=aspect, num=title, tight_layout=True)
            fig.suptitle(title)

        # Run k-means
        reduced_data = X_train_pca
        kmeans = KMeans(cluster_count)
        kmeans.fit(reduced_data)

        df_pca["kmeans_{}".format(idx)] = kmeans.labels_

        if True:

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
    df_pca_ext.to_csv(kmeans_filepath)

#if __name__ == '__main__':
#    """
#    # Run PCA and generate plots
#    """
#    # Set up argument parser
#    parser = argparse.ArgumentParser()
#    # Set up parser arguments
#    parser.add_argument(
#            "--input_filepath", default=None, required=True,
#            help="The csv input file containing data to perform quality"
#                " control on.")
#    parser.add_argument(
#            "--output_filepath", default=None, required=True,
#            help="The csv output file path to store the quality control data.")
#    parser.add_argument(
#            "--patients_filepath", default=None, required=False,
#            help="The csv input file containing data to perform quality"
#                " control on.")
#
#    args = parser.parse_args()
#
#    csv_input_filepath = args.csv_input_filepath
#    csv_output_filepath = args.csv_output_filepath
#
#    no_add_columns = args.no_add_columns
#    no_qc_pass_fail_output_folders = args.no_qc_pass_fail_output_folders
#    preprocessed_data_path = args.preprocessed_data_path
#
#
#    # Get exclusion criteria from file
#    criteria_file = args.criteria_file
#    if criteria_file:
#        with open(criteria_file,"r") as criteria_fp:
#            criteria = json.loads(criteria_fp.read())
#    else:
#        raise ValueError("Control criteria file required.")
#
#    main(
#            csv_input_filepath, csv_output_filepath, criteria, no_add_columns,
#            no_qc_pass_fail_output_folders, preprocessed_data_path)
#
