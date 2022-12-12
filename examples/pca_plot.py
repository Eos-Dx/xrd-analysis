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
        # 'cropped_intensity',
        # 'total_flux',
        # 'bright_pixel_count',
        'annulus_intensity_9A',
        'annulus_intensity_5A',
        'annulus_intensity_4A',
        'sector_intensity_equator_pair_9A',
        'sector_intensity_meridian_pair_9A',
        'sector_intensity_meridian_pair_5A',
        ]

if True:
    # Divide
    # divide_by = "cropped_intensity"
    divide_by = "total_flux"
    df = df.div(df[divide_by], axis="rows")

# Set features to use
df = df[feature_list]

df_AT = df[df.index.str.match(r"(CR_AT)")]
df_B = df[df.index.str.match(r"(CR_B)")]
df_A = df[df.index.str.match(r"(CR_A0|CR_AA)")]

# Exclude B-series
# df = df[~df.index.isin(df_B.index)]

# Set PCA to 2 components
pca = PCA(n_components=2)

# Create pipeline including standard scaling
estimator = make_pipeline(StandardScaler(), pca).fit(df.values)
# kmeans = KMeans(init=estimator['pca'].components_, n_clusters=cluster_count)


print("Explained variance ratios:")
print(estimator['pca'].explained_variance_ratio_)
print(
        "Total explained variance:",
        np.sum(estimator['pca'].explained_variance_ratio_))

# Print first two principal components
pca_components = estimator['pca'].components_
print(dict(zip(feature_list, pca_components[0,:])))
print(dict(zip(feature_list, pca_components[1,:])))

# Transform data using PCA
X_pca = estimator.transform(df.values)

################################
# PCA subspace projection plot #
################################

# Plot PCA-reduced dataset with file labels
title = "PCA-reduced Xena Dataset 2-D Subspace Projection"
fig, ax = plt.subplots(figsize=aspect, num=title, tight_layout=True)
fig.suptitle(title)
plt.scatter(X_pca[:,0], X_pca[:,1])

# Set offsets
x_label_offset = 0.01
y_label_offset = 0.01

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

if True:

    # Collect data subsets for plotting
    series_dict = {
            "AT_series": df_AT,
            "A_series": df_A,
            # "B_series": df_B,
            }

    # Plot all data subsets
    title = "PCA-Reduced Xena Dataset 2-D Subspace Projection, color by subset"
    fig, ax = plt.subplots(figsize=aspect, num=title, tight_layout=True)
    fig.suptitle(title)

    colors = {
            "AT_series": "#17becf",
            "A_series": "#bcbd22",
            "B_series": "#7f7f7f",
            }

    for series_name, df_sub in series_dict.items():
        X = estimator.transform(df_sub.values)
        plt.scatter(
                X[:,0], X[:,1], label=series_name, c=colors[series_name])


    plt.xlabel("PC0")
    plt.ylabel("PC1")

    plt.legend()
    plt.show()


#################
# Patients plot #
#################

if False:

    # Plot all data highlighting patients
    title = "PCA-reduced Xena Dataset Patient Highlights 2-D Subspace Projection"
    fig, ax = plt.subplots(figsize=aspect, num=title, tight_layout=True)
    fig.suptitle(title)

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
    title = "PCA-reduced Xena Dataset 2-D Subspace Projection, color by diagnosis"
    fig, ax = plt.subplots(figsize=aspect, num=title, tight_layout=True)
    fig.suptitle(title)

    # Add a Barcode column to the dataframe
    # Extract the first letter and all numbers in the filename before the subindex
    # E.g., from filename AB12345-01.txt -> A12345 is extracted
    # Note: Issue if the Barcode format changes
    extraction = df.index.str.extractall("CR_([A-Z]{1}).*?([0-9]+)")
    extraction_series = extraction[0] + extraction[1]
    extraction_list = extraction_series.tolist()

    assert(len(extraction_list) == df.shape[0])
    df_ext = df.copy()
    df_ext["Barcode"] = extraction_list

    df_ext = pd.merge(df_ext, db, left_on="Barcode", right_index=True)

    colors = {
            "cancer": "red",
            "healthy": "blue",
            }

    # Loop over series
    for diagnosis in df_ext["Diagnosis"].dropna().unique():
        df_diagnosis = df_ext[df_ext["Diagnosis"] == diagnosis]
        X_plot = estimator.transform(df_diagnosis[feature_list])
        plt.scatter(
                X_plot[:,0], X_plot[:,1], label=diagnosis, c=colors[diagnosis])

    plt.xlabel("PC0")
    plt.ylabel("PC1")

    plt.legend()
    plt.show()


#################
# K-means plots #
#################


# df_pca = data + kmeans cluster labels
df_pca = pd.DataFrame(data=X_pca, index=df.index)

cluster_count_min = 2
cluster_count_max = 6

# Run K-means on pca-reduced features
for idx in range(cluster_count_min, cluster_count_max+1):

    cluster_count = idx

    title = "K-Means on PCA-reduced Xena Dataset with {} clusters".format(idx)
    fig, ax = plt.subplots(figsize=aspect, num=title, tight_layout=True)
    fig.suptitle(title)

    # Run k-means
    reduced_data = X_pca
    kmeans = KMeans(cluster_count)
    kmeans.fit(reduced_data)

    df_pca["kmeans_{}".format(idx)] = kmeans.labels_

    # Plot decisision boundaries
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

# Save dataframe
df_pca.to_csv(kmeans_filepath)
