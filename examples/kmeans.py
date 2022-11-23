"""
Script to train multiple k-means unsupervised clustering models on a dataset

Future work: incorporate into eosdxanalysis/models/kmeans.py


Set the following variables:
* dataframe_filepath
* df_kmeans_path
* cluster_min
* cluster_max
* image_path
* image_cluster_path

"""
import os
import shutil

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set data path
dataframe_filepath = ""
# Load data into dataframe
df = pd.read_csv(dataframe_filepath, index_col="Filename")

# Set K-means output results path
df_kmeans_path = ""

# Set the K-means cluster numbers range
cluster_min = 2
cluster_max = 6

# Set the source image path
image_path = ""

# Set up standard scaler
scaler = StandardScaler()
# Fit standard scaler to data
scaler.fit(df)

# Transform data using standard scaler
X = scaler.transform(df)

# Train K-means models for each cluster number
for idx in range(cluster_min, cluster_max+1):
    kmeans = KMeans(idx, random_state=0).fit(X)
    df["kmeans_{}".format(idx)] = kmeans.labels_

# Save K-means results to file
df.to_csv(df_kmeans_path, index=True)

# Use K-means results to create cluster image preview folders

# Loop over files to copy the file to individual K-means cluster folders
for idx in df.index:
    filename = idx + ".png"

    # Copy the file to the appropriate directory or directories
    # Loop over K-means models
    for jdx in range(cluster_min, cluster_max+1):
        # Get the cluster label
        cluster = df["kmeans_{}".format(jdx)][idx]
        # Set the cluster image path
        image_cluster_path = \
                "/path/to/kmeans_clusters/StandardScalar/kmeans_n{}_c{}".format(
                        jdx, cluster)
        # Create the cluster image path
        os.makedirs(image_cluster_path, exist_ok=True)
        # Copy the file from the image source path to the image cluster path
        shutil.copy(
                os.path.join(image_path, filename),
                os.path.join(image_cluster_path, filename))
