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

# Load data into dataframe
df_path = "extracted_features.csv"
df = pd.read_csv(df_path, index_col="Filename")
df_AT = df[df.index.str.match(r"(CR_AT)")]
df_B = df[df.index.str.match(r"(CR_B)")]
df_A = df[df.index.str.match(r"(CR_A0|CR_AA)")]

# Set dataset to use
df = df

# Set PCA to capture 95% of variance
pca = PCA(n_components=0.95)

# Create pipeline including standard scaling
estimator = make_pipeline(StandardScaler(), pca).fit(df)
# kmeans = KMeans(init=estimator['pca'].components_, n_clusters=cluster_count)

print(estimator['pca'].explained_variance_ratio_)
print(estimator['pca'].components_)

# Transform data using PCA
X_pca = estimator.transform(df)

# print(estimator['pca'].components_)

# Plot results
fig, ax = plt.subplots()
plt.scatter(X_pca[:,0], X_pca[:,1])

# Set offsets
x_label_offset = 0.01
y_label_offset = 0.01

filename_list = df.index.to_list()

for i, filename in enumerate(filename_list):
    ax.annotate(
        filename.replace("CR_","").replace(".txt",""),
        (X_pca[i,0], X_pca[i,1]),
        xytext=(X_pca[i,0]+x_label_offset, X_pca[i,1]+y_label_offset))

# Label plot axes and title
plt.xlabel("PC0")
plt.ylabel("PC1")
plt.title("2-D Subspace Projection")
        
plt.show()

# Collect data subsets for plotting
series_dict = {
        "AT_series": df_AT,
        "A_series": df_A,
        "B_series": df_B,
        }

# Plot all data subsets
fig, ax = plt.subplots()
fig.suptitle("Series Plot")
for series_name, df_sub in series_dict.items():
    X = estimator.transform(df_sub)
    plt.scatter(X[:,0], X[:,1], label=series_name)

plt.xlabel("PC0")
plt.ylabel("PC1")

plt.legend()
plt.show()


# Color by cluster
# Here, df = data + kmeans cluster labels
df_pca = pd.DataFrame(data=X_pca, index=df.index)

cluster_count_min = 2
cluster_count_max = 6

# Run K-means on original features and plot pca-reduced version
for idx in range(cluster_count_min, cluster_count_max+1):

    cluster_count = idx
    kmeans = KMeans(cluster_count).fit(df_pca)
    df_pca["kmeans_{}".format(idx)] = kmeans.labels_

    fig, ax = plt.subplots()
    fig.suptitle("Number of clusters: {}".format(idx))

    for jdx in range(idx):
        X_pca = df_pca[df_pca["kmeans_{}".format(idx)] == jdx]
        plt.scatter(X_pca[0], X_pca[1])

        centroids = kmeans.cluster_centers_
        plt.scatter(
                centroids[:, 0],
                centroids[:, 1],
                marker="x",
                s=169,
                linewidths=3,
                color="k",
                zorder=10,
                alpha=0.25,
                )

        plt.xlabel("PC0")
        plt.ylabel("PC1")

plt.show()

# Save dataframe
df_pca.to_csv("kmeans_pca.csv")
