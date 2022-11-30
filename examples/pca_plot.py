"""
Example for plotting PCA-reduced K-means clusters
"""
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

# Load data into dataframe
df_path = ""
df = pd.read_csv(df_path, index_col="Filename")

# Set PCA to capture 95% of variance
pca = PCA(n_components=0.95)

# Create pipeline including standard scaling
estimator = make_pipeline(StandardScaler(), pca).fit(df)

print(estimator['pca'].explained_variance_ratio_)

# Transform data using PCA
X_pca = estimator.transform(df)

# print(estimator['pca'].components_)

# Plot results
fig, ax = plt.subplots()
plt.scatter(X_pca[:,0], X_pca[:,1])

# Set offsets
x_offset = 0.01
y_offset = 0.01

for i, filename in enumerate(filename_list):
    ax.annotate(
        filename,
        (X_pca[i,0], X_pca[i,1]),
        xytext=(X_pca[i,0]+x_label_offset, X_pca[i,1]+y_label_offset))

# Label plot axes and title
plt.xlabel("PC0")
plt.ylabel("PC1")
plt.title("2-D Subspace Projection AT Dataset")
        
plt.show()


# Color by cluster
# Here, df = data + kmeans cluster labels

for idx in range(2,6+1):
    df_pca["kmeans_{}".format(idx)] = df["kmeans_{}".format(idx)]

for idx in range(2,6+1):
    fig, ax = plt.subplots()
    fig.suptitle("Number of clusters: {}".format(idx))
    for jdx in range(idx):
        X_pca = df_pca[df_pca["kmeans_{}".format(idx)] == jdx]
        plt.scatter(X_pca[0], X_pca[1])

plt.show()
