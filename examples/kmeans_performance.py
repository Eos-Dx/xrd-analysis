"""
Code to build model from cancer clusters

Note: Ignore patients with no diagnosis
"""

import pandas as pd

from sklearn.metrics import confusion_matrix

from eosdxanalysis.models.utils import metrics_report


# Load k-means clustering results
kmeans_path = "kmeans_pca.csv"

# Set patients database
db_path = "patients_database.csv"
db = pd.read_csv(db_path, index_col="Barcode")

# Save patient predictions output
patient_predictions_filepath = "patient_predictions.csv"


# Set k
k = 20
# Set kmeans_column
kmeans_column = f"kmeans_{k}"


df_kmeans = pd.read_csv(kmeans_path, index_col="Filename")

# Set cancer cluster group
cancer_cluster_list = [2,5,7,14,18]

# Get cancer cluster group
cluster_group = df_kmeans[df_kmeans["kmeans_20"].isin(cancer_cluster_list)]
non_cluster_group = df_kmeans[~df_kmeans["kmeans_20"].isin(cancer_cluster_list)]

# Label all measurements in cancer cluster group as cancer
df_kmeans["label"] = df_kmeans[kmeans_column]
# Set cancer cluster group prediction equal to 1
df_kmeans.loc[cluster_group.index, "label"] = 1
# Set not in cluster group prediction equal to 0
df_kmeans.loc[non_cluster_group.index, "label"] = 0

# Set up patient predictions
df_patients = pd.DataFrame(columns={"Prediction"})

# Get the list of patient Ids
patient_id_list = df_kmeans["Patient_ID"].dropna().unique().tolist()

# Create an empty patients dataframe
df_patients = pd.DataFrame(columns={"Prediction"})

# Fill the patients dataframe with predictions
for patient_id in patient_id_list:
    patient_slice = df_kmeans[df_kmeans["Patient_ID"] == patient_id]
    if any(patient_slice["label"] == 1):
        df_patients.loc[patient_id] = 1
    else:
        df_patients.loc[patient_id] = 0

# Get patients and associated diagnosis
db_patients = db[["Patient_ID", "Diagnosis"]].drop_duplicates()
# Merge to get patient diagnosis and prediction in the same dataframe
df_patients_ext = pd.merge(
        df_patients, db_patients, left_index=True, right_on="Patient_ID")
# Set patient id as index
df_patients_ext.index = df_patients_ext["Patient_ID"]
# Extract diagnosis and prediction
df_patients_ext = df_patients_ext[["Diagnosis", "Prediction"]]
# Replace cancer diagnosis with 1, healthy diagnosis with 0
df_patients_ext.loc[df_patients_ext["Diagnosis"] == "cancer",
        "Diagnosis"] = 1
df_patients_ext.loc[df_patients_ext["Diagnosis"] == "healthy",
        "Diagnosis"] = 0

# Get true and predicted patient cancer diagnosis
y_true = df_patients_ext[~df_patients_ext["Diagnosis"].isna()]["Diagnosis"].astype(int)
y_pred = df_patients_ext[~df_patients_ext["Diagnosis"].isna()]["Prediction"].astype(int)

# Calculate performance metrics
TN, FP, FN, TP = confusion_matrix(y_true.values, y_pred.values).ravel()
metrics_report(TN=TN, FP=TP, FN=FN, TP=TP)

# Save predictions
df_patients_ext.to_csv(patient_predictions_filepath)
