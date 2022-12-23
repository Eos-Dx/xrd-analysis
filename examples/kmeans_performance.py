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


# Open measurement k-means cluster labels
df_kmeans = pd.read_csv(kmeans_path, index_col="Filename")

# Import patient data if not already present
if "Patient_ID" not in df_kmeans.columns:
    # Extract barcodes from preprocesed filenames
    extraction = df_kmeans.index.str.extractall("CR_([A-Z]{1}).*?([0-9]+)")
    extraction_series = extraction[0] + extraction[1].str.zfill(5)
    extraction_list = extraction_series.tolist()

    # Ensure the length of extracted barcodes matches the number of
    # preprocessed files
    assert(len(extraction_list) == df_kmeans.shape[0])

    # Set the barcode column
    df_kmeans["Barcode"] = extraction_list

    # Import patient id and diagnosis
    df_kmeans = pd.merge(df_kmeans, db, left_on="Barcode", right_index=True)

cluster_dict = {}
n_clusters = 20
cancer_cluster_list = []
cancer_cluster_threshold = 0.80

print("Cluster,Cancer_Measurements_Percent,Cancer_Measurements_Num,Cancer_Patients_Percent,Cancer_Patients_Num")

for idx in range(n_clusters):
    df_cluster = df_kmeans[df_kmeans[f"kmeans_{n_clusters}"] == idx]
    num_cancer = len(df_cluster[df_cluster["Diagnosis"] == "cancer"])
    num_healthy = len(df_cluster[df_cluster["Diagnosis"] == "healthy"])
    cancer_ratio = num_cancer / (num_cancer + num_healthy)
    cluster_name = "cluster_{}".format(idx)
    cluster_patients = df_kmeans[df_kmeans[f"kmeans_{n_clusters}"] == idx]["Patient_ID"].unique()
    num_cancer_patients = len(
            df_cluster[df_cluster["Diagnosis"] == "cancer"]["Patient_ID"].unique())
    num_healthy_patients = len(
            df_cluster[df_cluster["Diagnosis"] == "healthy"]["Patient_ID"].unique())
    patient_cancer_ratio = num_cancer_patients / (num_cancer_patients + num_healthy_patients)
    cluster_dict[cluster_name] = cluster_patients
    print(
            f"{idx},"
            f"{cancer_ratio*100:>3.1f},"
            f"{num_cancer},"
            f"{patient_cancer_ratio*100:>3.1f},"
            f"{num_cancer_patients}")
    # Store list of majority cancer clusters
    if cancer_ratio >= cancer_cluster_threshold:
        cancer_cluster_list += [idx]

# Get cancer cluster group
cluster_group = df_kmeans[df_kmeans[f"kmeans_{n_clusters}"].isin(cancer_cluster_list)]
non_cluster_group = df_kmeans[~df_kmeans[f"kmeans_{n_clusters}"].isin(cancer_cluster_list)]

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
    patient_slice = df_kmeans.loc[
            (df_kmeans["Patient_ID"] == patient_id) & \
            (df_kmeans["label"] == 1)]
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
metrics_report(TN=TN, FP=FP, FN=FN, TP=TP)

# Save predictions
df_patients_ext.to_csv(patient_predictions_filepath)
