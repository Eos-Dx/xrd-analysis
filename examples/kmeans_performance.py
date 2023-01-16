"""
Code to analyze cancer clusters from k-means
"""
import argparse

import pandas as pd

from sklearn.metrics import confusion_matrix

from eosdxanalysis.models.utils import metrics_report

def kmeans_performance(
        kmeans_results_filepath=None,
        patient_db_filepath=None,
        cluster_count=None,
        ):


    # Set kmeans_column
    kmeans_column = f"kmeans_{cluster_count}"

    # Open measurement k-means cluster labels
    df_kmeans = pd.read_csv(kmeans_results_filepath, index_col="Filename")

    # Import patient data if not already present
    if "Patient_ID" not in df_kmeans.columns:
        # Set patients database
        db = pd.read_csv(patient_db_filepath, index_col="Barcode")

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

    print("Cluster cancer measurements and patients composition:")
    print("Cluster,Measurements Percent,Measurements Count,Patients Percent,"
            "Patients Count")

    for idx in range(cluster_count):
        df_cluster = df_kmeans[df_kmeans[f"kmeans_{cluster_count}"] == idx]
        num_cancer = len(df_cluster[df_cluster["Diagnosis"] == "cancer"])
        num_healthy = len(df_cluster[df_cluster["Diagnosis"] == "healthy"])
        cancer_ratio = num_cancer / (num_cancer + num_healthy)
        cluster_name = "cluster_{}".format(idx)
        cluster_patients = df_kmeans[df_kmeans[f"kmeans_{cluster_count}"] == idx]["Patient_ID"].unique()
        num_cancer_patients = len(
                df_cluster[df_cluster["Diagnosis"] == "cancer"]["Patient_ID"].unique())
        num_healthy_patients = len(
                df_cluster[df_cluster["Diagnosis"] == "healthy"]["Patient_ID"].unique())
        patient_cancer_ratio = num_cancer_patients / (num_cancer_patients + num_healthy_patients)
        print(
                f"{idx},"
                f"{cancer_ratio*100:>3.1f},"
                f"{num_cancer},"
                f"{patient_cancer_ratio*100:>3.1f},"
                f"{num_cancer_patients}")

if __name__ == '__main__':
    """
    Run cancer predictions on preprocessed data
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--kmeans_results_filepath", type=str, default=None, required=True,
            help="The file containing training data with k-means labels.")
    parser.add_argument(
            "--patient_db_filepath", type=str, default=None, required=False,
            help="The file containing patient data with diagnosis.")
    parser.add_argument(
            "--cluster_count", type=int, default=None, required=True,
            help="The cluser count for k-means.")

    # Collect arguments
    args = parser.parse_args()

    kmeans_results_filepath = args.kmeans_results_filepath
    cluster_count = args.cluster_count
    patient_db_filepath = args.patient_db_filepath

    kmeans_performance(
        kmeans_results_filepath=kmeans_results_filepath,
        patient_db_filepath=patient_db_filepath,
        cluster_count=cluster_count,
        )
