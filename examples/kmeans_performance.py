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
        index_col=None,
        model_type=None,
        ):
    if model_type == "measurementwise":
        index_col = "Filename"
    elif model_type == "patientwise":
        index_col = "Patient_ID"

    # Set kmeans_column
    kmeans_column = f"kmeans_{cluster_count}"

    # Open measurement k-means cluster labels
    df_kmeans = pd.read_csv(kmeans_results_filepath, index_col=index_col)

    # Import patient data if not already present
    if "Patient_ID" not in df_kmeans.columns and model_type == "measurementwise":
        df_kmeans = add_patient_data(
                df_kmeans, patient_db_filepath, index_col="Barcode")

    print("Cluster cancer measurements and patients composition:")
    print("Cluster,Measurements Percent,Measurements Count,Patients Percent,"
            "Patients Count")

    for idx in range(cluster_count):
        df_cluster = df_kmeans[df_kmeans[f"kmeans_{cluster_count}"] == idx]
        num_cancer = len(df_cluster[df_cluster["Diagnosis"] == "cancer"])
        num_healthy = len(df_cluster[df_cluster["Diagnosis"] == "healthy"])
        cancer_ratio = num_cancer / (num_cancer + num_healthy)
        cluster_name = "cluster_{}".format(idx)

        if model_type == "measurementwise":
            num_cancer_patients = len(
                    df_cluster[df_cluster["Diagnosis"] == "cancer"]["Patient_ID"].unique())
            num_healthy_patients = len(
                    df_cluster[df_cluster["Diagnosis"] == "healthy"]["Patient_ID"].unique())
        elif model_type == "patientwise":
            num_cancer_patients = len(df_kmeans["Diagnosis"] == "cancer")
            num_healthy_patients = len(df_kmeans["Diagnosis"] == "healthy")

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
    parser.add_argument(
            "--model_type", default="measurementwise", type=str, required=False,
            help="Choice of ``measurementwise`` (default) or ``patientwise`` model.")

    # Collect arguments
    args = parser.parse_args()

    kmeans_results_filepath = args.kmeans_results_filepath
    cluster_count = args.cluster_count
    patient_db_filepath = args.patient_db_filepath
    model_type = args.model_type

    kmeans_performance(
        kmeans_results_filepath=kmeans_results_filepath,
        patient_db_filepath=patient_db_filepath,
        cluster_count=cluster_count,
        model_type=model_type,
        )
