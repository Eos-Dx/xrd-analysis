"""
Support Vector Machine
"""
import os
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

from joblib import dump

from eosdxanalysis.models.utils import scale_features
from eosdxanalysis.models.utils import add_patient_data


def run_svm(
        df_train_filepath, df_blind_filepath=None, db_path=None, output_path=None,
        scale_by=None, C=1.0):
    # Generate timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    df_train = pd.read_csv(df_train_filepath, index_col="Filename")
    if db_path:
        # Add patient data
        df_train = add_patient_data(df_train, db_path, index_col="Barcode").dropna().copy()

    feature_list = np.arange(1,51).astype(str)

    # Get training data
    if scale_by:
        df_train_scaled_features = scale_features(df_train, scale_by, feature_list)
        X_train = df_train_scaled_features[feature_list].values
    else:
        X_train = df_train[feature_list].values

    # Get training labels
    df_train["y_true"] = (df_train["Diagnosis"] == "cancer").astype(int)
    y = df_train["y_true"].values

    # Create classifier
    svm = SVC(C=C, kernel='rbf', gamma='auto', verbose=False)
    clf = make_pipeline(StandardScaler(), svm)

    # Train model
    clf.fit(X_train, y)

    save = True
    if save:
        model_output_filename = "svm_model_C_{}_{}.joblib".format(C, timestamp)
        model_output_filepath = os.path.join(output_path, model_output_filename)
        dump(clf, model_output_filepath)

    # Get patient-wise performance
    # Get patient labels
    y_true_patients = df_train.groupby("Patient_ID")["y_true"].max()

    # Get patient-wise predictions
    y_score_measurements = clf.decision_function(X_train)
    df_train["y_score"] = y_score_measurements
    # Calculate patient scores
    y_score_patients = df_train.groupby("Patient_ID")["y_score"].max()

    fpr, tpr, thresholds = roc_curve(y_true_patients, y_score_patients)

    # Find threshold closest to ideal classifier
    distances = np.sqrt((1-tpr)**2 + fpr**2)
    min_distance = np.min(distances)
    # Get the index of min distances
    optimal_index = np.where(distances == min_distance)
    optimal_threshold_array = thresholds[optimal_index]

    if optimal_threshold_array.size > 1:
        # Take the first one
        print("Info: {} optimal thresholds found".format(
            optimal_threshold_array.size))

    optimal_threshold = optimal_threshold_array[0]

    # Generate predictions for optimal threshold
    y_pred_patients = (y_score_patients.values >= optimal_threshold).astype(int)

    # Get the number of predicted "old" patients that have a healthy diagnosis
    train_patients_diagnosis_series = df_train.groupby("Patient_ID")["Diagnosis"].max()
    df_train_patients = pd.DataFrame(data=train_patients_diagnosis_series, columns={"Diagnosis"})
    df_train_patients["y_pred"] = y_pred_patients

    # Generate scores for optimal threshold
    accuracy = accuracy_score(y_true_patients, y_pred_patients)
    precision = precision_score(y_true_patients, y_pred_patients)
    sensitivity = recall_score(y_true_patients, y_pred_patients)
    specificity = recall_score(
            y_true_patients, y_pred_patients, pos_label=0)

    # Generate performance counts
    tn, fp, fn, tp = confusion_matrix(y_true_patients, y_pred_patients).ravel()

    # Print metrics
    print("tn,fp,fn,tp,threshold,accuracy,precision,sensitivity,specificity")
    print("{},{},{},{},{:.2f},{:2f},{:2f},{:2f},{:2f}".format(
        tn, fp, fn, tp,
        optimal_threshold, accuracy, precision, sensitivity,
        specificity))

    RocCurveDisplay.from_predictions(y_true_patients, y_score_patients)
    fig_roc = plt.gcf()
    title = "SVM ROC Curve C={}".format(C)
    fig_roc.suptitle(title)
    fig_roc_filename = "roc_curve_C_{}.png".format(C)
    fig_roc_filepath = os.path.join(output_path, fig_roc_filename)
    fig_roc.savefig(fig_roc_filepath)

    PrecisionRecallDisplay.from_predictions(y_true_patients, y_score_patients)
    fig_pr = plt.gcf()
    title = "SVM Precision-Recall Curve C={}".format(C)
    fig_pr.suptitle(title)
    fig_pr_filename = "precision_recall_curve_C_{}.png".format(C)
    fig_pr_filepath = os.path.join(output_path, fig_pr_filename)
    fig_pr.savefig(fig_pr_filepath)

    plt.show()



    # Run blind predictions
    if df_blind_filepath:
        # Run blind data through trained model and assess performance
        # Load dataframe
        df_blind = pd.read_csv(df_blind_filepath, index_col="Filename")

        # Get blind data
        if scale_by:
            df_blind_scaled_features = scale_features(df_blind, scale_by, feature_list)
            X_blind = df_blind_scaled_features[feature_list].values
        else:
            X_blind = df_blind[feature_list].values

        # Predict on measurements
        y_predict_blind = clf.predict(X_blind)

        # Save results
        df_blind["y_pred"] = y_predict_blind

        # Get patient predictions
        y_pred_blind_patients = df_blind.groupby("Patient_ID")["y_pred"].max()

        # Print patient statistics
        p_blind = y_pred_blind_patients.sum()
        n_blind = y_pred_blind_patients.shape[0] - p_blind
        print("n,p")
        print("{},{}".format(n_blind, p_blind))

        # Save blind measurement predictions
        measurement_output_prefix = "blind_measurement_predictions"

        measurement_csv_filename = "{}_C_{}_{}.csv".format(
                measurement_output_prefix, str(C), timestamp)
        measurement_csv_output_path = os.path.join(
                output_path, measurement_csv_filename)

        df_blind.to_csv(
                measurement_csv_output_path, columns=["y_pred"],
                index=True)

        print(
                "Blind measurement predictions saved to",
                measurement_csv_output_path)

        # Save blind patient predictions
        patient_output_prefix = "blind_patient_predictions"

        patient_csv_filename = "{}_C_{}_{}.csv".format(
                patient_output_prefix, str(C), timestamp)
        patient_csv_output_path = os.path.join(
                output_path, patient_csv_filename)

        y_pred_blind_patients.to_csv(
                patient_csv_output_path,
                index=True)

        print(
                "Blind patient predictions saved to",
                patient_csv_output_path)



if __name__ == '__main__':
    """
    Run svm
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--df_train_filepath", type=str, default=None, required=True,
            help="The path to training features")
    parser.add_argument(
            "--df_blind_filepath", type=str, default=None, required=False,
            help="The path to blind features")
    parser.add_argument(
            "--db_path", type=str, default=None, required=False,
            help="The path to patient database")
    parser.add_argument(
            "--output_path", type=str, default=None, required=False,
            help="The output path to save radial profiles and peak features")
    parser.add_argument(
            "--scale_by", type=str, default=None, required=False,
            help="The feature to scale by")
    parser.add_argument(
            "--C", type=float, default=1.0, required=False,
            help="The SVM C parameter")

    args = parser.parse_args()

    df_train_filepath = args.df_train_filepath
    df_blind_filepath = args.df_blind_filepath
    db_path = args.db_path
    output_path = args.output_path
    scale_by = args.scale_by
    C = args.C

    run_svm(df_train_filepath=df_train_filepath,
            df_blind_filepath=df_blind_filepath,
            db_path=db_path,
            output_path=output_path,
            scale_by=scale_by,
            C=C,
            )
