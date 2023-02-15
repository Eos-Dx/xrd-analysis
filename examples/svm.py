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


def run_svm(df_train_path, db_path, output_path=None, scale_by=None):
    # Generate timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    df_train = pd.read_csv(df_train_path, index_col="Filename")
    # Add patient data
    df_train = add_patient_data(df_train, db_path, index_col="Barcode").dropna().copy()
    feature_list = np.arange(1,51).astype(str)
    df_train["total_intensity"] = np.sum(df_train[feature_list].values, axis=1).astype(int)

    young_dogs_list = [
        "CR_A00120-10.txt",
        "CR_A00120-1.txt",
        "CR_A00120-2.txt",
        "CR_A00120-3.txt",
        "CR_A00120-4.txt",
        "CR_A00120-5.txt",
        "CR_A00120-6.txt",
        "CR_A00120-7.txt",
        "CR_A00120-8.txt",
        "CR_A00120-9.txt",
        "CR_A00131-10.txt",
        "CR_A00131-1.txt",
        "CR_A00131-2.txt",
        "CR_A00131-3.txt",
        "CR_A00131-4.txt",
        "CR_A00131-5.txt",
        "CR_A00131-6.txt",
        "CR_A00131-7.txt",
        "CR_A00131-8.txt",
        "CR_A00131-9.txt",
        "CR_A00134-1.txt",
        "CR_A00134-2.txt",
        "CR_A00134-3.txt",
        "CR_A00134-4.txt",
        "CR_A00134-5.txt",
        "CR_A00134-6.txt",
        "CR_A00134-7.txt",
        "CR_A00150-1.txt",
        "CR_A00150-2.txt",
        "CR_A00150-3.txt",
        "CR_A00150-4.txt",
        "CR_A00150-5.txt",
        "CR_A00150-6.txt",
        "CR_A00150-7.txt",
        "CR_A00150-8.txt",
        "CR_A00150-9.txt",
        "CR_A00153-10.txt",
        "CR_A00153-1.txt",
        "CR_A00153-2.txt",
        "CR_A00153-3.txt",
        "CR_A00153-4.txt",
        "CR_A00153-5.txt",
        "CR_A00153-6.txt",
        "CR_A00153-7.txt",
        "CR_A00153-8.txt",
        "CR_A00153-9.txt",
        "CR_A00156-1.txt",
        "CR_A00156-2.txt",
        "CR_A00156-3.txt",
        "CR_A00156-4.txt",
        "CR_A00156-5.txt",
        "CR_A00156-6.txt",
        "CR_A00156-7.txt",
        "CR_A00159-1.txt",
        "CR_A00159-2.txt",
        "CR_A00159-3.txt",
        "CR_A00159-4.txt",
        "CR_A00159-5.txt",
        "CR_A00159-6.txt",
        "CR_A00159-7.txt",
        "CR_A00171-1.txt",
        "CR_A00171-2.txt",
        "CR_A00171-3.txt",
        "CR_A00171-4.txt",
        "CR_A00171-5.txt",
        "CR_A00171-6.txt",
        "CR_A00171-7.txt",
        "CR_A00175-1.txt",
        "CR_A00175-2.txt",
        "CR_A00175-3.txt",
        "CR_A00175-4.txt",
        "CR_A00175-5.txt",
        "CR_A00178-1.txt",
        "CR_A00178-2.txt",
        "CR_A00178-3.txt",
        "CR_A00178-4.txt",
        "CR_A00178-5.txt",
        "CR_A00178-6.txt",
        "CR_A00178-7.txt",
        "CR_A00184-10.txt",
        "CR_A00184-1.txt",
        "CR_A00184-2.txt",
        "CR_A00184-3.txt",
        "CR_A00184-4.txt",
        "CR_A00184-5.txt",
        "CR_A00184-6.txt",
        "CR_A00184-7.txt",
        "CR_A00184-8.txt",
        "CR_A00184-9.txt",
        "CR_A00190-1.txt",
        "CR_A00190-2.txt",
        "CR_A00190-3.txt",
        "CR_A00190-4.txt",
        "CR_A00190-5.txt",
        "CR_A00190-6.txt",
        "CR_A00190-7.txt",
        "CR_A00193-1.txt",
        "CR_A00193-2.txt",
        "CR_A00193-3.txt",
        "CR_A00199-1.txt",
        "CR_A00199-2.txt",
        "CR_A00199-3.txt",
        "CR_A00199-4.txt",
        "CR_A00199-5.txt",
        "CR_A00199-6.txt",
        "CR_A00202-1.txt",
        "CR_A00202-2.txt",
        "CR_A00202-3.txt",
        "CR_A00202-4.txt",
        "CR_A00202-5.txt",
        "CR_A00202-6.txt",
        "CR_A00202-7.txt",
        "CR_A00250-1.txt",
        "CR_A00250-2.txt",
        "CR_A00250-3.txt",
        "CR_A00250-4.txt",
        "CR_A00262-1.txt",
        "CR_A00262-2.txt",
        "CR_A00262-3.txt",
        "CR_A00262-4.txt",
        "CR_A00266-10.txt",
        "CR_A00266-1.txt",
        "CR_A00266-2.txt",
        "CR_A00266-3.txt",
        "CR_A00266-4.txt",
        "CR_A00266-5.txt",
        "CR_A00266-6.txt",
        "CR_A00266-7.txt",
        "CR_A00266-8.txt",
        "CR_A00266-9.txt",
    ]


    df_young = df_train[df_train.index.isin(young_dogs_list)].copy()
    df_old = df_train[~df_train.index.isin(young_dogs_list)].copy()

    df_train.loc[df_young.index, "y_true"] = 0
    df_train.loc[df_old.index, "y_true"] = 1
    df_train["y_true"] = df_train["y_true"].astype(int)

    # Get training data
    if scale_by:
        df_train_scaled_features = scale_features(df_train, scale_by, feature_list)
        X = df_train_scaled_features[feature_list].values
    else:
        X = df_train[feature_list].values

    # Get training labels
    y = df_train["y_true"].values

    # Create classifier
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    # Train model
    clf.fit(X, y)

    save = True
    if save:
        model_output_filename = "svm_model_{}.joblib".format(timestamp)
        model_output_filepath = os.path.join(output_path, model_output_filename)
        dump(clf, model_output_filepath)

    # Get patient-wise performance
    # Get patient labels
    y_true_patients = df_train.groupby("Patient_ID")["y_true"].max()

    # Get patient-wise predictions
    y_score_measurements = clf.decision_function(X)
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
    plt.show()

    PrecisionRecallDisplay.from_predictions(y_true_patients, y_score_patients)
    plt.show()



if __name__ == '__main__':
    """
    Run azimuthal integration on an image
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--df_train_path", type=str, default=None, required=True,
            help="The path to training features")
    parser.add_argument(
            "--db_path", type=str, default=None, required=True,
            help="The path to patient database")
    parser.add_argument(
            "--output_path", type=str, default=None, required=False,
            help="The output path to save radial profiles and peak features")
    parser.add_argument(
            "--scale_by", type=str, default=None, required=False,
            help="The feature to scale by")

    args = parser.parse_args()

    df_train_path = args.df_train_path
    db_path = args.db_path
    output_path = args.output_path
    scale_by = args.scale_by

    run_svm(df_train_path=df_train_path,
            db_path=db_path,
            output_path=output_path,
            scale_by=scale_by,
            )
