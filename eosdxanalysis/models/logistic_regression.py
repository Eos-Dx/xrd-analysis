"""
Use the 2D Gaussian fit results to run logistic regression
"""
import os
import argparse
import glob
from collections import OrderedDict
from datetime import datetime
from time import time
from joblib import dump
from joblib import load

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay


from eosdxanalysis.models.curve_fitting import GaussianDecomposition
from eosdxanalysis.models.utils import metrics_report
from eosdxanalysis.models.utils import scale_features


def main(
        data_filepath=None, blind_data_filepath=None, output_path=None,
        max_iter=100, degree=1, use_cross_val=False, feature_list=[],
        joblib_filepath=None, balanced=None, scale_by=None):
    t0 = time()

    cmap="hot"

    # Set class_weight
    class_weight = balanced if balanced else None

    # Perform Logistic Regression
    # ---------------------------

    # Set timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    # Set output path with a timestamp if not specified
    if not output_path:
        output_dir = "logistic_regression_{}".format(timestamp)
        output_path = os.path.dirname(data_filepath)

    # Load an existing model if provided
    if joblib_filepath:
        clf = load(joblib_filepath)
    # Build a new model
    else:
        # Load dataframe
        df_train = pd.read_csv(data_filepath, index_col="Filename")

        diagnosis_series = (df_train["Diagnosis"] == "cancer").astype(int)
        df_train["y_true"] = diagnosis_series

        # Get training data
        if scale_by:
            df_train_scaled_features = scale_features(df_train, scale_by, feature_list)
            X_train = df_train_scaled_features[feature_list].values
        else:
            X_train = df_train[feature_list].values

        # Logistic Regression
        # -------------------
        # Create a polynomial
        poly = PolynomialFeatures(degree=degree)

        X = poly.fit_transform(X_train)

        # Get training labels
        y = df_train["y_true"].values

        # Create classifier
        logreg = LogisticRegression(
                C=1,class_weight=class_weight, solver="newton-cg",
                max_iter=max_iter)
        clf = make_pipeline(StandardScaler(), logreg)

        # Train model
        clf.fit(X, y)

        save = True
        if save:
            model_output_filename = "logistic_regression_degree_{}_model_{}.joblib".format(
                    degree, timestamp)
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
        fig_roc = plt.gcf()
        title = "Logistic Regression ROC Curve Degree {}".format(degree)
        fig_roc.suptitle(title)
        fig_roc_filename = "roc_curve_degree_{}.png".format(degree)
        fig_roc_filepath = os.path.join(output_path, fig_roc_filename)
        fig_roc.savefig(fig_roc_filepath)

        PrecisionRecallDisplay.from_predictions(y_true_patients, y_score_patients)
        fig_pr = plt.gcf()
        title = "Logistic Regression Precision-Recall Curve Degree {}".format(degree)
        fig_pr.suptitle(title)
        fig_pr_filename = "precision_recall_curve_degree_{}.png".format(degree)
        fig_pr_filepath = os.path.join(output_path, fig_pr_filename)
        fig_pr.savefig(fig_pr_filepath)

        plt.show()


    # Blind data predictions
    # ----------------------
    # Predict data on blind data set

    if blind_data_filepath:
        # Run blind data through trained model and assess performance
        # Load dataframe
        df_blind = pd.read_csv(blind_data_filepath, index_col="Filename")

        # Get blind data
        if scale_by:
            df_blind_scaled_features = scale_features(df_blind, scale_by, feature_list)
            X_blind = df_blind_scaled_features[feature_list].values
        else:
            X_blind = df_blind[feature_list].values

        # Create a polynomial
        poly = PolynomialFeatures(degree=degree)

        X_blind_poly = poly.fit_transform(X_blind)

        # Predict on measurements
        y_predict_blind = clf.predict(X_blind_poly)

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

        measurement_csv_filename = "{}_degree_{}_{}.csv".format(
                measurement_output_prefix, str(degree), timestamp)
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

        patient_csv_filename = "{}_degree_{}_{}.csv".format(
                patient_output_prefix, str(degree), timestamp)
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
    Run analysis on 2D Gaussian fit results
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--data_filepath", default=None, required=False,
            help="The file path containing 2D Gaussian fit results labeled data.")
    parser.add_argument(
            "--blind_data_filepath", default=None, required=False,
            help="The file path containing 2D Gaussian fit results for blind data.")
    parser.add_argument(
            "--output_path", default=None, required=False,
            help="The output path to save results in.")
    parser.add_argument(
            "--joblib_filepath", default=None, required=False,
            help="The path to the joblib model file.")
    parser.add_argument(
            "--feature_list", default=None, required=False,
            help="List of features to perform logistic regression on.")
    parser.add_argument(
            "--max_iter", type=int, default=None, required=False,
            help="The maximum iteration number for logistic regression.")
    parser.add_argument(
            "--degree", type=int, default=None, required=False,
            help="The logistic regression decision boundary polynomial degree.")
    parser.add_argument(
            "--use_cross_val", default=False, required=False,
            action="store_true",
            help="The logistic regression decision boundary polynomial degree.")
    parser.add_argument(
            "--balanced", default=None, required=False,
            action="store_true",
            help="Flag to perform balanced logistic regression training.")
    parser.add_argument(
            "--scale_by", type=str, default=None, required=False,
            help="The feature to scale by")

    # Collect arguments
    args = parser.parse_args()
    data_filepath = args.data_filepath
    blind_data_filepath = args.blind_data_filepath
    output_path = args.output_path
    max_iter = args.max_iter
    degree = args.degree
    use_cross_val = args.use_cross_val
    feature_list_kwarg = args.feature_list
    joblib_filepath = args.joblib_filepath
    balanced = args.balanced
    scale_by = args.scale_by

    feature_list = feature_list_kwarg.split(",") if feature_list_kwarg else []

    main(
            data_filepath, blind_data_filepath=blind_data_filepath,
            output_path=output_path, max_iter=max_iter, degree=degree,
            use_cross_val=use_cross_val, feature_list=feature_list,
            joblib_filepath=joblib_filepath, balanced=balanced,
            scale_by=scale_by)
