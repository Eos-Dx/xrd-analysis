"""
Use the 2D Gaussian fit results to run logistic regression
"""
import os
import argparse
import glob
from collections import OrderedDict
from datetime import datetime
from joblib import dump
from joblib import load

import numpy as np
from numpy.random import default_rng
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
        joblib_filepath=None, balanced=None, scale_by=None,
        random_state=0, test_size=None):

    # Set class_weight
    class_weight = balanced if balanced else None

    # Set rng
    rng = default_rng(random_state)

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
        df = pd.read_csv(data_filepath, index_col="Filename")

        # Scale data
        if scale_by:
            df_scaled_features = scale_features(df, scale_by, feature_list)
            X = df_scaled_features[feature_list].values
        else:
            X = df[feature_list].values

        diagnosis_series = (df["Diagnosis"] == "cancer").astype(int)
        df["y_true"] = diagnosis_series

        # Get training labels
        y = df["y_true"].values

        if test_size:
            # Split training data into training set and test set patientwise
            patient_series = df["Patient_ID"]
            patient_array = patient_series.unique()

            num_patients = patient_array.shape[0]
            num_test = int(test_size * num_patients)
            num_train = num_patients - num_test
            test_indices = rng.choice(np.arange(num_patients), num_test, replace=False)
            train_indices = list(set(np.arange(num_patients)) - set(test_indices))

            # Get the patient train/test split
            test_patient_array = patient_array[test_indices]
            train_patient_array = patient_array[train_indices]

            # Get the measurements
            test_measurements_index = df[df["Patient_ID"].isin(test_patient_array)].index
            train_measurements_index = df[df["Patient_ID"].isin(train_patient_array)].index

            # Store train/test data
            df_train = df[df.index.isin(train_measurements_index)].copy()
            df_test = df[df.index.isin(test_measurements_index)].copy()

            if scale_by:
                X_train_orig = df_scaled_features[df_scaled_features.index.isin(
                    train_measurements_index)][feature_list].values
                X_test_orig = df_scaled_features[df_scaled_features.index.isin(
                    test_measurements_index)][feature_list].values
            else:
                X_train_orig = df.loc[train_measurements_index, feature_list].values
                X_test_orig = df.loc[test_measurements_index, feature_list].values

            y_train = df[df.index.isin(train_measurements_index)]["y_true"].values
            y_test = df[df.index.isin(test_measurements_index)]["y_true"].values

        # Logistic Regression
        # -------------------
        # Create a polynomial
        poly = PolynomialFeatures(degree=degree)

        if test_size:
            X_train = poly.fit_transform(X_train_orig)
            X_test = poly.fit_transform(X_test_orig)
        else:
            df_train = df
            X_train = X
            y_train = y

        # Create classifier
        logreg = LogisticRegression(
                C=1,class_weight=class_weight, solver="newton-cg",
                max_iter=max_iter)
        clf = make_pipeline(StandardScaler(), logreg)

        # Train model
        clf.fit(X_train, y_train)

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
        df_train_patients["y_train_pred"] = y_pred_patients

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

        PrecisionRecallDisplay.from_predictions(y_true_patients, y_score_patients)
        fig_pr = plt.gcf()
        title = "Logistic Regression Precision-Recall Curve Degree {}".format(degree)
        fig_pr.suptitle(title)
        fig_pr_filename = "precision_recall_curve_degree_{}.png".format(degree)
        fig_pr_filepath = os.path.join(output_path, fig_pr_filename)
        fig_pr.savefig(fig_pr_filepath)

        RocCurveDisplay.from_predictions(y_true_patients, y_score_patients)
        fig_roc = plt.gcf()
        title = "Logistic Regression ROC Curve Degree {}".format(degree)
        fig_roc.suptitle(title)
        fig_roc_filename = "roc_curve_degree_{}.png".format(degree)
        fig_roc_filepath = os.path.join(output_path, fig_roc_filename)
        fig_roc.savefig(fig_roc_filepath)

        plt.show()

        if test_size:
            ###############################
            # Get performance on test set #
            ###############################

            # Get patient-wise performance
            # Get patient labels
            y_true_patients = df_test.groupby("Patient_ID")["y_true"].max()

            # Predict on measurements
            y_test_predict = clf.predict(X_test)

            # Save results
            df_test["y_test_pred"] = y_test_predict

            # Get patient predictions
            y_pred_patients = df_test.groupby("Patient_ID")["y_test_pred"].max()

            # Print patient statistics
            accuracy = accuracy_score(y_true_patients, y_pred_patients)
            precision = precision_score(y_true_patients, y_pred_patients)
            sensitivity = recall_score(y_true_patients, y_pred_patients)
            specificity = recall_score(
                    y_true_patients, y_pred_patients, pos_label=0)

            # Generate performance counts
            tn, fp, fn, tp = confusion_matrix(y_true_patients, y_pred_patients).ravel()

            # Print metrics
            print("tn,fp,fn,tp,accuracy,precision,sensitivity,specificity")
            print("{},{},{},{},{:2f},{:2f},{:2f},{:2f}".format(
                tn, fp, fn, tp,
                accuracy, precision, sensitivity,
                specificity))

            # Save test measurement predictions
            measurement_output_prefix = "test_measurement_predictions"

            measurement_csv_filename = "{}_degree_{}_{}.csv".format(
                    measurement_output_prefix, str(degree), timestamp)
            measurement_csv_output_path = os.path.join(
                    output_path, measurement_csv_filename)

            df_test.to_csv(
                    measurement_csv_output_path, columns=["y_test_pred"],
                    index=True)

            print(
                    "Test measurement predictions saved to",
                    measurement_csv_output_path)

            # Save test patient predictions
            patient_output_prefix = "test_patient_predictions"

            patient_csv_filename = "{}_degree_{}_{}.csv".format(
                    patient_output_prefix, str(degree), timestamp)
            patient_csv_output_path = os.path.join(
                    output_path, patient_csv_filename)

            y_pred_patients.to_csv(
                    patient_csv_output_path,
                    index=True)

            print(
                    "Test patient predictions saved to",
                    patient_csv_output_path)


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
    parser.add_argument(
            "--random_state", type=int, default=0, required=False,
            help="Random state seed.")
    parser.add_argument(
            "--test_size", type=float, default=None, required=False,
            help="Size of test set as a fraction of all training data.")

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
    random_state = args.random_state
    test_size = args.test_size

    feature_list = feature_list_kwarg.split(",") if feature_list_kwarg else []

    main(
            data_filepath, blind_data_filepath=blind_data_filepath,
            output_path=output_path, max_iter=max_iter, degree=degree,
            use_cross_val=use_cross_val, feature_list=feature_list,
            joblib_filepath=joblib_filepath, balanced=balanced,
            scale_by=scale_by, random_state=random_state,
            test_size=test_size)
