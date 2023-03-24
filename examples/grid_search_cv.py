"""
Grid Search CV
"""
import os
import argparse
import glob

from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline

from sklearn.metrics import euclidean_distances
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from eosdxanalysis.models.utils import scale_features
from eosdxanalysis.models.utils import specificity_score


def main(
        data_filepath=None, output_path=None,
        max_iter=1000, degree=1, feature_list=[],
        joblib_filepath=None, balanced=None, scale_by=None,
        random_state=0, patient_agg=None, feature_range=None,
        C_doublings=0, ideal_measurement_filename=None):

    # Set timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    # Set output path with a timestamp if not specified
    if not output_path:
        output_dir = "logistic_regression_{}".format(timestamp)
        output_path = os.path.dirname(data_filepath)

    if not feature_list:
        feature_list = feature_range

    # Load data
    df = pd.read_csv(data_filepath, index_col="Filename").dropna()

    # remove duplicates
    df = df[~df.duplicated(keep="first")].copy()
    df_train = df.copy()

    # Scale data by intensity
    if scale_by:
        df_scaled_features = scale_features(df_train, scale_by, feature_list)
        df_train[feature_list] = df_scaled_features[feature_list].values

    # Reduce dataset to get one data point per patient
    if ideal_measurement_filename:
        # Use the specified ideal measurement to calculate euclidean distances
        # for each measurement
        y_ideal = df_train.loc[ideal_measurement_filename, feature_list].values
        y_ideal = y_ideal.reshape(1,-1)

        # Calculate euclidean distances
        df_train["distance_to_ideal"] = euclidean_distances(
                df_train[feature_list], y_ideal)

        # Perform measurement aggregation
        if patient_agg in ["min", "max"]:
            keep = df_train.groupby(
                    "Patient_ID")["distance_to_ideal"].transform(
                            patient_agg).eq(
                                    df_train["distance_to_ideal"])
            df_train = df_train[keep].copy()
        elif patient_agg in ["mean", "median"]:
            agg_data_series = df_train.groupby(
                    "Patient_ID")["distance_to_ideal"].agg(
                            patient_agg)
            agg_feature_name = "{}_distance_to_ideal".format(patient_agg)
            feature_list = [agg_feature_name]

            # Get patient diagnosis series
            patient_series = df_train.groupby("Patient_ID")["Diagnosis"].max()

            # Create new dataframe with mean/median distances for
            # training data
            df_train = pd.DataFrame(
                    agg_data_series.values,
                    index=agg_data_series.index,
                    columns=feature_list)

            df_train["Diagnosis"] = patient_series

    # Get the true labels
    diagnosis_series = (df_train["Diagnosis"] == "cancer").astype(int)
    df_train["y_true"] = diagnosis_series

    # Get training labels
    y = df_train["y_true"].values

    # Set up polynomial
    poly = PolynomialFeatures(degree=degree)

    X = poly.fit_transform(df_train[feature_list].values)

    # Set grid search parameters
    C_range = 2**np.arange(C_doublings+1)
    parameters = {
            'max_iter': [max_iter],
            'solver':['newton-cg'],
            'C': C_range,
            }

    # Set scoring outputs
    scoring={
            "accuracy": make_scorer(accuracy_score),
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
            "precision": make_scorer(precision_score),
            "sensitivity": make_scorer(recall_score),
            "specificity": make_scorer(specificity_score),
            "roc_auc": make_scorer(roc_auc_score),
            }

    clf = make_pipeline(
            StandardScaler(), 
            GridSearchCV(
                LogisticRegression(),
                param_grid=parameters,
                scoring=scoring,
                return_train_score=True,
                cv=5,
                refit="roc_auc"))

    # Fit the classifier
    clf.fit(X, y)

    # Output results to csv file
    # Convert results to dataframe
    results = clf["gridsearchcv"].cv_results_
    df_results =  pd.DataFrame.from_dict(data=results, orient='index')

    # Set output filename
    if patient_agg:
        results_output_filename = "gridsearch_results_degree_{}_{}_{}.csv".format(
                degree,
                patient_agg,
                timestamp)
    else:
        results_output_filename = "gridsearch_results_degree_{}_{}.csv".format(
                degree,
                timestamp)
    results_output_filepath = os.path.join(output_path, results_output_filename)

    # Save results to file
    df_results.to_csv(
            results_output_filepath,
            header=C_range.astype(str),
            index_label="scoring")


if __name__ == '__main__':
    """
    Run logistic regression grid search cross-validation
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--data_filepath", default=None, required=False,
            help="The file path containing 2D Gaussian fit results labeled data.")
    parser.add_argument(
            "--output_path", default=None, required=False,
            help="The output path to save results in.")
    parser.add_argument(
            "--feature_list", default=None, required=False,
            help="List of features to perform logistic regression on.")
    parser.add_argument(
            "--max_iter", type=int, default=1000, required=False,
            help="The maximum iteration number for logistic regression.")
    parser.add_argument(
            "--degree", type=int, default=1, required=False,
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
            "--patient_agg", type=str, default=None, required=False,
            help="Score aggregation method (\"max\", \"min\", or \"mean\")")
    parser.add_argument(
            "--feature_range_count", type=int, default=None, required=False,
            help="Number of 1-indexed integer feature labels to use.")
    parser.add_argument(
            "--C_doublings", type=int, default=0, required=False,
            help="Number of C-doublings to use in grid search CV.")
    parser.add_argument(
            "--ideal_measurement_filename", type=str, default=None, required=False,
            help="Barcode of the measurement closest to the ideal keratin pattern.")

    # Collect arguments
    args = parser.parse_args()
    data_filepath = args.data_filepath
    output_path = args.output_path
    max_iter = args.max_iter
    degree = args.degree
    feature_list_kwarg = args.feature_list
    balanced = args.balanced
    scale_by = args.scale_by
    random_state = args.random_state
    patient_agg = args.patient_agg
    feature_range_count = args.feature_range_count
    C_doublings = args.C_doublings
    ideal_measurement_filename = args.ideal_measurement_filename

    feature_list = feature_list_kwarg.split(",") if feature_list_kwarg else []
    feature_range = (np.arange(feature_range_count) + 1).astype(str) if feature_range_count else None

    if feature_list == [] and feature_range_count < 1:
        raise ValueError("Must supply feature list or feature range count.")

    main(
            data_filepath,
            output_path=output_path,
            max_iter=max_iter,
            degree=degree,
            feature_list=feature_list,
            balanced=balanced,
            scale_by=scale_by,
            random_state=random_state,
            patient_agg=patient_agg,
            feature_range=feature_range,
            C_doublings=C_doublings,
            ideal_measurement_filename=ideal_measurement_filename,
            )
