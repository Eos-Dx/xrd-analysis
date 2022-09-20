"""
Use the 2D Gaussian fit results to analyze 5 A meridional "shoulder"
"""
import os
import argparse
import glob
from collections import OrderedDict
from datetime import datetime
from time import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

from eosdxanalysis.models.curve_fitting import GaussianDecomposition


def main(
        training_data_filepath, test_data_filepath=None, output_path=None,
        max_iter=100, degree=1):
    t0 = time()

    cmap="hot"

    # Load dataframe
    df_train = pd.read_csv(training_data_filepath, index_col=0)

    # Set up figure
    fig = plt.figure(figsize=(8,8))

    # Plot normal
    normal_rows = df_train[df_train["Cancer"] == 0]
    Xnormal = normal_rows["peak_location_radius_5A"]
    Ynormal = normal_rows["peak_location_radius_5_4A"]
    plt.scatter(Xnormal, Ynormal, color="blue", label="Normal", s=10)

    # Plot cancer
    cancer_rows = df_train[df_train["Cancer"] == 1]
    Xcancer = cancer_rows["peak_location_radius_5A"]
    Ycancer = cancer_rows["peak_location_radius_5_4A"]
    plt.scatter(Xcancer, Ycancer, color="red", label="Cancer", s=10)

    # Plot settings
    plt.xlim([45, 70])
    plt.ylim([55, 80])

    plt.xlabel("Peak location radius 5 A [Pixels]")
    plt.ylabel("Peak location radius 5-4 A [Pixels]")

    plt.title("Gaussian Fitting Features Subspace")
    plt.legend()

    # plt.show()
    fig.clear()
    plt.close(fig)

    print("Cancer:",cancer_rows.shape[0])
    print("Normal:",normal_rows.shape[0])


    # Logistic Regression
    # -------------------
    # Data

    feature_list = GaussianDecomposition.parameter_list()

    Xlinear = df_train[[*feature_list]].astype(float).values
    # Create a polynomial
    poly = PolynomialFeatures(degree=degree)

    X = poly.fit_transform(Xlinear)

    patient_averaging = False

    # Patient averaging
    if patient_averaging:
        csv_num = df_train["Patient"].nunique()

        print("There are " + str(csv_num) + " unique samples.")

        # Y = np.zeros((X.shape[0],1))
        # Y = df_train['Cancer'].values.reshape((-1,1))
        # print(Y.shape)

        # Labels
        Y = np.zeros((csv_num,1),dtype=bool)
        X_new = np.zeros((csv_num,2))

        # Loop over each sample
        # and average X and label Y

        for idx in np.arange(csv_num):
            # Get a sample
            sample = df_train.loc[df_train['Barcode'] == barcodes[idx]]
            patient = sample.values[0][1]
            # Get all specimens from the same patient
            df_train_rows = df_train.loc[df_train['Patient'] == patient]
            indices = df_train_rows.index
            # Now average across all samples
            X_new[idx,:] = np.mean(X[indices,:],axis=0)
            # Get the labels for the samples, first one is ok'
            Y[idx] = df_train_rows["Cancer"][indices[0]]


        X = X_new
        print("Total data count after averaging:")
        print(Y.shape)

        print("Normal data count:")
        print(np.sum(Y == False))
        print("Cancer data count:")
        print(np.sum(Y == True))

    # No patient averaging
    elif not patient_averaging:
        Y = df_train["Cancer"]

    # Check that X and Y have same number of rows
    assert(np.array_equal(X.shape[0], Y.shape[0]))

    # Perform Logistic Regression
    # ---------------------------


    # Perform logistic regression
    logreg = LogisticRegression(
            C=1e3,class_weight="balanced", solver="newton-cg",
            max_iter=max_iter)
    pipe = Pipeline([('scaler', StandardScaler()), ('logreg', logreg)])
    pipe.fit(X, Y)
    print("Score:")
    print("{:.2f}".format(pipe.score(X,Y)))

    thetas = pipe['logreg'].coef_.ravel()
    theta0 = pipe['logreg'].intercept_[0]
    theta_array = np.array([[theta0, *thetas]])
    # print("Theta array:")
    # print(theta_array)

    # Predict
    Y_predict = pipe.predict(X)

    # Get scores
    precision = precision_score(Y, Y_predict)
    print("Precision:")
    print("{:2.2}".format(precision))
    recall = recall_score(Y, Y_predict)
    print("Recall (Sensitivity):")
    print("{:2.2}".format(recall))
    # False positive rate: false positives
    # Of samples identified as positive, what percentage are false
    print("False positive rate:")
    false_positives = np.sum(Y[Y_predict == True] == False)
    predicted_positives = np.sum(Y_predict == True)
    false_positive_rate = false_positives/predicted_positives
    print("{:2.2}".format(false_positive_rate))

    # Accuracy = number of correct predictions / total predictions
    # Balanced accuracy score, weights by counts
    balanced_accuracy = balanced_accuracy_score(Y, Y_predict)
    print("Balanced accuracy:")
    print("{:2.2}".format(balanced_accuracy))
    # Unbalanced accuracy
    unbalanced_accuracy = accuracy_score(Y, Y_predict)
    print("Unbalanced accuracy:")
    print("{:2.2}".format(unbalanced_accuracy))

    # Test data
    # ---------
    if test_data_filepath:
        # Run test data through trained model and assess performance
        # Load dataframe
        df_test = pd.read_csv(test_data_filepath, index_col=0)

        X_test_linear = df_test[[*feature_list]].astype(float).values
        # Create a polynomial
        poly = PolynomialFeatures(degree=degree)

        X_test_poly = poly.fit_transform(X_test_linear)

        # Predict
        Y_predict = pipe.predict(X_test_poly)

        # Save results
        df_test["Prediction"] = Y_predict

        # Prefix
        output_prefix = "test_set_predictions"

        # Set output path with a timestamp if not specified
        if not output_path:
            # Set timestamp
            timestr = "%Y%m%dT%H%M%S.%f"
            timestamp = datetime.utcnow().strftime(timestr)

            output_dir = "predictions_".format(timestamp)
            output_path = os.path.dirname(training_data_filepath)

        csv_filename = "{}_degree_{}_{}.csv".format(
                output_prefix, str(degree), timestamp)
        csv_output_path = os.path.join(output_path, csv_filename)

        df_test["Prediction"].to_csv(csv_output_path)

    # TODO: Save model

if __name__ == '__main__':
    """
    Run shoulder analysis on 2D Gaussian fit results
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--training_data_filepath", default=None, required=True,
            help="The file path containing 2D Gaussian fit results training data.")
    parser.add_argument(
            "--test_data_filepath", default=None, required=False,
            help="The file path containing 2D Gaussian fit results for test data.")
    parser.add_argument(
            "--output_path", default=None, required=False,
            help="The output path to save results in.")
    parser.add_argument(
            "--max_iter", type=int, default=None, required=False,
            help="The maximum iteration number for logistic regression.")
    parser.add_argument(
            "--degree", type=int, default=None, required=False,
            help="The logistic regression decision boundary polynomial degree.")

    # Collect arguments
    args = parser.parse_args()
    training_data_filepath = args.training_data_filepath
    test_data_filepath = args.test_data_filepath
    output_path = args.output_path
    max_iter = args.max_iter
    degree = args.degree

    main(
            training_data_filepath, test_data_filepath=test_data_filepath,
            output_path=output_path, max_iter=max_iter, degree=degree)
