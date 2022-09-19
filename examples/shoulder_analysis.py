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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

from eosdxanalysis.models.curve_fitting import GaussianDecomposition


def main(fit_results_file, output_path=None, max_iter=100):
    t0 = time()

    cmap="hot"

    # Load dataframe
    df = pd.read_csv(fit_results_file, index_col=0)

    # Set up figure
    fig = plt.figure(figsize=(8,8))

    # Plot normal
    normal_rows = df[df["Cancer"] == 0]
    Xnormal = normal_rows["peak_location_radius_5A"]
    Ynormal = normal_rows["peak_location_radius_5_4A"]
    plt.scatter(Xnormal, Ynormal, color="blue", label="Normal", s=10)

    # Plot cancer
    cancer_rows = df[df["Cancer"] == 1]
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

    X = df[[*feature_list]].astype(float).values

    patient_averaging = False

    # Patient averaging
    if patient_averaging:
        csv_num = df["Patient"].nunique()

        print("There are " + str(csv_num) + " unique samples.")

        # Y = np.zeros((X.shape[0],1))
        # Y = df['Cancer'].values.reshape((-1,1))
        # print(Y.shape)

        # Labels
        Y = np.zeros((csv_num,1),dtype=bool)
        X_new = np.zeros((csv_num,2))

        # Loop over each sample
        # and average X and label Y

        for idx in np.arange(csv_num):
            # Get a sample
            sample = df.loc[df['Barcode'] == barcodes[idx]]
            patient = sample.values[0][1]
            # Get all specimens from the same patient
            df_rows = df.loc[df['Patient'] == patient]
            indices = df_rows.index
            # Now average across all samples
            X_new[idx,:] = np.mean(X[indices,:],axis=0)
            # Get the labels for the samples, first one is ok'
            Y[idx] = df_rows["Cancer"][indices[0]]


        X = X_new
        print("Total data count after averaging:")
        print(Y.shape)

        print("Normal data count:")
        print(np.sum(Y == False))
        print("Cancer data count:")
        print(np.sum(Y == True))

    # No patient averaging
    elif not patient_averaging:
        Y = df["Cancer"]

    # Check that X and Y have same number of rows
    assert(np.array_equal(X.shape[0], Y.shape[0]))

    # Perform Logistic Regression
    # ---------------------------

    # Perform logistic regression
    logreg = LogisticRegression(
            C=1e6,class_weight="balanced", solver="newton-cg",
            max_iter=max_iter)
    logreg.fit(X, Y)
    print("Score: {:.2f}".format(logreg.score(X,Y)))

    thetas = logreg.coef_.ravel()
    theta0 = logreg.intercept_[0]
    theta_array = np.array([[theta0, *thetas]])
    # print("Theta array:")
    # print(theta_array)

    # Predict
    Y_predict = logreg.predict(X)

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


if __name__ == '__main__':
    """
    Run shoulder analysis on 2D Gaussian fit results
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_filepath", default=None, required=True,
            help="The file path containing 2D Gaussian fit results.")
    parser.add_argument(
            "--output_path", default=None, required=False,
            help="The output path to save results in.")
    parser.add_argument(
            "--max_iter", type=int, default=None, required=False,
            help="The maximum iteration number for logistic regression.")

    # Collect arguments
    args = parser.parse_args()
    input_filepath = args.input_filepath
    output_path = args.output_path
    max_iter = args.max_iter

    main(input_filepath, output_path=output_path, max_iter=max_iter)
