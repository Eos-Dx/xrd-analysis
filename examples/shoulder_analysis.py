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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay

from eosdxanalysis.models.curve_fitting import GaussianDecomposition


def main(
        data_filepath, blind_data_filepath=None, output_path=None,
        max_iter=100, degree=1):
    t0 = time()

    cmap="hot"

    # Load dataframe
    df = pd.read_csv(data_filepath, index_col=0)

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

    Xlinear = df[[*feature_list]].astype(float).values
    # Create a polynomial
    poly = PolynomialFeatures(degree=degree)

    X = poly.fit_transform(Xlinear)

    patient_averaging = False

    # Patient averaging
    if patient_averaging:
        csv_num = df["Patient"].nunique()

        print("There are " + str(csv_num) + " unique samples.")

        # y = np.zeros((X.shape[0],1))
        # y = df['Cancer'].values.reshape((-1,1))
        # print(..shape)

        # Labels
        y = np.zeros((csv_num,1),dtype=bool)
        X_new = np.zeros((csv_num,2))

        # Loop over each sample
        # and average X and label y

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
            y[idx] = df_rows["Cancer"][indices[0]]


        X = X_new
        print("Total data count after averaging:")
        print(y.shape)

        print("Normal data count:")
        print(np.sum(y == False))
        print("Cancer data count:")
        print(np.sum(y == True))

    # No patient averaging
    elif not patient_averaging:
        y = df["Cancer"]

    # Check that X and y have same number of rows
    assert(np.array_equal(X.shape[0], y.shape[0]))

    # Randomly split up training and test set
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=0)

    # Perform Logistic Regression
    # ---------------------------

    # Perform logistic regression
    logreg = LogisticRegression(
            C=1e3,class_weight="balanced", solver="newton-cg",
            max_iter=max_iter)
    pipe = Pipeline([('scaler', StandardScaler()), ('logreg', logreg)])
    pipe.fit(X_train, y_train)

    scores = cross_val_score(pipe, X, y, cv=5)

    # Now check performance on entire set
    # Predict
    y_predict = pipe.predict(X)

    # Get true negatives, false positives, false negatives, true positives
    tn, fp, fn, tp = confusion_matrix(y, y_predict).ravel()

    # False positive rate
    false_positive_rate = fp/(fp + tp)
    # False negative rate
    false_negative_rate = fn/(fn + tn)

    # Get scores
    precision = precision_score(y, y_predict)
    recall = recall_score(y, y_predict)

    # Accuracy = number of correct predictions / total predictions
    # Balanced accuracy score, weights by counts
    balanced_accuracy = balanced_accuracy_score(y, y_predict)
    # Unbalanced accuracy
    unbalanced_accuracy = accuracy_score(y, y_predict)

    # Print scores
    print("Unbalanced Accuracy", end=" | ")
    print("Balanced Accuracy", end=" | ")
    print("Precision", end=" | ")
    print("Recall (Sensitivity)", end=" | ")
    print("False Positive Rate", end=" | ")
    print("False Negative Rate", end="\n")

    # 
    print("{:2.2}".format(balanced_accuracy), end=" | ")
    print("{:2.2}".format(unbalanced_accuracy), end=" | ")
    print("{:2.2}".format(precision), end=" | ")
    print("{:2.2}".format(recall), end=" | ")
    print("{:2.2}".format(false_positive_rate), end=" | ")
    print("{:2.2}".format(false_negative_rate), end="\n")

    return scores

if __name__ == '__main__':
    """
    Run shoulder analysis on 2D Gaussian fit results
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--data_filepath", default=None, required=True,
            help="The file path containing 2D Gaussian fit results labeled data.")
    parser.add_argument(
            "--blind_data_filepath", default=None, required=False,
            help="The file path containing 2D Gaussian fit results for blind data.")
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
    data_filepath = args.data_filepath
    blind_data_filepath = args.blind_data_filepath
    output_path = args.output_path
    max_iter = args.max_iter
    degree = args.degree

    main(
            data_filepath, blind_data_filepath=blind_data_filepath,
            output_path=output_path, max_iter=max_iter, degree=degree)
