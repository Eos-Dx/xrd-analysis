"""
Use the 2D Gaussian fit results to analyze 5 A meridional "shoulder"
"""
import os
import argparse
import glob
from collections import OrderedDict
from datetime import datetime
from time import time
from joblib import dump

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
        max_iter=100, degree=1, use_cross_val=False, feature_list=[]):
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

    if not feature_list:
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

    if use_cross_val == True:
        # Randomly split up training and test set
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4, random_state=0)
    else:
        X_train, y_train = X, y

    # Perform Logistic Regression
    # ---------------------------

    # Perform logistic regression
    logreg = LogisticRegression(
            C=1,class_weight="balanced", solver="newton-cg",
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

    print("TP",tp,"FP",fp,"TN",tn,"FN",fn)

    # Set timestamp
    timestr = "%y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    # Set output path with a timestamp if not specified
    if not output_path:
        output_dir = "logistic_regression_{}".format(timestamp)
        output_path = os.path.dirname(data_filepath)

    # Save the model
    model_filename = "logistic_regression_model_{}.joblib".format(timestamp)
    model_filepath = os.path.join(output_path, model_filename)
    dump(pipe, model_filepath)

    # Blind data predictions
    # ----------------------
    # Predict data on blind data set

    if blind_data_filepath:
        # Run blind data through trained model and assess performance
        # Load dataframe
        df_blind = pd.read_csv(blind_data_filepath)

        X_blind_linear = df_blind[[*feature_list]].astype(float).values
        # Create a polynomial
        poly = PolynomialFeatures(degree=degree)

        X_blind_poly = poly.fit_transform(X_blind_linear)

        # Predict
        y_predict = pipe.predict(X_blind_poly)

        # Save results
        df_blind["Prediction"] = y_predict

        # Prefix
        output_prefix = "blind_set_predictions"

        csv_filename = "{}_degree_{}_{}.csv".format(
                output_prefix, str(degree), timestamp)
        csv_output_path = os.path.join(output_path, csv_filename)

        df_blind.to_csv(
                csv_output_path, columns=["Filename", "Prediction"],
                index=False)

        print("Blind predictions saved to", csv_output_path)

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

    # Collect arguments
    args = parser.parse_args()
    data_filepath = args.data_filepath
    blind_data_filepath = args.blind_data_filepath
    output_path = args.output_path
    max_iter = args.max_iter
    degree = args.degree
    use_cross_val = args.use_cross_val
    feature_list_kwarg = args.feature_list

    feature_list = feature_list_kwarg.split(",") if feature_list_kwarg else []

    main(
            data_filepath, blind_data_filepath=blind_data_filepath,
            output_path=output_path, max_iter=max_iter, degree=degree,
            use_cross_val=use_cross_val, feature_list=feature_list)
