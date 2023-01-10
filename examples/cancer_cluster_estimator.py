"""
Code to run cancer predictions using cancer cluster estimator
"""
import numpy as np
import pandas as pd
import argparse

from joblib import load

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

from eosdxanalysis.models.estimators import CancerClusterEstimator
from eosdxanalysis.models.estimators import PatientCancerClusterEstimator


def run_cancer_cluster_predictions_on_df(
        df_train, df_predict=None, output_path=None, cancer_cluster_list=None,
        cancer_label=1, distance_threshold=0.5):
    """
    Run cancer predictions on preprocessed dataframe
    """
    estimator = CancerClusterEstimator(distance_threshold=0.5, cancer_label=1)
    X = df_train
    y = df_train.index.astype(int)
    estimator.fit(X, y)
    predictions = estimator.predict(df_predict)
    return predictions

def train_cancer_cluster_model_on_df(
        df_train, df_predict=None, output_path=None, cancer_cluster_list=None,
        cancer_label=1, distance_threshold=0.5):
    """
    Train model on labeled cancer cluster centers
    """
    estimator = CancerClusterEstimator(distance_threshold=0.5, cancer_label=1)
    X = df_train

    # Rename y_true using cancer_cluster_list
    y_true = df_train.index.isin(cancer_cluster_list).astype(int)
    estimator.fit(X, y_true)
    y_pred = estimator.predict(df_train)
    return y_true, y_pred

def train_cancer_cluster_model_on_df_file(
        training_data_filepath, output_path=None,
        cancer_cluster_list=None, cancer_label=1, distance_threshold=0.5):
    """
    Train model on labeled cancer cluster centers
    """
    # Load training data
    df_train = pd.read_csv(training_data_filepath, index_col="cluster")

    # Run predictions
    y_true, y_pred = train_cancer_cluster_model_on_df(
            df_train, output_path=output_path,
            cancer_cluster_list=cancer_cluster_list, cancer_label=cancer_label,
            distance_threshold=distance_threshold)
    return y_true, y_pred

def run_cancer_cluster_predictions_on_df_file(
        training_data_filepath, data_filepath=None, output_path=None,
        cancer_cluster_list=None, cancer_label=1, distance_threshold=0.5):
    """
    Run cancer predictions on preprocessed data from file
    """
    # Load training data
    df_train = pd.read_csv(training_data_filepath, index_col="cluster")
    # Load data to predict on
    if data_filepath is not None:
        df_predict = pd.read_csv(data_filepath, index_col="Filename")
    else:
        df_predict = None

    # Run predictions
    predictions = run_cancer_cluster_predictions_on_df(
            df_train, df_predict=df_predict, output_path=output_path,
            cancer_cluster_list=cancer_cluster_list, cancer_label=cancer_label,
            distance_threshold=distance_threshold)
    return predictions

def grid_search_cluster_model_on_df(
        df_train, output_path=None,
        cancer_cluster_list=None, cancer_label=1, distance_threshold=0.5):
    """
    Perform grid search with cross-validation on training data
    """
    # Get training data
    X = df_train[df_train.columns[:-1]]
    # Get training data labels
    y_true = df_train["kmeans_20"].isin(cancer_cluster_list).astype(int)

    # Set parameter search grid
    threshold_range = np.arange(0,5,0.5)
    param_grid = {
            "distance_threshold": threshold_range,
            }

    clf = GridSearchCV(CancerClusterEstimator(cancer_label=1), param_grid)
    # Run grid search
    clf.fit(X, y_true)

    return clf

def grid_search_cluster_model_on_df_file(
        training_data_filepath, output_path=None,
        cancer_cluster_list=None, cancer_label=1, distance_threshold=0.5):
    """
    Perform grid search with cross-validation on training data
    """
    # Load training data
    df_train = pd.read_csv(training_data_filepath, index_col="Filename")

    # Run predictions
    results = grid_search_cluster_model_on_df(
            df_train, output_path=output_path,
            cancer_cluster_list=cancer_cluster_list, cancer_label=cancer_label,
            distance_threshold=distance_threshold)
    return results

def run_patient_predictions_kmeans(
        training_data_filepath=None, data_filepath=None,
        feature_list=None, cancer_cluster_list=None, divide_by=None,
        normal_cluster_list=None, cluster_model_name=None,
        unsupervised_estimator_filepath=None,
        patient_db=None,
        blind_predictions=None,
        ):
    # Load training data
    df_train = pd.read_csv(training_data_filepath, index_col="Filename")

    # Scale all features
    if divide_by:
        df_train = df_train.div(df_train[divide_by], axis="rows")
        df_train = df_train[feature_list]
        if divide_by in feature_list:
            # Drop divide_by column
            df_train = df_train.drop(columns=[divide_by])

    # Load test data
    df_test = None
    if data_filepath:
        df_test = pd.read_csv(data_filepath, index_col="Filename")
        # Scale all features
        if divide_by:
            df_test = df_test.div(df_test[divide_by], axis="rows")
            df_test = df_test[feature_list]
            if divide_by in feature_list:
                # Drop divide_by column
                df_test = df_test.drop(columns=[divide_by])

    # Transform kmeans cluster centers
    unsupervised_estimator = load(unsupervised_estimator_filepath)
    scaler = unsupervised_estimator["standardscaler"]
    kmeans_model = unsupervised_estimator["kmeans"]

    # Get cluster data
    X_train_scaled = scaler.transform(df_train[feature_list])
    df_train[cluster_model_name] = kmeans_model.predict(X_train_scaled)

    # Use patient database to get patient diagnosis
    db = pd.read_csv(patient_db, index_col="Barcode")
    extraction = df_train.index.str.extractall(
            "CR_([A-Z]{1}).*?([0-9]+)")
    extraction_series = extraction[0] + extraction[1].str.zfill(5)
    extraction_list = extraction_series.tolist()

    assert(len(extraction_list) == df_train.shape[0])
    df_train_ext = df_train.copy()
    df_train_ext["Barcode"] = extraction_list

    df_train_ext = pd.merge(
            df_train_ext, db, left_on="Barcode", right_index=True)

    # Set up measurement-wise true labels of the training data
    y_true_measurements = pd.Series(index=df_train_ext.index, dtype=int)
    if cancer_cluster_list in ("", None):
        y_true_measurements[df_train_ext[cluster_model_name].isin(normal_cluster_list)] = 0
        y_true_measurements[~df_train_ext[cluster_model_name].isin(normal_cluster_list)] = 1
    if normal_cluster_list in ("", None):
        y_true_measurements[df_train_ext[cluster_model_name].isin(cancer_cluster_list)] = 1
        y_true_measurements[~df_train_ext[cluster_model_name].isin(cancer_cluster_list)] = 0
    # y_true_measurements = y_true_measurements.values
    df_train_ext["predictions"] = y_true_measurements.astype(int)

    # Generate patient-wise true labels of the training data
    # s_true_measurements = pd.Series(y_true_measurements, index=df_train_ext["Patient_ID"], dtype=int)
    y_true_patients = df_train_ext.groupby("Patient_ID")["Diagnosis"].max()
    y_true_patients.loc[y_true_patients == "cancer"] = 1
    y_true_patients.loc[y_true_patients == "healthy"] = 0
    y_true_patients = y_true_patients.astype(int)

    y_pred_patients = df_train_ext.groupby("Patient_ID")["predictions"].max().values

    print("tn,fp,fn,tp,accuracy,precision,sensitivity,specificity")

    # Generate prediction counts
    tn, fp, fn, tp = confusion_matrix(y_true_patients, y_pred_patients).ravel()

    # Generate scores
    accuracy = accuracy_score(y_true_patients, y_pred_patients)
    precision = precision_score(y_true_patients, y_pred_patients)
    sensitivity = recall_score(y_true_patients, y_pred_patients)
    specificity = recall_score(
            y_true_patients, y_pred_patients, pos_label=0)

    print("{:d},{:d},{:d},{:d},{:2f},{:2f},{:2f},{:2f}".format(
                tn, fp, fn, tp,
                accuracy, precision, sensitivity, specificity))

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    # Make blind predictions
    if df_test is not None:
        extraction = df_test.index.str.extractall(
                "CR_([A-Z]{1}).*?([0-9]+)")
        extraction_series = extraction[0] + extraction[1].str.zfill(5)
        extraction_list = extraction_series.tolist()

        assert(len(extraction_list) == df_test.shape[0])
        df_test_ext = df_test.copy()
        df_test_ext["Barcode"] = extraction_list

        df_test_ext = pd.merge(
                df_test_ext, db, left_on="Barcode", right_index=True)

        # Get cluster data
        X_test_scaled = scaler.transform(df_test_ext[feature_list])
        df_test_ext[cluster_model_name] = kmeans_model.predict(X_test_scaled)

        # Predict measurements
        y_test_pred_measurements = pd.Series(index=df_test_ext.index, dtype=int)
        if cancer_cluster_list in ("", None):
            y_test_pred_measurements[df_test_ext[cluster_model_name].isin(normal_cluster_list)] = 0
            y_test_pred_measurements[~df_test_ext[cluster_model_name].isin(normal_cluster_list)] = 1
        if normal_cluster_list in ("", None):
            y_test_pred_measurements[df_test_ext[cluster_model_name].isin(cancer_cluster_list)] = 1
            y_test_pred_measurements[~df_test_ext[cluster_model_name].isin(cancer_cluster_list)] = 0

        # y_true_measurements = y_true_measurements.values
        df_test_ext["predictions"] = y_test_pred_measurements.astype(int)

        # Group by patients
        df_test_pred_patients = df_test_ext.groupby("Patient_ID")["predictions"].max()

        # df_test_pred_patients.to_csv()
        cancer_prediction_num = df_test_pred_patients.sum()
        patient_num = df_test_pred_patients.shape[0]
        normal_prediction_num = patient_num - cancer_prediction_num
        print("normal,cancer,total")
        print("{},{},{}".format(
            normal_prediction_num,
            cancer_prediction_num,
            patient_num))

        # Save blind predictions to file
        df_test_pred_patients.to_csv(blind_predictions)

def run_patient_predictions_pointwise(
        training_data_filepath=None, data_filepath=None,
        feature_list=None, cancer_cluster_list=None,
        normal_cluster_list=None, cluster_model_name=None,
        unsupervised_estimator_filepath=None,
        blind_predictions=None,
        threshold=None,
        divide_by=None,
        patient_db=None,
        ):
    # Load training data
    df_train = pd.read_csv(training_data_filepath, index_col="Filename")

    # Scale all features
    if divide_by:
        df_train = df_train.div(df_train[divide_by], axis="rows")
        df_train = df_train[feature_list]
        if divide_by in feature_list:
            # Drop divide_by column
            df_train = df_train.drop(columns=[divide_by])

    # Load test data
    df_test = None
    if data_filepath:
        df_test = pd.read_csv(data_filepath, index_col="Filename")
        # Scale all features
        if divide_by:
            df_test = df_test.div(df_test[divide_by], axis="rows")
            df_test = df_test[feature_list]
            if divide_by in feature_list:
                # Drop divide_by column
                df_test = df_test.drop(columns=[divide_by])

    # Transform kmeans cluster centers
    unsupervised_estimator = load(unsupervised_estimator_filepath)
    scaler = unsupervised_estimator["standardscaler"]
    kmeans_model = unsupervised_estimator["kmeans"]

    # Use patient database to get patient diagnosis
    db = pd.read_csv(patient_db, index_col="Barcode")
    extraction = df_train.index.str.extractall(
            "CR_([A-Z]{1}).*?([0-9]+)")
    extraction_series = extraction[0] + extraction[1].str.zfill(5)
    extraction_list = extraction_series.tolist()

    assert(len(extraction_list) == df_train.shape[0])
    df_train_ext = df_train.copy()
    df_train_ext["Barcode"] = extraction_list

    df_train_ext = pd.merge(
            df_train_ext, db, left_on="Barcode", right_index=True)

    # Run k-means model to label data points
    df_train_ext[cluster_model_name] = kmeans_model.predict(df_train_ext[feature_list])

    # Set up measurement-wise true labels of the training data
    y_true_measurements = pd.Series(index=df_train_ext.index, dtype=int)
    if cancer_cluster_list in ("", None):
        y_true_measurements[df_train_ext[cluster_model_name].isin(normal_cluster_list)] = 0
        y_true_measurements[~df_train_ext[cluster_model_name].isin(normal_cluster_list)] = 1
    if normal_cluster_list in ("", None):
        y_true_measurements[df_train_ext[cluster_model_name].isin(cancer_cluster_list)] = 1
        y_true_measurements[~df_train_ext[cluster_model_name].isin(cancer_cluster_list)] = 0

    # Generate patient-wise true labels of the training data
    y_true_patients = df_train_ext.groupby("Patient_ID")["Diagnosis"].max()
    y_true_patients.loc[y_true_patients == "cancer"] = 1
    y_true_patients.loc[y_true_patients == "healthy"] = 0
    y_true_patients = y_true_patients.astype(int)

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    # Loop over thresholds
    # Set the threshold range to loop over
    # threshold_range = np.arange(0, 4, 0.2)
    # Create the estimator
    threshold_range = np.arange(0, 10, 0.1)
    print("tn,fp,fn,tp,threshold,accuracy,precision,sensitivity,specificity")
    # Store values for plotting
    sensitivity_array = np.zeros_like(threshold_range)
    specificity_array = np.zeros_like(threshold_range)
    precision_array = np.zeros_like(threshold_range)
    for idx in range(threshold_range.size):
        distance_threshold = threshold_range[idx]
        estimator = PatientCancerClusterEstimator(
                distance_threshold=distance_threshold,
                cancer_label=1,
                normal_label=0,
                cancer_cluster_list=cancer_cluster_list,
                normal_cluster_list=normal_cluster_list,
                feature_list=feature_list, label_name=cluster_model_name)

        # Fit the estimator the training data
        estimator.fit(df_train_ext, y_true_measurements)

        # Generate measurement-wise predictions of the training data
        y_pred_patients = estimator.predict(df_train_ext)

        # Generate scores
        accuracy = accuracy_score(y_true_patients, y_pred_patients)
        precision = precision_score(y_true_patients, y_pred_patients)
        sensitivity = recall_score(
                y_true_patients, y_pred_patients, pos_label=1)
        specificity = recall_score(
                y_true_patients, y_pred_patients, pos_label=0)

        # Generate performance counts
        tn, fp, fn, tp = confusion_matrix(y_true_patients, y_pred_patients).ravel()

        # Store scores
        sensitivity_array[idx] = sensitivity
        specificity_array[idx] = specificity
        precision_array[idx] = precision

        print("{:.2f},{:2f},{:2f},{:2f},{:.2f},{:2f},{:2f},{:2f},{:2f}".format(
                    tn, fp, fn, tp,
                    distance_threshold, accuracy, precision, sensitivity,
                    specificity))

    # Manually create ROC and precision-recall curves
    subtitle = "Pointwise Cancer Distance Model"
    tpr = sensitivity_array
    fpr = 1 - specificity_array
    x_offset = 0
    y_offset = 0.002

    # ROC Curve
    title = "ROC Curve: {}".format(subtitle)
    fig = plt.figure(title, figsize=(12,12))
    plt.step(fpr, tpr, where="post")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title(title)

    # Annotate by threshold
    for x, y, s in zip(fpr, tpr, threshold_range):
        plt.text(x+x_offset, y+y_offset, np.round(s,1))
    
    # Precision-Recall Curve
    title = "Precision-Recall Curve: {}".format(subtitle)
    fig = plt.figure(title, figsize=(12,12))
    recall_array = sensitivity_array
    plt.step(recall_array, precision_array, where="pre")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title(title)

    # Annotate by threshold
    for x, y, s in zip(recall_array, precision_array, threshold_range):
        plt.text(x+x_offset, y+y_offset, np.round(s,1))

    plt.show()

    if df_test is not None:
        # Make predictions on test data
        distance_threshold = threshold
        # Transform df_test data using kmeans_model
        unsupervised_estimator = load(unsupervised_estimator_filepath)
        kmeans_model = unsupervised_estimator["kmeans"]
        scaler = unsupervised_estimator["standardscaler"]
        X_test_scaled = scaler.transform(df_test[feature_list])
        df_test_scaled = pd.DataFrame(
                data=X_test_scaled, columns=feature_list, index=df_test.index)
        df_test_scaled["Patient_ID"] = df_test["Patient_ID"]

        # Create estimator
        estimator = PatientCancerClusterEstimator(
                distance_threshold=distance_threshold,
                cancer_label=1,
                normal_label=0,
                cancer_cluster_list=cancer_cluster_list,
                normal_cluster_list=normal_cluster_list,
                feature_list=feature_list, label_name=cluster_model_name)

        # Fit the estimator to the cluster training data
        estimator.fit(df_train, y_true_measurements)

        # Generate measurement-wise predictions of the training data
        y_test_pred_patients = estimator.predict(df_test_scaled).astype(int)

        patient_list = df_test["Patient_ID"].unique()
        df_test_pred_patients = pd.DataFrame(
                data=y_test_pred_patients, columns=["prediction"], index=patient_list)

        # df_test_patients_predictions.to_csv()

        patient_num = df_test_pred_patients.shape[0]
        cancer_prediction_num = df_test_pred_patients.sum()
        normal_prediction_num = patient_num - cancer_prediction_num
        print("Total blind patients:", patients_num)
        print("Total blind cancer predictions:",cancer_prediction_num)
        print("Total blind normal predictions:",normal_prediction_num)

        # Save blind predictions to file
        df_test_pred_patients.to_csv(blind_predictions)

def run_patient_predictions_clusterwise(
        training_data_filepath=None, data_filepath=None,
        feature_list=None, cancer_cluster_list=None,
        normal_cluster_list=None, cluster_model_name=None,
        unsupervised_estimator_filepath=None,
        divide_by=None, patient_db=None,
        blind_predictions=None,
        threshold=None,
        ):
    # Load training data
    df_train = pd.read_csv(training_data_filepath, index_col="Filename")

    # Scale all features
    if divide_by:
        df_train = df_train.div(df_train[divide_by], axis="rows")
        df_train = df_train[feature_list]
        if divide_by in feature_list:
            # Drop divide_by column
            df_train = df_train.drop(columns=[divide_by])

    # Load test data
    df_test = None
    if data_filepath:
        df_test = pd.read_csv(data_filepath, index_col="Filename")
        # Scale all features
        if divide_by:
            df_test = df_test.div(df_test[divide_by], axis="rows")
            df_test = df_test[feature_list]
            if divide_by in feature_list:
                # Drop divide_by column
                df_test = df_test.drop(columns=[divide_by])

    # Transform kmeans cluster centers
    unsupervised_estimator = load(unsupervised_estimator_filepath)
    scaler = unsupervised_estimator["standardscaler"]
    kmeans_model = unsupervised_estimator["kmeans"]

    # Use patient database to get patient diagnosis
    db = pd.read_csv(patient_db, index_col="Barcode")
    extraction = df_train.index.str.extractall(
            "CR_([A-Z]{1}).*?([0-9]+)")
    extraction_series = extraction[0] + extraction[1].str.zfill(5)
    extraction_list = extraction_series.tolist()

    assert(len(extraction_list) == df_train.shape[0])
    df_train_ext = df_train.copy()
    df_train_ext["Barcode"] = extraction_list

    df_train_ext = pd.merge(
            df_train_ext, db, left_on="Barcode", right_index=True)

    df_train_ext[cluster_model_name] = kmeans_model.predict(df_train_ext[feature_list])

    # Set up measurement-wise true labels of the training data
    y_true_measurements = pd.Series(index=df_train_ext.index, dtype=int)
    if cancer_cluster_list in ("", None):
        y_true_measurements[df_train_ext[cluster_model_name].isin(normal_cluster_list)] = 0
        y_true_measurements[~df_train_ext[cluster_model_name].isin(normal_cluster_list)] = 1
    if normal_cluster_list in ("", None):
        y_true_measurements[df_train_ext[cluster_model_name].isin(cancer_cluster_list)] = 1
        y_true_measurements[~df_train_ext[cluster_model_name].isin(cancer_cluster_list)] = 0
    # y_true_measurements = y_true_measurements.values

    # Generate patient-wise true labels of the training data
    # s_true_measurements = pd.Series(y_true_measurements, index=df_train["Patient_ID"], dtype=int)
    y_true_patients = df_train_ext.groupby("Patient_ID")["Diagnosis"].max()
    y_true_patients.loc[y_true_patients == "cancer"] = 1
    y_true_patients.loc[y_true_patients == "healthy"] = 0
    y_true_patients = y_true_patients.astype(int)

    # Transform kmeans cluster centers
    unsupervised_estimator = load(unsupervised_estimator_filepath)
    kmeans_model = unsupervised_estimator["kmeans"]
    clusters = kmeans_model.cluster_centers_

    df_clusters = pd.DataFrame(data=clusters, columns=feature_list)
    n_clusters = clusters.shape[0]
    df_clusters["kmeans_{}".format(n_clusters)] = np.arange(n_clusters)
    y_true_clusters = np.zeros((n_clusters))
    y_true_clusters[cancer_cluster_list] = 1

    # Loop over thresholds
    # Set the threshold range to loop over
    threshold_range = np.arange(0, 10, 0.1)
    # Store values for plotting
    sensitivity_array = np.zeros_like(threshold_range)
    specificity_array = np.zeros_like(threshold_range)
    precision_array = np.zeros_like(threshold_range)
    print("tn,fp,fn,tp,threshold,accuracy,precision,sensitivity,specificity")
    for idx in range(threshold_range.size):
        distance_threshold = threshold_range[idx]
        estimator = PatientCancerClusterEstimator(
                distance_threshold=distance_threshold,
                cancer_label=1,
                normal_label=0,
                cancer_cluster_list=cancer_cluster_list,
                normal_cluster_list=normal_cluster_list,
                feature_list=feature_list, label_name=cluster_model_name)

        # Fit the estimator to the cluster training data
        estimator.fit(df_clusters, y_true_clusters)

        # Generate measurement-wise predictions of the training data
        X_train_scaled = scaler.transform(df_train_ext[feature_list])
        df_train_scaled_ext = df_train_ext.copy()
        df_train_scaled_ext[feature_list] = X_train_scaled
        y_pred_patients = estimator.predict(df_train_scaled_ext)

        # Generate scores
        accuracy = accuracy_score(y_true_patients, y_pred_patients)
        precision = precision_score(y_true_patients, y_pred_patients)
        sensitivity = recall_score(y_true_patients, y_pred_patients)
        specificity = recall_score(
                y_true_patients, y_pred_patients, pos_label=0)


        # Generate performance counts
        tn, fp, fn, tp = confusion_matrix(y_true_patients, y_pred_patients).ravel()

        # Store scores
        sensitivity_array[idx] = sensitivity
        specificity_array[idx] = specificity
        precision_array[idx] = precision

        print("{:.2f},{:2f},{:2f},{:2f},{:.2f},{:2f},{:2f},{:2f},{:2f}".format(
                    tn, fp, fn, tp,
                    distance_threshold, accuracy, precision, sensitivity,
                    specificity))

    # Manually create ROC and precision-recall curves
    subtitle = "Cluster Centerwise Cancer Distance Model"
    tpr = sensitivity_array
    fpr = 1 - specificity_array
    x_offset = 0
    y_offset = 0.002

    # ROC Curve
    title = "ROC Curve: {}".format(subtitle)
    fig = plt.figure(title, figsize=(12,12))
    plt.step(fpr, tpr, where="post")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title(title)

    # Annotate by threshold
    for x, y, s in zip(fpr, tpr, threshold_range):
        plt.text(x+x_offset, y+y_offset, np.round(s,1))
    
    # Precision-Recall Curve
    title = "Precision-Recall Curve: {}".format(subtitle)
    fig = plt.figure(title, figsize=(12,12))
    recall_array = sensitivity_array
    plt.step(recall_array, precision_array, where="pre")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title(title)

    # Annotate by threshold
    for x, y, s in zip(recall_array, precision_array, threshold_range):
        plt.text(x+x_offset, y+y_offset, np.round(s,1))

    plt.show()

    if df_test is not None:
        # Add in patient data
        extraction = df_test.index.str.extractall(
                "CR_([A-Z]{1}).*?([0-9]+)")
        extraction_series = extraction[0] + extraction[1].str.zfill(5)
        extraction_list = extraction_series.tolist()

        assert(len(extraction_list) == df_test.shape[0])
        df_test_ext = df_test.copy()
        df_test_ext["Barcode"] = extraction_list

        df_test_ext = pd.merge(
                df_test_ext, db, left_on="Barcode", right_index=True)

        # Get cluster data
        X_test_scaled = scaler.transform(df_test_ext[feature_list])

        X_test_scaled = scaler.transform(df_test[feature_list])
        df_test_scaled = pd.DataFrame(
                data=X_test_scaled, columns=feature_list, index=df_test.index)
        df_test_scaled["Patient_ID"] = df_test_ext["Patient_ID"]

        # Make predictions on test data
        distance_threshold = threshold

        # Create estimator
        estimator = PatientCancerClusterEstimator(
                distance_threshold=distance_threshold,
                cancer_label=1,
                normal_label=0,
                cancer_cluster_list=cancer_cluster_list,
                normal_cluster_list=normal_cluster_list,
                feature_list=feature_list, label_name=cluster_model_name)

        # Fit the estimator to the cluster training data
        estimator.fit(df_clusters, y_true_clusters)

        # Generate measurement-wise predictions of the training data
        y_test_pred_patients = estimator.predict(df_test_scaled).astype(int)

        patient_list = df_test_scaled["Patient_ID"].unique()
        df_test_pred_patients = pd.DataFrame(
                data=y_test_pred_patients, columns=["prediction"], index=patient_list)

        # df_test_patients_predictions.to_csv()

        cancer_prediction_num = df_test_pred_patients.values.sum().astype(int)
        patient_num = df_test_pred_patients.shape[0]
        normal_prediction_num = (patients_num - cancer_prediction_num).astype(int)
        print("Total blind patients:", patients_num)
        print("Total blind cancer predictions:",cancer_prediction_num)
        print("Total blind normal predictions:",normal_prediction_num)

        # Save blind predictions to file
        with open(blind_predictions, "w") as outfile:
            outfile.write("cancer,healthy,total\n")
            outfile.write("{},{},{}".format(
                cancer_prediction_num,
                normal_prediction_num,
                patient_num,
                ))

def run_patient_predictions_cv(
        training_data_filepath=None, feature_list=None, cancer_cluster_list=None,
        normal_cluster_list=None, cluster_model_name=None,
        ):
    # Load training data
    df_train = pd.read_csv(training_data_filepath, index_col="Filename")

    # Set up measurement-wise true labels of the training data
    y_true_measurements = pd.Series(index=df_train.index, dtype=int)
    if cancer_cluster_list in ("", None):
        y_true_measurements[df_train[cluster_model_name].isin(normal_cluster_list)] = 0
        y_true_measurements[~df_train[cluster_model_name].isin(normal_cluster_list)] = 1
    if normal_cluster_list in ("", None):
        y_true_measurements[df_train[cluster_model_name].isin(cancer_cluster_list)] = 1
        y_true_measurements[~df_train[cluster_model_name].isin(cancer_cluster_list)] = 0
    # y_true_measurements = y_true_measurements.values

    # Generate patient-wise true labels of the training data
    # s_true_measurements = pd.Series(y_true_measurements, index=df_train["Patient_ID"], dtype=int)
    y_true_patients = df_train.groupby("Patient_ID")["Diagnosis"].max()
    y_true_patients.loc[y_true_patients == "cancer"] = 1
    y_true_patients.loc[y_true_patients == "healthy"] = 0
    y_true_patients = y_true_patients.astype(int)

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    # Loop over thresholds
    # Set the threshold range to loop over
    threshold_range = np.arange(0, 4, 0.2)
    print("threshold,accuracy,precision,sensitivity,specificity,harmonic_mean_sens_spec")
    for idx in range(5):

        # Split patients 80/20
        patients_train, patients_test, y_true_patients_train, y_true_patients_test = \
                train_test_split(
                y_true_patients.index, y_true_patients.values,
                train_size=0.8, random_state=idx)

        for jdx in range(threshold_range.size):
            # Set the threshold
            threshold = threshold_range[jdx]

            # Create the estimator
            estimator = PatientCancerClusterEstimator(
                    distance_threshold=threshold, cancer_label=1,
                    normal_label=0,
                    cancer_cluster_list=cancer_cluster_list,
                    normal_cluster_list=normal_cluster_list,
                    feature_list=feature_list, label_name=cluster_model_name)


            # Split the measurements according to the patient train/test split
            measurements_train = df_train[df_train["Patient_ID"].isin(patients_train)]
            y_true_measurements_train = y_true_measurements.loc[measurements_train.index].astype(int)
            measurements_test = df_train[df_train["Patient_ID"].isin(patients_test)]
            y_true_measurements_test = y_true_measurements.loc[measurements_test.index].astype(int)

            # Fit the estimator the training data
            estimator.fit(measurements_train, y_true_measurements_train)

            # Generate measurement-wise predictions of the training data
            y_test_pred_patients = estimator.predict(measurements_test)

            # Get y_test_true_patients
            y_test_true_patients = y_true_patients.loc[patients_test].values.astype(int)

            # Generate scores
            accuracy = accuracy_score(y_test_true_patients, y_test_pred_patients)
            # roc_auc = roc_auc_score(y_test_true_patients, y_test_pred_patients)
            precision = precision_score(y_test_true_patients, y_test_pred_patients)
            sensitivity = recall_score(y_test_true_patients, y_test_pred_patients)
            specificity = recall_score(
                    y_test_true_patients, y_test_pred_patients, pos_label=0)
            harmonic_mean_sens_spec = 2*sensitivity*specificity/(sensitivity + specificity)

            print(
                    "{:.2f},{:2f},{:2f},{:2f},{:2f},{:2f}".format(
                        threshold, accuracy, precision, sensitivity,
                        specificity, harmonic_mean_sens_spec))

def run_patient_predictions_knearest_neighbors(
        training_data_filepath=None, feature_list=None, cancer_cluster_list=None,
        normal_cluster_list=None, cluster_model_name=None,
        ):
    # Load training data
    df_train = pd.read_csv(training_data_filepath, index_col="Filename")

    X = df_train[feature_list]

    # Set up measurement-wise true labels of the training data
    y_true_measurements = pd.Series(index=df_train.index, dtype=int)
    y_true_measurements[df_train[cluster_model_name].isin(cancer_cluster_list)] = 1
    y_true_measurements[~df_train[cluster_model_name].isin(cancer_cluster_list)] = 0
    y_true_measurements = y_true_measurements.values

    # Generate patient-wise true labels of the training data
    # s_true_measurements = pd.Series(y_true_measurements, index=df_train["Patient_ID"], dtype=int)
    y_true_patients = df_train.groupby("Patient_ID")["Diagnosis"].max()
    y_true_patients.loc[y_true_patients == "cancer"] = 1
    y_true_patients.loc[y_true_patients == "healthy"] = 0
    y_true_patients = y_true_patients.astype(int)

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    # Loop over thresholds
    # Set the threshold range to loop over
    threshold_range = np.arange(0, 4, 0.2)
    print("threshold,roc_auc,accuracy,precision,sensitivity,specificity")
    for idx in range(threshold_range.size):
        # Set the threshold
        threshold = threshold_range[idx]

        # Create the estimator
        estimator = KNeighborsClassifier(n_neighbors=1)

        # Fit the estimator the training data
        estimator.fit(X, y_true_measurements)

        # Generate measurement-wise predictions of the training data
        y_pred_measurements = estimator.predict(X).astype(int)

        # Generate patient-wise predictions
        X_copy = df_train.copy()[feature_list + ["Patient_ID"]]

        X_copy["predictions"] = y_pred_measurements

        y_pred_patients = X_copy.groupby(
                    "Patient_ID")["predictions"].max().values

        # Generate scores
        accuracy = accuracy_score(y_true_patients, y_pred_patients)
        roc_auc = roc_auc_score(y_true_patients, y_pred_patients)
        precision = precision_score(y_true_patients, y_pred_patients)
        sensitivity = recall_score(y_true_patients, y_pred_patients)
        specificity = recall_score(
                y_true_patients, y_pred_patients, pos_label=0)

        print(
                "{:.2f},{:.2f},{:2f},{:2f},{:2f},{:2f}".format(
                    threshold, roc_auc, accuracy, precision, sensitivity,
                    specificity))

def run_patient_predictions_radius_neighbors(
        training_data_filepath=None, feature_list=None, cancer_cluster_list=None,
        normal_cluster_list=None, cluster_model_name=None,
        ):
    # Load training data
    df_train = pd.read_csv(training_data_filepath, index_col="Filename")

    X = df_train[feature_list]

    # Set up measurement-wise true labels of the training data
    y_true_measurements = pd.Series(index=df_train.index, dtype=int)
    y_true_measurements[df_train[cluster_model_name].isin(cancer_cluster_list)] = 1
    y_true_measurements[~df_train[cluster_model_name].isin(cancer_cluster_list)] = 0
    y_true_measurements = y_true_measurements.values

    # Generate patient-wise true labels of the training data
    # s_true_measurements = pd.Series(y_true_measurements, index=df_train["Patient_ID"], dtype=int)
    y_true_patients = df_train.groupby("Patient_ID")["Diagnosis"].max()
    y_true_patients.loc[y_true_patients == "cancer"] = 1
    y_true_patients.loc[y_true_patients == "healthy"] = 0
    y_true_patients = y_true_patients.astype(int)

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    # Loop over thresholds
    # Set the threshold range to loop over
    threshold_range = np.arange(0, 4, 0.2)
    print("threshold,roc_auc,accuracy,precision,sensitivity,specificity")
    for idx in range(threshold_range.size):
        # Set the threshold
        threshold = threshold_range[idx]

        # Create the estimator
        estimator = RadiusNeighborsClassifier(radius=threshold)

        # Fit the estimator the training data
        estimator.fit(X, y_true_measurements)

        # Generate measurement-wise predictions of the training data
        y_pred_measurements = estimator.predict(X).astype(int)

        # Generate patient-wise predictions
        X_copy = df_train.copy()[feature_list + ["Patient_ID"]]

        X_copy["predictions"] = y_pred_measurements

        y_pred_patients = X_copy.groupby(
                    "Patient_ID")["predictions"].max().values

        # Generate scores
        accuracy = accuracy_score(y_true_patients, y_pred_patients)
        roc_auc = roc_auc_score(y_true_patients, y_pred_patients)
        precision = precision_score(y_true_patients, y_pred_patients)
        sensitivity = recall_score(y_true_patients, y_pred_patients)
        specificity = recall_score(
                y_true_patients, y_pred_patients, pos_label=0)

        print(
                "{:.2f},{:.2f},{:2f},{:2f},{:2f},{:2f}".format(
                    threshold, roc_auc, accuracy, precision, sensitivity,
                    specificity))


if __name__ == '__main__':
    """
    Run cancer predictions on preprocessed data
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--training_data_filepath", type=str, default=None, required=True,
            help="The file containing training data to train estimator on.")
    parser.add_argument(
            "--data_filepath", type=str, default=None, required=False,
            help="The file containing data to perform predictions on.")
    parser.add_argument(
            "--output_path", type=str, default=None, required=False,
            help="The output path to save results in.")
    parser.add_argument(
            "--cancer_cluster_list", type=str, default="", required=False,
            help="List of cancer clusters.")
    parser.add_argument(
            "--normal_cluster_list", type=str, default="", required=False,
            help="List of normal clusters.")
    parser.add_argument(
            "--cancer_label", type=int, default=1, required=False,
            help="The cancerl label to use for saving results.")
    parser.add_argument(
            "--distance_threshold", type=float, default=None, required=False,
            help="The distance threshold to use for cancer predictions.")
    parser.add_argument(
            "--feature_list", type=str, default=None, required=False,
            help="The list of features to use.")
    parser.add_argument(
            "--divide_by", type=str, default=None, required=False,
            help="The feature to scale all other features by.")
    parser.add_argument(
            "--cluster_model_name", type=str, default=None, required=False,
            help="Name of the column containing cluster labels.")
    parser.add_argument(
            "--unsupervised_estimator_filepath", type=str, default=None, required=False,
            help="Name of the unsupervised estimator containing scaler and kmeans model.")
    parser.add_argument(
            "--patient_db", type=str, default=None, required=False,
            help="File containing patient diagnosis.")
    parser.add_argument(
            "--blind_predictions", type=str, default=None, required=False,
            help="File to save blind predictions.")

    # Collect arguments
    args = parser.parse_args()

    training_data_filepath = args.training_data_filepath
    data_filepath = args.data_filepath
    output_path = args.output_path
    cancer_cluster_list = args.cancer_cluster_list
    normal_cluster_list = args.normal_cluster_list
    cancer_label = args.cancer_label
    distance_threshold = args.distance_threshold
    feature_list = str(args.feature_list).split(",")
    cluster_model_name = args.cluster_model_name
    unsupervised_estimator_filepath = args.unsupervised_estimator_filepath
    divide_by = args.divide_by
    patient_db = args.patient_db
    blind_predictions = args.blind_predictions
    
    # Convert cancer_cluster_list csv to list of ints
    if cancer_cluster_list:
        cancer_cluster_list = cancer_cluster_list.split(",")
        cancer_cluster_list = [int(x) for x in cancer_cluster_list]
    # Convert normal_cluster_list csv to list of ints
    if normal_cluster_list:
        normal_cluster_list = normal_cluster_list.split(",")
        normal_cluster_list = [int(x) for x in normal_cluster_list]

    if cancer_cluster_list is None:
        raise ValueError("Cancer cluster list must not be empty.")

    if False:
        results = grid_search_cluster_model_on_df_file(
            training_data_filepath=training_data_filepath,
            output_path=output_path,
            cancer_cluster_list=cancer_cluster_list,
            cancer_label=cancer_label,
            distance_threshold=distance_threshold)

        print(results.cv_results_)

    run_patient_predictions_kmeans(
            training_data_filepath=training_data_filepath,
            data_filepath=data_filepath,
            cancer_cluster_list=cancer_cluster_list,
            normal_cluster_list=normal_cluster_list,
            feature_list=feature_list,
            cluster_model_name=cluster_model_name,
            unsupervised_estimator_filepath=unsupervised_estimator_filepath,
            divide_by=divide_by,
            patient_db=patient_db,
            blind_predictions=blind_predictions,
            # threshold=distance_threshold,
            )
