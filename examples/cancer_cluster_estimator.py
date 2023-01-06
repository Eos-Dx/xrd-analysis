"""
Code to run cancer predictions using cancer cluster estimator
"""
import numpy as np
import pandas as pd
import argparse

from joblib import load

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
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
        training_data_filepath=None, feature_list=None, cancer_cluster_list=None,
        normal_cluster_list=None, cluster_model_name=None,
        unsupervised_estimator_filepath=None,
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
    df_train["predictions"] = y_true_measurements.astype(int)

    # Generate patient-wise true labels of the training data
    # s_true_measurements = pd.Series(y_true_measurements, index=df_train["Patient_ID"], dtype=int)
    y_true_patients = df_train.groupby("Patient_ID")["Diagnosis"].max()
    y_true_patients.loc[y_true_patients == "cancer"] = 1
    y_true_patients.loc[y_true_patients == "healthy"] = 0
    y_true_patients = y_true_patients.astype(int)

    # Transform kmeans cluster centers
    unsupervised_estimator = load(unsupervised_estimator_filepath)
    kmeans_model = unsupervised_estimator["kmeans"]
    clusters = kmeans_model.cluster_centers_

    # Loop over thresholds
    # Set the threshold range to loop over
    # threshold_range = np.arange(0, 4, 0.2)
    # Create the estimator
    print("accuracy,precision,sensitivity,specificity")

    # Make predictions
    y_pred_patients = df_train.groupby("Patient_ID")["predictions"].max().values

    # Generate scores
    accuracy = accuracy_score(y_true_patients, y_pred_patients)
    precision = precision_score(y_true_patients, y_pred_patients)
    sensitivity = recall_score(y_true_patients, y_pred_patients)
    specificity = recall_score(
            y_true_patients, y_pred_patients, pos_label=0)

    print("{:2f},{:2f},{:2f},{:2f}".format(
                accuracy, precision, sensitivity, specificity))

def run_patient_predictions_pointwise(
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

    # Loop over thresholds
    # Set the threshold range to loop over
    # threshold_range = np.arange(0, 4, 0.2)
    # Create the estimator
    threshold_range = np.arange(0, 4, 0.2)
    print("threshold,accuracy,precision,sensitivity,specificity")
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
        estimator.fit(df_train, y_true_measurements)

        # Generate measurement-wise predictions of the training data
        y_pred_patients = estimator.predict(df_train)

        # Generate scores
        accuracy = accuracy_score(y_true_patients, y_pred_patients)
        precision = precision_score(y_true_patients, y_pred_patients)
        sensitivity = recall_score(y_true_patients, y_pred_patients)
        specificity = recall_score(
                y_true_patients, y_pred_patients, pos_label=0)

        print("{:.2f},{:2f},{:2f},{:2f},{:2f}".format(
                    distance_threshold, accuracy, precision, sensitivity,
                    specificity))

    # ROC analysis
    distance_threshold = 1
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
    y_score_patients = estimator.decision_function(df_train)

    import matplotlib.pyplot as plt
    RocCurveDisplay.from_predictions(y_true_patients, y_score_patients)
    plt.show()
    PrecisionRecallDisplay.from_predictions(y_true_patients, y_score_patients)
    plt.show()

def run_patient_predictions_clusterwise(
        training_data_filepath=None, feature_list=None, cancer_cluster_list=None,
        normal_cluster_list=None, cluster_model_name=None,
        unsupervised_estimator_filepath=None,
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
    # threshold_range = np.arange(0, 4, 0.2)
    # Create the estimator
    threshold_range = np.arange(0, 5, 0.1)
    print("threshold,accuracy,precision,sensitivity,specificity")
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
        y_pred_patients = estimator.predict(df_train)

        # Generate scores
        accuracy = accuracy_score(y_true_patients, y_pred_patients)
        precision = precision_score(y_true_patients, y_pred_patients)
        sensitivity = recall_score(y_true_patients, y_pred_patients)
        specificity = recall_score(
                y_true_patients, y_pred_patients, pos_label=0)

        print("{:.2f},{:2f},{:2f},{:2f},{:2f}".format(
                    distance_threshold, accuracy, precision, sensitivity,
                    specificity))

    # ROC analysis
    distance_threshold = 1
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
    y_score_patients = estimator.decision_function(df_train)

    # Compute threshold corresponding best point in ROC curve
    fpr, tpr, roc_thresholds = roc_curve(
            y_true_patients, y_score_patients, pos_label=1)

    # Compute threshold corresponding to best point in precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(
            y_true_patients, y_score_patients, pos_label=1)

    import matplotlib.pyplot as plt
    RocCurveDisplay.from_predictions(y_true_patients, y_score_patients)
    plt.show()
    PrecisionRecallDisplay.from_predictions(y_true_patients, y_score_patients)
    plt.show()

    import ipdb
    ipdb.set_trace()

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
            "--distance_threshold", type=float, default=0.5, required=False,
            help="The distance threshold to use for cancer predictions.")
    parser.add_argument(
            "--feature_list", type=str, default=None, required=False,
            help="The list of features to use.")
    parser.add_argument(
            "--cluster_model_name", type=str, default=None, required=False,
            help="Name of the column containing cluster labels.")
    parser.add_argument(
            "--unsupervised_estimator_filepath", type=str, default=None, required=False,
            help="Name of the unsupervised estimator containing scaler and kmeans model.")

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
            cancer_cluster_list=cancer_cluster_list,
            normal_cluster_list=normal_cluster_list,
            feature_list=feature_list,
            cluster_model_name=cluster_model_name,
            unsupervised_estimator_filepath=unsupervised_estimator_filepath,
            )
