"""
Code for implementing custom cluster-based models
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.utils.estimator_checks import check_estimator

from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import euclidean_distances


class CancerClusterEstimator(BaseEstimator, ClassifierMixin):
    """
    Measurement-based estimator based on cancer cluster definition
    Cancer patients must have label 1.
    Healthy patients must have label 0.
    """

    def __init__(self, distance_threshold=0, cancer_label=1):

        self.distance_threshold = distance_threshold
        self.cancer_label = cancer_label

    def fit(self, X, y):

        cancer_label = self.cancer_label

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        labels = unique_labels(y)
        # Check if label types are numbers
        if not np.issubdtype(labels.dtype, np.number):
            raise ValueError("Labels must be digits.")

        # Store the classes seen during fit
        self.classes_ = labels

        self.X_ = X
        self.y_ = y

        # TODO: Insert code to train the model
        # Store the data belonging to cancer clusters
        # We could use cancer cluster centers
        cancer_data = X[y == cancer_label]
        self.cancer_data_ = cancer_data

        # Return the classifier
        return self

    def predict(self, X):

        distance_threshold = self.distance_threshold
        cancer_label = self.cancer_label

        # Calculate the distance to the closest clusters
        closest_distances = self.decision_function(X)

        # If sample is close enough to any cancer cluster, predict cancer
        cancer_predictions = closest_distances <= distance_threshold
        cancer_predictions[cancer_predictions] = cancer_label

        return cancer_predictions.astype(int)

    def decision_function(self, X):

        cancer_label = self.cancer_label

        # Get distance threshold
        distance_threshold = self.distance_threshold

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Find the closest cancer clusters
        # Get the distance matrix
        distances = euclidean_distances(X, self.cancer_data_)
        # Find the indices of the closest cancer data
        closest_indices = np.argmin(distances, axis=1)
        # Find the distances to the closest cancer data
        closest_distances = np.min(distances, axis=1)

        return closest_distances


class PatientCancerClusterEstimator(BaseEstimator, ClassifierMixin):
    """
    Patient-based estimator based on cancer cluster definition
    Cancer clusters must have label 1.
    All other clusters must have label 0.
    Uses the CancerClusterEstimator as a subestimator.
    """
    def __init__(self, distance_threshold=0, cancer_label=1, normal_label=0,
            feature_list=None, label_name=None, cancer_cluster_list=None,
            normal_cluster_list=None, indeterminate_label=2):

        self.distance_threshold = distance_threshold
        self.cancer_label = cancer_label
        self.normal_label = normal_label
        self.feature_list = feature_list
        self.label_name = label_name
        self.cancer_cluster_list = cancer_cluster_list
        self.normal_cluster_list = normal_cluster_list
        self.indeterminate_label = indeterminate_label
        self.tol = 1e-6

    def fit(self, X, y):
        """
        X must be a dataframe with 7 feature columns, one Patient_ID column
        y are the patient diagnoses
        """
        distance_threshold = self.distance_threshold
        cancer_label = self.cancer_label
        feature_list = self.feature_list
        label_name = self.label_name
        cancer_cluster_list = self.cancer_cluster_list
        normal_cluster_list = self.normal_cluster_list

        # Check if feature_list is empty
        if feature_list in (None, ""):
            raise ValueError("Feature list must be specified.")

        X_features = X[feature_list]

        # Check that X and y have correct shape
        X_features, y = check_X_y(X_features, y)
        labels = unique_labels(y)
        # Check if label types are numbers
        if not np.issubdtype(labels.dtype, np.number):
            raise ValueError("Labels must be digits.")

        # Ensure cancer_cluster_list and normal_cluster_list are not both empty
        if cancer_cluster_list in ([], None, "") and normal_cluster_list in ([], None, ""):
            raise ValueError("Must provide cancer cluster list, normal cluster list, or both")

        # Store the classes seen during fit
        self.classes_ = labels

        self.X_ = X_features
        self.y_ = y

        # Get values from cancer clusters
        if cancer_cluster_list in ([], None, ""):
            y_cancer_cluster = np.zeros_like(X[label_name], dtype=bool)
        else:
            y_cancer_cluster = X[label_name].isin(cancer_cluster_list)

        # Get values from normal clusters
        if normal_cluster_list in ([], None, ""):
            y_normal_cluster = np.zeros_like(X[label_name], dtype=bool)
        else:
            y_normal_cluster = X[label_name].isin(normal_cluster_list)

        # Store the data belonging to cancer clusters
        cancer_data_ = X_features[y_cancer_cluster]
        self.cancer_data_ = cancer_data_
        # Store the data belonging to normal clusters
        normal_data_ = X_features[y_normal_cluster]
        self.normal_data_ = normal_data_

        # Return the classifier
        return self

    def predict(self, X):

        distance_threshold = self.distance_threshold
        cancer_label = self.cancer_label
        tol = self.tol

        # Calculate the distance to the closest clusters
        decisions = self.decision_function(X)

        # Cancer predictions
        # If sample is close enough to any cancer cluster, predict cancer
        cancer_patient_predictions = (decisions - distance_threshold) <= tol
        cancer_patient_predictions[cancer_patient_predictions] = cancer_label

        # Normal predictions

        return cancer_patient_predictions.astype(int)

    def decision_function(self, X):

        tol=1e-6
        distance_threshold = self.distance_threshold
        cancer_label = self.cancer_label
        normal_label = self.normal_label
        indeterminate_label = self.indeterminate_label
        feature_list = self.feature_list
        distance_threshold = self.distance_threshold

        X = pd.DataFrame(X)

        # Check if fit has been called
        check_is_fitted(self)

        X_features = X[feature_list]

        # Input validation
        X_features = check_array(X_features)

        # Copy data
        X_copy = X.copy()

        # Find the closest cancer clusters
        # Get the distance matrix
        cancer_distances = euclidean_distances(X_features, self.cancer_data_)
        closest_cancer_distances = np.min(cancer_distances, axis=1)
        X_copy["closest_cancer_distances"] = closest_cancer_distances
        # Find the minimum distances per patient
        closest_cancer_patient_distances = X_copy.groupby(
                "Patient_ID")["closest_cancer_distances"].min().values

        # For normal, find the closest normal clusters for each measurement
        decisions = closest_cancer_patient_distances

        return decisions
