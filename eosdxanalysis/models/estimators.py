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
    def __init__(self, distance_threshold=0, cancer_label=1,
            feature_list=None, label_name="kmeans_20"):

        self.distance_threshold = distance_threshold
        self.cancer_label = cancer_label
        self.feature_list = feature_list
        self.label_name = label_name

    def fit(self, X, y):
        """
        X must be a dataframe with 7 feature columns, one Patient_ID column
        """
        distance_threshold = self.distance_threshold
        cancer_label = self.cancer_label
        feature_list = self.feature_list
        label_name = self.label_name

        if feature_list is None:
            raise ValueError("Feature list must be specified.")

        X_patient_ids = X["Patient_ID"]
        X_features = X[feature_list]

        # Check that X and y have correct shape
        X_features, y = check_X_y(X_features, y)
        labels = unique_labels(y)
        # Check if label types are numbers
        if not np.issubdtype(labels.dtype, np.number):
            raise ValueError("Labels must be digits.")

        # Store the classes seen during fit
        self.classes_ = labels

        self.X_ = X_features
        self.y_ = y

        # Store the data belonging to cancer clusters
        cancer_data = X_features[y == cancer_label]
        self.cancer_data_ = cancer_data

        # Return the classifier
        return self

    def predict(self, X):
        """
        X must be a dataframe with 7 feature columns, one Patient_ID column
        """

        distance_threshold = self.distance_threshold
        cancer_label = self.cancer_label

        # Calculate the distance to the closest clusters
        closest_distances = self.decision_function(X)

        # If sample is close enough to any cancer cluster, predict cancer
        cancer_predictions = closest_distances <= distance_threshold
        cancer_predictions[cancer_predictions] = cancer_label

        return cancer_predictions.astype(int)

    def decision_function(self, X):
        """
        X must be a dataframe with 7 feature columns, one Patient_ID column
        """

        X = pd.DataFrame(X)

        cancer_label = self.cancer_label
        distance_threshold = self.distance_threshold
        feature_list = self.feature_list

        # Check if fit has been called
        check_is_fitted(self)

        X_patient_ids = X["Patient_ID"]
        X_features = X[feature_list]

        # Input validation
        X_features = check_array(X_features)

        # Find the closest cancer clusters per patient
        # Get the distance matrix
        distances = euclidean_distances(X_features, self.cancer_data_)
        # Find the indices of the closest cancer data
        closest_indices = np.argmin(distances, axis=1)
        # Find the distances to the closest cancer data
        closest_distances = np.min(distances, axis=1)
        X_copy = X.copy()
        X_copy["closest_distances"] = closest_distances
        # Find the minimum distances per patient
        closest_patient_distances = X_copy.groupby(
                "Patient_ID")["closest_distances"].transform('min').values

        return closest_patient_distances
