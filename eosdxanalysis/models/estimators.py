"""
Code for implementing custom cluster-based models
"""

import numpy as np
import pandas as pd

from scipy.stats import zscore

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.utils.estimator_checks import check_estimator

from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import euclidean_distances
from sklearn.metrics import manhattan_distances


class CancerClusterEstimator(BaseEstimator, ClassifierMixin):
    """
    Measurement-based estimator based on cancer cluster definition
    Cancer patients must have label 1.
    Healthy patients must have label 0.
    """

    def __init__(
            self, distance_threshold=0, cancer_label=1,
            distance_function="euclidean"):

        self.distance_threshold = distance_threshold
        self.cancer_label = cancer_label
        self.distance_function = distance_function

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
        distance_type = self.distance_type

        # Get distance threshold
        distance_threshold = self.distance_threshold

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Find the closest cancer clusters
        # Get the distance matrix
        if distance_function == "euclidean":
            distances = euclidean_distances(X, self.cancer_data_)
        elif distance_function == "manhattan":
            distances = manhattan_distances(X, self.cancer_data_)
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
            normal_cluster_list=None, indeterminate_label=2,
            distance_type="worst_distance", projection=False,
            normal_cluster_center=None, abnormal_cluster_center=None,
            z_threshold=3.0):
        """
        Parameters
        ----------

        distance_type : str
            Choice of ``worst_distance`` (default) or ``mean_distance`` model.

        projection : str
            Choice of ``normal``, ``abnormal``, or ``False`` (default).
        """

        self.distance_threshold = distance_threshold
        self.cancer_label = cancer_label
        self.normal_label = normal_label
        self.feature_list = feature_list
        self.label_name = label_name
        self.cancer_cluster_list = cancer_cluster_list
        self.normal_cluster_list = normal_cluster_list
        self.indeterminate_label = indeterminate_label
        self.tol = 1e-6
        self.distance_type = distance_type
        self.projection = projection
        self.normal_cluster_center = normal_cluster_center
        self.abnormal_cluster_center = abnormal_cluster_center
        self.z_threshold = z_threshold

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
        cancer_cluster_list = self.cancer_cluster_list
        normal_cluster_list = self.normal_cluster_list
        projection = self.projection

        # Calculate the distance to the closest clusters
        decisions = self.decision_function(X)

        if normal_cluster_list in (None, "") or projection == "abnormal":
            # Cancer predictions
            # If sample is close enough to any cancer cluster, predict cancer
            cancer_patient_predictions = (np.abs(decisions) <= (np.abs(distance_threshold) + tol))
            cancer_patient_predictions[cancer_patient_predictions] = cancer_label

        if cancer_cluster_list in (None, "") or projection == "normal":
            # Normal predictions
            # If sample is too far from the closest normal cluster, predict cancer
            cancer_patient_predictions = (decisions > (distance_threshold + tol))
            cancer_patient_predictions[cancer_patient_predictions] = cancer_label

        return cancer_patient_predictions.astype(int)

    def decision_function(self, X):
        """
        Returns a score to be thresholded. Higher scores have increased
        probability of being cancer.
        """

        tol=1e-6
        cancer_label = self.cancer_label
        normal_label = self.normal_label
        indeterminate_label = self.indeterminate_label
        feature_list = self.feature_list
        cancer_cluster_list = self.cancer_cluster_list
        normal_cluster_list = self.normal_cluster_list
        distance_type = self.distance_type
        projection = self.projection
        normal_cluster_center = self.normal_cluster_center
        abnormal_cluster_center = self.abnormal_cluster_center
        z_threshold = self.z_threshold

        if normal_cluster_center is not None and abnormal_cluster_center is not None:
            # Create the vector pointing from the normal cluster to the abnormal cluster
            normal_abnormal_vector = abnormal_cluster_center - normal_cluster_center
            normal_abnormal_vector = normal_abnormal_vector.reshape(-1,1)

        X = pd.DataFrame(X)

        # Check if fit has been called
        check_is_fitted(self)

        X_features = X[feature_list]

        # Input validation
        X_features = check_array(X_features)

        # Copy data
        X_copy = X.copy()

        if projection is None:
            if normal_cluster_list in (None, ""):
                # Find the closest cancer clusters
                # Get the distance matrix
                if distance_function == "euclidean":
                    cancer_distances = euclidean_distances(X_features, self.cancer_data_)
                elif distance_function == "manhattan":
                    cancer_distances = manhattan_distances(X_features, self.cancer_data_)

                closest_cancer_distances = np.min(cancer_distances, axis=1)
                X_copy["closest_cancer_distances"] = closest_cancer_distances

                if distance_type == "worst_distance":
                    # Find the minimum distances per patient
                    closest_cancer_patient_distances = X_copy.groupby(
                            "Patient_ID")["closest_cancer_distances"].min().values
                elif distance_type == "best_distance":
                    # Find the max distances per patient
                    closest_cancer_patient_distances = X_copy.groupby(
                            "Patient_ID")["closest_cancer_distances"].max().values
                elif distance_type == "mean_distance":
                    # Find the mean distances per patient
                    closest_cancer_patient_distances = X_copy.groupby(
                            "Patient_ID")["closest_cancer_distances"].mean().values
                elif distance_type == "filtered_mean_distance":
                    # Calculate z-scores of measurements on a per-patient basis
                    X_copy_filtered = X_copy.copy()
                    X_copy["z_score"] = X_copy.groupby(
                            "Patient_ID")["closest_cancer_distances"].apply(
                                    zscore).replace(np.nan, 0)
                    X_copy_filtered = X_copy[np.abs(X_copy["z_score"]) < z_threshold]

                    # Find the mean distances per patient on the filtered data
                    closest_cancer_patient_distances = X_copy_filtered.groupby(
                            "Patient_ID")["closest_cancer_distances"].mean().values

                # Take the inverse of distances, smaller distance has higher
                # probability of being cancer
                decisions = -closest_cancer_patient_distances

            if cancer_cluster_list in (None, ""):
                # Find the closest normal clusters
                # Get the distance matrix
                if distance_function == "euclidean":
                    normal_distances = euclidean_distances(X_features, self.normal_data_)
                elif distance_function == "manhattan":
                    normal_distances = manhattan_distances(X_features, self.normal_data_)

                closest_normal_distances = np.min(normal_distances, axis=1)
                X_copy["closest_normal_distances"] = closest_normal_distances

                if distance_type == "worst_distance":
                    # Find the maximum distances per patient
                    closest_normal_patient_distances = X_copy.groupby(
                            "Patient_ID")["closest_normal_distances"].max().values
                elif distance_type == "best_distance":
                    # Find the minimum distances per patient
                    closest_normal_patient_distances = X_copy.groupby(
                            "Patient_ID")["closest_normal_distances"].min().values
                elif distance_type == "mean_distance":
                    # Find the mean distances per patient
                    closest_normal_patient_distances = X_copy.groupby(
                            "Patient_ID")["closest_normal_distances"].mean().values
                elif distance_type == "filtered_mean_distance":
                    # Calculate z-scores of measurements on a per-patient basis
                    X_copy_filtered = X_copy.copy()
                    X_copy["z_score"] = X_copy.groupby(
                            "Patient_ID")["closest_normal_distances"].apply(
                                    zscore).replace(np.nan, 0)
                    X_copy_filtered = X_copy[np.abs(X_copy["z_score"]) < z_threshold]

                    # Find the mean distances per patient on the filtered data
                    closest_normal_patient_distances = X_copy_filtered.groupby(
                            "Patient_ID")["closest_normal_distances"].mean().values

                # Return distances from normal, the larger the distance, the higher
                # probability of being cancer
                decisions = closest_normal_patient_distances
        elif projection == "normal":
            # Project distance from normal cluster to normal-abnormal axis
            # Find the closest normal clusters
            # Get the distance matrix

            # Project X_features onto normal_abnormal_vector
            X_projected_features = \
                    (np.dot(X_features, normal_abnormal_vector) @ normal_abnormal_vector.T) \
                    / np.dot(normal_abnormal_vector.T, normal_abnormal_vector)

            # Calculated distances of projected measurements to normal
            if distance_function == "euclidean":
                projected_normal_distances = euclidean_distances(
                        X_projected_features, self.normal_data_)
            elif distance_function == "manhattan":
                projected_normal_distances = manhattan_distances(
                        X_projected_features, self.normal_data_)
            closest_projected_normal_distances = np.min(projected_normal_distances, axis=1)

            X_copy["closest_projected_normal_distances"] = closest_projected_normal_distances

            if distance_type == "worst_distance":
                # Find the maximum distances per patient
                closest_projected_normal_patient_distances = X_copy.groupby(
                        "Patient_ID")["closest_projected_normal_distances"].max(
                                ).values
            elif distance_type == "best_distance":
                # Find the minimum distances per patient
                closest_projected_normal_patient_distances = X_copy.groupby(
                        "Patient_ID")["closest_projected_normal_distances"].min(
                                ).values
            elif distance_type == "mean_distance":
                # Find the mean distances per patient
                closest_projected_normal_patient_distances = X_copy.groupby(
                        "Patient_ID")["closest_projected_normal_distances"].mean(
                                ).values
            elif distance_type == "filtered_mean_distance":
                # Calculate z-scores of measurements on a per-patient basis
                X_copy_filtered = X_copy.copy()
                X_copy["z_score"] = X_copy.groupby(
                        "Patient_ID")["closest_projected_normal_distances"].apply(
                                zscore).replace(np.nan, 0)
                X_copy_filtered = X_copy[np.abs(X_copy["z_score"]) < z_threshold]

                # Find the mean distances per patient on the filtered data
                closest_projected_normal_patient_distances = X_copy_filtered.groupby(
                        "Patient_ID")["closest_projected_normal_distances"].mean().values

            # Return projected distances from normal, the larger the distance,
            # the higher probability of being cancer
            decisions = closest_projected_normal_patient_distances

        elif projection == "abnormal":
            # Project distance from abnormal cluster to normal-abnormal axis
            # Find the closest cancer clusters
            # Get the distance matrix

            # Project X_features onto normal_abnormal_vector
            X_projected_features = \
                    (np.dot(X_features, normal_abnormal_vector) @ normal_abnormal_vector.T) \
                    / np.dot(normal_abnormal_vector.T, normal_abnormal_vector)

            if distance_function == "euclidean":
                projected_cancer_distances = euclidean_distances(
                        X_projected_features, self.cancer_data_)
            elif distance_function == "manhattan":
                projected_cancer_distances = manhattan_distances(
                        X_projected_features, self.cancer_data_)
            closest_projected_cancer_distances = np.min(projected_cancer_distances, axis=1)

            X_copy["closest_projected_cancer_distances"] = closest_projected_cancer_distances

            if distance_type == "worst_distance":
                # Find the minimum distances per patient
                closest_projected_cancer_patient_distances = X_copy.groupby(
                        "Patient_ID")["closest_projected_cancer_distances"].min(
                                ).values
            elif distance_type == "best_distance":
                # Find the maximum distances per patient
                closest_projected_cancer_patient_distances = X_copy.groupby(
                        "Patient_ID")["closest_projected_cancer_distances"].max(
                                ).values
            elif distance_type == "mean_distance":
                # Find the mean distances per patient
                closest_projected_cancer_patient_distances = X_copy.groupby(
                        "Patient_ID")["closest_projected_cancer_distances"].mean(
                                ).values
            elif distance_type == "filtered_mean_distance":
                # Calculate z-scores of measurements on a per-patient basis
                X_copy_filtered = X_copy.copy()
                X_copy["z_score"] = X_copy.groupby(
                        "Patient_ID")["closest_projected_cancer_distances"].apply(
                                zscore).replace(np.nan, 0)
                X_copy_filtered = X_copy[np.abs(X_copy["z_score"]) < z_threshold]

                # Find the mean distances per patient on the filtered data
                closest_projected_cancer_patient_distances = X_copy_filtered.groupby(
                        "Patient_ID")["closest_projected_cancer_distances"].mean().values

            # Take the inverse of distances, smaller distance has higher
            # probability of being cancer
            decisions = -closest_projected_cancer_patient_distances

        return decisions
