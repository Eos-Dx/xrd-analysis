from dataclasses import dataclass
from typing import Callable, Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import Normalizer, StandardScaler
from xgboost import XGBClassifier

SCALED_DATA = "radial_profile_data_norm_scaled"


@dataclass
class MLClusterBlind:
    """
    Stores information about a machine learning cluster for blind data.

    :param q_range: The q-range of the cluster.
    :type q_range: np.ndarray
    :param normalizer: The normalizer used for the data.
    :type normalizer: Normalizer
    :param std: The standard scaler used for the data.
    :type std: StandardScaler
    :param model: The machine learning model.
    :type model: Union[XGBClassifier, RandomForestClassifier]
    :param accuracy: The accuracy of the model.
    :type accuracy: float
    :param roc_auc: The ROC AUC score of the model.
    :type roc_auc: float
    :param important_features: Array of important features, defaults to None.
    :type important_features: np.ndarray, optional
    :param model_reduced: The reduced machine learning model, defaults to None.
    :type model_reduced: Union[XGBClassifier, RandomForestClassifier], optional
    """

    q_range: np.ndarray
    normalizer: Normalizer
    std: StandardScaler
    model: Union[XGBClassifier, RandomForestClassifier]
    accuracy: float
    roc_auc: float
    important_features: np.array = None
    model_reduced: Union[XGBClassifier, RandomForestClassifier] = None


@dataclass
class MLBlindContainer:
    """
    Container for holding multiple `MLClusterBlind` instances.

    :param model_name: The name of the model.
    :type model_name: str
    :param wavelength: The wavelength used in the analysis.
    :type wavelength: int
    :param pixel_size: The pixel size used in the analysis.
    :type pixel_size: int
    :param faulty_pixel_array: Array of faulty pixel coordinates.
    :type faulty_pixel_array: np.ndarray
    :param fill_method: The method used for filling missing data.
    :type fill_method: str
    :param filter_size: The size of the filter used.
    :type filter_size: int
    :param n_clusters: The number of clusters.
    :type n_clusters: int
    :param z_score_threshold: The threshold for Z-score based outlier removal.
    :type z_score_threshold: float
    :param direction: The direction of outlier removal.
    :type direction: str
    :param clusters: Dictionary of `MLClusterBlind` instances.
    :type clusters: Dict[int, MLClusterBlind]
    """

    model_name: str
    wavelength: int
    pixel_size: int
    faulty_pixel_array: np.array
    fill_method: str
    filter_size: int
    n_clusters: int
    z_score_threshold: float
    direction: str
    clusters: Dict[int, MLClusterBlind]


@dataclass
class MLCluster:
    """
    Stores information about a machine learning cluster.

    :param df: DataFrame containing the data.
    :type df: pd.DataFrame
    :param q_range: The q-range of the cluster.
    :type q_range: np.ndarray
    :param q_cluster: The cluster label.
    :type q_cluster: int
    :param normalizer: The normalizer used for the data, defaults to None.
    :type normalizer: Normalizer, optional
    :param std: The standard scaler used for the data, defaults to None.
    :type std: StandardScaler, optional
    :param model: The machine learning model, defaults to None.
    :type model: Union[XGBClassifier, RandomForestClassifier], optional
    :param accuracy: The accuracy of the model, defaults to 0.
    :type accuracy: float, optional
    :param roc_auc: The ROC AUC score of the model, defaults to 0.
    :type roc_auc: float, optional
    :param X_test: Test features, defaults to None.
    :type X_test: np.ndarray, optional
    :param y_test: Test labels, defaults to None.
    :type y_test: np.ndarray, optional
    :param train_func: Function used to train the model, defaults to None.
    :type train_func: Callable, optional
    :param important_features: Array of important features, defaults to None.
    :type important_features: np.ndarray, optional
    :param model_reduced: The reduced machine learning model, defaults to None.
    :type model_reduced: Union[XGBClassifier, RandomForestClassifier], optional
    """

    df: pd.DataFrame
    q_range: np.ndarray
    q_cluster: int
    normalizer: Normalizer = None
    std: StandardScaler = None
    model: Union[XGBClassifier, RandomForestClassifier] = None
    accuracy: float = 0
    roc_auc: float = 0
    X_test: np.ndarray = None
    y_test: np.ndarray = None
    train_func: Callable = None
    important_features: np.array = None
    model_reduced: Union[XGBClassifier, RandomForestClassifier] = None

    def predict_proba(self, reduced=False):
        """
        Predicts the probability of the positive class for each sample.

        :param reduced: Whether to use the reduced model, defaults to False.
        :type reduced: bool, optional
        """
        transformed_data = np.vstack(self.df[SCALED_DATA].values)
        if reduced:
            model = self.model_reduced
            transformed_data = transformed_data[:, self.important_features]
        else:
            model = self.model
        res = model.predict_proba(transformed_data)[:, 1]
        self.df["prediction_proba"] = res

    def predict(self, reduced=False):
        """
        Predicts the class for each sample.

        :param reduced: Whether to use the reduced model, defaults to False.
        :type reduced: bool, optional
        """
        transformed_data = np.vstack(self.df[SCALED_DATA].values)
        if reduced:
            model = self.model_reduced
            transformed_data = transformed_data[:, self.important_features]
        else:
            model = self.model
        res = model.predict(transformed_data)
        self.df["prediction"] = res

    def calc_accuracy(self, reduced=False):
        """
        Calculates the accuracy of the model.

        :param reduced: Whether to use the reduced model, defaults to False.
        :type reduced: bool, optional
        """
        self.predict(reduced)
        y_data = self.df["cancer_diagnosis"].astype(int).values
        accuracy = accuracy_score(
            y_data, self.df["prediction"].astype(int).values
        )
        self.accuracy = round(accuracy, 3)

    def calc_roc_auc(self, reduced=False):
        """
        Calculates the ROC AUC score of the model.

        :param reduced: Whether to use the reduced model, defaults to False.
        :type reduced: bool, optional
        """
        self.predict_proba(reduced)
        y_data = self.df["cancer_diagnosis"].astype(int).values
        roc_auc = roc_auc_score(
            y_data, self.df["prediction_proba"].astype(float).values
        )
        self.roc_auc = round(roc_auc, 3)

    def model_train(self, **kwargs):
        r"""
        Trains the model using the provided training function.

        :param \**kwargs: Additional arguments for the training function.
        """
        self.train_func(self, **kwargs)

    def set_df(self, df):
        """
        Sets the DataFrame for the cluster.

        :param df: The DataFrame to set.
        :type df: pd.DataFrame
        """
        self.df = df

    @property
    def q_max_min(self):
        """
        Gets the maximum q value.

        :returns: The maximum q value.
        :rtype: float
        """
        return self.q_range[-1]

    @property
    def q_min_max(self):
        """
        Gets the minimum q value.

        :returns: The minimum q value.
        :rtype: float
        """
        return self.q_range[0]


@dataclass
class MLClusterContainer:
    """
    Container for holding multiple `MLCluster` instances.

    :param model_name: The name of the model.
    :type model_name: str
    :param clusters: Dictionary of `MLCluster` instances.
    :type clusters: Dict[int, MLCluster]
    """

    model_name: str
    clusters: Dict[int, MLCluster]
