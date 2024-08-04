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
    Class to store information about an ML cluster for blind data.

    Args:
        q_range (np.ndarray): The q-range of the cluster.
        normalizer (Normalizer): The normalizer used for the data.
        std (StandardScaler): The standard scaler used for the data.
        model (Union[XGBClassifier, RandomForestClassifier]): The machine
            learning model.
        accuracy (float): The accuracy of the model.
        roc_auc (float): The ROC AUC score of the model.
        important_features (np.array, optional): Array of important features.
            Defaults to None.
        model_reduced (Union[XGBClassifier, RandomForestClassifier], optional):
            The reduced machine learning model. Defaults to None.
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
    Container class to hold multiple MLClusterBlind instances.

    Args:
        model_name (str): The name of the model.
        wavelength (int): The wavelength used in the analysis.
        pixel_size (int): The pixel size used in the analysis.
        faulty_pixel_array (np.array): Array of faulty pixel coordinates.
        fill_method (str): The method used for filling missing data.
        filter_size (int): The size of the filter used.
        n_clusters (int): The number of clusters.
        z_score_threshold (float): The threshold for Z-score based
            outlier removal.
        direction (str): The direction of outlier removal.
        clusters (Dict[int, MLClusterBlind]): Dictionary of MLClusterBlind
            instances.
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
    Class to store information about an ML cluster.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        q_range (np.ndarray): The q-range of the cluster.
        q_cluster (int): The cluster label.
        normalizer (Normalizer, optional): The normalizer
            used for the data. Defaults to None.
        std (StandardScaler, optional): The standard scaler used
            for the data. Defaults to None.
        model (Union[XGBClassifier, RandomForestClassifier], optional):
            The machine learning model. Defaults to None.
        accuracy (float, optional): The accuracy of the model. Defaults to 0.
        roc_auc (float, optional): The ROC AUC score of the model.
            Defaults to 0.
        X_test (np.ndarray, optional): Test features. Defaults to None.
        y_test (np.ndarray, optional): Test labels. Defaults to None.
        train_func (Callable, optional): Function used to train the model.
            Defaults to None.
        important_features (np.array, optional): Array of important features.
            Defaults to None.
        model_reduced (Union[XGBClassifier, RandomForestClassifier], optional):
            The reduced machine learning model. Defaults to None.
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
        Predict the probability of the positive class for each sample.

        Args:
            reduced (bool, optional): Whether to use the reduced model.
                Defaults to False.
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
        Predict the class for each sample.

        Args:
            reduced (bool, optional): Whether to use
                the reduced model. Defaults to False.
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
        Calculate the accuracy of the model.

        Args:
            reduced (bool, optional): Whether to use
                the reduced model. Defaults to False.
        """
        self.predict(reduced)
        y_data = self.df["cancer_diagnosis"].astype(int).values
        accuracy = accuracy_score(
            y_data, self.df["prediction"].astype(int).values
        )
        self.accuracy = round(accuracy, 3)

    def calc_roc_auc(self, reduced=False):
        """
        Calculate the ROC AUC score of the model.

        Args:
            reduced (bool, optional): Whether to use
                the reduced model. Defaults to False.
        """
        self.predict_proba(reduced)
        y_data = self.df["cancer_diagnosis"].astype(int).values
        roc_auc = roc_auc_score(
            y_data, self.df["prediction_proba"].astype(float).values
        )
        self.roc_auc = round(roc_auc, 3)

    def model_train(self, **kwargs):
        """
        Train the model using the provided training function.

        Args:
            **kwargs: Additional arguments for the training function.
        """
        self.train_func(self, **kwargs)

    def set_df(self, df):
        """
        Set the DataFrame for the cluster.

        Args:
            df (pd.DataFrame): The DataFrame to set.
        """
        self.df = df

    @property
    def q_max_min(self):
        """
        Get the maximum q value.

        Returns:
            float: The maximum q value.
        """
        return self.q_range[-1]

    @property
    def q_min_max(self):
        """
        Get the minimum q value.

        Returns:
            float: The minimum q value.
        """
        return self.q_range[0]


@dataclass
class MLClusterContainer:
    """
    Container class to hold multiple MLCluster instances.

    Args:
        model_name (str): The name of the model.
        clusters (Dict[int, MLCluster]): Dictionary
            of MLCluster instances.
    """

    model_name: str
    clusters: Dict[int, MLCluster]


@dataclass
class ModelScale:
    norm: bool = True
    normt: str = 'l1'


@dataclass
class Limits:
    q_min_saxs: float
    q_max_saxs: float
    q_min_waxs: float
    q_max_waxs: float
