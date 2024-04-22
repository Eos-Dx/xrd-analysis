from dataclasses import dataclass
from typing import Callable, Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import Normalizer, StandardScaler
from xgboost import XGBClassifier

SCALED_DATA = "interpolated_radial_profile_data_norm_scaled"


@dataclass
class MLClusterBlind:
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
class MLDataCluster:
    df: pd.DataFrame
    q_cluster: int  # cluster number
    normalizer: Normalizer = None
    scaler: StandardScaler = None
    model: Union[XGBClassifier, RandomForestClassifier] = None
    X_test: np.ndarray = None
    y_test: np.ndarray = None
    train_func: Callable = None
    shap_values: np.array = None  # SHAP values

    def accuracy(self):
        """
        return NamedTuple of values such as:
        accurac, roc_auc, etc.
        """
        # TODO: make it
        pass

    def predict_proba(self):
        transformed_data = np.vstack(self.df[SCALED_DATA].values)
        res = self.model.predict_proba(transformed_data)[:, 1]
        self.df["prediction_proba"] = res

    def predict(self):
        transformed_data = np.vstack(self.df[SCALED_DATA].values)
        res = self.model.predict(transformed_data)
        self.df["prediction"] = res

    def calc_accuracy(self):
        self.predict()
        y_data = self.df["cancer_diagnosis"].astype(int).values
        accuracy = accuracy_score(
            y_data, self.df["prediction"].astype(int).values
        )
        return round(accuracy, 3)

    def calc_roc_auc(self):
        self.predict_proba()
        y_data = self.df["cancer_diagnosis"].astype(int).values
        roc_auc = roc_auc_score(
            y_data, self.df["prediction_proba"].astype(float).values
        )
        return round(roc_auc, 3)

    def model_train(
        self,
        learning_rate=0.01,
        n_estimators=100,
        max_depth=10,
        random_state=32,
    ):
        self.train_func(
            self,
            learning_rate=0.01,
            n_estimators=100,
            max_depth=10,
            random_state=32,
            split=0.3,
        )

    def set_df(self, df):
        self.df = df

    @property
    def q_max_min(self):
        return self.q_range[-1]

    @property
    def q_min_max(self):
        return self.q_range[0]

    @property
    def q_range(self):
        """
        return q_range for this cluster as np.array
        """
        # TODO: make it work
        pass


@dataclass
class MLDataClusterContainer:
    model_name: str
    clusters: Dict[int, MLDataCluster]
