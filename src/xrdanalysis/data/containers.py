import numpy as np
import pandas as pd

from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, roc_curve,
                             RocCurveDisplay, roc_auc_score)

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from dataclasses import dataclass
from typing import Dict, Union, Callable


SCALED_DATA = 'radial_profile_data_norm_scaled'


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
class MLCluster:
    df: pd.DataFrame
    q_range: np.ndarray
    q_cluster: int = None
    normalizer: Normalizer = None
    std: StandardScaler = None
    model: Union[XGBClassifier, RandomForestClassifier] = None
    accuracy: float = 0
    roc_auc: float = 0
    X_test: np.ndarray = None
    y_test: np.ndarray = None
    idx_test: np.array = None
    idx_train: np.array = None
    train_func: Callable = None
    important_features: np.array = None
    model_reduced: Union[XGBClassifier, RandomForestClassifier] = None

    def predict_proba(self, reduced=False):
        transformed_data = np.vstack(self.df[SCALED_DATA].values)
        if reduced:
            model = self.model_reduced
            transformed_data = transformed_data[:, self.important_features]
        else:
            model = self.model
        res = model.predict_proba(transformed_data)[:, 1]
        self.df['prediction_proba'] = res

    def predict(self, reduced=False):
        transformed_data = np.vstack(self.df[SCALED_DATA].values)
        if reduced:
            model = self.model_reduced
            transformed_data = transformed_data[:, self.important_features]
        else:
            model = self.model
        res = model.predict(transformed_data)
        self.df['prediction'] = res

    def calc_accuracy(self, reduced=False):
        self.predict(reduced)
        y_data = self.df["cancer_diagnosis"].astype(int).values
        accuracy = accuracy_score(y_data, self.df['prediction'].astype(int).values)
        self.accuracy = round(accuracy, 3)

    def calc_roc_auc(self, reduced=False):
        self.predict_proba(reduced)
        y_data = self.df["cancer_diagnosis"].astype(int).values
        roc_auc = roc_auc_score(y_data, self.df['prediction_proba'].astype(float).values)
        self.roc_auc = round(roc_auc, 3)

    def model_train(self, **kwargs):
        self.train_func(self, **kwargs)

    def set_df(self, df):
        self.df = df

    @property
    def q_max_min(self):
        return self.q_range[-1]

    @property
    def q_min_max(self):
        return self.q_range[0]


@dataclass
class MLClusterContainer:
    model_name: str
    clusters: Dict[int, MLCluster]
