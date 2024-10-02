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
class Limits:
    q_min_saxs: float
    q_max_saxs: float
    q_min_waxs: float
    q_max_waxs: float


@dataclass
class Rule:
    q_value: float
    lower: float = None
    upper: float = None
    cancer_diagnosis: bool = None
