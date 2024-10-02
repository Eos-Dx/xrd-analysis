from dataclasses import dataclass

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
