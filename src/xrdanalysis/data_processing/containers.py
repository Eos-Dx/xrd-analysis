from dataclasses import dataclass
from typing import Any
from enum import Enum

SCALED_DATA = "radial_profile_data_norm_scaled"


@dataclass
class Limits:
    q_min_saxs: float
    q_max_saxs: float
    q_min_waxs: float
    q_max_waxs: float


class Action(Enum):
    DELETE = 1
    REPLACE = 2


@dataclass
class Rule:
    column_name: str


@dataclass
class RuleV:
    action: Action
    x: Any = None  # Value for replacement


@dataclass
class RuleQ(Rule):
    """
    Defines a rule that removes entries if value at certain q is lower or higher or both
    """
    q_column_name: str
    q_value: float
    lower: float = None
    upper: float = None
    cancer_diagnosis: bool = None

