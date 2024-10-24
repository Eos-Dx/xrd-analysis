from dataclasses import dataclass
from enum import Enum
from typing import Any

SCALED_DATA = "radial_profile_data_norm_scaled"


@dataclass
class Limits:
    """
    Data class that defines the SAXS and WAXS q-range limits.

    :param q_min_saxs: Minimum q value for SAXS.
    :type q_min_saxs: float
    :param q_max_saxs: Maximum q value for SAXS.
    :type q_max_saxs: float
    :param q_min_waxs: Minimum q value for WAXS.
    :type q_min_waxs: float
    :param q_max_waxs: Maximum q value for WAXS.
    :type q_max_waxs: float
    """

    q_min_saxs: float
    q_max_saxs: float
    q_min_waxs: float
    q_max_waxs: float


class Action(Enum):
    """
    Enum class defining actions that can be taken on data: DELETE or REPLACE.
    """

    DELETE = 1
    REPLACE = 2


@dataclass
class Rule:
    """
    Data class that defines a basic rule for operations on a specified column.

    :param column_name: The name of the column to apply the rule to.
    :type column_name: str
    """

    column_name: str


@dataclass
class RuleV:
    """
    Data class that defines a rule to perform an action on a value.

    :param action: The action to take (DELETE or REPLACE).
    :type action: Action
    :param x: The value used for replacement (if action is REPLACE).
    :type x: Any
    """

    action: Action
    x: Any = None  # Value for replacement


@dataclass
class RuleQ(Rule):
    """
    Defines a rule that removes entries based on intensity at a specific \
    q value.

    :param q_column_name: The name of the column containing q values.
    :type q_column_name: str
    :param q_value: The q value at which the intensity is evaluated.
    :type q_value: float
    :param lower: The lower intensity threshold for filtering. \
    If None, no lower threshold.
    :type lower: float, optional
    :param upper: The upper intensity threshold for filtering. \
    If None, no upper threshold.
    :type upper: float, optional
    :param cancer_diagnosis: Additional boolean flag for filtering based \
    on diagnosis.
    :type cancer_diagnosis: bool, optional
    """

    q_column_name: str
    q_value: float
    lower: float = None
    upper: float = None
    cancer_diagnosis: bool = None
