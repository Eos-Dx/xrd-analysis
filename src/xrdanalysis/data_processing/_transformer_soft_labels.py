"""Private soft-label transformers re-exported by ``transformers``."""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class SpecimenStatusToSoftLabels(TransformerMixin):
    """Map specimen status and rule columns to normalized soft-label vectors."""

    def __init__(
        self,
        status_col: str = "specimen_status",
        rule_cols: Optional[List[str]] = None,
        output_col: str = "cancer_status_soft",
        class_order: Optional[List[Union[str, int]]] = None,
        rules: Optional[Dict[Tuple[Optional[str], Any, str], List[float]]] = None,
        normalize: bool = True,
        strict: bool = False,
        capitalize_status: bool = True,
    ) -> None:
        self.status_col = status_col
        self.rule_cols = list(rule_cols) if rule_cols is not None else ["biopsy"]
        self.output_col = output_col
        self.class_order = (
            list(class_order)
            if class_order is not None
            else ["CANCER", "BENIGN", "NORMAL"]
        )
        self.normalize = bool(normalize)
        self.strict = bool(strict)
        self.capitalize_status = bool(capitalize_status)
        default_rules = {
            ("biopsy", True, "CANCER"): [0.89, 0.09, 0.02],
            ("biopsy", True, "BENIGN"): [0.09, 0.89, 0.02],
            ("biopsy", False, "NORMAL"): [0.10, 0.30, 0.60],
            ("biopsy", False, "BENIGN"): [0.20, 0.60, 0.20],
            ("biopsy", False, "CANCER"): [0.60, 0.30, 0.10],
            (None, None, "NORMAL"): [0.10, 0.30, 0.60],
            (None, None, "BENIGN"): [0.20, 0.60, 0.20],
            (None, None, "CANCER"): [0.60, 0.30, 0.10],
        }
        self.rules = rules if rules is not None else default_rules
        for key, vector in list(self.rules.items()):
            if not isinstance(vector, (list, tuple, np.ndarray)):
                raise ValueError(
                    f"Rule for {key} must be a vector-like; got {type(vector)}"
                )
            if len(vector) != len(self.class_order):
                raise ValueError(
                    f"Rule for {key} length {len(vector)} != len(class_order) {len(self.class_order)}"
                )

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def _to_bool(self, value) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, (int, np.integer)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "1", "yes", "y", "t"}:
                return True
            if text in {"false", "0", "no", "n", "f"}:
                return False
        try:
            return bool(value)
        except Exception:
            return None

    def _canon_status(self, status) -> Optional[str]:
        if status is None:
            return None
        try:
            value = (
                status.strip().upper()
                if isinstance(status, str) and self.capitalize_status
                else str(status)
            )
            if self.capitalize_status and not isinstance(status, str):
                value = value.upper()
            return "CANCER" if value == "MALIGNANT" else value
        except Exception:
            return None

    def _norm_vec(self, vector: List[float]) -> List[float]:
        arr = np.clip(np.asarray(vector, dtype=float), 0.0, None)
        if self.normalize and float(arr.sum()) > 0:
            arr = arr / float(arr.sum())
        return arr.astype(float).tolist()

    def _key_variants(self, column: str, value, status: str):
        keys = [(column, value, status)]
        if isinstance(value, str):
            keys.append((column, value.strip().upper(), status))
        boolean_value = self._to_bool(value)
        if boolean_value is not None:
            keys.append((column, boolean_value, status))
        keys.append((column, None, status))
        return keys

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        if self.status_col not in X.columns:
            raise KeyError(f"Status column '{self.status_col}' not found in DataFrame")
        missing = [column for column in self.rule_cols if column not in X.columns]
        if missing:
            raise KeyError(f"Rule columns not found in DataFrame: {missing}")
        results: List[Optional[List[float]]] = []
        for _, row in X.iterrows():
            status = self._canon_status(row[self.status_col])
            vector = None
            if status is not None:
                for column in self.rule_cols:
                    for key in self._key_variants(column, row[column], status):
                        value = self.rules.get(key)
                        if value is not None:
                            vector = self._norm_vec(value)
                            break
                    if vector is not None:
                        break
                if vector is None:
                    value = self.rules.get((None, None, status)) or self.rules.get(
                        (None, status)
                    )
                    if value is not None:
                        vector = self._norm_vec(value)
            if vector is None:
                if self.strict:
                    raise KeyError(
                        f"No soft-label rule matched for status={status} using columns {self.rule_cols}"
                    )
                vector = [np.nan] * len(self.class_order)
            results.append(vector)
        X[self.output_col] = results
        return X


class SoftLabelToWeightedSamples(TransformerMixin):
    """Expand each soft-label vector into weighted hard-label sample rows."""

    def __init__(
        self,
        soft_col: str = "cancer_status_soft",
        label_col: str = "cancer_status",
        weight_col: str = "cancer_status_weighted",
        class_names: Optional[List[Union[str, int]]] = None,
        label_col_numeric: Optional[str] = None,
        label_codes: Optional[List[Any]] = None,
        label_code_map: Optional[Dict[Any, Any]] = None,
        min_weight: float = 0.0,
        normalize: bool = True,
        drop_soft_col: bool = False,
    ) -> None:
        self.soft_col = soft_col
        self.label_col = label_col
        self.weight_col = weight_col
        self.class_names = class_names
        self.label_col_numeric = label_col_numeric
        self.label_codes = label_codes
        self.label_code_map = label_code_map
        self.min_weight = float(min_weight)
        self.normalize = bool(normalize)
        self.drop_soft_col = bool(drop_soft_col)
        if (
            self.label_codes is not None
            and self.class_names is not None
            and len(self.label_codes) != len(self.class_names)
        ):
            raise ValueError(
                f"label_codes length {len(self.label_codes)} must equal "
                f"class_names length {len(self.class_names)}"
            )

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def _to_prob_list(self, value) -> Optional[List[float]]:
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(value, dtype=float)
        elif isinstance(value, str):
            try:
                arr = np.asarray(ast.literal_eval(value), dtype=float)
            except Exception:
                try:
                    arr = np.asarray(
                        [
                            float(item)
                            for item in value.strip().strip("[]()").split(",")
                            if item != ""
                        ],
                        dtype=float,
                    )
                except Exception:
                    return None
        else:
            try:
                arr = np.asarray(value, dtype=float)
            except Exception:
                return None
        if arr.ndim != 1 or arr.size == 0:
            return None
        arr = np.clip(arr, 0.0, None)
        if self.normalize and float(arr.sum()) > 0:
            arr = arr / float(arr.sum())
        return arr.astype(float).tolist()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.soft_col not in df.columns:
            raise KeyError(
                f"Soft-label column '{self.soft_col}' not found in DataFrame"
            )
        rows = []
        for _, row in df.iterrows():
            probabilities = self._to_prob_list(row[self.soft_col])
            if probabilities is None:
                continue
            if self.class_names is not None and len(self.class_names) != len(
                probabilities
            ):
                raise ValueError(
                    f"class_names length ({len(self.class_names)}) does not match "
                    f"soft vector length ({len(probabilities)})"
                )
            for index, weight in enumerate(probabilities):
                if weight <= self.min_weight:
                    continue
                new_row = row.copy()
                new_row[self.weight_col] = float(weight)
                label_value = (
                    self.class_names[index] if self.class_names is not None else index
                )
                new_row[self.label_col] = label_value
                if self.label_col_numeric is not None:
                    code = None
                    if self.label_code_map is not None:
                        code = self.label_code_map.get(
                            label_value, self.label_code_map.get(index)
                        )
                    if (
                        code is None
                        and self.label_codes is not None
                        and self.class_names is not None
                    ):
                        code = self.label_codes[index]
                    new_row[self.label_col_numeric] = index if code is None else code
                rows.append(new_row)
        if not rows:
            columns = list(df.columns)
            if self.weight_col not in columns:
                columns.append(self.weight_col)
            if self.label_col not in columns:
                columns.append(self.label_col)
            return pd.DataFrame(columns=columns)
        output = pd.DataFrame(rows)
        if self.drop_soft_col and self.soft_col in output.columns:
            output = output.drop(columns=[self.soft_col])
        output.reset_index(drop=True, inplace=True)
        return output
