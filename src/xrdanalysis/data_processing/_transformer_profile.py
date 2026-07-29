"""Private profile-column transformers re-exported by ``transformers``."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class ColumnStandardizer(TransformerMixin):
    """Standardize array values stored in one DataFrame column."""

    def __init__(self, column):
        self.column = column
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """Fit the scaler to arrays in the configured column."""
        column_data = pd.DataFrame(X[self.column].tolist())
        self.scaler.fit(column_data)
        return self

    def transform(self, X, y=None):
        """Return a copy with standardized arrays in the configured column."""
        X_copy = X.copy()
        column_data = pd.DataFrame(X_copy[self.column].tolist())
        X_copy[self.column] = list(self.scaler.transform(column_data))
        return X_copy


class RuleBasedProfileFilter(TransformerMixin, BaseEstimator):
    """Apply configured nearest-point or interpolated profile rules."""

    def __init__(
        self,
        rules,
        q_col: str = "q_range",
        data_col: str = "radial_profile_data",
        type_col: str = "type_measurement",
        evaluation_mode: str = "nearest",
        window_half_width: float = 0.0,
        keep_unruled: bool = True,
        keep_column: str = "passes_rules",
        failed_rule_column: str = "failed_rule",
        details_column: str = "rule_details",
        trace_column: str = "rule_trace",
        rules_checked_column: str = "rules_checked",
        drop_failures: bool = False,
        reset_index: bool = False,
    ):
        self.rules = rules
        self.q_col = q_col
        self.data_col = data_col
        self.type_col = type_col
        self.evaluation_mode = evaluation_mode
        self.window_half_width = float(window_half_width)
        self.keep_unruled = keep_unruled
        self.keep_column = keep_column
        self.failed_rule_column = failed_rule_column
        self.details_column = details_column
        self.trace_column = trace_column
        self.rules_checked_column = rules_checked_column
        self.drop_failures = drop_failures
        self.reset_index = reset_index
        self.stats_ = None

    @staticmethod
    def parse_rules_text(rules_text: str) -> List[Tuple[float, str, float]]:
        """Parse one ``q operator threshold`` rule per nonblank line."""
        allowed_ops = {"<", "<=", ">", ">=", "==", "!="}
        parsed_rules = []
        for line_idx, raw_line in enumerate(str(rules_text).splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid rule on line {line_idx}: '{line}'. "
                    "Expected format: q operator threshold."
                )
            q_target_str, op_str, threshold_str = parts
            if op_str not in allowed_ops:
                raise ValueError(
                    f"Invalid operator on line {line_idx}: '{op_str}'. "
                    "Use one of <, <=, >, >=, ==, !=."
                )
            parsed_rules.append((float(q_target_str), op_str, float(threshold_str)))
        return parsed_rules

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def _format_rule(self, q_target: float, op_str: str, threshold: float) -> str:
        return f"{q_target:g} {op_str} {threshold:g}"

    def _rule_count(self) -> int:
        if isinstance(self.rules, dict):
            return int(sum(len(rule_list) for rule_list in self.rules.values()))
        return int(len(self.rules))

    def _rule_list_for_row(self, row: pd.Series) -> List[Tuple[float, str, float]]:
        if isinstance(self.rules, dict):
            return list(self.rules.get(row.get(self.type_col, None), []))
        return list(self.rules)

    def _evaluate_row(
        self, row: pd.Series
    ) -> Tuple[bool, str, List[Dict[str, Any]], int]:
        ops = {
            "<": lambda lhs, rhs: lhs < rhs,
            "<=": lambda lhs, rhs: lhs <= rhs,
            ">": lambda lhs, rhs: lhs > rhs,
            ">=": lambda lhs, rhs: lhs >= rhs,
            "==": lambda lhs, rhs: lhs == rhs,
            "!=": lambda lhs, rhs: lhs != rhs,
        }
        rule_list = self._rule_list_for_row(row)
        if not rule_list:
            return bool(self.keep_unruled), "", [], 0
        q = np.asarray(row.get(self.q_col, []), dtype=float)
        y = np.asarray(row.get(self.data_col, []), dtype=float)
        if q.size == 0 or y.size == 0:
            return False, "missing q/profile data", [], len(rule_list)
        n = min(q.size, y.size)
        q, y = q[:n], y[:n]
        details = []
        for q_target, op_str, threshold in rule_list:
            if self.evaluation_mode == "nearest" and self.window_half_width > 0:
                window_mask = np.abs(q - q_target) <= self.window_half_width
                if np.any(window_mask):
                    window_indices = np.flatnonzero(window_mask)
                    nearest_idx = int(
                        window_indices[np.argmin(np.abs(q[window_indices] - q_target))]
                    )
                else:
                    nearest_idx = int(np.abs(q - q_target).argmin())
                q_eval, intensity = float(q[nearest_idx]), float(y[nearest_idx])
            elif self.evaluation_mode == "nearest":
                nearest_idx = int(np.abs(q - q_target).argmin())
                q_eval, intensity = float(q[nearest_idx]), float(y[nearest_idx])
            elif self.evaluation_mode == "interp":
                q_eval, intensity = float(q_target), float(np.interp(q_target, q, y))
            else:
                raise ValueError("evaluation_mode must be either 'nearest' or 'interp'")
            passed = bool(ops[op_str](intensity, threshold))
            rule_text = self._format_rule(q_target, op_str, threshold)
            details.append(
                {
                    "rule": rule_text,
                    "target_q": float(q_target),
                    "nearest_q": q_eval,
                    "intensity": intensity,
                    "operator": op_str,
                    "threshold": float(threshold),
                    "passed": passed,
                }
            )
            if not passed:
                return False, rule_text, details, len(rule_list)
        return True, "", details, len(rule_list)

    def transform(self, X, y=None):
        """Annotate rows and optionally retain only rows passing all rules."""
        X_copy = X.copy()
        evaluations = X_copy.apply(self._evaluate_row, axis=1)

        def _format_rule_trace(details):
            if not details:
                return ""
            return " | ".join(
                (
                    f"{detail['rule']} "
                    f"(nearest_q={detail['nearest_q']:.3f}, "
                    f"I={detail['intensity']:.6f}, "
                    f"{'pass' if detail['passed'] else 'fail'})"
                )
                for detail in details
            )

        X_copy[self.keep_column] = [bool(item[0]) for item in evaluations]
        X_copy[self.failed_rule_column] = [str(item[1]) for item in evaluations]
        X_copy[self.details_column] = [list(item[2]) for item in evaluations]
        X_copy[self.trace_column] = X_copy[self.details_column].apply(
            _format_rule_trace
        )
        X_copy[self.rules_checked_column] = [int(item[3]) for item in evaluations]
        passed_rows, total_rows = int(X_copy[self.keep_column].sum()), int(len(X_copy))
        self.stats_ = {
            "rows_total": total_rows,
            "rows_passed": passed_rows,
            "rows_failed": total_rows - passed_rows,
            "rules_configured": self._rule_count(),
            "evaluation_mode": self.evaluation_mode,
            "window_half_width": self.window_half_width,
        }
        if self.drop_failures:
            X_copy = X_copy.loc[X_copy[self.keep_column]].copy()
            if self.reset_index:
                X_copy = X_copy.reset_index(drop=True)
        elif self.reset_index:
            X_copy = X_copy.reset_index(drop=True)
        return X_copy
