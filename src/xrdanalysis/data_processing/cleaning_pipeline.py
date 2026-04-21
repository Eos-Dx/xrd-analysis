"""
Reusable dataframe cleaning pipeline with optional stage artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


StageOutput = pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]] | dict[str, Any]
StageCallable = Callable[[pd.DataFrame], StageOutput]


@dataclass
class DataFrameStageArtifact:
    """
    Snapshot of one dataframe-cleaning stage.

    Parameters
    ----------
    name : str
        Stage name.
    df : pd.DataFrame
        Output dataframe after the stage.
    stats : dict[str, Any]
        Optional stage metadata such as counts, thresholds, or auxiliary frames.
    """

    name: str
    df: pd.DataFrame
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataFramePipelineResult:
    """
    Result object returned by ``DataFrameCleaningPipeline.transform(..., return_report=True)``.
    """

    final_df: pd.DataFrame
    stage_artifacts: list[DataFrameStageArtifact]
    summary_df: pd.DataFrame

    def get_stage(self, name: str) -> DataFrameStageArtifact | None:
        for artifact in self.stage_artifacts:
            if artifact.name == name:
                return artifact
        return None


class DataFrameCleaningPipeline(TransformerMixin, BaseEstimator):
    """
    Execute a named sequence of dataframe cleaning steps with stage snapshots.

    Each step receives a dataframe and may return one of:
    - a dataframe
    - ``(dataframe, stats_dict)``
    - ``{"df": dataframe, "stats": {...}}``

    When ``return_report=True``, the pipeline returns a ``DataFramePipelineResult``
    containing the final dataframe plus per-stage artifacts and a compact summary.
    This makes it easy to plot intermediate stages and compute counts without
    re-implementing the cleaning chain in every notebook.
    """

    def __init__(
        self,
        steps: Sequence[tuple[str, StageCallable | Any]],
        *,
        copy_input: bool = True,
        stop_on_empty: bool = True,
    ) -> None:
        self.steps = list(steps)
        self.copy_input = bool(copy_input)
        self.stop_on_empty = bool(stop_on_empty)

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def _run_step(self, step: StageCallable | Any, df: pd.DataFrame) -> StageOutput:
        if hasattr(step, "fit_transform"):
            return step.fit_transform(df)
        if hasattr(step, "transform"):
            return step.transform(df)
        if callable(step):
            return step(df)
        raise TypeError("Pipeline step must be callable or implement transform().")

    def _coerce_stage_output(
        self,
        name: str,
        stage_output: StageOutput,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if isinstance(stage_output, pd.DataFrame):
            return stage_output, {}
        if isinstance(stage_output, tuple) and len(stage_output) == 2:
            _df, _stats = stage_output
            if not isinstance(_df, pd.DataFrame):
                raise TypeError(f"Stage '{name}' did not return a dataframe.")
            if _stats is None:
                _stats = {}
            if not isinstance(_stats, dict):
                raise TypeError(f"Stage '{name}' stats must be a dict.")
            return _df, _stats
        if isinstance(stage_output, dict) and "df" in stage_output:
            _df = stage_output["df"]
            _stats = stage_output.get("stats", {})
            if not isinstance(_df, pd.DataFrame):
                raise TypeError(f"Stage '{name}' did not return a dataframe.")
            if _stats is None:
                _stats = {}
            if not isinstance(_stats, dict):
                raise TypeError(f"Stage '{name}' stats must be a dict.")
            return _df, _stats
        raise TypeError(
            f"Stage '{name}' must return a DataFrame, (DataFrame, dict), or {{'df': DataFrame, 'stats': dict}}."
        )

    def transform(self, X: pd.DataFrame, return_report: bool = False):
        _current = X.copy() if self.copy_input else X
        _artifacts: list[DataFrameStageArtifact] = []
        _summary_rows: list[dict[str, Any]] = []

        for _stage_idx, (_name, _step) in enumerate(self.steps, start=1):
            _rows_in = int(len(_current))
            _stage_output = self._run_step(_step, _current)
            _next_df, _stats = self._coerce_stage_output(_name, _stage_output)
            if hasattr(_step, "stats_") and isinstance(getattr(_step, "stats_", None), dict):
                _stats = {**getattr(_step, "stats_"), **_stats}
            _rows_out = int(len(_next_df))
            _stage_stats = {
                "stage_name": _name,
                "stage_idx": int(_stage_idx),
                "rows_in": _rows_in,
                "rows_out": _rows_out,
                "rows_removed": int(max(_rows_in - _rows_out, 0)),
                **_stats,
            }
            _summary_rows.append(
                {
                    "stage_name": _name,
                    "stage_idx": int(_stage_idx),
                    "rows_in": _rows_in,
                    "rows_out": _rows_out,
                    "rows_removed": int(max(_rows_in - _rows_out, 0)),
                }
            )
            if return_report:
                _artifacts.append(
                    DataFrameStageArtifact(
                        name=_name,
                        df=_next_df.copy(),
                        stats=_stage_stats,
                    )
                )
            _current = _next_df
            if self.stop_on_empty and len(_current) == 0:
                break

        if return_report:
            return DataFramePipelineResult(
                final_df=_current.copy(),
                stage_artifacts=_artifacts,
                summary_df=pd.DataFrame(_summary_rows),
            )
        return _current


class ColumnValueFilter(TransformerMixin, BaseEstimator):
    """
    Generic row filter based on one dataframe column.

    Examples
    --------
    Keep biopsy rows::

        ColumnValueFilter("biopsy_flag", op="in", values=[True])

    Keep rows with status in a whitelist::

        ColumnValueFilter("specimen_status", op="in", values=["BENIGN", "CANCER"])

    Keep rows where score is above threshold::

        ColumnValueFilter("goodness", op=">=", value=30.0)
    """

    def __init__(
        self,
        column: str,
        *,
        op: str = "in",
        value: Any = None,
        values: Sequence[Any] | None = None,
        lower: Any = None,
        upper: Any = None,
        keep_na: bool = False,
        reset_index: bool = False,
    ) -> None:
        self.column = column
        self.op = str(op)
        self.value = value
        self.values = list(values) if values is not None else None
        self.lower = lower
        self.upper = upper
        self.keep_na = bool(keep_na)
        self.reset_index = bool(reset_index)
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def _build_mask(self, series: pd.Series) -> pd.Series:
        _op = self.op.lower()
        if _op == "in":
            if self.values is None:
                raise ValueError("`values` must be provided for op='in'.")
            return series.isin(self.values)
        if _op == "not_in":
            if self.values is None:
                raise ValueError("`values` must be provided for op='not_in'.")
            return ~series.isin(self.values)
        if _op in {"==", "eq"}:
            return series.eq(self.value)
        if _op in {"!=", "ne"}:
            return series.ne(self.value)
        if _op in {">", ">=", "<", "<="}:
            _numeric = pd.to_numeric(series, errors="coerce")
            _value = float(self.value)
            if _op == ">":
                return _numeric.gt(_value)
            if _op == ">=":
                return _numeric.ge(_value)
            if _op == "<":
                return _numeric.lt(_value)
            return _numeric.le(_value)
        if _op == "between":
            if self.lower is None or self.upper is None:
                raise ValueError("`lower` and `upper` must be provided for op='between'.")
            _numeric = pd.to_numeric(series, errors="coerce")
            return _numeric.between(float(self.lower), float(self.upper), inclusive="both")
        if _op == "contains":
            return series.fillna("").astype(str).str.contains(str(self.value), regex=True, na=False)
        if _op == "isna":
            return series.isna()
        if _op == "notna":
            return series.notna()
        raise ValueError(f"Unsupported column filter op: {self.op}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.column not in X.columns:
            raise KeyError(f"Column '{self.column}' not found in DataFrame.")
        _out = X.copy()
        _mask = self._build_mask(_out[self.column])
        if self.keep_na:
            _mask = _mask | _out[self.column].isna()
        _filtered = _out[_mask].copy()
        if self.reset_index:
            _filtered.reset_index(drop=True, inplace=True)
        self.stats_ = {
            "filter_type": "column_value",
            "filter_column": self.column,
            "filter_op": self.op,
            "filter_value": self.value,
            "filter_values": self.values,
            "filter_lower": self.lower,
            "filter_upper": self.upper,
            "keep_na": self.keep_na,
        }
        return _filtered


class DataFrameQueryFilter(TransformerMixin, BaseEstimator):
    """
    Generic dataframe filter using ``pandas.DataFrame.query``.

    Example
    -------
    ``DataFrameQueryFilter(\"biopsy_flag == True and specimen_status in ['BENIGN', 'CANCER']\")``
    """

    def __init__(self, query: str, *, local_dict: dict[str, Any] | None = None, reset_index: bool = False) -> None:
        self.query = str(query)
        self.local_dict = dict(local_dict or {})
        self.reset_index = bool(reset_index)
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _filtered = X.query(self.query, local_dict=self.local_dict).copy()
        if self.reset_index:
            _filtered.reset_index(drop=True, inplace=True)
        self.stats_ = {
            "filter_type": "query",
            "query": self.query,
        }
        return _filtered
