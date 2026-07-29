"""Private goodness-score transformers re-exported by ``transformers``."""

from __future__ import annotations

import pandas as pd
from sklearn.base import TransformerMixin

from xrdanalysis.data_processing._goodness import (
    filter_goodness_dataframe,
    transform_goodness_dataframe,
)


class GoodnessTransformer(TransformerMixin):
    """Calculate the established high-frequency goodness score per row."""

    def __init__(
        self,
        column: str = "polar_data",
        skip_bins: int = 30,
        hf_cutoff_fraction: float = 0.25,
        output_col: str = "goodness",
        save_dev: bool = False,
        diff_col: str = "data_diff",
    ):
        self.column = column
        self.skip_bins = skip_bins
        self.hf_cutoff_fraction = hf_cutoff_fraction
        self.output_col = output_col
        self.save_dev = save_dev
        self.diff_col = diff_col

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return transform_goodness_dataframe(
            df,
            column=self.column,
            skip_bins=self.skip_bins,
            hf_cutoff_fraction=self.hf_cutoff_fraction,
            output_col=self.output_col,
            save_dev=self.save_dev,
            diff_col=self.diff_col,
        )


class GoodnessFilter(TransformerMixin):
    """Filter profile rows using measurement-type goodness thresholds."""

    def __init__(
        self,
        goodness_column: str = "goodness",
        type_column: str = "type_measurement",
        thresholds: dict = None,
        rule: str = ">",
        default_threshold: float = 50.0,
        verbose: bool = False,
    ):
        self.goodness_column = goodness_column
        self.type_column = type_column
        self.thresholds = (
            thresholds if thresholds is not None else {"SAXS": 50, "WAXS": 30}
        )
        self.rule = rule
        self.default_threshold = default_threshold
        self.verbose = verbose

        # Validate rule parameter
        valid_rules = [">", ">=", "<", "<="]
        if self.rule not in valid_rules:
            raise ValueError(f"Rule must be one of {valid_rules}, got '{self.rule}'")

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return filter_goodness_dataframe(
            df,
            goodness_column=self.goodness_column,
            type_column=self.type_column,
            thresholds=self.thresholds,
            rule=self.rule,
            default_threshold=self.default_threshold,
            verbose=self.verbose,
        )
