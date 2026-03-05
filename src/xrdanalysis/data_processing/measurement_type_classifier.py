"""
Measurement Type Classifier Transformer

Classifies measurements as WAXS, SAXS, etc. based on detector distance from PONI files.
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, Tuple, Union, Any
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


class MeasurementTypeClassifier(TransformerMixin, BaseEstimator):
    """
    Classify measurements by type based on detector distance read from PONI files.
    
    The classifier reads the Distance field from PONI file content and applies
    user-defined classification rules.
    
    Parameters
    ----------
    poni_col : str, default='ponifile'
        Column name containing PONI file content as string
    output_col : str, default='type_measurement'
        Column name for output classification
    type_rules : dict, optional
        Dictionary mapping type names to (operator, threshold) tuples.
        Operators: '<', '<=', '>', '>=', '==', 'between'
        Examples:
            {'WAXS': ('<', 0.05), 'SAXS': ('>', 0.1)}
            {'WAXS': ('<', 0.05), 'INTERMEDIATE': ('between', (0.05, 0.1)), 'SAXS': ('>', 0.1)}
        If None, defaults to WAXS < 0.05, SAXS > 0.1
    unknown_label : str, default='UNKNOWN'
        Label for measurements that don't match any rule
    debug : bool, default=False
        Print debug information
    """
    
    def __init__(
        self,
        poni_col: str = 'ponifile',
        output_col: str = 'type_measurement',
        type_rules: Dict[str, Tuple[str, Union[float, Tuple[float, float]]]] = None,
        unknown_label: str = 'UNKNOWN',
        debug: bool = False,
    ):
        self.poni_col = poni_col
        self.output_col = output_col
        self.unknown_label = unknown_label
        self.debug = debug
        
        # Default rules if none provided
        if type_rules is None:
            type_rules = {
                'WAXS': ('<', 0.05),
                'SAXS': ('>', 0.1),
            }
        self.type_rules = type_rules
        self.stats_ = None

    def _read_distance_from_poni(self, poni_content: str) -> float:
        """
        Extract Distance value from PONI file content string.
        
        Parameters
        ----------
        poni_content : str
            PONI file content as string
        
        Returns
        -------
        float
            Distance value in meters, or NaN if not found
        """
        if pd.isna(poni_content) or not isinstance(poni_content, str):
            return np.nan
        
        # Look for Distance: <value> anywhere in the content
        # Use word boundary instead of line start to handle malformed ponifiles without line breaks
        match = re.search(r'\bDistance:\s*([0-9.eE+-]+)', poni_content)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return np.nan
        return np.nan
    
    def _apply_rule(self, distance: float, operator: str, threshold: Union[float, Tuple[float, float]]) -> bool:
        """
        Apply a classification rule to a distance value.
        
        Parameters
        ----------
        distance : float
            Distance value in meters
        operator : str
            Comparison operator: '<', '<=', '>', '>=', '==', 'between'
        threshold : float or tuple
            Threshold value(s) for comparison
        
        Returns
        -------
        bool
            True if rule matches, False otherwise
        """
        if not np.isfinite(distance):
            return False
        
        if operator == '<':
            return distance < threshold
        elif operator == '<=':
            return distance <= threshold
        elif operator == '>':
            return distance > threshold
        elif operator == '>=':
            return distance >= threshold
        elif operator == '==':
            return distance == threshold
        elif operator == 'between':
            if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
                low, high = threshold
                return low <= distance <= high
        return False
    
    def fit(self, df: pd.DataFrame, y=None):
        """
        Fit transformer (no-op, but allows sklearn compatibility).
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        y : ignored
        
        Returns
        -------
        self
        """
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify measurements and add type_measurement column.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with ponifile column
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame with new type_measurement column
        """
        df = df.copy()
        
        if self.poni_col not in df.columns:
            raise ValueError(f"Column '{self.poni_col}' not found in dataframe")
        
        # Extract distances from PONI files
        if self.debug:
            print(f"Reading distances from {self.poni_col} column...")
        
        distances = df[self.poni_col].apply(self._read_distance_from_poni)
        
        # Initialize with unknown label
        df[self.output_col] = self.unknown_label
        
        # Apply classification rules in order
        type_counts = {}
        for type_name, (operator, threshold) in self.type_rules.items():
            mask = distances.apply(lambda d: self._apply_rule(d, operator, threshold))
            df.loc[mask, self.output_col] = type_name
            type_counts[type_name] = int(mask.sum())
            
            if self.debug:
                print(f"  {type_name} ({operator} {threshold}): {mask.sum()} measurements")
        
        # Count unknowns
        unknown_mask = df[self.output_col] == self.unknown_label
        type_counts[self.unknown_label.lower() + '_count'] = int(unknown_mask.sum())
        type_counts['nan_distance_count'] = int(distances.isna().sum())
        type_counts['total_count'] = len(df)
        
        # Compute statistics
        self.stats_ = type_counts
        
        if self.debug:
            print(f"\nClassification complete:")
            for k, v in type_counts.items():
                print(f"  {k}: {v}")
        
        return df

    def get_stats(self) -> Dict:
        """Get classification statistics from last transform."""
        if self.stats_ is None:
            raise ValueError("Must call transform() first")
        return self.stats_.copy()
