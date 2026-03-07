"""
Data Drift Detector
===================
Detects schema changes and statistical drift between two datasets.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional

class DriftDetector:
    """
    Detects data drift between a baseline dataset and a current dataset.
    """
    
    def __init__(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame):
        self.baseline = baseline_df
        self.current = current_df
        self.report: Dict[str, Any] = {
            'schema_drift': {},
            'statistical_drift': {},
            'score': 100
        }
    
    def run(self) -> Dict[str, Any]:
        """Run all drift checks."""
        self._check_schema_drift()
        self._check_statistical_drift()
        self._calculate_score()
        return self.report
    
    def _check_schema_drift(self):
        """Check for missing or new columns."""
        base_cols = set(self.baseline.columns)
        curr_cols = set(self.current.columns)
        
        missing = list(base_cols - curr_cols)
        new = list(curr_cols - base_cols)
        
        self.report['schema_drift'] = {
            'missing_columns': missing,
            'new_columns': new,
            'has_drift': len(missing) > 0
        }
    
    def _check_statistical_drift(self):
        """
        Check for statistical drift in shared numeric columns using KS-Test.
        KS-Test (Kolmogorov-Smirnov) checks if two samples come from same distribution.
        """
        base_cols = set(self.baseline.select_dtypes(include=[np.number]).columns)
        curr_cols = set(self.current.select_dtypes(include=[np.number]).columns)
        shared_cols = list(base_cols.intersection(curr_cols))
        
        drifted_features = []
        
        for col in shared_cols:
            # Drop NaNs for valid test
            b_data = self.baseline[col].dropna()
            c_data = self.current[col].dropna()
            
            if len(b_data) == 0 or len(c_data) == 0:
                continue
                
            # KS Test
            # Null hypothesis: samples are from same distribution.
            # If p_value < 0.05, we reject null hypothesis -> DRIFT DETECTED.
            stat, p_value = stats.ks_2samp(b_data, c_data)
            
            is_drifted = p_value < 0.05
            
            # Additional metrics
            b_mean = b_data.mean()
            c_mean = c_data.mean()
            delta_mean = abs(b_mean - c_mean)
            perc_change = (delta_mean / b_mean) * 100 if b_mean != 0 else 0
            
            self.report['statistical_drift'][col] = {
                'p_value': float(p_value),
                'is_drifted': bool(is_drifted),
                'baseline_mean': float(b_mean),
                'current_mean': float(c_mean),
                'pct_change': float(perc_change)
            }
            
            if is_drifted:
                drifted_features.append(col)
                
        self.report['drifted_features'] = drifted_features
        self.report['drift_detected'] = len(drifted_features) > 0

    def _calculate_score(self):
        """Calculate a simple health score (0-100)."""
        # Penalty for schema drift
        score = 100
        if self.report['schema_drift']['has_drift']:
            score -= 20
        
        # Penalty for statistical drift
        total_feats = len(self.report['statistical_drift'])
        drifted = len(self.report['drifted_features'])
        
        if total_feats > 0:
            drift_ratio = drifted / total_feats
            score -= (drift_ratio * 50)
            
        self.report['score'] = max(0, int(score))

