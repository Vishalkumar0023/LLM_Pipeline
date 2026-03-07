"""
Synthetic Data Generator
========================
Generates privacy-safe statistical clones of datasets using Kernel Density Estimation (KDE)
for numeric features and probabilistic sampling for categorical features.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from typing import Optional, Dict, Any

class SyntheticGenerator:
    """
    Learns the statistical properties of a DataFrame and generates synthetic rows.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.generators: Dict[str, Any] = {}
        self.column_order = df.columns.tolist()
        self.dtypes = df.dtypes

    def fit(self):
        """Learn statistical properties (distributions) of the dataset."""
        for col in self.df.columns:
            # Drop missing values for learning
            series = self.df[col].dropna()
            
            if series.empty:
                continue

            if pd.api.types.is_numeric_dtype(series):
                # 1. Learn Numerical Distribution via KDE
                # Use simple heuristics for bandwidth: std * n^(-1/5)
                std = series.std()
                if std == 0:
                    self.generators[col] = {'type': 'constant', 'value': series.iloc[0]}
                else:
                    # Basic bandwidth selection
                    n = len(series)
                    bandwidth = 1.06 * std * (n ** (-0.2))
                    
                    data = series.values.reshape(-1, 1)
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
                    self.generators[col] = {
                        'type': 'kde', 
                        'model': kde, 
                        'min': series.min(), 
                        'max': series.max(),
                        'is_integer': pd.api.types.is_integer_dtype(series)
                    }
            else:
                # 2. Learn Categorical Distribution
                probs = series.value_counts(normalize=True)
                self.generators[col] = {'type': 'categorical', 'probs': probs}

    def generate(self, n_rows: int = 100) -> pd.DataFrame:
        """Generate synthetic rows based on learned distributions."""
        synth_data = {}
        
        for col in self.column_order:
            if col not in self.generators:
                # Fill with NaN if we couldn't learn anything
                synth_data[col] = [np.nan] * n_rows
                continue

            model = self.generators[col]
            
            if model['type'] == 'constant':
                synth_data[col] = [model['value']] * n_rows
                
            elif model['type'] == 'kde':
                # Sample from KDE
                samples = model['model'].sample(n_rows).flatten()
                
                # Post-process: Round integers and clip to original range (optional but good for realism)
                # Ensure we don't produce negative salaries if original was positive, etc.
                # Clipping is debatable for synthetic data but safer for "realistic looking" data
                # samples = np.clip(samples, model['min'], model['max']) 
                
                if model['is_integer']:
                    samples = np.round(samples).astype(int)
                    
                synth_data[col] = samples
                
            elif model['type'] == 'categorical':
                categories = model['probs'].index.values
                weights = model['probs'].values
                # Handle case where weights don't sum exactly to 1 due to float precision
                weights = weights / weights.sum()
                
                samples = np.random.choice(categories, size=n_rows, p=weights)
                synth_data[col] = samples

        return pd.DataFrame(synth_data, columns=self.column_order)
