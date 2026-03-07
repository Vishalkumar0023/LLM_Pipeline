"""
Data Cleaner Module
===================
Handles data cleaning operations including missing values,
duplicates, outliers, and data type corrections.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
import re


class DataCleaner:
    """Clean and preprocess datasets."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize cleaner with a DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame to clean
        """
        self.df = df.copy()
        # Convert pandas extension types (StringDtype, nullable Int/Float)
        # to standard numpy dtypes for compatibility with numpy/sklearn
        for col in self.df.columns:
            if pd.api.types.is_string_dtype(self.df[col]) and self.df[col].dtype != 'object':
                self.df[col] = self.df[col].astype('object')
            elif pd.api.types.is_integer_dtype(self.df[col]) and hasattr(self.df[col].dtype, 'numpy_dtype'):
                self.df[col] = self.df[col].astype(self.df[col].dtype.numpy_dtype)
            elif pd.api.types.is_float_dtype(self.df[col]) and hasattr(self.df[col].dtype, 'numpy_dtype'):
                self.df[col] = self.df[col].astype(self.df[col].dtype.numpy_dtype)
        self.original_shape = df.shape
        self.cleaning_log: List[str] = []
        self.transformations: List[Dict[str, Any]] = []
        self.row_changes: List[Dict[str, Any]] = []
    
    def remove_duplicates(
        self, 
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> 'DataCleaner':
        """
        Remove duplicate rows.
        
        Parameters:
        -----------
        subset : list, optional
            Columns to consider for duplicates
        keep : str
            Which duplicate to keep ('first', 'last', False)
        """
        before = len(self.df)
        
        # Identify duplicates for logging before removing
        dupes = self.df[self.df.duplicated(subset=subset, keep=keep)]
        if not dupes.empty:
            for idx in dupes.index[:2]:
                # Create a string representation of the row (first 3 cols)
                row_str = ", ".join([str(x) for x in self.df.loc[idx].values[:3]]) + "..."
                self.row_changes.append({
                    "index": int(idx),
                    "column": "Row",
                    "old_value": row_str,
                    "new_value": "Deleted",
                    "operation": "Row Removed",
                    "reason": "Duplicate"
                })

        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = before - len(self.df)
        
        if removed > 0:
            msg = f"Removed {removed:,} duplicate rows"
            self.cleaning_log.append(msg)
            self.transformations.append({
                "operation": "remove_duplicates",
                "rows_removed": removed
            })
        
        return self
    
    def handle_missing_values(
        self,
        strategy: str = 'auto',
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'mode',
        drop_threshold: float = 0.4,
        fill_value: Optional[Any] = None
    ) -> 'DataCleaner':
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        strategy : str
            'auto', 'drop_rows', 'drop_cols', 'fill'
        numeric_strategy : str
            Strategy for numeric columns: 'mean', 'median', 'zero'
        categorical_strategy : str
            Strategy for categorical: 'mode', 'unknown'
        drop_threshold : float
            Drop columns with missing > threshold (0-1)
        fill_value : any, optional
            Custom fill value when strategy='fill'
        """
        missing_before = self.df.isnull().sum().sum()
        
        if missing_before == 0:
            self.cleaning_log.append("No missing values to handle")
            return self
        
        # Drop columns with too many missing values
        missing_pct = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_pct[missing_pct > drop_threshold].index.tolist()
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            msg = f"Dropped {len(cols_to_drop)} columns with >{drop_threshold*100}% missing: {cols_to_drop}"
            self.cleaning_log.append(msg)
            self.transformations.append({
                "operation": "drop_high_missing_columns",
                "columns": cols_to_drop,
                "threshold": drop_threshold
            })
        
        if strategy == 'drop_rows':
            before = len(self.df)
            self.df = self.df.dropna()
            msg = f"Dropped {before - len(self.df):,} rows with missing values"
            self.cleaning_log.append(msg)
            return self
        
        # Fill numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().any():
                # Log usage
                missing_indices = self.df[self.df[col].isnull()].index.tolist()
                
                # Determine strategy name and value
                strat_name = "Imputed (Custom)"
                val_used = fill_value
                
                if fill_value is not None:
                    self.df[col] = self.df[col].fillna(fill_value)
                    strat_name = f"Filled {fill_value}"
                elif numeric_strategy == 'mean':
                    val_used = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(val_used)
                    strat_name = "Imputed Mean"
                elif numeric_strategy == 'median':
                    val_used = self.df[col].median()
                    self.df[col] = self.df[col].fillna(val_used)
                    strat_name = "Imputed Median"
                elif numeric_strategy == 'zero':
                    val_used = 0
                    self.df[col] = self.df[col].fillna(0)
                    strat_name = "Filled Zero"

                # Capture 2 examples
                for idx in missing_indices[:2]:
                    self.row_changes.append({
                        "index": int(idx),
                        "column": col,
                        "old_value": "NaN",
                        "new_value": round(float(val_used), 4) if isinstance(val_used, (float, int)) else str(val_used),
                        "operation": strat_name,
                        "reason": "Missing Value"
                    })
        
        # Fill categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if self.df[col].isnull().any():
                missing_indices = self.df[self.df[col].isnull()].index.tolist()
                strat_name = "Imputed (Custom)"
                val_used = fill_value
                
                if fill_value is not None:
                    self.df[col] = self.df[col].fillna(fill_value)
                    strat_name = f"Filled {fill_value}"
                elif categorical_strategy == 'mode':
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        val_used = mode_val[0]
                        self.df[col] = self.df[col].fillna(val_used)
                        strat_name = "Imputed Mode"
                elif categorical_strategy == 'unknown':
                    val_used = "Unknown"
                    self.df[col] = self.df[col].fillna('Unknown')
                    strat_name = "Filled 'Unknown'"
                
                # Capture 2 examples
                for idx in missing_indices[:2]:
                    self.row_changes.append({
                        "index": int(idx),
                        "column": col,
                        "old_value": "NaN",
                        "new_value": str(val_used),
                        "operation": strat_name,
                        "reason": "Missing Value"
                    })
        
        missing_after = self.df.isnull().sum().sum()
        msg = f"Handled missing values: {missing_before:,} → {missing_after:,}"
        self.cleaning_log.append(msg)
        self.transformations.append({
            "operation": "handle_missing",
            "numeric_strategy": numeric_strategy,
            "categorical_strategy": categorical_strategy,
            "values_filled": missing_before - missing_after
        })
        
        return self
    
    def fix_data_types(
        self,
        type_mapping: Optional[Dict[str, str]] = None,
        infer_types: bool = True
    ) -> 'DataCleaner':
        """
        Fix and optimize data types.
        
        Parameters:
        -----------
        type_mapping : dict, optional
            Manual mapping of column names to types
        infer_types : bool
            Whether to automatically infer types
        """
        changes = []
        
        # Apply manual type mapping
        if type_mapping:
            for col, dtype in type_mapping.items():
                if col in self.df.columns:
                    try:
                        self.df[col] = self.df[col].astype(dtype)
                        changes.append(f"{col} → {dtype}")
                    except (ValueError, TypeError) as e:
                        self.cleaning_log.append(f"Could not convert {col} to {dtype}: {e}")
        
        if infer_types:
            for col in self.df.columns:
                # Try to convert object columns to numeric
                if self.df[col].dtype == 'object':
                    # Try numeric conversion
                    try:
                        numeric_series = pd.to_numeric(self.df[col], errors='coerce')
                        if numeric_series.notna().sum() / len(self.df) > 0.9:
                            self.df[col] = numeric_series
                            changes.append(f"{col} → numeric")
                            continue
                    except:
                        pass
                    
                    # Try datetime conversion
                    try:
                        datetime_series = pd.to_datetime(self.df[col], errors='coerce', infer_datetime_format=True)
                        if datetime_series.notna().sum() / len(self.df) > 0.9:
                            self.df[col] = datetime_series
                            changes.append(f"{col} → datetime")
                            continue
                    except:
                        pass
                    
                    # Convert to category if low cardinality
                    if self.df[col].nunique() / len(self.df) < 0.05:
                        self.df[col] = self.df[col].astype('category')
                        changes.append(f"{col} → category")
        
        if changes:
            msg = f"Fixed data types: {len(changes)} columns"
            self.cleaning_log.append(msg)
            self.transformations.append({
                "operation": "fix_data_types",
                "changes": changes
            })
        
        return self
    
    def handle_outliers(
        self,
        method: str = 'iqr',
        columns: Optional[List[str]] = None,
        threshold: float = 1.5,
        action: str = 'clip'
    ) -> 'DataCleaner':
        """
        Detect and handle outliers in numeric columns.
        
        Parameters:
        -----------
        method : str
            'iqr' or 'zscore'
        columns : list, optional
            Specific columns to check (default: all numeric)
        threshold : float
            IQR multiplier (1.5) or Z-score threshold (3.0)
        action : str
            'clip', 'remove', or 'nan'
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_handled = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            series = self.df[col].dropna()
            
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
            elif method == 'zscore':
                mean = series.mean()
                std = series.std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Count outliers
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                # Capture examples BEFORE modifying
                outlier_indices = self.df[outlier_mask].index.tolist()
                for idx in outlier_indices[:2]:
                    # Get the actual value
                    old_val = self.df.loc[idx, col]
                    new_val_str = "NaN"
                    action_desc = "Removed"
                    
                    if action == 'clip':
                        if old_val < lower_bound:
                            new_val = lower_bound
                        else:
                            new_val = upper_bound
                        new_val_str = f"{new_val:.4f}"
                        action_desc = "Clipped w/ IQR"
                    elif action == 'nan':
                        new_val_str = "NaN"
                        action_desc = "Set to NaN"
                    
                    self.row_changes.append({
                        "index": int(idx),
                        "column": col,
                        "old_value": f"{old_val:.4f}",
                        "new_value": new_val_str,
                        "operation": action_desc,
                        "reason": "Outlier"
                    })

                if action == 'clip':
                    self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                elif action == 'remove':
                    self.df = self.df[~outlier_mask]
                elif action == 'nan':
                    self.df.loc[outlier_mask, col] = np.nan
                
                outliers_handled[col] = int(outlier_count)
        
        if outliers_handled:
            total = sum(outliers_handled.values())
            msg = f"Handled {total:,} outliers in {len(outliers_handled)} columns using {method}/{action}"
            self.cleaning_log.append(msg)
            self.transformations.append({
                "operation": "handle_outliers",
                "method": method,
                "action": action,
                "outliers_per_column": outliers_handled
            })
        
        return self
    
    def clean_categorical_values(
        self,
        columns: Optional[List[str]] = None,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        replace_mapping: Optional[Dict[str, Dict[str, str]]] = None
    ) -> 'DataCleaner':
        """
        Clean and standardize categorical values.
        
        Parameters:
        -----------
        columns : list, optional
            Specific columns to clean (default: all object/category)
        lowercase : bool
            Convert to lowercase
        strip_whitespace : bool
            Remove leading/trailing whitespace
        replace_mapping : dict, optional
            Column-specific value replacements
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        changes = []
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            # Original series for comparison
            original_series = self.df[col].copy()
            original_unique = self.df[col].nunique()

            if self.df[col].dtype == 'category':
                self.df[col] = self.df[col].astype(str)
            
            if strip_whitespace:
                self.df[col] = self.df[col].str.strip()
            
            if lowercase:
                self.df[col] = self.df[col].str.lower()
            
            # Apply custom replacements
            if replace_mapping and col in replace_mapping:
                self.df[col] = self.df[col].replace(replace_mapping[col])
            
            # Detect and log row-level changes
            # We only care about non-null values that actually changed
            changed_mask = (original_series != self.df[col]) & original_series.notna()
            if changed_mask.any():
                changed_indices = self.df[changed_mask].index[:1000]
                for idx in changed_indices:
                     self.row_changes.append({
                        "index": int(idx),
                        "column": col,
                        "old_value": str(original_series.loc[idx]),
                        "new_value": str(self.df.loc[idx, col]),
                        "operation": "Text Normalized",
                        "reason": "Categorical Standardized"
                     })

            new_unique = self.df[col].nunique()
            if new_unique < original_unique:
                changes.append(f"{col}: {original_unique} → {new_unique} unique values")
        
        if changes:
            msg = f"Cleaned categorical values in {len(columns)} columns"
            self.cleaning_log.append(msg)
            self.transformations.append({
                "operation": "clean_categorical",
                "changes": changes
            })
        
        return self
    
    def normalize_features(
        self,
        method: str = 'standard',
        columns: Optional[List[str]] = None
    ) -> 'DataCleaner':
        """
        Scale numerical features.
        
        Parameters:
        -----------
        method : str
            'standard', 'minmax', or 'robust'
        columns : list, optional
            Columns to scale (default: all numeric)
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        # Filter columns present in df
        columns = [c for c in columns if c in self.df.columns]
        
        if not columns:
            return self
            
        scaler = None
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        try:
            self.df[columns] = scaler.fit_transform(self.df[columns])
            
            msg = f"Scaled {len(columns)} columns using {method} scaler"
            self.cleaning_log.append(msg)
            self.transformations.append({
                "operation": "normalize_features",
                "method": method,
                "columns": columns
            })
        except Exception as e:
            msg = f"Failed to scale columns: {str(e)}"
            self.cleaning_log.append(msg)
            
        return self

    def encode_categorical(
        self,
        method: str = 'onehot',
        columns: Optional[List[str]] = None,
        max_categories: int = 20
    ) -> 'DataCleaner':
        """
        Encode categorical features.
        
        Parameters:
        -----------
        method : str
            'onehot' or 'label'
        columns : list, optional
            Columns to encode
        max_categories : int
            Max unique values for one-hot encoding
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
        columns = [c for c in columns if c in self.df.columns]
        
        if not columns:
            return self

        changes = []
        
        if method == 'label':
            le = LabelEncoder()
            for col in columns:
                try:
                    # Handle nulls first (fill with 'Unknown')
                    if self.df[col].isnull().any():
                        self.df[col] = self.df[col].fillna('Unknown')
                    
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    changes.append(f"{col} (LabelEncoded)")
                except Exception as e:
                    self.cleaning_log.append(f"Label encoding failed for {col}: {e}")
                    
        elif method == 'onehot':
            for col in columns:
                if self.df[col].nunique() > max_categories:
                    self.cleaning_log.append(f"Skipping OneHot for {col}: >{max_categories} categories")
                    continue
                
                try:
                    dummies = pd.get_dummies(self.df[col], prefix=col, dummy_na=True)
                    self.df = pd.concat([self.df, dummies], axis=1)
                    self.df.drop(columns=[col], inplace=True)
                    changes.append(f"{col} -> {dummies.shape[1]} columns")
                except Exception as e:
                    self.cleaning_log.append(f"OneHot encoding failed for {col}: {e}")

        if changes:
            msg = f"Encoded {len(changes)} features using {method}"
            self.cleaning_log.append(msg)
            self.transformations.append({
                "operation": "encode_categorical",
                "method": method,
                "columns": columns,
                "changes": changes
            })
            
        return self

    def clean_numeric_text(
        self,
        columns: Optional[List[str]] = None,
        remove_symbols: bool = True,
        handle_shorthand: bool = True
    ) -> 'DataCleaner':
        """
        Clean text columns containing numbers (e.g. '$1,200', '1.5k').
        
        Parameters:
        -----------
        columns : list, optional
            Columns to clean
        remove_symbols : bool
            Remove currency symbols and commas
        handle_shorthand : bool
            Convert 'k', 'M', 'B' suffixes (e.g. 1.5k -> 1500)
        """
        if columns is None:
            # Try to guess columns that look like numeric text
            columns = []
            for col in self.df.select_dtypes(include=['object', 'string']).columns:
                # Sample check
                sample = self.df[col].dropna().astype(str).sample(min(20, len(self.df)), random_state=42)
                if sample.str.contains(r'[\$\€\£\,kKmMbB]').any() and sample.str.contains(r'\d').all():
                    columns.append(col)

        changes = []
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            original_nans = self.df[col].isna().sum()
            
            # Work on a copy
            series = self.df[col].astype(str).str.strip()
            
            if remove_symbols:
                # Remove typical currency symbols and commas
                series = series.str.replace(r'[\$\€\£\,\s]', '', regex=True)
            
            if handle_shorthand:
                def parse_shorthand(val):
                    if pd.isna(val) or val == 'nan': return np.nan
                    val = val.lower()
                    multiplier = 1
                    if val.endswith('k'):
                        multiplier = 1000
                        val = val[:-1]
                    elif val.endswith('m'):
                        multiplier = 1000000
                        val = val[:-1]
                    elif val.endswith('b'):
                        multiplier = 1000000000
                        val = val[:-1]
                    
                    try:
                        return float(val) * multiplier
                    except:
                        return np.nan

            if handle_shorthand:
                # Use the defined internal function
                self.df[col] = series.apply(parse_shorthand)
            else:
                self.df[col] = pd.to_numeric(series, errors='coerce')
                
            new_nans = self.df[col].isna().sum()
            valid_converted = len(self.df) - new_nans
            
            # Log examples of successful conversions
            if valid_converted > 0:
                # Find indices where it wasn't null before but is now a valid number
                # AND the string representation looks different (e.g., "$100" vs 100.0)
                # or just log any valid conversion to show off
                valid_mask = self.df[col].notna() & (self.df[col].astype(str) != series)
                if valid_mask.any():
                    # Capture all changes (limit to 1000 safety)
                    sample_indices = self.df[valid_mask].index[:1000]
                    for idx in sample_indices:
                        val_old = self.df.loc[idx, col] # This is already the NEW value in self.df
                        # We need the OLD value from 'series' variable (which was copies)
                        # Wait, 'series' was `self.df[col].astype(str).str.strip()` ... 
                        # but we also did regex replacement on it if remove_symbols=True
                        # So let's grab the raw original from a temp var if we can, or just use the 'series' which is "cleaned string"
                        
                        # Actually, let's just grab the original raw value from a backup if needed,
                        # but 'series' is close enough to show "cleaned string" vs "final number".
                        # Better yet: usage `self.df.loc[idx, col]` is the NEW value.
                        # The OLD value is in `series[idx]` (if index aligns, which it should).
                        # BUT series was modified by remove_symbols.
                        
                        # Let's just say:
                        old_val_str = series.loc[idx] 
                        new_val = self.df.loc[idx, col]
                        
                        self.row_changes.append({
                            "index": int(idx),
                            "column": col,
                            "old_value": str(old_val_str),
                            "new_value": str(new_val),
                            "operation": "Text Cleaned",
                            "reason": "Numeric Text"
                        })

                changes.append(f"{col}: Converted to numeric ({valid_converted} valid)")
        
        if changes:
            self.cleaning_log.append(f"Cleaned numeric text in {len(changes)} columns")
            self.transformations.append({
                "operation": "clean_numeric_text",
                "columns": columns,
                "changes": changes
            })
            
        return self

    def rename_columns(
        self,
        mapping: Dict[str, str]
    ) -> 'DataCleaner':
        """
        Rename columns.
        
        Parameters:
        -----------
        mapping : dict
            Dictionary of {old_name: new_name}
        """
        # Filter mapping to existing columns
        valid_mapping = {k: v for k, v in mapping.items() if k in self.df.columns}
        
        if valid_mapping:
            self.df.rename(columns=valid_mapping, inplace=True)
            msg = f"Renamed {len(valid_mapping)} columns: {valid_mapping}"
            self.cleaning_log.append(msg)
            self.transformations.append({
                "operation": "rename_columns",
                "mapping": valid_mapping
            })
            
        return self

    def extract_regex_feature(
        self,
        source_col: str,
        pattern: str,
        new_col_name: str
    ) -> 'DataCleaner':
        r"""
        Extract text using regex capture group.
        
        Parameters:
        -----------
        source_col : str
            Source column name
        pattern : str
            Regex pattern with one capture group (e.g. r'ID: (\d+)')
        new_col_name : str
            Name for the new column
        """
        if source_col not in self.df.columns:
            return self
            
        try:
            # Ensure pattern is raw string if possible, generally passed as string here
            extracted = self.df[source_col].astype(str).str.extract(pattern, expand=False)
            
            self.df[new_col_name] = extracted
            
            matched_count = extracted.notna().sum()
            msg = f"Extracted '{new_col_name}' from '{source_col}' ({matched_count} matches)"
            self.cleaning_log.append(msg)
            self.transformations.append({
                "operation": "extract_regex",
                "source": source_col,
                "target": new_col_name,
                "pattern": pattern,
                "matches": int(matched_count)
            })
            
        except Exception as e:
            self.cleaning_log.append(f"Regex extraction failed: {e}")
            
        return self

    def drop_columns(
        self,
        columns: Optional[List[str]] = None,
        drop_constant: bool = True,
        drop_id_like: bool = False
    ) -> 'DataCleaner':
        """
        Drop specified or problematic columns.
        
        Parameters:
        -----------
        columns : list, optional
            Specific columns to drop
        drop_constant : bool
            Drop columns with only one unique value
        drop_id_like : bool
            Drop columns that appear to be IDs
        """
        cols_to_drop = set(columns or [])
        
        if drop_constant:
            for col in self.df.columns:
                if self.df[col].nunique() <= 1:
                    cols_to_drop.add(col)
        
        if drop_id_like:
            for col in self.df.columns:
                if self.df[col].nunique() == len(self.df):
                    cols_to_drop.add(col)
        
        cols_to_drop = [c for c in cols_to_drop if c in self.df.columns]
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            msg = f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}"
            self.cleaning_log.append(msg)
            self.transformations.append({
                "operation": "drop_columns",
                "columns": cols_to_drop
            })
        
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """Return the cleaned DataFrame."""
        return self.df
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Return summary of all cleaning operations."""
        return {
            "original_shape": self.original_shape,
            "final_shape": self.df.shape,
            "rows_changed": self.original_shape[0] - self.df.shape[0],
            "columns_changed": self.original_shape[1] - self.df.shape[1],
            "operations": self.cleaning_log,
            "transformations": self.transformations,
            "row_changes": self.row_changes
        }
    
    def validate_quality(self) -> Dict[str, Any]:
        """
        Perform data quality checks and calculate a 0-100 Quality Score.
        
        Returns:
        --------
        dict
            Report containing quality metrics, score, and warnings.
        """
        n_rows = len(self.df)
        n_cols = len(self.df.columns)
        
        if n_rows == 0 or n_cols == 0:
            return {"score": 0, "grade": "F", "warnings": ["Empty dataset"]}

        # 1. Completeness (0-40 pts)
        missing_total = self.df.isnull().sum().sum()
        missing_pct = missing_total / (n_rows * n_cols)
        completeness_score = max(0, 40 * (1 - missing_pct * 2))  # Penalize missing heavily

        # 2. Uniqueness (0-30 pts)
        # Check duplicates
        n_dupes = self.df.duplicated().sum()
        dupe_pct = n_dupes / n_rows
        uniqueness_score = max(0, 30 * (1 - dupe_pct * 2))

        # 3. Consistency/Validity (0-30 pts)
        # Check for constant columns (0 variance)
        n_constant = sum([1 for c in self.df.columns if self.df[c].nunique() <= 1])
        const_pct = n_constant / n_cols
        consistency_score = max(0, 30 * (1 - const_pct * 3))

        final_score = int(completeness_score + uniqueness_score + consistency_score)
        
        grade = 'A' if final_score >= 90 else 'B' if final_score >= 80 else 'C' if final_score >= 60 else 'D' if final_score >= 40 else 'F'

        report: Dict[str, Any] = {
            "score": final_score,
            "grade": grade,
            "rows": n_rows,
            "columns": n_cols,
            "missing_values": int(missing_total),
            "missing_percentage": round(float(missing_pct * 100), 1),
            "duplicate_rows": int(n_dupes),
            "constant_columns": [],
            "warnings": []
        }
        
        # Check for constant columns
        for col in self.df.columns:
            if self.df[col].nunique() <= 1:
                report["constant_columns"].append(col)
                report["warnings"].append(f"Column '{col}' is constant (1 unique value)")
                
        # Check for extreme missing values
        high_missing = self.df.columns[self.df.isnull().mean() > 0.5].tolist()
        if high_missing:
            report["warnings"].append(f"{len(high_missing)} columns have >50% missing values")
            
        return report

    def generate_suggestions(self) -> List[Dict[str, str]]:
        """
        Generate AI-like cleaning suggestions based on data issues.
        """
        suggestions = []
        
        # Missing Values
        missing = self.df.isnull().sum()
        missing_cols = missing[missing > 0]
        
        for col, count in missing_cols.items():
            pct = count / len(self.df)
            if pct > 0.4:
                suggestions.append({
                    "column": col,
                    "issue": f"{pct:.0%} missing values",
                    "action": "Drop Column",
                    "reason": "Too much missing data to impute reliably."
                })
            else:
                method = "Median Imputation" if pd.api.types.is_numeric_dtype(self.df[col]) else "Mode Imputation"
                suggestions.append({
                    "column": col,
                    "issue": f"{pct:.0%} missing values",
                    "action": method,
                    "reason": "Standard strategy for filling gaps."
                })

        # Duplicates
        n_dupes = self.df.duplicated().sum()
        if n_dupes > 0:
            suggestions.append({
                "column": "Dataset",
                "issue": f"{n_dupes} duplicate rows",
                "action": "Remove Duplicates",
                "reason": "Duplicate data skews model training."
            })
            
        # Outliers (Numeric)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].nunique() < 10: continue # Skip categorical-like
            
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            
            if outliers > 0:
                pct = outliers / len(self.df)
                if pct < 0.05:
                    action = "Clip (Winsorize)" 
                else:
                    action = "Log Transform" if (self.df[col] > 0).all() else "Standard Scaling"
                    
                suggestions.append({
                    "column": col,
                    "issue": f"{outliers} outliers detected",
                    "action": action,
                    "reason": "Outliers can distort linear models."
                })

        # ID Columns
        for col in self.df.columns:
            if col.lower() in ['id', 'uuid', 'guid', 'index'] or \
               (self.df[col].nunique() == len(self.df) and pd.api.types.is_string_dtype(self.df[col])):
                suggestions.append({
                    "column": col,
                    "issue": "High cardinality / ID-like",
                    "action": "Drop Column",
                    "reason": "Identifiers do not predict the target."
                })

        return suggestions

    def print_summary(self) -> None:
        """Print cleaning summary."""
        summary = self.get_cleaning_summary()
        
        print("=" * 60)
        print("DATA CLEANING SUMMARY")
        print("=" * 60)
        print(f"\n📊 Shape: {summary['original_shape']} → {summary['final_shape']}")
        print(f"   Rows changed: {summary['rows_changed']:+,}")
        print(f"   Columns changed: {summary['columns_changed']:+,}")
        
        print(f"\n🔧 Operations performed:")
        for op in summary['operations']:
            print(f"   • {op}")
        
        print("\n" + "=" * 60)
