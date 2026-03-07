"""
Data Loader Module
==================
Handles loading and initial validation of datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any


class DataLoader:
    """Load and validate datasets from various file formats."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.validation_report: Dict[str, Any] = {}
    
    def load(
        self, 
        source: Union[str, pd.DataFrame], 
        **kwargs
    ) -> pd.DataFrame:
        """
        Load dataset from file path or DataFrame.
        
        Parameters:
        -----------
        source : str or pd.DataFrame
            File path (CSV/Excel) or existing DataFrame
        **kwargs : dict
            Additional arguments passed to pandas read functions
            
        Returns:
        --------
        pd.DataFrame : Loaded dataset
        """
        if isinstance(source, pd.DataFrame):
            self.df = source.copy()
            self.file_path = "DataFrame input"
        elif isinstance(source, str):
            self.file_path = source
            self.df = self._load_from_file(source, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
        
        return self.df
    
    def _load_from_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from file based on extension."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension == '.csv':
            return pd.read_csv(file_path, **kwargs)
        elif extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, **kwargs)
        elif extension == '.json':
            return pd.read_json(file_path, **kwargs)
        elif extension == '.parquet':
            return pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def validate(self) -> Dict[str, Any]:
        """
        Perform initial validation on loaded dataset.
        
        Returns:
        --------
        dict : Validation report with dataset information
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call load() first.")
        
        df = self.df
        
        # Basic info
        self.validation_report = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        }
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        self.validation_report["missing_values"] = {
            "counts": missing[missing > 0].to_dict(),
            "percentages": missing_pct[missing_pct > 0].to_dict(),
            "total_missing_cells": int(missing.sum()),
            "columns_with_missing": int((missing > 0).sum())
        }
        
        # Duplicates
        duplicate_count = df.duplicated().sum()
        self.validation_report["duplicates"] = {
            "count": int(duplicate_count),
            "percentage": round(duplicate_count / len(df) * 100, 2)
        }
        
        # Data type analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
        
        self.validation_report["column_types"] = {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols,
            "boolean": boolean_cols
        }
        
        # Constant/near-constant columns
        constant_cols = []
        near_constant_cols = []
        
        for col in df.columns:
            nunique = df[col].nunique()
            if nunique == 1:
                constant_cols.append(col)
            elif nunique <= 2 and len(df) > 100:
                near_constant_cols.append(col)
        
        self.validation_report["low_variance_columns"] = {
            "constant": constant_cols,
            "near_constant": near_constant_cols
        }
        
        # Potential ID columns (high cardinality)
        potential_id_cols = []
        for col in df.columns:
            if df[col].nunique() == len(df):
                potential_id_cols.append(col)
        
        self.validation_report["potential_id_columns"] = potential_id_cols
        
        return self.validation_report
    
    def print_summary(self) -> None:
        """Print a formatted summary of the validation report."""
        if not self.validation_report:
            self.validate()
        
        report = self.validation_report
        
        print("=" * 60)
        print("DATASET VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 Shape: {report['shape'][0]:,} rows × {report['shape'][1]} columns")
        print(f"💾 Memory Usage: {report['memory_usage_mb']:.2f} MB")
        
        print(f"\n📋 Column Types:")
        for ctype, cols in report['column_types'].items():
            if cols:
                print(f"   • {ctype.capitalize()}: {len(cols)} columns")
        
        missing = report['missing_values']
        if missing['columns_with_missing'] > 0:
            print(f"\n⚠️  Missing Values:")
            print(f"   • Columns affected: {missing['columns_with_missing']}")
            print(f"   • Total missing cells: {missing['total_missing_cells']:,}")
            for col, pct in list(missing['percentages'].items())[:5]:
                print(f"   • {col}: {pct}%")
            if len(missing['percentages']) > 5:
                print(f"   ... and {len(missing['percentages']) - 5} more columns")
        else:
            print(f"\n✅ No missing values detected")
        
        dups = report['duplicates']
        if dups['count'] > 0:
            print(f"\n⚠️  Duplicates: {dups['count']:,} rows ({dups['percentage']}%)")
        else:
            print(f"\n✅ No duplicate rows detected")
        
        if report['low_variance_columns']['constant']:
            print(f"\n⚠️  Constant columns: {report['low_variance_columns']['constant']}")
        
        if report['potential_id_columns']:
            print(f"\n🔑 Potential ID columns: {report['potential_id_columns']}")
        
        print("\n" + "=" * 60)
