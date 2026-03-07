"""
Data Pipeline Module
====================
Main orchestrator that combines all modules into a unified pipeline.
"""

import pandas as pd
import numpy as np
import os
import io
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from .data_loader import DataLoader
from .data_cleaner import DataCleaner
# EDAAnalyzer and FeatureEngineer are imported lazily inside methods
# to avoid loading matplotlib/seaborn/scipy at module import time
from .model_trainer import ModelTrainer
from .report_generator import ReportGenerator


class DataPipeline:
    """
    Complete data pipeline for cleaning, EDA, and feature engineering.
    
    This class orchestrates all data processing steps from raw data
    to model-ready features.
    
    Example:
    --------
    >>> pipeline = DataPipeline()
    >>> pipeline.load("data.csv")
    >>> pipeline.run_full_pipeline(target_col="price", problem_type="regression")
    >>> cleaned_df = pipeline.get_cleaned_data()
    >>> final_df = pipeline.get_final_data()
    """
    
    def __init__(self):
        """Initialize the data pipeline."""
        self.loader: Optional[DataLoader] = None
        self.cleaner: Optional[DataCleaner] = None
        self.eda: Optional['EDAAnalyzer'] = None
        self.engineer: Optional['FeatureEngineer'] = None
        self.trainer: Optional[ModelTrainer] = None
        
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.final_df: Optional[pd.DataFrame] = None
        
        self.target_col: Optional[str] = None
        self.problem_type: Optional[str] = None
        
        self.pipeline_report: Dict[str, Any] = {}
        self.model_results: Optional[Dict[str, Any]] = None
    
    def load(
        self,
        source,
        **kwargs
    ) -> 'DataPipeline':
        """
        Load dataset from file or DataFrame.
        
        Parameters:
        -----------
        source : str or pd.DataFrame
            File path or DataFrame
        **kwargs : dict
            Additional arguments for pandas read functions
        """
        self.loader = DataLoader()
        self.raw_df = self.loader.load(source, **kwargs)
        
        print(f"✅ Loaded dataset: {self.raw_df.shape[0]:,} rows × {self.raw_df.shape[1]} columns")
        
        return self
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the loaded dataset.
        
        Returns:
        --------
        dict : Validation report
        """
        if self.loader is None:
            raise ValueError("No data loaded. Call load() first.")
        
        report = self.loader.validate()
        self.loader.print_summary()
        self.pipeline_report['validation'] = report
        
        return report
    
    def clean(
        self,
        remove_duplicates: bool = True,
        handle_missing: bool = True,
        missing_numeric_strategy: str = 'median',
        missing_categorical_strategy: str = 'mode',
        missing_drop_threshold: float = 0.4,
        fix_types: bool = True,
        handle_outliers: bool = True,
        outlier_method: str = 'iqr',
        outlier_action: str = 'clip',
        clean_categorical: bool = True,
        drop_constant: bool = True
    ) -> 'DataPipeline':
        """
        Clean the dataset.
        
        Parameters:
        -----------
        (See DataCleaner for parameter descriptions)
        """
        if self.raw_df is None:
            raise ValueError("No data loaded. Call load() first.")
        
        self.cleaner = DataCleaner(self.raw_df)
        
        if remove_duplicates:
            self.cleaner.remove_duplicates()
        
        if handle_missing:
            self.cleaner.handle_missing_values(
                numeric_strategy=missing_numeric_strategy,
                categorical_strategy=missing_categorical_strategy,
                drop_threshold=missing_drop_threshold
            )
        
        if fix_types:
            self.cleaner.fix_data_types()
        
        if handle_outliers:
            self.cleaner.handle_outliers(
                method=outlier_method,
                action=outlier_action
            )
        
        if clean_categorical:
            self.cleaner.clean_categorical_values()
        
        if drop_constant:
            self.cleaner.drop_columns(drop_constant=True)
        
        self.cleaned_df = self.cleaner.get_cleaned_data()
        self.cleaner.print_summary()
        self.pipeline_report['cleaning'] = self.cleaner.get_cleaning_summary()
        
        return self
    
    def analyze(
        self,
        target_col: Optional[str] = None,
        show_plots: bool = True,
        save_plots: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform exploratory data analysis.
        
        Parameters:
        -----------
        target_col : str, optional
            Target column for analysis
        show_plots : bool
            Whether to display plots
        save_plots : bool
            Whether to save plots to files
        output_dir : str, optional
            Directory to save plots
        """
        df = self.cleaned_df if self.cleaned_df is not None else self.raw_df
        
        if df is None:
            raise ValueError("No data available. Call load() first.")
        
        if target_col:
            self.target_col = target_col
        
        from .eda import EDAAnalyzer
        self.eda = EDAAnalyzer(df, target_col=self.target_col)
        results = self.eda.run_full_analysis(
            show_plots=show_plots,
            save_plots=save_plots,
            output_dir=output_dir
        )
        
        self.pipeline_report['eda'] = results
        
        return results
    
    def engineer_features(
        self,
        target_col: Optional[str] = None,
        problem_type: Optional[str] = None,
        encode_categorical: bool = True,
        scale_features: bool = True,
        scale_method: str = 'standard',
        create_datetime_features: bool = True,
        create_polynomial_features: bool = False,
        polynomial_degree: int = 2,
        drop_low_variance: bool = True,
        drop_high_correlation: bool = True,
        correlation_threshold: float = 0.95,
        handle_imbalance: bool = False,
        imbalance_method: str = 'smote',
        auto_evolve_features: bool = False
    ) -> 'DataPipeline':
        """
        Perform feature engineering.
        
        Parameters:
        -----------
        (See FeatureEngineer for parameter descriptions)
        """
        df = self.cleaned_df if self.cleaned_df is not None else self.raw_df
        
        if df is None:
            raise ValueError("No data available. Call load() first.")
        
        if target_col:
            self.target_col = target_col
        if problem_type:
            self.problem_type = problem_type
        
        from .feature_engineer import FeatureEngineer
        self.engineer = FeatureEngineer(
            df, 
            target_col=self.target_col,
            problem_type=self.problem_type
        )
        
        if create_datetime_features:
            self.engineer.create_datetime_features()
        
        if encode_categorical:
            self.engineer.encode_categorical()
        
        if create_polynomial_features:
            self.engineer.create_polynomial_features(degree=polynomial_degree)

        if auto_evolve_features:
            self.engineer.auto_evolve()
        
        if drop_low_variance:
            self.engineer.drop_low_variance_features()
        
        if drop_high_correlation:
            self.engineer.drop_highly_correlated(threshold=correlation_threshold)
        
        if scale_features:
            self.engineer.scale_features(method=scale_method)
        
        if handle_imbalance and self.problem_type == 'classification':
            self.engineer.handle_class_imbalance(method=imbalance_method)
        
        # Compute feature importance if target specified
        if self.target_col:
            self.engineer.compute_feature_importance()
        
        self.final_df = self.engineer.get_transformed_data()
        self.engineer.print_summary()
        self.pipeline_report['feature_engineering'] = self.engineer.get_summary()
        
        return self
    
    def run_full_pipeline(
        self,
        target_col: Optional[str] = None,
        problem_type: Optional[str] = None,
        show_eda_plots: bool = True,
        cleaning_config: Optional[Dict[str, Any]] = None,
        feature_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete pipeline from validation to feature engineering.
        
        Parameters:
        -----------
        target_col : str, optional
            Target column name
        problem_type : str, optional
            'classification', 'regression', or 'clustering'
        show_eda_plots : bool
            Whether to show EDA visualizations
        cleaning_config : dict, optional
            Override cleaning parameters
        feature_config : dict, optional
            Override feature engineering parameters
            
        Returns:
        --------
        tuple : (cleaned_df, final_df)
        """
        self.target_col = target_col
        self.problem_type = problem_type
        
        print("\n" + "=" * 70)
        print("🚀 STARTING DATA PIPELINE")
        print("=" * 70)
        
        # Step 1: Validation
        print("\n📋 STEP 1: DATA VALIDATION")
        print("-" * 40)
        self.validate()
        
        # Step 2: Cleaning
        print("\n🧹 STEP 2: DATA CLEANING")
        print("-" * 40)
        clean_params = cleaning_config or {}
        self.clean(**clean_params)
        
        # Step 3: EDA
        print("\n📊 STEP 3: EXPLORATORY DATA ANALYSIS")
        print("-" * 40)
        self.analyze(target_col=target_col, show_plots=show_eda_plots)
        
        # Step 4: Feature Engineering
        print("\n⚙️  STEP 4: FEATURE ENGINEERING")
        print("-" * 40)
        feature_params = feature_config or {}
        feature_params['target_col'] = target_col
        feature_params['problem_type'] = problem_type
        self.engineer_features(**feature_params)
        
        # Summary
        self._print_pipeline_summary()
        
        return self.cleaned_df, self.final_df
    
    def _print_pipeline_summary(self) -> None:
        """Print final pipeline summary."""
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE")
        print("=" * 70)
        
        if self.raw_df is not None:
            print(f"\n📊 Data Transformation:")
            print(f"   Raw:     {self.raw_df.shape[0]:,} rows × {self.raw_df.shape[1]} columns")
        
        if self.cleaned_df is not None:
            print(f"   Cleaned: {self.cleaned_df.shape[0]:,} rows × {self.cleaned_df.shape[1]} columns")
        
        if self.final_df is not None:
            print(f"   Final:   {self.final_df.shape[0]:,} rows × {self.final_df.shape[1]} columns")
        
        print(f"\n🎯 Target: {self.target_col or 'Not specified'}")
        print(f"📈 Problem Type: {self.problem_type or 'Not specified'}")
        
        # Data quality check
        if self.final_df is not None:
            missing = self.final_df.isnull().sum().sum()
            print(f"\n✅ Final Data Quality:")
            print(f"   • Missing values: {missing}")
            print(f"   • Ready for modeling: {'Yes' if missing == 0 else 'No (handle remaining missing)'}")
        
        print("\n" + "=" * 70)
    
    def get_raw_data(self) -> Optional[pd.DataFrame]:
        """Return the raw DataFrame."""
        return self.raw_df
    
    def get_cleaned_data(self) -> Optional[pd.DataFrame]:
        """Return the cleaned DataFrame."""
        return self.cleaned_df
    
    def get_final_data(self) -> Optional[pd.DataFrame]:
        """Return the feature-engineered DataFrame (model-ready)."""
        return self.final_df
    
    def get_report(self) -> Dict[str, Any]:
        """Return the complete pipeline report."""
        return self.pipeline_report
    
    def save_data(
        self,
        output_dir: str,
        save_cleaned: bool = True,
        save_final: bool = True,
        format: str = 'csv'
    ) -> None:
        """
        Save processed datasets to files.
        
        Parameters:
        -----------
        output_dir : str
            Output directory path
        save_cleaned : bool
            Save cleaned data
        save_final : bool
            Save final engineered data
        format : str
            Output format: 'csv' or 'parquet'
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_cleaned and self.cleaned_df is not None:
            if format == 'csv':
                self.cleaned_df.to_csv(output_path / 'cleaned_data.csv', index=False)
            else:
                self.cleaned_df.to_parquet(output_path / 'cleaned_data.parquet', index=False)
            print(f"✅ Saved cleaned data to {output_path / f'cleaned_data.{format}'}")
        
        if save_final and self.final_df is not None:
            if format == 'csv':
                self.final_df.to_csv(output_path / 'final_data.csv', index=False)
            else:
                self.final_df.to_parquet(output_path / 'final_data.parquet', index=False)
            print(f"✅ Saved final data to {output_path / f'final_data.{format}'}")

    def train_model(
        self,
        target_col: Optional[str] = None,
        problem_type: Optional[str] = None,
        export_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train ML models on the processed data.
        
        Parameters:
        -----------
        target_col : str, optional
            Target column name (auto-detected if None)
        problem_type : str, optional
            'classification' or 'regression' (auto-detected if None)
        export_path : str, optional
            Path to export the best model (.pkl)
        
        Returns:
        --------
        dict : Training results dashboard
        """
        df = self.final_df if self.final_df is not None else self.cleaned_df
        if df is None:
            raise ValueError("No data available. Run clean() or load() first.")
        
        t_col = target_col or self.target_col
        p_type = problem_type or self.problem_type
        
        self.trainer = ModelTrainer(df, target_col=t_col, problem_type=p_type, raw_df=self.raw_df)
        self.model_results = self.trainer.run()
        
        if export_path:
            self.trainer.export_model(export_path)
        
        self.pipeline_report['model_training'] = self.model_results
        return self.model_results

    def generate_html_report(self, output_path: str) -> str:
        """
        Generate a detailed HTML report using ReportGenerator.
        """
        generator = ReportGenerator(self, self.model_results)
        html_content = generator.generate_html()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"✅ Generated detailed HTML report: {output_path}")
        return output_path

    def generate_markdown_report(self, output_path: str) -> str:
        """
        Generate a comprehensive Markdown report of the pipeline run.
        """
        report = []
        report.append(f"# Mini Data Clean Tool - Model Report")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 1. Dataset Overview
        report.append("## 1. Data Processing Journey")
        if self.raw_df is not None:
             report.append(f"- **Raw Dataset:** {self.raw_df.shape[0]:,} rows × {self.raw_df.shape[1]} columns")
        if self.cleaned_df is not None:
             report.append(f"- **Cleaned Dataset:** {self.cleaned_df.shape[0]:,} rows × {self.cleaned_df.shape[1]} columns")
        
        # Pipeline Steps
        if 'preprocessing' in self.pipeline_report:
            report.append("\n### Cleaning Steps Executed:")
            steps = self.pipeline_report['preprocessing'].get('steps_executed', [])
            if steps:
                for step in steps:
                    report.append(f"- {step}")
            else:
                report.append("- No major cleaning issues found.")

        # 2. Model Performance
        report.append("\n## 2. Model Performance Analysis")
        
        if self.model_results and 'comparison' in self.model_results:
            comp = self.model_results['comparison']
            metric_name = comp.get('metric', 'Metric')
            
            report.append("| Model | Score | Reliability |")
            report.append("| :--- | :--- | :--- |")
            
            # Raw Row
            raw_score = comp.get('raw_score', 'N/A')
            raw_rel = comp.get('raw_reliability', {})
            raw_grade = raw_rel.get('grade', 'N/A')
            raw_pts = raw_rel.get('score', 0)
            report.append(f"| **Raw Baseline** | {raw_score}% ({metric_name}) | **{raw_grade}** ({raw_pts}/100) |")
            
            # Cleaned Row
            clean_score = comp.get('cleaned_score', 'N/A')
            # Get reliability from best model metadata or calculate
            # For now, let's grab it from export metadata if available, or approximate
            # Actually, model_trainer export includes it. 
            # We can grab it from best_model dict if available
            clean_grade = 'N/A'
            clean_pts = 0
            
            # Try to dig reliability out of best model
            # This is a bit tricky as it's not directly in comparison dict usually
            # But we can look at reliability score if we re-calculated it or stored it
            # The backend calc logic is in ModelTrainer.
            # For the report sake, let's look at the 'model_training' specific section
            best_model_info = self.model_results.get('best_model', {})
            # We don't have reliability directly here unless we added it to the dict in ModelTrainer.run
            # Let's check ModelTrainer.run output structure. 
            # It returns 'best_model': {... 'metrics': ...}
            # The 'reliability' key is in the *export*, not immediately in run return?
            # Wait, line 832 in ModelTrainer adds 'reliability' to export.
            # We should probably expose it in the main result dict too for easy access.
            
            report.append(f"| **Cleaned Model** | **{clean_score}%** ({metric_name}) | *(See Dashboard)* |")
            
            report.append(f"\n**Improvement:** {comp.get('improvement_pct', 0)}% improvement over baseline.")

        # 3. Key Observations
        report.append("\n## 3. Key Observations")
        report.append("> This model report was generated automatically by the Mini Data Clean Tool.")
        report.append(f"- **Target Variable:** `{self.target_col}`")
        report.append(f"- **Problem Type:** `{self.problem_type}`")
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Generated model report: {output_path}")
        return output_path
