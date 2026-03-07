"""
Data Pipeline Tool - Example Usage
==================================
This script demonstrates how to use the data pipeline tool
for data cleaning, EDA, and feature engineering.
"""

import pandas as pd
import numpy as np
from data_pipeline import DataPipeline

# ============================================================
# QUICK START - Run full pipeline with one command
# ============================================================

def quick_start_example(file_path: str, target_col: str, problem_type: str):
    """
    Quick start example - run entire pipeline with one command.
    
    Parameters:
    -----------
    file_path : str
        Path to your CSV or Excel file
    target_col : str
        Name of target/label column
    problem_type : str
        'classification', 'regression', or 'clustering'
    """
    # Initialize and run
    pipeline = DataPipeline()
    pipeline.load(file_path)
    
    cleaned_df, final_df = pipeline.run_full_pipeline(
        target_col=target_col,
        problem_type=problem_type,
        show_eda_plots=True
    )
    
    # Save results
    pipeline.save_data("./output", format='csv')
    
    return cleaned_df, final_df


# ============================================================
# STEP-BY-STEP - More control over each stage
# ============================================================

def step_by_step_example(file_path: str):
    """
    Step-by-step example with full control over each stage.
    """
    pipeline = DataPipeline()
    
    # ----- Step 1: Load and Validate -----
    print("STEP 1: Loading data...")
    pipeline.load(file_path)
    validation_report = pipeline.validate()
    
    # ----- Step 2: Clean (with custom settings) -----
    print("\nSTEP 2: Cleaning data...")
    pipeline.clean(
        remove_duplicates=True,
        handle_missing=True,
        missing_numeric_strategy='median',      # Options: 'mean', 'median', 'zero'
        missing_categorical_strategy='mode',     # Options: 'mode', 'unknown'
        missing_drop_threshold=0.4,              # Drop cols with >40% missing
        fix_types=True,
        handle_outliers=True,
        outlier_method='iqr',                    # Options: 'iqr', 'zscore'
        outlier_action='clip',                   # Options: 'clip', 'remove', 'nan'
        clean_categorical=True,
        drop_constant=True
    )
    
    # Get cleaned data at any point
    cleaned_df = pipeline.get_cleaned_data()
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    # ----- Step 3: EDA -----
    print("\nSTEP 3: Running EDA...")
    eda_results = pipeline.analyze(
        target_col='target_column_name',  # Replace with your target
        show_plots=True,
        save_plots=False
    )
    
    # ----- Step 4: Feature Engineering -----
    print("\nSTEP 4: Engineering features...")
    pipeline.engineer_features(
        target_col='target_column_name',        # Replace with your target
        problem_type='classification',           # Options: 'classification', 'regression'
        encode_categorical=True,
        scale_features=True,
        scale_method='standard',                 # Options: 'standard', 'minmax'
        create_datetime_features=True,
        create_polynomial_features=False,        # Enable for interaction terms
        drop_low_variance=True,
        drop_high_correlation=True,
        correlation_threshold=0.95,
        handle_imbalance=False                   # Enable for imbalanced classification
    )
    
    # Get final model-ready data
    final_df = pipeline.get_final_data()
    print(f"Final data shape: {final_df.shape}")
    
    return pipeline


# ============================================================
# USING INDIVIDUAL MODULES
# ============================================================

def individual_modules_example(file_path: str):
    """
    Example using individual modules for maximum flexibility.
    """
    from data_pipeline import DataLoader, DataCleaner, EDAAnalyzer, FeatureEngineer
    
    # ----- Data Loader -----
    loader = DataLoader()
    df = loader.load(file_path)
    loader.validate()
    loader.print_summary()
    
    # ----- Data Cleaner -----
    cleaner = DataCleaner(df)
    cleaner.remove_duplicates()
    cleaner.handle_missing_values(
        numeric_strategy='median',
        categorical_strategy='mode'
    )
    cleaner.fix_data_types()
    cleaner.handle_outliers(method='iqr', action='clip')
    cleaner.clean_categorical_values(lowercase=True, strip_whitespace=True)
    cleaner.drop_columns(drop_constant=True)
    cleaner.print_summary()
    
    cleaned_df = cleaner.get_cleaned_data()
    
    # ----- EDA Analyzer -----
    eda = EDAAnalyzer(cleaned_df, target_col='your_target')
    eda.summary_statistics()
    eda.categorical_summary()
    eda.correlation_analysis(threshold=0.7)
    eda.distribution_analysis()
    eda.target_analysis()
    
    # Generate plots
    eda.plot_distributions()
    eda.plot_boxplots()
    eda.plot_correlation_heatmap()
    eda.plot_categorical()
    
    insights = eda.get_insights()
    print("Insights:", insights)
    
    # ----- Feature Engineer -----
    engineer = FeatureEngineer(
        cleaned_df,
        target_col='your_target',
        problem_type='classification'
    )
    
    engineer.encode_categorical(method='auto')
    engineer.scale_features(method='standard')
    engineer.create_datetime_features()
    engineer.drop_low_variance_features()
    engineer.drop_highly_correlated(threshold=0.95)
    engineer.compute_feature_importance()
    engineer.print_summary()
    
    final_df = engineer.get_transformed_data()
    
    return final_df


# ============================================================
# WORKING WITH DATAFRAME DIRECTLY (No file)
# ============================================================

def dataframe_example():
    """
    Example starting with an existing DataFrame instead of file.
    """
    # Create sample data
    df = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000) * 10,
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Add some missing values
    df.loc[0:50, 'feature1'] = np.nan
    df.loc[100:120, 'category'] = np.nan
    
    # Run pipeline on DataFrame
    pipeline = DataPipeline()
    pipeline.load(df)  # Pass DataFrame directly
    
    cleaned_df, final_df = pipeline.run_full_pipeline(
        target_col='target',
        problem_type='classification',
        show_eda_plots=False
    )
    
    return final_df


# ============================================================
# MAIN - Run examples
# ============================================================

if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # Option 1: Quick start with your data
    # cleaned_df, final_df = quick_start_example(
    #     file_path="your_data.csv",
    #     target_col="target_column",
    #     problem_type="classification"  # or "regression"
    # )
    
    # Option 2: Step-by-step with custom settings
    # pipeline = step_by_step_example("your_data.csv")
    
    # Option 3: Using individual modules
    # final_df = individual_modules_example("your_data.csv")
    
    # Option 4: Working with DataFrame directly
    # final_df = dataframe_example()
    
    print("=" * 60)
    print("DATA PIPELINE TOOL READY")
    print("=" * 60)
    print("""
Usage:
------
1. Import the pipeline:
   from data_pipeline import DataPipeline

2. Load your data:
   pipeline = DataPipeline()
   pipeline.load("your_data.csv")

3. Run full pipeline:
   cleaned_df, final_df = pipeline.run_full_pipeline(
       target_col="your_target",
       problem_type="classification"  # or "regression"
   )

4. Or run steps individually:
   pipeline.validate()
   pipeline.clean()
   pipeline.analyze()
   pipeline.engineer_features()

5. Get your data:
   cleaned_df = pipeline.get_cleaned_data()
   final_df = pipeline.get_final_data()

6. Save results:
   pipeline.save_data("./output", format='csv')
    """)
