"""
Run Pipeline Script
===================
Edit the variables below and run this file in VS Code.
"""

from data_pipeline import DataPipeline

# ============================================
# EDIT THESE VALUES
# ============================================

FILE_PATH = "Google_Cleaned_Data.csv"        # Your CSV or Excel file path
TARGET_COLUMN = "Size"           # Your target column name
PROBLEM_TYPE = "regression"    # "classification" or "regression"

# ============================================
# RUN PIPELINE
# ============================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Load data
    pipeline.load(FILE_PATH)
    
    # Run full pipeline
    cleaned_df, final_df = pipeline.run_full_pipeline(
        target_col=TARGET_COLUMN,
        problem_type=PROBLEM_TYPE,
        show_eda_plots=True
    )
    
    # Save results to output folder
    pipeline.save_data("./output")
    
    print("\n✅ Done! Check the 'output' folder for cleaned data.")
