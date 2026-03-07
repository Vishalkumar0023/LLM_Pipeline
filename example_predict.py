import joblib
import pandas as pd
import sys
import os

def load_and_predict(model_path, csv_data_path=None):
    """
    Loads a model from a .pkl file and makes predictions.
    If no data is provided, it just checks if the model loads.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    if csv_data_path:
        if not os.path.exists(csv_data_path):
             print(f"Error: Data file '{csv_data_path}' not found.")
             return
        
        print(f"Loading data from {csv_data_path}...")
        try:
            data = pd.read_csv(csv_data_path)
            # Basic check: The model usually expects specific features. 
            # This try/except block handles shape mismatches generally.
            predictions = model.predict(data)
            print("\n🔮 Predictions:")
            print(predictions)
            
            # Save predictions
            output_file = "predictions.csv"
            data['prediction'] = predictions
            data.to_csv(output_file, index=False)
            print(f"\n✅ Predictions saved to {output_file}")
            
        except Exception as e:
             print(f"❌ Error during prediction: {e}")
             print("Tip: Ensure your input CSV has the same columns (features) as the data used to train the model.")
    else:
        print("\nℹ️  To make predictions, provide a CSV file with features:")
        print(f"python {sys.argv[0]} {model_path} your_new_data.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example_predict.py <path_to_model.pkl> [path_to_data.csv]")
    else:
        model_file = sys.argv[1]
        data_file = sys.argv[2] if len(sys.argv) > 2 else None
        load_and_predict(model_file, data_file)
