import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

DEMO_DATA_DIR = 'demo_data'
DEMO_MODELS_DIR = 'demo_models'

def train_sales_model():
    print("Training Sales Model...")
    data_path = os.path.join(DEMO_DATA_DIR, 'sales_demo.csv')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    # Features: TV_Ad_Budget, Radio_Ad_Budget, Newspaper_Ad_Budget
    X = df[['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget']]
    y = df['Sales']

    model = LinearRegression()
    model.fit(X, y)

    output_path = os.path.join(DEMO_MODELS_DIR, 'sales_model.pkl')
    joblib.dump(model, output_path)
    print(f"Sales model saved to {output_path}")

def train_student_model():
    print("Training Student Marks Model...")
    data_path = os.path.join(DEMO_DATA_DIR, 'student_marks_demo.csv')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    # Features: Study_Hours, Attendance_Percentage, Previous_Score
    X = df[['Study_Hours', 'Attendance_Percentage', 'Previous_Score']]
    y = df['Final_Marks']

    model = LinearRegression()
    model.fit(X, y)

    output_path = os.path.join(DEMO_MODELS_DIR, 'student_model.pkl')
    joblib.dump(model, output_path)
    print(f"Student model saved to {output_path}")

if __name__ == "__main__":
    os.makedirs(DEMO_MODELS_DIR, exist_ok=True)
    train_sales_model()
    train_student_model()
