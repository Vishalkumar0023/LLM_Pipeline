import requests
import json

BASE_URL = 'http://127.0.0.1:8080'

def test_sales_prediction():
    print("\nTesting Sales Prediction...")
    url = f"{BASE_URL}/api/predict"
    payload = {
        "model_type": "sales",
        "features": {
            "TV_Ad_Budget": 150,
            "Radio_Ad_Budget": 25,
            "Newspaper_Ad_Budget": 50
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Success:", response.json())
        else:
            print("❌ Failed:", response.status_code, response.text)
    except Exception as e:
        print("❌ Error:", e)

def test_student_prediction():
    print("\nTesting Student Prediction...")
    url = f"{BASE_URL}/api/predict"
    payload = {
        "model_type": "student",
        "features": {
            "Study_Hours": 5,
            "Attendance_Percentage": 80,
            "Previous_Score": 75
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Success:", response.json())
        else:
            print("❌ Failed:", response.status_code, response.text)
    except Exception as e:
        print("❌ Error:", e)

if __name__ == "__main__":
    print(f"Testing API at {BASE_URL}")
    test_sales_prediction()
    test_student_prediction()
