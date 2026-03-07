import requests
import json
import time

BASE_URL = 'http://127.0.0.1:8080'
DEMO_FILE = 'demo_data/sales_demo.csv'

# Login first
session = requests.Session()
login_payload = {'username': 'testuser', 'password': 'testpassword'}

print("1. Registering/Logging in...")
try:
    # Try signup
    r = session.post(f"{BASE_URL}/signup", json=login_payload)
except:
    pass
# Try login
r = session.post(f"{BASE_URL}/login", json=login_payload)
if r.status_code != 200:
    print(f"Login failed: {r.text}")
    exit(1)
print("Logged in.")

# Upload file
print("\n2. Uploading file...")
files = {'file': open(DEMO_FILE, 'rb')}
r = session.post(f"{BASE_URL}/upload", files=files)
if r.status_code != 200:
    print(f"Upload failed: {r.text}")
    exit(1)
print("File uploaded.")

# Process (Train)
print("\n3. Processing (Training)...")
process_payload = {
    "original_filename": "sales_demo.csv",
    "dataset_name": "Sales Test Model",
    "target_column": "Sales",
    "problem_type": "regression"
}
r = session.post(f"{BASE_URL}/process", json=process_payload)
if r.status_code != 200:
    print(f"Processing failed: {r.text}")
    exit(1)
data = r.json()
dataset_id = data['dataset_id']
print(f"Dataset created with ID: {dataset_id}")

# Train Model
print("\n4. Training Model...")
train_payload = {
    "target_column": "Sales",
    "problem_type": "regression"
}
r = session.post(f"{BASE_URL}/dataset/{dataset_id}/train", json=train_payload)
if r.status_code != 200:
    print(f"Training failed: {r.text}")
    exit(1)
print("Model trained.")

# Test User Model Info
print("\n5. Testing /api/user_model_info...")
r = session.get(f"{BASE_URL}/api/user_model_info/{dataset_id}")
if r.status_code == 200:
    print(f"✅ User Model Info Success: {r.json()['features']}")
else:
    print(f"❌ User Model Info Failed: {r.text}")

# Test Prediction
print("\n6. Testing /api/predict_user_model...")
predict_payload = {
    "dataset_id": dataset_id,
    "features": {
        "TV_Ad_Budget": 200,
        "Radio_Ad_Budget": 40,
        "Newspaper_Ad_Budget": 20
    }
}
r = session.post(f"{BASE_URL}/api/predict_user_model", json=predict_payload)
if r.status_code == 200:
    print(f"✅ Prediction Success: {r.json()}")
else:
    print(f"❌ Prediction Failed: {r.text}")
