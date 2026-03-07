
import requests
import json
import os
import pandas as pd

# Create dummy classification file
with open('classification_demo.csv', 'w') as f:
    f.write("StudyHours,Attendance,Pass\n10,95,1\n9,90,1\n2,40,0\n3,50,0\n8,85,1\n1,30,0\n5,70,1\n4,45,0\n")

BASE_URL = 'http://127.0.0.1:8080'
DEMO_FILE = 'classification_demo.csv'

# Login first
session = requests.Session()
login_payload = {'username': 'testuser', 'password': 'testpassword'}

import time
timestamp = int(time.time())
username = f"testuser_{timestamp}"
login_payload = {'username': username, 'password': 'testpassword', 'email': f"{username}@example.com"}

print(f"1. Registering user {username}...")
r = session.post(f"{BASE_URL}/signup", json=login_payload)
print(f"Signup response: {r.status_code} {r.text}")
r = session.post(f"{BASE_URL}/login", json=login_payload)
if r.status_code != 200:
    print(f"Login failed: {r.text}")
    exit(1)

# Upload file
print("\n2. Uploading file...")
files = {'file': open(DEMO_FILE, 'rb')}
r = session.post(f"{BASE_URL}/upload", files=files)
if r.status_code != 200:
    print(f"Upload failed: {r.text}")
    exit(1)
filename = r.json()['filename']
print(f"File uploaded: {filename}")

# Process
print("\n3. Processing...")
process_payload = {
    "filename": filename,
    "target_column": "Pass",
    "problem_type": "classification"
}
r = session.post(f"{BASE_URL}/process", json=process_payload)
if r.status_code != 200:
    print(f"Process failed: {r.text}")
    exit(1)
data = r.json()
dataset_id = data['dataset_id']
print(f"Dataset created with ID: {dataset_id}")

# Train
print("\n4. Training Model...")
train_payload = {
    "target_column": "Pass",
    "problem_type": "classification"
}
r = session.post(f"{BASE_URL}/dataset/{dataset_id}/train", json=train_payload)
if r.status_code != 200:
    print(f"Training failed: {r.text}")
    print(r.text)
    exit(1)

print("\n=== TRAINING RESULTS ===")
print(json.dumps(r.json(), indent=2))
