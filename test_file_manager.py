import requests
import json

BASE_URL = 'http://127.0.0.1:8080'

def test_file_manager():
    session = requests.Session()
    username = 'test_user_fm'
    password = 'TestPass123'
    
    # Signup/Login
    print(f"Creating user {username}...")
    session.post(f"{BASE_URL}/signup", data={
        'username': username,
        'email': 'test_fm@example.com',
        'password': password
    })
    
    login = session.post(f"{BASE_URL}/login", data={'username': username, 'password': password})
    if login.status_code != 200:
        print("Login failed")
        return

    # Upload 1
    files = {'file': ('dummy1.csv', 'a,b,c\n1,2,3')}
    res = session.post(f"{BASE_URL}/upload", files=files)
    print(f"Upload 1 Status: {res.status_code}")
    res = session.post(f"{BASE_URL}/process", json={'dataset_name': 'D1', 'original_filename': 'dummy1.csv'})
    print(f"Process 1 Status: {res.status_code}")
    
    # Upload 2
    files = {'file': ('dummy2.csv', 'x,y,z\n4,5,6')}
    res = session.post(f"{BASE_URL}/upload", files=files)
    print(f"Upload 2 Status: {res.status_code}")
    res = session.post(f"{BASE_URL}/process", json={'dataset_name': 'D2', 'original_filename': 'dummy2.csv'})
    print(f"Process 2 Status: {res.status_code}")
    
    # List files
    print("\n--- Listing Files ---")
    res = session.get(f"{BASE_URL}/api/user_files")
    print(f"Status: {res.status_code}")
    files = res.json()
    print(f"Files found: {len(files)}")
    for f in files:
        print(f" - {f['name']} ({f['type']})")
        
    # We expect: 
    # 1. cleaned_...csv (D1)
    # 2. final_...csv (D1)
    # 3. cleaned_...csv (D2)
    # 4. final_...csv (D2)
    # Total >= 4 (temp_upload.csv is deleted by process endpoint)
    if len(files) < 4:
        print(f"Error: Expected at least 4 files, got {len(files)}")
        return

    # Delete 2 files
    to_delete = [files[1]['name'], files[2]['name']] # Skip first potentially (temp)
    print(f"\n--- Deleting {to_delete} ---")
    res = session.post(f"{BASE_URL}/api/delete_files", json={'filenames': to_delete})
    print(f"Status: {res.status_code}")
    print(res.json())
    
    # List again
    print("\n--- Listing Files After Delete ---")
    res = session.get(f"{BASE_URL}/api/user_files")
    files_after = res.json()
    print(f"Files found: {len(files_after)}")
    
    if len(files_after) == len(files) - 2:
        print("SUCCESS: 2 files deleted")
    else:
        print(f"FAILURE: File count mismatch (Expected {len(files)-2}, got {len(files_after)})")

if __name__ == '__main__':
    test_file_manager()
