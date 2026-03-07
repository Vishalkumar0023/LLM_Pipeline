import requests

session = requests.Session()
login_data = {
    "username": "demotest",
    "email": "demotest@example.com",
    "password": "password123",
}

r = session.post("http://127.0.0.1:8080/register", data=login_data)
r = session.post(
    "http://127.0.0.1:8080/login",
    json={"username": "demotest", "password": "password123"},
)

r = session.post(
    "http://127.0.0.1:8080/api/llm/ingest", data={"urls": '["https://example.com"]'}
)
print("Ingest:", r.json())
session_id = r.json().get("session_id")

if session_id:
    r = session.post(
        "http://127.0.0.1:8080/api/llm/process",
        json={"session_id": session_id, "chunk_size": 512, "template": "alpaca"},
    )
    print("Process:", r.json())

    # Export
    r = session.post(
        "http://127.0.0.1:8080/api/llm/export",
        json={
            "session_id": session_id,
            "version": "vTEST",
            "model": "meta-llama/Meta-Llama-3-8B",
            "method": "lora",
        },
    )
    print("Export:", r.status_code, r.json())
