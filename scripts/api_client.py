r"""A Python program acting as a user of our API.

Run (with the server running in another terminal):
    python scripts\api_client.py
"""
import requests

BASE_URL = "http://127.0.0.1:8000"
HEADERS = {"X-API-Key": "supersecret123"}

# 1. Health check (no key needed)
r = requests.get(f"{BASE_URL}/health")
print("health  :", r.status_code, r.json())

# 2. Single prediction
r = requests.post(f"{BASE_URL}/predict", json={"x": 0.0, "y": 1.0}, headers=HEADERS)
print("predict :", r.status_code, r.json())

# 3. Batch prediction
batch = {"points": [{"x": 0.0, "y": 1.0}, {"x": 1.0, "y": -0.4}, {"x": 0.5, "y": 0.5}]}
r = requests.post(f"{BASE_URL}/predict/batch", json=batch, headers=HEADERS)
print("batch   :", r.status_code, f"count={r.json()['count']}")
for p in r.json()["predictions"]:
    print("   ", p)