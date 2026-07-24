r"""Consume the streaming endpoint chunk by chunk, like a chat UI does.

Run (with the server running in another terminal):
    python scripts\stream_client.py
"""
import requests

URL = "http://127.0.0.1:8000/predict/stream"
HEADERS = {"X-API-Key": "supersecret123"}

with requests.get(URL, params={"x": 0.0, "y": 1.0}, headers=HEADERS, stream=True) as r:
    r.raise_for_status()
    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
        # print each piece the moment it arrives, no newline, no buffering
        print(chunk, end="", flush=True)
print()  # final newline