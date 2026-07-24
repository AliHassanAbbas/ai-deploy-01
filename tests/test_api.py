r"""API tests. Run from the project root:

    python -m pytest tests\test_api.py -v

TestClient runs the app IN-PROCESS: no server, no port, no network -
requests go straight into FastAPI. Fast enough to run on every save.
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.deps import _request_log
from app.main import app

HEADERS = {"X-API-Key": settings.api_key}


@pytest.fixture(scope="module")
def client():
    # `with` triggers the lifespan: the real model loads once for all tests.
    with TestClient(app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "docs" in r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "1.0.0"


def test_predict_requires_api_key(client):
    r = client.post("/predict", json={"x": 0.0, "y": 1.0})  # no header
    assert r.status_code == 401


def test_predict_ok(client):
    r = client.post("/predict", json={"x": 0.0, "y": 1.0}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["class_name"] == "outer_moon"
    assert len(body["probabilities"]) == 2


def test_predict_rejects_bad_input(client):
    r = client.post("/predict", json={"x": 0.0, "y": "banana"}, headers=HEADERS)
    assert r.status_code == 422


def test_batch(client):
    payload = {"points": [{"x": 0.0, "y": 1.0}, {"x": 1.0, "y": -0.4}]}
    r = client.post("/predict/batch", json=payload, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["predictions"][0]["class_name"] == "outer_moon"
    assert body["predictions"][1]["class_name"] == "inner_moon"


def test_image_upload(client):
    # Build a small dark PNG in memory - no file on disk needed.
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32), color=(10, 10, 10)).save(buffer, format="PNG")
    buffer.seek(0)

    r = client.post(
        "/predict/image",
        files={"file": ("test.png", buffer, "image/png")},
        headers=HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["width"] == 64
    assert body["height"] == 32
    assert body["predicted_label"] == "dark_image"


def test_rate_limit(client):
    old_limit = settings.rate_limit_per_minute
    settings.rate_limit_per_minute = 3
    _request_log.clear()                      # forget earlier tests' requests
    try:
        for _ in range(3):
            assert client.post(
                "/predict", json={"x": 0.0, "y": 1.0}, headers=HEADERS
            ).status_code == 200
        r = client.post("/predict", json={"x": 0.0, "y": 1.0}, headers=HEADERS)
        assert r.status_code == 429           # the 4th one is refused
    finally:
        settings.rate_limit_per_minute = old_limit
        _request_log.clear()