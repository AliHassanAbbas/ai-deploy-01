"""Dependencies: reusable checks/objects that endpoints declare they need.

FastAPI runs each dependency BEFORE the endpoint. If a dependency raises
HTTPException, the endpoint never runs and the client gets that error.
"""
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from app.config import settings


# ---- 1. API-key check ---------------------------------------------------
def verify_api_key(x_api_key: str | None = Header(default=None)):
    """Reject any request that doesn't carry the correct X-API-Key header."""
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Send it in the X-API-Key header.",
        )


# ---- 2. Rate limiting (simple, in-memory, per client IP) -----------------
_request_log: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request):
    """Allow at most settings.rate_limit_per_minute requests per IP per minute."""
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _request_log[ip]

    while window and now - window[0] > 60:   # drop entries older than 60 s
        window.popleft()

    if len(window) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again in a minute.",
        )
    window.append(now)


# ---- 3. Access to the loaded model ----------------------------------------
def get_predictor(request: Request):
    """Hand the endpoint the ONE predictor loaded at startup."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return predictor