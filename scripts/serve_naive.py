r"""NAIVE serving: each request runs its own batch-of-1 forward pass.

Deliberately minimal (no auth/rate limit) so the load test measures the
model-serving strategy, nothing else. Same patterns as Tutorial #2.

Run from the project root:
    uvicorn scripts.serve_naive:app --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.text_predictor import TextPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = TextPredictor("models/sentiment_transformer")
    print("model loaded (naive server)")
    yield


app = FastAPI(title="Sentiment API - naive", lifespan=lifespan)


class TextIn(BaseModel):
    text: str


@app.post("/predict")
def predict(body: TextIn):
    # One request = one forward pass with batch size 1.
    return app.state.predictor.predict_one(body.text)