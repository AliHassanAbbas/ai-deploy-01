r"""Free-tier-friendly sentiment API: ONNX Runtime, no PyTorch.

Serves the model exported in Tutorial #3 (models/sentiment_transformer/
model.onnx). Deliberately torch-free: on tiny cloud instances (512 MB),
importing torch alone risks an out-of-memory kill; onnxruntime + numpy
fit comfortably.

Run from the project root:
    uvicorn scripts.serve_onnx:app --host 0.0.0.0 --port 8000
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel

BUNDLE = Path(__file__).resolve().parent.parent / "models" / "sentiment_transformer"


def encode(text: str, vocab: dict, max_len: int) -> list[int]:
    """Text -> fixed-length ids (same logic as src/textdata.py, torch-free)."""
    unk = vocab["<unk>"]
    ids = [vocab.get(w, unk) for w in text.lower().split()][:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(BUNDLE / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    app.state.vocab = config["vocab"]
    app.state.max_len = config["architecture"]["max_len"]
    app.state.class_names = config["class_names"]
    app.state.session = ort.InferenceSession(
        str(BUNDLE / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    print("ONNX model loaded")
    yield


app = FastAPI(title="Sentiment API (ONNX, free-tier friendly)", lifespan=lifespan)


class TextIn(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Sentiment Transformer API - try POST /predict or open /docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: TextIn):
    ids = encode(body.text, app.state.vocab, app.state.max_len)
    input_ids = np.array([ids], dtype=np.int64)
    logits = app.state.session.run(None, {"input_ids": input_ids})[0][0]
    probs = softmax(logits)
    class_id = int(probs.argmax())
    return {
        "label": app.state.class_names[class_id],
        "confidence": round(float(probs[class_id]), 4),
    }