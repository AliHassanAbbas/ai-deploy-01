"""Load-once predictor for the sentiment transformer (Tutorial #1 pattern)."""
import json
from pathlib import Path

import torch

from src.textdata import encode
from src.textmodel import TinySentimentTransformer


class TextPredictor:
    def __init__(self, bundle_dir):
        bundle_dir = Path(bundle_dir)
        with open(bundle_dir / "config.json", encoding="utf-8") as f:
            self.config = json.load(f)

        arch = self.config["architecture"]
        self.vocab = self.config["vocab"]
        self.max_len = arch["max_len"]
        self.class_names = self.config["class_names"]

        self.model = TinySentimentTransformer(
            vocab_size=arch["vocab_size"],
            d_model=arch["d_model"],
            nhead=arch["nhead"],
            num_layers=arch["num_layers"],
            dim_feedforward=arch["dim_feedforward"],
            max_len=arch["max_len"],
            num_classes=arch["num_classes"],
        )
        state = torch.load(
            bundle_dir / "model_weights.pt", map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """ONE forward pass for the whole list of texts."""
        rows = [encode(t, self.vocab, self.max_len) for t in texts]
        input_ids = torch.tensor(rows, dtype=torch.int64)
        probs = torch.softmax(self.model(input_ids), dim=1)
        class_ids = probs.argmax(dim=1)
        return [
            {
                "label": self.class_names[int(class_ids[i])],
                "confidence": round(float(probs[i, class_ids[i]]), 4),
            }
            for i in range(len(texts))
        ]

    def predict_one(self, text: str) -> dict:
        return self.predict_batch([text])[0]