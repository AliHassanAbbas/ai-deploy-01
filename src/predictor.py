"""The Predictor: loads the inference bundle ONCE, then answers forever.

Loading a model from disk is slow (milliseconds to seconds). Answering
a question with a loaded model is fast (microseconds to milliseconds).
So we load in __init__ (once, at startup) and keep .predict() cheap.
Every serving framework you will ever use (FastAPI, TorchServe, etc.)
is built around this exact pattern.
"""
import json
from pathlib import Path

import torch

from src.model import MoonClassifier


class Predictor:
    def __init__(self, bundle_dir):
        bundle_dir = Path(bundle_dir)

        # 1. Read the bundle config (architecture + preprocessing + labels).
        with open(bundle_dir / "config.json", encoding="utf-8") as f:
            self.config = json.load(f)

        arch = self.config["architecture"]
        prep = self.config["preprocessing"]

        # 2. Rebuild the empty model with the SAME architecture...
        self.model = MoonClassifier(
            input_dim=arch["input_dim"],
            hidden_dim=arch["hidden_dim"],
            num_classes=arch["num_classes"],
        )

        # 3. ...then pour the learned weights into it.
        state_dict = torch.load(
            bundle_dir / "model_weights.pt",
            map_location="cpu",   # load on CPU even if trained on GPU
            weights_only=True,    # safety: only tensors, no arbitrary code
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()         # switch off training-only behavior

        # 4. Preprocessing constants, ready as tensors.
        self.mean = torch.tensor(prep["mean"], dtype=torch.float32)
        self.std = torch.tensor(prep["std"], dtype=torch.float32)
        self.class_names = self.config["class_names"]

    @torch.no_grad()   # inference only: skip gradient bookkeeping (faster)
    def predict(self, points):
        """Classify one point [x, y] or a batch [[x1,y1], [x2,y2], ...].

        Always returns a LIST of result dicts, one per input point:
        {"class_id": int, "class_name": str, "probabilities": [float, float]}
        """
        x = torch.tensor(points, dtype=torch.float32)
        if x.ndim == 1:              # single point -> make it a batch of 1
            x = x.unsqueeze(0)

        x = (x - self.mean) / self.std          # SAME normalization as training
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)    # scores -> probabilities
        class_ids = probs.argmax(dim=1)

        results = []
        for i in range(len(x)):
            cid = int(class_ids[i])
            results.append({
                "class_id": cid,
                "class_name": self.class_names[cid],
                "probabilities": [round(p, 4) for p in probs[i].tolist()],
            })
        return results