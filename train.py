r"""Train the moon classifier and save a self-contained inference bundle.

Run from the project root:
    python train.py

Produces models\moon_classifier\ containing:
    model_weights.pt  - the learned weights (state_dict)
    config.json       - everything needed to rebuild and use the model
"""
import json

import torch
import torch.nn as nn

from src import config
from src.data import make_moons
from src.model import MoonClassifier


def main():
    torch.manual_seed(config.SEED)

    # ---- 1. Data -------------------------------------------------------
    X, y = make_moons(config.N_SAMPLES, config.NOISE, config.SEED)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # Normalization: shift/scale inputs so training is stable.
    # We MUST remember mean/std - at prediction time, new inputs must be
    # normalized with these exact same numbers, or predictions are garbage.
    mean = X.mean(dim=0)
    std = X.std(dim=0)
    X_norm = (X - mean) / std

    # ---- 2. Model, loss, optimizer --------------------------------------
    model = MoonClassifier(config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_CLASSES)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # ---- 3. Training loop ------------------------------------------------
    model.train()
    for epoch in range(1, config.EPOCHS + 1):
        optimizer.zero_grad()
        logits = model(X_norm)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

    # ---- 4. Final accuracy ----------------------------------------------
    model.eval()
    with torch.no_grad():
        preds = model(X_norm).argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
    print(f"final training accuracy: {accuracy:.1%}")

    # ---- 5. Save the inference bundle -------------------------------------
    config.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    # Weights only - NOT the whole model object.
    torch.save(model.state_dict(), config.BUNDLE_DIR / "model_weights.pt")

    # Everything the predictor needs to rebuild the model and preprocess
    # inputs, in one human-readable file.
    bundle_config = {
        "model_version": config.MODEL_VERSION,
        "architecture": {
            "input_dim": config.INPUT_DIM,
            "hidden_dim": config.HIDDEN_DIM,
            "num_classes": config.NUM_CLASSES,
        },
        "preprocessing": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "class_names": config.CLASS_NAMES,
        "training_accuracy": round(accuracy, 4),
    }
    with open(config.BUNDLE_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(bundle_config, f, indent=2)

    print(f"saved inference bundle to: {config.BUNDLE_DIR}")


if __name__ == "__main__":
    main()