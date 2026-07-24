r"""Train the sentiment transformer and save its inference bundle.

Run from the project root:
    python train_text.py

Produces models\sentiment_transformer\ containing:
    model_weights.pt  - learned weights (state_dict)
    config.json       - architecture + vocab + labels + test accuracy
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.textdata import accuracy, build_vocab, make_dataset
from src.textmodel import TinySentimentTransformer

BUNDLE_DIR = Path(__file__).resolve().parent / "models" / "sentiment_transformer"

MAX_LEN = 32
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 256
EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
SEED = 42


def main():
    torch.manual_seed(SEED)

    vocab = build_vocab()
    X_train, y_train = make_dataset(4000, vocab, MAX_LEN, seed=SEED)
    X_test, y_test = make_dataset(1000, vocab, MAX_LEN, seed=SEED + 1)

    model = TinySentimentTransformer(
        vocab_size=len(vocab), d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FF, max_len=MAX_LEN,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        for i in range(0, len(X_train), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        train_acc = accuracy(model, X_train, y_train)
        model.train()
        print(f"epoch {epoch}  loss {total_loss / len(X_train):.4f}  train acc {train_acc:.1%}")

    test_acc = accuracy(model, X_test, y_test)
    print(f"test accuracy: {test_acc:.1%}")

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), BUNDLE_DIR / "model_weights.pt")

    bundle_config = {
        "model_version": "1.0.0",
        "architecture": {
            "vocab_size": len(vocab),
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FF,
            "max_len": MAX_LEN,
            "num_classes": 2,
        },
        "vocab": vocab,
        "class_names": ["negative", "positive"],
        "test_accuracy": round(test_acc, 4),
    }
    with open(BUNDLE_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(bundle_config, f, indent=2)

    print(f"saved inference bundle to: {BUNDLE_DIR}")


if __name__ == "__main__":
    main()