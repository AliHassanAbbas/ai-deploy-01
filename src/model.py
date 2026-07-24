"""The model architecture, and nothing else.

This file must NOT load data, train, or print. It only answers one
question: "what does the network look like?" Both train.py and
predictor.py import it, so training and serving are guaranteed to
use the exact same architecture.
"""
import torch
import torch.nn as nn


class MoonClassifier(nn.Module):
    """A small MLP: 2 inputs -> 16 -> 16 -> 2 class scores (logits)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)