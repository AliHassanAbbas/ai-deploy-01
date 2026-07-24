"""Data generation for the two-moons toy dataset.

Kept in its own file so train.py stays short, and so tests could
import and reuse the same data function later.
"""
import numpy as np


def make_moons(n_samples: int = 1000, noise: float = 0.15, seed: int = 42):
    """Return (X, y): two interleaving half-circles ("moons").

    X: float32 array of shape (n_samples, 2) - the points.
    y: int64 array of shape (n_samples,)     - 0 = outer moon, 1 = inner moon.
    """
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    theta = np.linspace(0, np.pi, n)

    outer = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    inner = np.stack([1.0 - np.cos(theta), 0.5 - np.sin(theta)], axis=1)

    X = np.concatenate([outer, inner]).astype(np.float32)
    X += rng.normal(scale=noise, size=X.shape).astype(np.float32)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.int64)
    return X, y