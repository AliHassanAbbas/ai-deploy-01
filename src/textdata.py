"""Synthetic sentiment dataset: sentences built from real English words.

Sentences are random mixes of positive/negative/neutral words; the label
is the polarity that dominates. Toy, but it trains a REAL transformer
and gives us a held-out set to measure accuracy honestly.
"""
import numpy as np
import torch

POSITIVE = [
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "brilliant", "superb", "delightful", "enjoyable", "beautiful", "perfect",
    "masterpiece", "stunning", "charming", "gripping", "hilarious", "touching",
    "impressive", "memorable", "fresh", "clever", "satisfying", "powerful",
    "inspired",
]

NEGATIVE = [
    "bad", "terrible", "awful", "horrible", "boring", "disappointing",
    "dreadful", "poor", "weak", "messy", "painful", "forgettable",
    "predictable", "clumsy", "tedious", "annoying", "bland", "shallow",
    "confusing", "lifeless", "cheap", "lazy", "pointless", "unbearable",
    "disaster",
]

NEUTRAL = [
    "the", "a", "an", "this", "that", "it", "was", "is", "were", "and",
    "but", "with", "movie", "film", "story", "plot", "acting", "scene",
    "screen", "camera", "director", "actor", "actress", "cast", "music",
    "score", "script", "dialogue", "ending", "beginning", "middle", "part",
    "moment", "time", "hour", "minute", "character", "hero", "villain",
    "audience", "cinema", "sequel", "series", "season", "episode", "watch",
    "watched", "seen", "felt", "thought", "quite", "very", "really",
    "somewhat", "rather", "mostly", "overall", "honestly", "definitely",
    "again",
]

PAD, UNK = "<pad>", "<unk>"


def build_vocab():
    """word -> integer id. Id 0 is padding, id 1 is unknown."""
    words = [PAD, UNK] + POSITIVE + NEGATIVE + NEUTRAL
    return {word: idx for idx, word in enumerate(words)}


def encode(text: str, vocab: dict, max_len: int) -> list[int]:
    """Text -> fixed-length list of ids (truncate or pad with 0)."""
    ids = [vocab.get(w, vocab[UNK]) for w in text.lower().split()][:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


def make_sentence(rng, label: int) -> str:
    """One random sentence whose majority sentiment matches the label."""
    main = POSITIVE if label == 1 else NEGATIVE
    other = NEGATIVE if label == 1 else POSITIVE

    words = list(rng.choice(main, size=rng.integers(2, 6)))       # 2-5 main
    if rng.random() < 0.3:                                        # sometimes 1
        words += list(rng.choice(other, size=1))                  # opposite word
    words += list(rng.choice(NEUTRAL, size=rng.integers(4, 15)))  # filler
    rng.shuffle(words)
    return " ".join(words)


def make_dataset(n: int, vocab: dict, max_len: int, seed: int):
    """Return (X, y): X int64 tensor (n, max_len), y int64 tensor (n,)."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    rows = [encode(make_sentence(rng, int(lab)), vocab, max_len) for lab in labels]
    X = torch.tensor(rows, dtype=torch.int64)
    y = torch.tensor(labels, dtype=torch.int64)
    return X, y


@torch.inference_mode()
def accuracy(model, X, y, batch_size: int = 256) -> float:
    """Fraction of correct predictions, evaluated in batches."""
    model.eval()
    correct = 0
    for i in range(0, len(X), batch_size):
        logits = model(X[i : i + batch_size])
        correct += (logits.argmax(dim=1) == y[i : i + batch_size]).sum().item()
    return correct / len(X)