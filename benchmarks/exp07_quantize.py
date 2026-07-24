r"""Experiment 07: dynamic int8 quantization - size, speed, and accuracy.

Run from the project root:
    python -m benchmarks.exp07_quantize
"""
import os
import warnings

import torch
import torch.nn as nn

from src.bench import benchmark, report
from src.text_predictor import TextPredictor
from src.textdata import accuracy, build_vocab, make_dataset

warnings.filterwarnings("ignore")   # quantization APIs print deprecation noise

predictor = TextPredictor("models/sentiment_transformer")
model = predictor.model
model.eval()

example = torch.randint(2, 100, (1, 32), dtype=torch.int64)

# --- Measure fp32 FIRST (with all its fast paths intact) --------------------
def fp32_run():
    with torch.inference_mode():
        return model(example)

fp32_stats = benchmark(fp32_run)

# --- Quantize: every nn.Linear stores int8 weights instead of fp32 ----------
quantized = torch.ao.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Sharp edge (real): the quantized attention trips the transformer's fused
# "fast path" check with AttributeError: 'function' object has no attribute
# 'device'. The documented escape hatch is to disable that fast path:
torch.backends.mha.set_fastpath_enabled(False)


def int8_run():
    with torch.inference_mode():
        return quantized(example)


# --- 1. Size on disk ---------------------------------------------------------
fp32_path = "models/sentiment_transformer/model_weights.pt"
int8_path = "models/sentiment_transformer/model_weights_int8.pt"
torch.save(quantized.state_dict(), int8_path)

fp32_mb = os.path.getsize(fp32_path) / 1e6
int8_mb = os.path.getsize(int8_path) / 1e6
print(f"fp32 weights: {fp32_mb:.2f} MB")
print(f"int8 weights: {int8_mb:.2f} MB   ({fp32_mb / int8_mb:.2f}x smaller)\n")

# --- 2. Speed ------------------------------------------------------------------
report("fp32 (baseline)", fp32_stats)
report("int8 (dynamic quant)", benchmark(int8_run))

# --- 3. Accuracy - the honest part -------------------------------------------
vocab = build_vocab()
X_test, y_test = make_dataset(1000, vocab, predictor.max_len, seed=43)
acc_fp32 = accuracy(model, X_test, y_test)
acc_int8 = accuracy(quantized, X_test, y_test)
print(f"\naccuracy fp32: {acc_fp32:.1%}")
print(f"accuracy int8: {acc_int8:.1%}   (change: {(acc_int8 - acc_fp32) * 100:+.2f} points)")