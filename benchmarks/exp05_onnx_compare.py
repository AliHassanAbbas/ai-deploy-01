r"""Experiment 05: verify ONNX gives the SAME answers as PyTorch.

An optimized model that answers differently is not optimized - it's broken.
We compare outputs numerically before trusting any speed numbers.

Run from the project root:
    python -m benchmarks.exp05_onnx_compare
"""
import numpy as np
import onnxruntime as ort
import torch

from src.text_predictor import TextPredictor

TOLERANCE = 1e-4   # fp32 op reordering causes tiny differences; this is fine

predictor = TextPredictor("models/sentiment_transformer")
model = predictor.model
model.eval()

session = ort.InferenceSession(
    "models/sentiment_transformer/model.onnx",
    providers=["CPUExecutionProvider"],
)

worst = 0.0
for batch_size in (1, 4, 32):
    input_ids = torch.randint(2, 100, (batch_size, 32), dtype=torch.int64)

    with torch.inference_mode():
        torch_out = model(input_ids).numpy()

    onnx_out = session.run(None, {"input_ids": input_ids.numpy()})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    worst = max(worst, max_diff)
    print(f"batch {batch_size:2d}: max absolute difference = {max_diff:.8f}")

print(f"\nworst difference {worst:.8f} vs tolerance {TOLERANCE}")
print("MATCH - safe to benchmark" if worst < TOLERANCE else "MISMATCH - do NOT use this export")