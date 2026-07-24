r"""Experiment 06: benchmark ONNX Runtime vs the PyTorch baseline.

Run from the project root:
    python -m benchmarks.exp06_onnx_bench
"""
import numpy as np
import onnxruntime as ort
import torch

from src.bench import benchmark, report
from src.text_predictor import TextPredictor

predictor = TextPredictor("models/sentiment_transformer")
model = predictor.model
model.eval()

session = ort.InferenceSession(
    "models/sentiment_transformer/model.onnx",
    providers=["CPUExecutionProvider"],
)

for batch_size in (1, 32):
    example_torch = torch.randint(2, 100, (batch_size, 32), dtype=torch.int64)
    example_numpy = example_torch.numpy()

    def eager():
        with torch.inference_mode():
            return model(example_torch)

    def onnx_run():
        return session.run(None, {"input_ids": example_numpy})

    print(f"--- batch size {batch_size} ---")
    report("PyTorch eager", benchmark(eager), batch_size)
    report("ONNX Runtime (CPU)", benchmark(onnx_run), batch_size)
    print()