r"""Experiment 04: export the model to ONNX with dynamic axes.

Run from the project root:
    python -m benchmarks.exp04_onnx_export
"""
import torch

from src.text_predictor import TextPredictor

predictor = TextPredictor("models/sentiment_transformer")
model = predictor.model
model.eval()

example = torch.randint(2, 100, (1, 32), dtype=torch.int64)
onnx_path = "models/sentiment_transformer/model.onnx"

torch.onnx.export(
    model,
    (example,),
    onnx_path,
    input_names=["input_ids"],
    output_names=["logits"],
    # dynamic axes: WITHOUT this, the ONNX model only accepts the exact
    # shape we exported with (batch 1). WITH it, dimension 0 (batch) may
    # be any size at runtime.
    dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
    dynamo=False,   # use the classic exporter (most portable today)
)
print(f"exported to {onnx_path}")