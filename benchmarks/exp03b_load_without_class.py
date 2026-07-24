r"""Experiment 03b: load the TorchScript file with ZERO project imports.

Notice what this script does NOT import: src.textmodel. The .pt file
carries the whole computation - the Python class is no longer needed.
This is what "freezing your model into a file" means.

Run from anywhere:
    python -m benchmarks.exp03b_load_without_class
"""
import torch   # <- the ONLY import

model = torch.jit.load("models/sentiment_transformer/model_traced.pt")
model.eval()

input_ids = torch.randint(2, 100, (1, 32), dtype=torch.int64)
with torch.inference_mode():
    logits = model(input_ids)

print("loaded and ran the model without importing its class")
print("output shape:", tuple(logits.shape))
print("logits:", [round(v, 3) for v in logits[0].tolist()])