r"""Experiment 01: the baseline - and the cost of forgetting eval/inference_mode.

Run from the project root:
    python -m benchmarks.exp01_baseline
"""
import torch

from src.bench import benchmark, report
from src.text_predictor import TextPredictor

predictor = TextPredictor("models/sentiment_transformer")
model = predictor.model
example = torch.randint(2, 100, (1, 32), dtype=torch.int64)  # batch 1, seq 32

# --- The WRONG way: train mode, gradients being tracked --------------------
model.train()
def wrong():
    return model(example)

# --- The RIGHT way: eval mode + inference_mode ------------------------------
model.eval()
def right():
    with torch.inference_mode():
        return model(example)

print("same input, three configurations:\n")
report("train mode + grad (WRONG)", benchmark(wrong))

model.eval()
def eval_but_grad():
    return model(example)
report("eval mode, grad on (still bad)", benchmark(eval_but_grad))

report("eval + inference_mode (BASELINE)", benchmark(right))

# --- And the correctness angle: train mode gives RANDOM outputs -------------
model.train()
a = model(example)
b = model(example)
print(f"\ntrain mode, same input twice -> max difference: {(a - b).abs().max().item():.4f}")
model.eval()
with torch.inference_mode():
    a = model(example)
    b = model(example)
print(f"eval  mode, same input twice -> max difference: {(a - b).abs().max().item():.4f}")