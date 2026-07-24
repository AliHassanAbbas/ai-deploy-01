r"""Experiment 08: half precision (fp16/bf16) - a GPU technique, measured if
you have one, explained if you don't.

Run from the project root:
    python -m benchmarks.exp08_half_precision
"""
import torch

from src.bench import benchmark, report
from src.text_predictor import TextPredictor

if not torch.cuda.is_available():
    print("No CUDA GPU detected - skipping the measurement.")
    print()
    print("What you would measure on a GPU: the same model in fp16/bf16")
    print("(2 bytes per number instead of 4) typically runs 1.5-3x faster")
    print("and uses half the VRAM, because modern GPUs have dedicated")
    print("tensor cores for half-precision math. Run this script again on")
    print("a machine with an NVIDIA GPU to fill the fp16 row of your table.")
    raise SystemExit(0)

predictor = TextPredictor("models/sentiment_transformer")
model = predictor.model.cuda().eval()
example = torch.randint(2, 100, (32, 32), dtype=torch.int64, device="cuda")


def fp32_run():
    with torch.inference_mode():
        return model(example)


def fp16_run():
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        return model(example)


report("fp32 on GPU (batch 32)", benchmark(fp32_run), batch_size=32)
report("fp16 autocast (batch 32)", benchmark(fp16_run), batch_size=32)

with torch.inference_mode():
    a = model(example).float()
    with torch.autocast("cuda", dtype=torch.float16):
        b = model(example).float()
print(f"\nmax difference fp32 vs fp16: {(a - b).abs().max().item():.5f}")
print("(half precision is APPROXIMATE - always check accuracy, not just speed)")