r"""Experiment 02: torch.compile - try it, measure it, or skip gracefully.

Run from the project root:
    python -m benchmarks.exp02_compile
"""
import torch

from src.bench import benchmark, report
from src.text_predictor import TextPredictor

predictor = TextPredictor("models/sentiment_transformer")
model = predictor.model
model.eval()
example = torch.randint(2, 100, (1, 32), dtype=torch.int64)


def eager():
    with torch.inference_mode():
        return model(example)


report("baseline (eager)", benchmark(eager))

# --- Try to compile. On Windows this may fail - that is EXPECTED and OK. ----
try:
    compiled_model = torch.compile(model)

    def compiled():
        with torch.inference_mode():
            return compiled_model(example)

    print("compiling on first calls (can take 30-90 s, one time only)...")
    stats = benchmark(compiled)          # warmup inside triggers compilation
    report("torch.compile", stats)
except Exception as e:
    print(f"\ntorch.compile not usable here: {type(e).__name__}: {e}")
    print("This is common on Windows - continuing with the other methods.")