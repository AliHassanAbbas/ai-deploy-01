r"""Experiment 03: TorchScript - freeze the model into a self-contained file.

Run from the project root:
    python -m benchmarks.exp03_torchscript
"""
import torch

from src.bench import benchmark, report
from src.text_predictor import TextPredictor

predictor = TextPredictor("models/sentiment_transformer")
model = predictor.model
model.eval()
example = torch.randint(2, 100, (1, 32), dtype=torch.int64)

# --- TRACE: run the model once with an example input and RECORD every
# --- tensor operation. The recording IS the saved model.
# --- We trace under no_grad so the recording captures the same fast code
# --- path the model uses when serving (tracing with grad enabled records
# --- a DIFFERENT path for nn.TransformerEncoder and fails its sanity check).
with torch.no_grad():
    traced = torch.jit.trace(model, example)

traced_path = "models/sentiment_transformer/model_traced.pt"
traced.save(traced_path)
print(f"saved traced model to {traced_path}\n")


def eager():
    with torch.inference_mode():
        return model(example)


def torchscript():
    with torch.inference_mode():
        return traced(example)


report("baseline (eager)", benchmark(eager))
report("TorchScript (traced)", benchmark(torchscript))

# --- Correctness: the traced model must agree with the original -------------
with torch.inference_mode():
    diff = (model(example) - traced(example)).abs().max().item()
print(f"\nmax difference eager vs traced: {diff:.8f}")