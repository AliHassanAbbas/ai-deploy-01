"""Central place for all project settings.

If a number or path is used in more than one file, it lives here.
Change it once, and every script picks up the change.
"""
from pathlib import Path

# ---- Paths (computed, never hardcoded) --------------------------------
# __file__ is this file's location. parent.parent walks up to the
# project root, so these paths work on ANY computer, any username.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BUNDLE_DIR = MODELS_DIR / "moon_classifier"   # the "inference bundle" folder

# ---- Model architecture ----------------------------------------------
INPUT_DIM = 2          # each data point is (x, y)
HIDDEN_DIM = 16        # neurons in each hidden layer
NUM_CLASSES = 2        # two moons -> two classes
CLASS_NAMES = ["outer_moon", "inner_moon"]

# ---- Training hyperparameters ----------------------------------------
LEARNING_RATE = 0.01
EPOCHS = 300
N_SAMPLES = 1000
NOISE = 0.15
SEED = 42

# ---- Bundle metadata --------------------------------------------------
MODEL_VERSION = "1.0.0"