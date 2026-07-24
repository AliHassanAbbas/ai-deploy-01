"""Tests for the Predictor. Run from the project root:

    python -m pytest tests -v
"""
import math

import pytest

from src.config import BUNDLE_DIR, CLASS_NAMES
from src.predictor import Predictor


@pytest.fixture(scope="module")
def predictor():
    """Load the bundle ONCE and share it across all tests in this file."""
    return Predictor(BUNDLE_DIR)


def test_bundle_loads(predictor):
    # If __init__ raised (missing file, wrong shapes), we never get here.
    assert predictor.model is not None
    assert predictor.class_names == CLASS_NAMES


def test_single_point_output_format(predictor):
    results = predictor.predict([0.0, 1.0])

    assert isinstance(results, list)
    assert len(results) == 1

    r = results[0]
    assert isinstance(r["class_id"], int)
    assert r["class_name"] in CLASS_NAMES
    assert len(r["probabilities"]) == 2


def test_probabilities_are_valid(predictor):
    r = predictor.predict([0.5, 0.5])[0]
    probs = r["probabilities"]

    assert all(0.0 <= p <= 1.0 for p in probs)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-3)


def test_batch_prediction(predictor):
    batch = [[0.0, 1.0], [1.0, -0.4], [0.5, 0.5]]
    results = predictor.predict(batch)
    assert len(results) == len(batch)


def test_known_easy_points(predictor):
    # (0, 1) sits on top of the outer moon; (1, -0.4) deep in the inner moon.
    assert predictor.predict([0.0, 1.0])[0]["class_name"] == "outer_moon"
    assert predictor.predict([1.0, -0.4])[0]["class_name"] == "inner_moon"