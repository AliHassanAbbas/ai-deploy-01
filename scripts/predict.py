"""Command-line prediction tool.

Run from the project root (-m makes Python treat the project root as
the import base, so "from src..." works):

    python -m scripts.predict --x 0.0 --y 1.0
"""
import argparse

from src.config import BUNDLE_DIR
from src.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description="Classify a 2D point.")
    parser.add_argument("--x", type=float, required=True, help="x coordinate")
    parser.add_argument("--y", type=float, required=True, help="y coordinate")
    args = parser.parse_args()

    predictor = Predictor(BUNDLE_DIR)
    result = predictor.predict([args.x, args.y])[0]

    print(f"input point : ({args.x}, {args.y})")
    print(f"prediction  : {result['class_name']} (class {result['class_id']})")
    print(f"probabilities: {result['probabilities']}")


if __name__ == "__main__":
    main()