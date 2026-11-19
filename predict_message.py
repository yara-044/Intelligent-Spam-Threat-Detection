"""Classify one message with the trained best model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    prediction = int(model.predict([args.message])[0])
    label = "spam" if prediction == 1 else "ham"
    print(f"model: {bundle['name']}")
    print(f"prediction: {label}")


if __name__ == "__main__":
    main()