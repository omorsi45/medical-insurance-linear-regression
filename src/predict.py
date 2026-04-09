from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import add_bmi_category, add_smoker_bmi_interaction

_VALID_SEX = {"male", "female"}
_VALID_SMOKER = {"yes", "no"}
_VALID_REGIONS = {"northeast", "northwest", "southeast", "southwest"}


def _validate_inputs(args: argparse.Namespace) -> None:
    if not (0 <= args.age <= 120):
        raise ValueError(f"age must be between 0 and 120, got {args.age}")
    if not (0 < args.bmi <= 100):
        raise ValueError(f"bmi must be between 0 and 100, got {args.bmi}")
    if args.children < 0:
        raise ValueError(f"children must be non-negative, got {args.children}")
    if args.sex.strip().lower() not in _VALID_SEX:
        raise ValueError(f"sex must be one of {_VALID_SEX}, got '{args.sex}'")
    if args.smoker.strip().lower() not in _VALID_SMOKER:
        raise ValueError(f"smoker must be one of {_VALID_SMOKER}, got '{args.smoker}'")
    if args.region.strip().lower() not in _VALID_REGIONS:
        raise ValueError(f"region must be one of {_VALID_REGIONS}, got '{args.region}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/best_model.joblib")
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--sex", type=str, required=True)
    parser.add_argument("--bmi", type=float, required=True)
    parser.add_argument("--children", type=float, required=True)
    parser.add_argument("--smoker", type=str, required=True)
    parser.add_argument("--region", type=str, required=True)

    args = parser.parse_args()
    _validate_inputs(args)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first with src/train.py")

    model = joblib.load(model_path)

    row = pd.DataFrame(
        [
            {
                "age": args.age,
                "sex": args.sex.strip().lower(),
                "bmi": args.bmi,
                "children": args.children,
                "smoker": args.smoker.strip().lower(),
                "region": args.region.strip().lower(),
            }
        ]
    )

    row = add_bmi_category(row)
    row = add_smoker_bmi_interaction(row)

    pred = float(model.predict(row)[0])
    print(pred)


if __name__ == "__main__":
    main()
