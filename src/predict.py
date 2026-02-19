# src/predict.py (new)
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import add_bmi_category, add_smoker_bmi_interaction


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
