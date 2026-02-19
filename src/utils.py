# src/utils.py (overwrite to include summary helper)
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    missing = df.isna().sum().to_dict()
    dup_count = int(df.duplicated().sum())
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": dup_count,
        "missing_values": {k: int(v) for k, v in missing.items()},
        "columns": list(df.columns),
    }
