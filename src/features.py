# src/features.py (same as before; keep or overwrite)
from __future__ import annotations

import pandas as pd


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [-float("inf"), 18.5, 25.0, 30.0, float("inf")]
    labels = ["underweight", "normal", "overweight", "obese"]
    df["bmi_category"] = pd.cut(df["bmi"], bins=bins, labels=labels, right=False)
    return df


def add_smoker_bmi_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    smoker_yes = (df["smoker"].astype(str).str.lower() == "yes").astype(int)
    df["smoker_bmi"] = df["bmi"] * smoker_yes
    return df
