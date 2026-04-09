from __future__ import annotations

import pandas as pd

from config import BMI_BINS, BMI_LABELS


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bmi_category"] = pd.cut(df["bmi"], bins=BMI_BINS, labels=BMI_LABELS, right=False)
    return df


def add_smoker_bmi_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    smoker_yes = (df["smoker"].astype(str).str.lower() == "yes").astype(int)
    df["smoker_bmi"] = df["bmi"] * smoker_yes
    return df
