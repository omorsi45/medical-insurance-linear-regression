# src/preprocess.py (same as before; keep or overwrite)
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REQUIRED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]


@dataclass(frozen=True)
class FeatureSpec:
    target: str = "charges"
    numeric_features: Tuple[str, ...] = ("age", "bmi", "children", "smoker_bmi")
    categorical_features: Tuple[str, ...] = ("sex", "smoker", "region", "bmi_category")


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),

        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(spec.numeric_features)),
            ("cat", categorical_pipe, list(spec.categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    return list(preprocessor.get_feature_names_out())
