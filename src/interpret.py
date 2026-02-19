# src/interpret.py (same as before; keep or overwrite)
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline


def extract_coefficients(model: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    reg = model.named_steps["regressor"]
    coefs = reg.coef_.ravel()
    out = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    out["abs_coefficient"] = out["coefficient"].abs()
    out = out.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return out


def save_top_coeff_plot(df_coefs: pd.DataFrame, out_path: str | Path, top_n: int = 15) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    top = df_coefs.head(top_n).iloc[::-1]
    plt.figure()
    plt.barh(top["feature"], top["coefficient"])
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
