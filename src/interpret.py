from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline

from plots import _save_figure


def extract_coefficients(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    reg = model.named_steps["regressor"]
    coefs = reg.coef_.ravel()
    out = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    out["abs_coefficient"] = out["coefficient"].abs()
    out = out.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return out


def save_top_coeff_plot(df_coefs: pd.DataFrame, out_path: str | Path, top_n: int = 15) -> None:
    top = df_coefs.head(top_n).iloc[::-1]
    plt.figure()
    plt.barh(top["feature"], top["coefficient"])
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    _save_figure(Path(out_path))
