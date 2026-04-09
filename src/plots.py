from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from config import FIGURE_DPI


def _save_figure(out_path: Path, dpi: int = FIGURE_DPI) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def save_predicted_vs_actual(y_true: ArrayLike, y_pred: ArrayLike, out_path: str | Path) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))

    plt.figure()
    plt.scatter(y_true, y_pred, s=12)
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Actual charges")
    plt.ylabel("Predicted charges")
    _save_figure(Path(out_path))


def save_residuals_hist(residuals: ArrayLike, out_path: str | Path) -> None:
    residuals = np.asarray(residuals)

    plt.figure()
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Count")
    _save_figure(Path(out_path))


def save_residuals_vs_predicted(y_pred: ArrayLike, residuals: ArrayLike, out_path: str | Path) -> None:
    y_pred = np.asarray(y_pred)
    residuals = np.asarray(residuals)

    plt.figure()
    plt.scatter(y_pred, residuals, s=12)
    plt.axhline(0)
    plt.xlabel("Predicted charges")
    plt.ylabel("Residual (actual - predicted)")
    _save_figure(Path(out_path))
