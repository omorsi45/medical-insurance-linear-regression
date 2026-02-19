from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_predicted_vs_actual(y_true, y_pred, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure()
    plt.scatter(y_true, y_pred, s=12)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Actual charges")
    plt.ylabel("Predicted charges")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_residuals_hist(residuals, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    residuals = np.asarray(residuals)

    plt.figure()
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_residuals_vs_predicted(y_pred, residuals, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_pred = np.asarray(y_pred)
    residuals = np.asarray(residuals)

    plt.figure()
    plt.scatter(y_pred, residuals, s=12)
    plt.axhline(0)
    plt.xlabel("Predicted charges")
    plt.ylabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
