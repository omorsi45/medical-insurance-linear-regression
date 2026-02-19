from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import sys

def main():
    seeds = [1, 7, 21, 42, 99]
    data_path = "data/insurance.csv"

    rows = []
    for s in seeds:
        run_id = f"seed_{s}"
        subprocess.check_call(
    [
        sys.executable,
        "src/train.py",
        "--data_path", data_path,
        "--model", "ridge",
        "--random_state", str(s),
        "--run_id", run_id,
    ]
)

        metrics_path = Path("reports") / "runs" / run_id / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as f:
            m = json.load(f)

        rows.append(
            {
                "seed": s,
                "r2": m["r2"],
                "rmse": m["rmse"],
                "mae": m["mae"],
            }
        )

    df = pd.DataFrame(rows).sort_values("seed")
    summary = {
        "r2_mean": float(df["r2"].mean()),
        "r2_std": float(df["r2"].std(ddof=1)),
        "rmse_mean": float(df["rmse"].mean()),
        "rmse_std": float(df["rmse"].std(ddof=1)),
        "mae_mean": float(df["mae"].mean()),
        "mae_std": float(df["mae"].std(ddof=1)),
    }

    out_csv = Path("reports") / "seed_sweep_summary.csv"
    df.to_csv(out_csv, index=False)

    out_json = Path("reports") / "seed_sweep_stats.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:")
    print("-", out_csv)
    print("-", out_json)


if __name__ == "__main__":
    main()
