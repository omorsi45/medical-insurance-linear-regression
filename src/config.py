from __future__ import annotations

# Plotting
FIGURE_DPI: int = 200

# BMI categorisation (used in features.py)
BMI_BINS: list[float] = [-float("inf"), 18.5, 25.0, 30.0, float("inf")]
BMI_LABELS: list[str] = ["underweight", "normal", "overweight", "obese"]

# Seeds used in the stability sweep
SEED_SWEEP_SEEDS: list[int] = [1, 7, 21, 42, 99]
