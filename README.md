# Medical Insurance Cost Prediction (Linear / Ridge Regression)

Predict medical insurance `charges` using a linear model with feature engineering, cross-validation, and stability checks.

**Timeline:** May 2025 – Jul 2025

## Results

**Test set (seed=42):**
- R2: **0.867**
- RMSE: **4,546**
- MAE: **2,721**

**Stability (5 random splits):**
- R2: **0.818–0.867** (mean **0.844 ± 0.021**)
- RMSE: **4,516–5,076** (mean **4,810 ± 264**)
- MAE: **2,601–3,064** (mean **2,867 ± 201**)

Files:
- `reports/metrics.json`
- `reports/seed_sweep_summary.csv`
- `reports/seed_sweep_stats.json`

## Project overview
This repo trains a Linear/Ridge regression model on the common Kaggle insurance dataset and outputs:
- evaluation metrics (R2, RMSE, MAE)
- cross-validation results from GridSearchCV
- coefficient/feature impact (linear model coefficients)
- prediction + residual plots
- a saved model you can reuse for new predictions

Minor update to documentation.

## Dataset
Download the dataset ("Medical Cost Personal Datasets") and place the CSV here:
- `data/insurance.csv`

Expected columns:
- `age, sex, bmi, children, smoker, region, charges`

Note: `data/` is in `.gitignore` so the dataset is not uploaded to GitHub.

## Project structure

```
src/
├── config.py        # Shared constants (DPI, BMI bins, sweep seeds)
├── features.py      # Feature engineering (BMI category, smoker×BMI interaction)
├── preprocess.py    # Data cleaning, validation, sklearn preprocessing pipeline
├── train.py         # Main training script (GridSearchCV, metrics, plots, model export)
├── predict.py       # Single-row inference from a saved model
├── interpret.py     # Coefficient extraction and bar chart
├── plots.py         # Diagnostic plots (predicted vs actual, residuals)
├── seed_sweep.py    # Stability check across multiple random seeds
└── utils.py         # RMSE, JSON/CSV save helpers, dataset summary
```

## Setup (Windows / macOS / Linux)
From the repo root:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

## Training

```bash
python src/train.py
```

Optional arguments:

| Flag | Default | Description |
|---|---|---|
| `--data_path` | `data/insurance.csv` | Path to the dataset |
| `--model` | `ridge` | `linear` or `ridge` |
| `--test_size` | `0.2` | Fraction held out for testing |
| `--random_state` | `42` | Random seed |
| `--cv_folds` | `5` | Cross-validation folds |
| `--run_id` | `latest` | Tag for output subdirectory |

Outputs go to `reports/`, `figures/`, and `models/`.

## Prediction (single row)

```bash
python src/predict.py \
  --age 35 --sex female --bmi 27.5 \
  --children 2 --smoker no --region northeast
```

Valid values: `sex` = male/female, `smoker` = yes/no, `region` = northeast/northwest/southeast/southwest.

## Stability sweep

Runs training across 5 seeds and reports mean ± std of all metrics:

```bash
python src/seed_sweep.py
```

Results saved to `reports/seed_sweep_summary.csv` and `reports/seed_sweep_stats.json`.
