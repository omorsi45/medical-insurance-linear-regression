# Medical Insurance Cost Prediction (Linear / Ridge Regression)

Predict medical insurance `charges` using a linear model with feature engineering, cross-validation, and stability checks.

## Project overview
This repo trains a Linear/Ridge regression model on the common Kaggle insurance dataset and outputs:
- evaluation metrics (R2, RMSE, MAE)
- cross-validation results from GridSearchCV
- coefficient/feature impact (linear model coefficients)
- prediction + residual plots
- a saved model you can reuse for new predictions

## Dataset
Download the dataset (“Medical Cost Personal Datasets”) and place the CSV here:

- `data/insurance.csv`

Expected columns:
- `age, sex, bmi, children, smoker, region, charges`

Note: `data/` is in `.gitignore` so the dataset is not uploaded to GitHub.

## Setup (Windows / macOS / Linux)
From the repo root:

```bash
python -m venv .venv
