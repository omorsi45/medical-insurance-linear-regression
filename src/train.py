from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline

from features import add_bmi_category, add_smoker_bmi_interaction
from interpret import extract_coefficients, save_top_coeff_plot
from plots import save_predicted_vs_actual, save_residuals_hist, save_residuals_vs_predicted
from preprocess import FeatureSpec, build_preprocessor, get_feature_names, validate_columns
from utils import dataset_summary, rmse, save_json


def get_output_dirs(run_id: str):
    if run_id == "latest":
        return Path("reports"), Path("figures"), Path("models")

    return (
        Path("reports") / "runs" / run_id,
        Path("figures") / "runs" / run_id,
        Path("models") / "runs" / run_id,
    )


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. Put your dataset at data/insurance.csv (or pass --data_path)."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["sex", "smoker", "region"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    df = df.dropna(subset=["charges"])

    for col in ["age", "bmi", "children", "charges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["age", "bmi", "children", "charges"])
    return df


def build_model_pipeline(model_type: str, random_state: int) -> tuple[Pipeline, dict]:
    spec = FeatureSpec()
    pre = build_preprocessor(spec)

    if model_type == "linear":
        reg = LinearRegression()
        param_grid = {"regressor__fit_intercept": [True, False]}
    elif model_type == "ridge":
        reg = Ridge(random_state=random_state)
        param_grid = {
            "regressor__alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
            "regressor__fit_intercept": [True, False],
        }
    else:
        raise ValueError("--model must be one of: linear, ridge")

    pipe = Pipeline(steps=[("preprocessor", pre), ("regressor", reg)])
    return pipe, param_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/insurance.csv")
    parser.add_argument("--model", type=str, default="ridge", choices=["linear", "ridge"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--run_id", type=str, default="latest")
    args = parser.parse_args()

    # IMPORTANT: this must be inside main(), after args is defined
    reports_dir, figures_dir, models_dir = get_output_dirs(args.run_id)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_data(args.data_path)
    validate_columns(df_raw)
    save_json(reports_dir / "data_summary.json", dataset_summary(df_raw))

    df = clean_data(df_raw)

    df = add_bmi_category(df)
    df = add_smoker_bmi_interaction(df)

    spec = FeatureSpec()
    X = df.drop(columns=[spec.target])
    y = df[spec.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    pipe, param_grid = build_model_pipeline(args.model, args.random_state)
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    pd.DataFrame(search.cv_results_).to_csv(reports_dir / "cv_results.csv", index=False)

    y_pred = best_model.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, y_pred))
    test_rmse = rmse(y_test, y_pred)
    test_r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "model": args.model,
        "best_params": search.best_params_,
        "r2": test_r2,
        "rmse": test_rmse,
        "mae": test_mae,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": int(args.random_state),
        "test_size": float(args.test_size),
        "cv_folds": int(args.cv_folds),
        "run_id": args.run_id,
    }
    save_json(reports_dir / "metrics.json", metrics)

    joblib.dump(best_model, models_dir / "best_model.joblib")

    preds = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
    preds["residual"] = preds["y_true"] - preds["y_pred"]
    preds.to_csv(reports_dir / "test_predictions.csv", index=False)

    save_predicted_vs_actual(preds["y_true"], preds["y_pred"], figures_dir / "predicted_vs_actual.png")
    save_residuals_hist(preds["residual"], figures_dir / "residuals_hist.png")
    save_residuals_vs_predicted(preds["y_pred"], preds["residual"], figures_dir / "residuals_vs_predicted.png")

    pre = best_model.named_steps["preprocessor"]
    feature_names = get_feature_names(pre)

    coefs_df = extract_coefficients(best_model, feature_names)
    coefs_df.to_csv(reports_dir / "top_coefficients.csv", index=False)
    save_top_coeff_plot(coefs_df, figures_dir / "top_coefficients.png", top_n=15)

    print("=== Metrics (test set) ===")
    print(f"R2:   {test_r2}")
    print(f"RMSE: {test_rmse}")
    print(f"MAE:  {test_mae}")
    print(f"Run:  {args.run_id}")


if __name__ == "__main__":
    main()
