"""
Phase 2 - Model 1: Cost Forecasting (regression)

Predicts NEXT-DAY cost_usd per instance from per-instance lag/rolling features.
To get a 7-day forecast total: predict day t+1, append it, recompute the lag/rolling
features, and roll the model forward 7 steps; summing those 7 predictions gives the
7-day cost. (Here we evaluate the single next-day target, which is what drives that loop.)

Trains XGBoost (with a graceful fallback) AND RandomForestRegressor, compares RMSE/MAPE
on a TEMPORAL test split (last ~30 days), and keeps the better model by MAPE.

Run:  py models/train/train_forecast.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Import the shared feature transforms (same code the API serves with -> no skew).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.features import build_forecast_features, FORECAST_BASE_FEATURES

# XGBoost is optional: fall back to sklearn's HistGradientBoosting if unavailable.
try:
    from xgboost import XGBRegressor

    HAVE_XGB = True
    GB_NAME = "XGBoost"
except Exception:  # pragma: no cover - environment dependent
    from sklearn.ensemble import HistGradientBoostingRegressor

    HAVE_XGB = False
    GB_NAME = "HistGradientBoosting (xgboost not installed)"

# ---------------------------------------------------------------------------
# Paths (relative to project root so the script runs from anywhere)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "usage_processed.csv"
GT = ROOT / "data" / "raw" / "ground_truth.csv"
SAVED = ROOT / "models" / "saved"
EVAL = ROOT / "models" / "evaluate"
SAVED.mkdir(parents=True, exist_ok=True)
EVAL.mkdir(parents=True, exist_ok=True)

TEST_DAYS = 30          # temporal hold-out window
MAPE_TARGET = 15.0      # success threshold (%)


def mape(y_true, y_pred):
    """Mean absolute percentage error in percent (guards against divide-by-zero)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


# Feature engineering now lives in common/features.build_forecast_features so that the
# Phase 3 API builds identical features at serve time (no train/serve skew).


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])

    # Attach is_anomaly (INNER join on date + instance_id) so we can de-spike the
    # cost series for the baseline forecaster.
    gt = pd.read_csv(GT, parse_dates=["date"])
    df = df.merge(gt[["date", "instance_id", "is_anomaly"]],
                  on=["date", "instance_id"], how="inner")

    df = build_forecast_features(df)

    feature_cols = list(FORECAST_BASE_FEATURES)

    # One-hot encode instance_type
    type_dummies = pd.get_dummies(df["instance_type"], prefix="itype")
    df = pd.concat([df, type_dummies], axis=1)
    feature_cols += list(type_dummies.columns)

    # Drop rows without a target or with missing lag features. Also exclude rows whose
    # target day is a planted spike: the baseline forecaster is not meant to predict
    # anomalies (Model 3 catches those), so scoring on them would be unfair and wrong.
    model_df = df.dropna(subset=feature_cols + ["target"]).copy()
    model_df = model_df[~model_df["target_is_anomaly"]]

    # Temporal split: last TEST_DAYS days are the test set (no shuffling)
    cutoff = model_df["date"].max() - pd.Timedelta(days=TEST_DAYS)
    train = model_df[model_df["date"] <= cutoff]
    test = model_df[model_df["date"] > cutoff]

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    print(f"Train rows: {len(train)}  Test rows: {len(test)}  "
          f"(test = last {TEST_DAYS} days, cutoff {cutoff.date()})")

    # --- Gradient boosting model (XGBoost or fallback) ---
    if HAVE_XGB:
        gb = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor

        gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                           random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_rmse = float(np.sqrt(mean_squared_error(y_test, gb_pred)))
    gb_mape = mape(y_test, gb_pred)

    # --- RandomForest ---
    rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_pred)))
    rf_mape = mape(y_test, rf_pred)

    print("\n--- Forecast model comparison (test set) ---")
    print(f"{GB_NAME:40s}  RMSE={gb_rmse:.4f}  MAPE={gb_mape:.2f}%")
    print(f"{'RandomForestRegressor':40s}  RMSE={rf_rmse:.4f}  MAPE={rf_mape:.2f}%")

    # Keep the better model by MAPE (the success metric)
    if gb_mape <= rf_mape:
        best_name, best_model, best_rmse, best_mape = GB_NAME, gb, gb_rmse, gb_mape
    else:
        best_name, best_model = "RandomForestRegressor", rf
        best_rmse, best_mape = rf_rmse, rf_mape

    print(f"\nBest model: {best_name}  (RMSE={best_rmse:.4f}, MAPE={best_mape:.2f}%)")
    passed = best_mape < MAPE_TARGET
    print(f"Success target MAPE < {MAPE_TARGET}% -> {'PASS' if passed else 'FAIL'}")

    # Persist model + feature list
    joblib.dump({"model": best_model, "features": feature_cols, "name": best_name},
                SAVED / "forecast_model.joblib")

    metrics = {
        "model": "forecast",
        "best_model": best_name,
        "rmse": best_rmse,
        "mape": best_mape,
        "gb_model": GB_NAME,
        "gb_rmse": gb_rmse,
        "gb_mape": gb_mape,
        "rf_rmse": rf_rmse,
        "rf_mape": rf_mape,
        "test_days": TEST_DAYS,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "threshold_mape": MAPE_TARGET,
        "pass": bool(passed),
    }
    with open(EVAL / "forecast_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model -> {SAVED / 'forecast_model.joblib'}")
    print(f"Saved metrics -> {EVAL / 'forecast_metrics.json'}")


if __name__ == "__main__":
    main()
