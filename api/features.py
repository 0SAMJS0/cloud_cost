"""
Serving-side feature helpers.

This module deliberately contains NO feature math of its own — it imports the exact
transforms from common.features (the same ones the training scripts use) and only adds the
online plumbing: turning request payloads into the single feature row each model expects.
That is what guarantees no train/serve skew.
"""

import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is importable when uvicorn is launched from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.features import (  # noqa: E402
    add_anomaly_features,
    build_forecast_features,
    ANOMALY_FEATURES,
    FORECAST_BASE_FEATURES,
    WASTE_FEATURES,
)

__all__ = [
    "add_anomaly_features",
    "build_forecast_features",
    "ANOMALY_FEATURES",
    "FORECAST_BASE_FEATURES",
    "WASTE_FEATURES",
    "run_anomaly_detection",
    "forecast_pipeline",
]


def run_anomaly_detection(df: pd.DataFrame, iso_model) -> pd.DataFrame:
    """Add anomaly features, run IsolationForest, return df sorted by date with
    `is_anomaly` (bool) and `anomaly_score` (higher = more anomalous) columns.

    Expects columns: instance_id, date, cost_usd.
    """
    feat = add_anomaly_features(df)
    X = feat[ANOMALY_FEATURES].to_numpy()
    pred = iso_model.predict(X)                  # -1 anomaly, 1 normal
    scores = iso_model.decision_function(X)      # lower = more anomalous
    feat["is_anomaly"] = pred == -1
    feat["anomaly_score"] = -scores              # flip so higher = more anomalous
    return feat


def forecast_pipeline(history_df: pd.DataFrame, iso_model, forecast_bundle: dict) -> dict:
    """Wire Model 3 -> Model 1: flag anomalous days in the input window, de-spike them to
    the per-instance median, build lag/rolling features, and forecast next-day baseline
    cost.

    Expects history_df columns: instance_id, instance_type, date, cpu_avg, memory_avg,
    network_in, network_out, cost_usd (>= ~8 rows; 30 recommended).
    """
    # Step 1: anomaly detection on the input cost series.
    feat = run_anomaly_detection(history_df, iso_model)

    # Step 2: build_forecast_features() consumes `is_anomaly` and replaces flagged days
    # with the per-instance median (computed over this input window) before lags/rollings.
    fdf = build_forecast_features(feat)
    last = fdf.iloc[-1]

    # Step 3: assemble the single feature row, aligned to the model's saved feature list
    # (which includes the one-hot instance_type columns). Missing columns -> 0.
    model = forecast_bundle["model"]
    feat_list = forecast_bundle["features"]
    row = {c: last[c] for c in FORECAST_BASE_FEATURES}
    X = pd.DataFrame([row]).reindex(columns=feat_list, fill_value=0)

    itype = str(history_df["instance_type"].iloc[0])
    itype_col = f"itype_{itype}"
    if itype_col in feat_list:
        X.at[0, itype_col] = 1  # else: unknown type -> all itype cols stay 0

    prediction = float(model.predict(X)[0])

    # Confidence heuristic: stability of the recent de-spiked baseline. Low volatility
    # (coefficient of variation) -> high confidence. Bounded to [0, 1].
    recent = fdf["cost_base"].tail(7)
    mean = float(recent.mean())
    std = float(recent.std(ddof=0)) if len(recent) > 1 else 0.0
    cv = (std / mean) if mean else 0.0
    confidence = max(0.0, min(1.0, 1.0 - cv))

    flagged = feat.loc[feat["is_anomaly"], "date"].dt.strftime("%Y-%m-%d").tolist()
    last_cost = float(history_df.sort_values("date")["cost_usd"].iloc[-1])

    return {
        "prediction": prediction,
        "confidence": confidence,
        "recent_mean": mean,
        "last_cost": last_cost,
        "flagged_dates": flagged,
    }
