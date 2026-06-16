"""
Shared feature engineering — the SINGLE source of truth for how raw usage rows become
model inputs.

This module is imported by BOTH the Phase 2 training scripts (models/train/*.py) and the
Phase 3 serving API (api/*). Keeping the transforms in one place is what prevents
train/serve skew: the API cannot accidentally compute features differently from how the
models were trained, because it runs the exact same functions.

The three models use three different feature sets:
  - forecast : per-instance lag/rolling features built on a *de-spiked* cost series
  - anomaly  : per-instance z-score of cost + day-over-day pct change
  - waste    : raw utilisation metrics (no engineering)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Waste model — raw utilisation features (no engineering).
# The project doc lists "uptime hours" but the dataset has no uptime column, so
# network in/out is used as the extra utilisation signal instead.
# ---------------------------------------------------------------------------
WASTE_FEATURES = ["cpu_avg", "memory_avg", "network_in", "network_out"]

# ---------------------------------------------------------------------------
# Anomaly model — per-instance normalised features (never raw cost_usd, so that
# expensive instance types are not flagged just for being expensive).
# ---------------------------------------------------------------------------
ANOMALY_FEATURES = ["cost_z", "cost_pct_chg"]

# ---------------------------------------------------------------------------
# Forecast model — engineered lag/rolling features (excluding the one-hot
# instance_type columns, which are appended at train time from the data).
# ---------------------------------------------------------------------------
FORECAST_BASE_FEATURES = [
    "lag_1",
    "lag_7",
    "cost_roll7",
    "cost_roll30",
    "cpu_roll7",
    "cpu_roll30",
    "memory_roll7",
    "memory_roll30",
    "net_in_roll7",
    "net_in_roll30",
    "net_out_roll7",
    "net_out_roll30",
    "day_of_week",
]


def add_anomaly_features(df):
    """Add per-instance z-score and day-over-day pct change of cost.

    Requires columns: instance_id, date, cost_usd. Returns a sorted copy with
    ANOMALY_FEATURES added. Identical transform used by train_anomaly.py.
    """
    df = df.sort_values(["instance_id", "date"]).reset_index(drop=True).copy()
    g = df.groupby("instance_id", group_keys=False)

    mean = g["cost_usd"].transform("mean")
    std = g["cost_usd"].transform("std").replace(0, 1e-6)
    df["cost_z"] = (df["cost_usd"] - mean) / std

    df["cost_pct_chg"] = g["cost_usd"].pct_change().fillna(0.0)
    return df


def build_forecast_features(df):
    """Per-instance lag + rolling features; target is next-day baseline cost.

    Requires columns: instance_id, date, cost_usd, is_anomaly, cpu_avg, memory_avg,
    network_in, network_out. Returns a sorted copy with FORECAST_BASE_FEATURES (plus
    cost_base, target, target_is_anomaly).

    FEATURE DECISION: the dataset contains planted cost spikes (3-5x normal) that are, by
    construction, unpredictable from prior usage — exactly what Model 3 (anomaly) flags.
    Leaving them in (a) corrupts the lag/rolling features and (b) makes the target
    impossible to forecast. So the forecaster models the *baseline* cost: anomaly-day cost
    is replaced by the per-instance median before building lags/rollings and the target.
    At serve time the API runs the anomaly model first to populate `is_anomaly`, so this
    de-spiking happens identically online and offline.
    """
    df = df.sort_values(["instance_id", "date"]).copy()
    g = df.groupby("instance_id", group_keys=False)

    # De-spiked cost: replace planted-anomaly days with the instance's median cost.
    inst_median = g["cost_usd"].transform("median")
    df["cost_base"] = np.where(df["is_anomaly"].astype(bool), inst_median, df["cost_usd"])

    g = df.groupby("instance_id", group_keys=False)  # regroup with new column

    # Lag features of (de-spiked) cost
    df["lag_1"] = g["cost_base"].shift(1)
    df["lag_7"] = g["cost_base"].shift(7)

    # Rolling 7/30-day means (shifted by 1 to avoid leaking the current day).
    roll_specs = {
        "cost_base": "cost",
        "cpu_avg": "cpu",
        "memory_avg": "memory",
        "network_in": "net_in",
        "network_out": "net_out",
    }
    for col, short in roll_specs.items():
        for win in (7, 30):
            df[f"{short}_roll{win}"] = g[col].apply(
                lambda s: s.shift(1).rolling(win, min_periods=1).mean()
            )

    # Calendar feature
    df["day_of_week"] = df["date"].dt.dayofweek

    # Target: next-day baseline cost per instance
    df["target"] = g["cost_base"].shift(-1)
    df["target_is_anomaly"] = g["is_anomaly"].shift(-1).fillna(False).astype(bool)
    return df
