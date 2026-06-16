"""
Phase 2 - Model 3: Anomaly Detection (unsupervised, IsolationForest)

Detects planted cost spikes (3-5x normal). The CRITICAL trick is to NOT feed raw
cost_usd: expensive instance types would otherwise be flagged just for being expensive.
Instead we normalise cost PER INSTANCE:
  - cost_z       : z-score of cost_usd within each instance_id
  - cost_pct_chg : day-over-day percentage change of cost within each instance_id
A real spike stands out on both axes regardless of the instance's baseline price.

contamination ~0.015 matches the true anomaly rate. Evaluated against is_anomaly:
recall (fraction of planted spikes caught) and false-positive rate.

Run:  py models/train/train_anomaly.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# Shared feature transforms (same code the API serves with -> no skew).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.features import add_anomaly_features, ANOMALY_FEATURES

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "usage_processed.csv"
GT = ROOT / "data" / "raw" / "ground_truth.csv"
SAVED = ROOT / "models" / "saved"
EVAL = ROOT / "models" / "evaluate"
SAVED.mkdir(parents=True, exist_ok=True)
EVAL.mkdir(parents=True, exist_ok=True)

# The true planted-anomaly rate is 65/3635 = 0.0179, so contamination is set to the
# real rate (~0.018). With per-instance normalised features the FP rate stays far below
# the 5% budget, so matching the true rate maximises recall essentially for free.
CONTAMINATION = 0.018
FP_TARGET = 0.05  # false-positive rate must stay under 5%


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    gt = pd.read_csv(GT, parse_dates=["date"])

    # Per-instance normalisation (do NOT use raw cost_usd). Built by the shared module.
    df = add_anomaly_features(df)
    features = ANOMALY_FEATURES
    X = df[features].to_numpy()

    iso = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X)
    pred = iso.predict(X)          # -1 = anomaly, 1 = normal
    df["pred_anomaly"] = (pred == -1).astype(int)

    # --- Evaluate against ground truth (inner join on date + instance_id) ---
    merged = df.merge(gt[["date", "instance_id", "is_anomaly"]],
                      on=["date", "instance_id"], how="inner")
    merged["is_anomaly"] = merged["is_anomaly"].astype(int)

    y_true = merged["is_anomaly"].to_numpy()
    y_pred = merged["pred_anomaly"].to_numpy()

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    n_anom = int(y_true.sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    print(f"Rows: {len(merged)}  planted anomalies: {n_anom}  flagged: {tp + fp}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"\nRecall (spikes caught): {recall:.3f}  ({tp}/{n_anom})")
    print(f"False-positive rate:    {fp_rate:.4f}")
    print(f"Precision:              {precision:.3f}")

    passed = (fp_rate < FP_TARGET) and (recall > 0.0)
    print(f"\nSuccess target FP rate < {FP_TARGET:.0%} -> "
          f"{'PASS' if passed else 'FAIL'}")

    joblib.dump({"model": iso, "features": features, "contamination": CONTAMINATION},
                SAVED / "anomaly_model.joblib")

    metrics = {
        "model": "anomaly",
        "recall": float(recall),
        "fp_rate": float(fp_rate),
        "precision": float(precision),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_anomalies": n_anom,
        "contamination": CONTAMINATION,
        "threshold_fp": FP_TARGET,
        "pass": bool(passed),
    }
    with open(EVAL / "anomaly_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model -> {SAVED / 'anomaly_model.joblib'}")
    print(f"Saved metrics -> {EVAL / 'anomaly_metrics.json'}")


if __name__ == "__main__":
    main()
