"""
Phase 2 - Model 2: Waste Detection (binary classification)

Flags under-utilised instances and emits a 0-1 waste score (predict_proba).

Labels: is_wasteful from ground_truth.csv. This label is CONSTANT per instance, so a
naive row-level split would let the same instance appear in both train and test and leak
the answer. We therefore use a GROUPED split by instance_id (GroupShuffleSplit) so an
instance is wholly in train OR test.

Trains LogisticRegression (interpretable) then RandomForestClassifier, keeps the better
one by precision (the success metric).

Run:  py models/train/train_waste.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Shared feature list (same code the API serves with -> no skew).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.features import WASTE_FEATURES

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

# The project doc lists "uptime hours" as a feature, but the dataset has no uptime
# column, so we use network in/out as the extra utilisation signal instead.
FEATURES = WASTE_FEATURES
PRECISION_TARGET = 0.80


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    gt = pd.read_csv(GT, parse_dates=["date"])

    # Attach labels via INNER join on (date, instance_id)
    df = df.merge(gt[["date", "instance_id", "is_wasteful"]],
                  on=["date", "instance_id"], how="inner")
    df["is_wasteful"] = df["is_wasteful"].astype(int)

    X = df[FEATURES]
    y = df["is_wasteful"]
    groups = df["instance_id"]

    n_inst = groups.nunique()
    n_waste_inst = df.groupby("instance_id")["is_wasteful"].first().sum()
    print(f"Instances: {n_inst} ({n_waste_inst} wasteful)  rows: {len(df)}")

    # Grouped split: same instance never in both train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    test_inst = groups.iloc[test_idx].nunique()
    test_waste_inst = int(df.iloc[test_idx].groupby("instance_id")["is_wasteful"].first().sum())
    print(f"Held-out test instances: {test_inst} ({test_waste_inst} wasteful)")

    # --- LogisticRegression (interpretable, scaled) ---
    lr = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prec = precision_score(y_test, lr_pred, zero_division=0)
    lr_rec = recall_score(y_test, lr_pred, zero_division=0)

    # --- RandomForestClassifier ---
    rf = RandomForestClassifier(n_estimators=300, random_state=42,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prec = precision_score(y_test, rf_pred, zero_division=0)
    rf_rec = recall_score(y_test, rf_pred, zero_division=0)

    print("\n--- Waste model comparison (held-out instances) ---")
    print(f"{'LogisticRegression':22s}  precision={lr_prec:.3f}  recall={lr_rec:.3f}")
    print(f"{'RandomForestClassifier':22s}  precision={rf_prec:.3f}  recall={rf_rec:.3f}")

    # Keep the better model by precision (tie-break on recall)
    if (lr_prec, lr_rec) >= (rf_prec, rf_rec):
        best_name, best_model = "LogisticRegression", lr
        best_prec, best_rec, best_pred = lr_prec, lr_rec, lr_pred
    else:
        best_name, best_model = "RandomForestClassifier", rf
        best_prec, best_rec, best_pred = rf_prec, rf_rec, rf_pred

    best_f1 = f1_score(y_test, best_pred, zero_division=0)
    print(f"\nBest model: {best_name}  precision={best_prec:.3f}  "
          f"recall={best_rec:.3f}  f1={best_f1:.3f}")
    passed = best_prec > PRECISION_TARGET
    print(f"Success target precision > {PRECISION_TARGET:.0%} -> "
          f"{'PASS' if passed else 'FAIL'}")

    # Demonstrate the 0-1 waste score
    scores = best_model.predict_proba(X_test)[:, 1]
    print(f"Waste score (predict_proba) range on test: "
          f"[{scores.min():.3f}, {scores.max():.3f}]")

    joblib.dump({"model": best_model, "features": FEATURES, "name": best_name},
                SAVED / "waste_model.joblib")

    metrics = {
        "model": "waste",
        "best_model": best_name,
        "precision": float(best_prec),
        "recall": float(best_rec),
        "f1": float(best_f1),
        "lr_precision": float(lr_prec),
        "lr_recall": float(lr_rec),
        "rf_precision": float(rf_prec),
        "rf_recall": float(rf_rec),
        "test_instances": int(test_inst),
        "threshold_precision": PRECISION_TARGET,
        "pass": bool(passed),
    }
    with open(EVAL / "waste_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model -> {SAVED / 'waste_model.joblib'}")
    print(f"Saved metrics -> {EVAL / 'waste_metrics.json'}")


if __name__ == "__main__":
    main()
