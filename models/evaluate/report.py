"""
Phase 2 scorecard.

Loads the three metrics JSON files written by the training scripts and prints a
PASS/FAIL summary against each success threshold:
  - Forecast MAPE      < 15%
  - Waste precision    > 80%
  - Anomaly FP rate    < 5%   (also reports recall)

Run:  py models/evaluate/report.py
"""

import json
from pathlib import Path

EVAL = Path(__file__).resolve().parent


def load(name):
    path = EVAL / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def line(label, value, ok):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}]  {label:32s} {value}")


def main():
    forecast = load("forecast_metrics.json")
    waste = load("waste_metrics.json")
    anomaly = load("anomaly_metrics.json")

    print("=" * 60)
    print("        PHASE 2 SCORECARD - AI Cloud Cost Optimizer")
    print("=" * 60)

    results = []

    if forecast:
        mape = forecast["mape"]
        ok = mape < 15.0
        line(f"Forecast MAPE < 15%", f"MAPE = {mape:.2f}%  "
             f"(model: {forecast['best_model']}, RMSE={forecast['rmse']:.3f})", ok)
        results.append(ok)
    else:
        print("  [MISS]  forecast_metrics.json not found - run train_forecast.py")
        results.append(False)

    if waste:
        prec = waste["precision"]
        ok = prec > 0.80
        line("Waste precision > 80%",
             f"precision = {prec:.1%}  (recall={waste['recall']:.1%}, "
             f"model: {waste['best_model']})", ok)
        results.append(ok)
    else:
        print("  [MISS]  waste_metrics.json not found - run train_waste.py")
        results.append(False)

    if anomaly:
        fp = anomaly["fp_rate"]
        ok = fp < 0.05
        line("Anomaly FP rate < 5%",
             f"FP rate = {fp:.2%}  (recall={anomaly['recall']:.1%}, "
             f"{anomaly['tp']}/{anomaly['n_anomalies']} spikes caught)", ok)
        results.append(ok)
    else:
        print("  [MISS]  anomaly_metrics.json not found - run train_anomaly.py")
        results.append(False)

    print("=" * 60)
    overall = "ALL THRESHOLDS PASSED" if all(results) else "SOME THRESHOLDS FAILED"
    print(f"  OVERALL: {overall}  ({sum(results)}/{len(results)} passed)")
    print("=" * 60)


if __name__ == "__main__":
    main()
