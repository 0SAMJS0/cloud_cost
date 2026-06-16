"""
Generate realistic sample payloads for the four endpoints straight from the processed
data + ground truth, and write them to tests/samples.json. Run once:

    py tests/make_samples.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "usage_processed.csv"
GT = ROOT / "data" / "raw" / "ground_truth.csv"
OUT = Path(__file__).resolve().parent / "samples.json"


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    gt = pd.read_csv(GT, parse_dates=["date"])
    labelled = df.merge(gt, on=["date", "instance_id"], how="inner")

    # --- forecast: last 30 days of one instance ---
    inst = df["instance_id"].iloc[0]
    hist = (df[df["instance_id"] == inst].sort_values("date").tail(30))
    forecast_payload = {
        "instance_id": inst,
        "instance_type": str(hist["instance_type"].iloc[0]),
        "history": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "cpu_avg": round(float(c), 2),
                "memory_avg": round(float(m), 2),
                "network_in": round(float(ni), 4),
                "network_out": round(float(no), 4),
                "cost_usd": round(float(cost), 4),
            }
            for d, c, m, ni, no, cost in zip(
                hist["date"], hist["cpu_avg"], hist["memory_avg"],
                hist["network_in"], hist["network_out"], hist["cost_usd"]
            )
        ],
    }

    # --- waste: a representative day from a wasteful instance ---
    waste_ids = labelled.groupby("instance_id")["is_wasteful"].first()
    wasteful_inst = waste_ids[waste_ids].index[0]
    healthy_inst = waste_ids[~waste_ids].index[0]
    wrow = df[df["instance_id"] == wasteful_inst].iloc[-1]
    waste_payload = {
        "cpu_avg": round(float(wrow["cpu_avg"]), 2),
        "memory_avg": round(float(wrow["memory_avg"]), 2),
        "network_in": round(float(wrow["network_in"]), 4),
        "network_out": round(float(wrow["network_out"]), 4),
        "instance_type": str(wrow["instance_type"]),
        "current_monthly_cost": round(float(wrow["cost_usd"]) * 30, 2),
    }
    hrow = df[df["instance_id"] == healthy_inst].iloc[-1]
    waste_healthy_payload = {
        "cpu_avg": round(float(hrow["cpu_avg"]), 2),
        "memory_avg": round(float(hrow["memory_avg"]), 2),
        "network_in": round(float(hrow["network_in"]), 4),
        "network_out": round(float(hrow["network_out"]), 4),
        "instance_type": str(hrow["instance_type"]),
        "current_monthly_cost": round(float(hrow["cost_usd"]) * 30, 2),
    }

    # --- anomaly: a window around a known planted spike for some instance ---
    anom_inst = labelled[labelled["is_anomaly"]]["instance_id"].iloc[0]
    spike_date = labelled[(labelled["instance_id"] == anom_inst) &
                          (labelled["is_anomaly"])]["date"].iloc[0]
    series = (labelled[labelled["instance_id"] == anom_inst]
              .sort_values("date"))
    window = series[(series["date"] >= spike_date - pd.Timedelta(days=20)) &
                    (series["date"] <= spike_date + pd.Timedelta(days=20))]
    anomaly_payload = {
        "instance_id": anom_inst,
        "series": [
            {"date": d.strftime("%Y-%m-%d"), "cost_usd": round(float(c), 4)}
            for d, c in zip(window["date"], window["cost_usd"])
        ],
    }

    samples = {
        "forecast": forecast_payload,
        "waste_wasteful": waste_payload,
        "waste_healthy": waste_healthy_payload,
        "anomaly": anomaly_payload,
        "_meta": {
            "forecast_instance": inst,
            "wasteful_instance": wasteful_inst,
            "healthy_instance": healthy_inst,
            "anomaly_instance": anom_inst,
            "spike_date": spike_date.strftime("%Y-%m-%d"),
        },
    }

    OUT.write_text(json.dumps(samples, indent=2))
    print(f"Wrote {OUT}")
    print(json.dumps(samples["_meta"], indent=2))


if __name__ == "__main__":
    main()
