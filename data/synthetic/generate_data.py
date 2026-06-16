"""
generate_data.py
================
Phase 1 — Data First.

Generates 6 months of *realistic-looking* daily AWS EC2 usage data across
20 simulated instances, then writes it to data/raw/usage_2024.csv.

Why synthetic? Because we control the ground truth. We know which instances
are wasteful and which days are anomalies, so in Phase 2 we can actually
measure whether our models found them. Real AWS Cost & Usage Reports never
hand you labels.

The raw file is deliberately a little DIRTY (missing values, dupes, a few
impossible numbers, messy instance-type strings). That is on purpose — real
infrastructure data is messy, and process_data.py exists to clean it. Do NOT
"fix" the dirt here; fix it in the cleaning step.

Ground truth (which instances are wasteful, which rows are planted spikes) is
written to a SEPARATE file, data/raw/ground_truth.csv, so it never leaks into
the features your models train on.

Run:
    python data/synthetic/generate_data.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility. Same seed => same dataset. This is how we "version" the
# raw data: anyone who runs this script gets byte-identical numbers, so we can
# retrain later and know exactly what we trained on.
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

# Resolve paths relative to the project root, no matter where you run from.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
RAW_PATH = os.path.join(RAW_DIR, "usage_2024.csv")
GROUND_TRUTH_PATH = os.path.join(RAW_DIR, "ground_truth.csv")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START_DATE = "2024-01-01"
N_DAYS = 182                # ~6 months: Jan 1 -> Jun 30 2024
N_INSTANCES = 20
WASTEFUL_FRACTION = 0.40    # 8 of 20 instances are wasteful (CPU 2-8%)
ANOMALY_RATE = 0.015        # ~1.5% of instance-days get a cost spike
SPIKE_MIN, SPIKE_MAX = 3.0, 5.0  # spikes are 3-5x normal cost

# Approx us-east-1 on-demand $/hr. Realistic ballpark, not billing-accurate.
INSTANCE_CATALOG = {
    "t3.micro":   0.0104,
    "t3.small":   0.0208,
    "t3.medium":  0.0416,
    "t3.large":   0.0832,
    "m5.large":   0.0960,
    "m5.xlarge":  0.1920,
    "m5.2xlarge": 0.3840,
    "c5.large":   0.0850,
    "c5.xlarge":  0.1700,
    "r5.large":   0.1260,
    "r5.xlarge":  0.2520,
}
INSTANCE_TYPES = list(INSTANCE_CATALOG.keys())

EGRESS_USD_PER_GB = 0.09   # AWS data-transfer-out pricing, first tier


def build_instance_profiles() -> pd.DataFrame:
    """Decide, per instance, its type and whether it is healthy or wasteful."""
    n_wasteful = int(round(N_INSTANCES * WASTEFUL_FRACTION))
    is_wasteful = np.array([True] * n_wasteful + [False] * (N_INSTANCES - n_wasteful))
    rng.shuffle(is_wasteful)

    rows = []
    for i in range(N_INSTANCES):
        inst_type = rng.choice(INSTANCE_TYPES)
        rows.append(
            {
                "instance_id": f"i-{i:04d}{rng.integers(0, 0xFFFF):04x}",
                "instance_type": inst_type,
                "hourly_rate": INSTANCE_CATALOG[inst_type],
                "is_wasteful": bool(is_wasteful[i]),
                # Healthy instances follow a business-hours pattern; wasteful
                # ones are roughly flat (nobody is using them).
                "business_hours": (not is_wasteful[i]) and bool(rng.random() < 0.8),
            }
        )
    return pd.DataFrame(rows)


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    profiles = build_instance_profiles()

    records = []
    truth_records = []

    for _, prof in profiles.iterrows():
        # A gentle upward trend over the 6 months (the company is scaling).
        trend = np.linspace(1.0, 1.25, N_DAYS)

        for d_idx, date in enumerate(dates):
            weekday = date.weekday()  # 0=Mon ... 6=Sun
            is_weekend = weekday >= 5

            # --- CPU ---------------------------------------------------------
            if prof["is_wasteful"]:
                cpu = rng.normal(5.0, 1.8)            # idle box, ~2-8%
            else:
                base = rng.normal(55.0, 8.0)          # healthy, ~40-70%
                if prof["business_hours"] and is_weekend:
                    base *= 0.45                       # quiet on weekends
                cpu = base
            cpu = float(np.clip(cpu, 0.5, 99.0))

            # --- Memory: loosely tracks CPU but has its own life ------------
            memory = float(np.clip(cpu * rng.uniform(0.7, 1.3) + rng.normal(8, 5), 1.0, 99.0))

            # --- Network (GB/day): activity-driven --------------------------
            activity = cpu / 100.0
            seasonal = 0.6 if (prof["business_hours"] and is_weekend) else 1.0
            net_in = max(0.0, rng.normal(8 * activity, 2) * seasonal * trend[d_idx])
            net_out = max(0.0, rng.normal(5 * activity, 1.5) * seasonal * trend[d_idx])

            # --- Uptime: occasionally instances are stopped part of the day -
            uptime_frac = 1.0
            if rng.random() < 0.05:
                uptime_frac = float(rng.uniform(0.3, 0.9))

            # --- Cost -------------------------------------------------------
            compute_cost = prof["hourly_rate"] * 24 * uptime_frac
            egress_cost = net_out * EGRESS_USD_PER_GB
            cost = compute_cost + egress_cost
            cost *= rng.normal(1.0, 0.03)  # small day-to-day billing noise

            # --- Planted anomaly: rare cost spike ---------------------------
            is_anomaly = rng.random() < ANOMALY_RATE
            if is_anomaly:
                cost *= float(rng.uniform(SPIKE_MIN, SPIKE_MAX))
                # spikes usually come with a data-transfer burst
                net_out *= float(rng.uniform(2.0, 4.0))

            records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "instance_id": prof["instance_id"],
                    "instance_type": prof["instance_type"],
                    "cpu_avg": round(cpu, 2),
                    "memory_avg": round(memory, 2),
                    "network_in": round(net_in, 4),
                    "network_out": round(net_out, 4),
                    "cost_usd": round(max(cost, 0.0), 4),
                }
            )
            truth_records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "instance_id": prof["instance_id"],
                    "is_wasteful": prof["is_wasteful"],
                    "is_anomaly": is_anomaly,
                }
            )

    df = pd.DataFrame(records)
    truth = pd.DataFrame(truth_records)
    return df, truth


def inject_dirt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the raw file realistically messy so the cleaning step has real work.
    Everything here is the kind of junk you genuinely see in exported infra
    data. None of it touches the planted cost spikes — those are real signal.
    """
    df = df.copy()
    n = len(df)

    # 1) Missing values in cpu_avg / memory_avg (sensor gaps).
    miss_cpu = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[miss_cpu, "cpu_avg"] = np.nan
    miss_mem = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[miss_mem, "memory_avg"] = np.nan

    # 2) A few impossible values (bad sensor reads).
    bad = rng.choice(n, size=12, replace=False)
    df.loc[bad[:4], "cpu_avg"] = rng.uniform(101, 130, size=4)   # CPU > 100%
    df.loc[bad[4:8], "cpu_avg"] = rng.uniform(-10, -1, size=4)   # negative CPU
    df.loc[bad[8:], "network_in"] = rng.uniform(-5, -0.1, size=4)  # negative net

    # 3) Messy instance_type strings (casing + whitespace).
    messy = rng.choice(n, size=int(n * 0.03), replace=False)
    df.loc[messy, "instance_type"] = df.loc[messy, "instance_type"].apply(
        lambda s: f"  {s.upper()} " if rng.random() < 0.5 else f" {s.capitalize()}"
    )

    # 4) A handful of blank/garbled dates.
    bad_dates = rng.choice(n, size=5, replace=False)
    df.loc[bad_dates, "date"] = ""

    # 5) Some exact duplicate rows.
    dupes = df.sample(n=int(n * 0.01), random_state=SEED)
    df = pd.concat([df, dupes], ignore_index=True)

    # Shuffle so the dirt isn't all at the bottom.
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(RAW_PATH):
        # The spec is explicit: never overwrite raw data once generated.
        print(f"[skip] {RAW_PATH} already exists — refusing to overwrite raw data.")
        print("       Delete it manually if you really want to regenerate.")
        return

    clean_df, truth = generate()
    raw_df = inject_dirt(clean_df)

    raw_df.to_csv(RAW_PATH, index=False)
    truth.to_csv(GROUND_TRUTH_PATH, index=False)

    print(f"[ok] wrote {len(raw_df):,} raw rows -> {RAW_PATH}")
    print(f"[ok] wrote {len(truth):,} ground-truth rows -> {GROUND_TRUTH_PATH}")
    print(f"     instances: {raw_df['instance_id'].nunique()}  "
          f"wasteful: {int(truth.groupby('instance_id')['is_wasteful'].first().sum())}  "
          f"anomalies: {int(truth['is_anomaly'].sum())}")


if __name__ == "__main__":
    main()
