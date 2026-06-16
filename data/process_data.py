"""
process_data.py
===============
Phase 1 — Data First (the cleaning half).

Loads the dirty raw file (data/raw/usage_2024.csv), cleans it, and writes a
trustworthy version to data/processed/usage_processed.csv. This is the file
every later phase will actually train and serve on.

Design rule: clean the JUNK, keep the SIGNAL. The raw file has two very
different kinds of "weird" values:

  * Junk    -> bad sensor reads (CPU = 130%, negative network), missing cells,
               duplicates, messy strings. We fix or drop these.
  * Signal  -> the planted 3-5x cost spikes. Those are real anomalies the
               Phase 2 model has to catch, so we MUST NOT remove them, even
               though they look like outliers.

That distinction is the whole point of this step. A naive "drop all outliers"
clean would silently delete the exact rows the project is built to detect.

Run (after generate_data.py):
    python data/process_data.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "usage_2024.csv")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "usage_processed.csv")

VALID_TYPES = {
    "t3.micro", "t3.small", "t3.medium", "t3.large",
    "m5.large", "m5.xlarge", "m5.2xlarge",
    "c5.large", "c5.xlarge", "r5.large", "r5.xlarge",
}

EXPECTED_COLS = [
    "date", "instance_id", "instance_type",
    "cpu_avg", "memory_avg", "network_in", "network_out", "cost_usd",
]


def load_raw() -> pd.DataFrame:
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Run data/synthetic/generate_data.py first."
        )
    return pd.read_csv(RAW_PATH)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict[str, int] = {}
    report["rows_in"] = len(df)

    # 1) Drop exact duplicate rows. -----------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_dropped"] = before - len(df)

    # 2) Parse dates; drop rows with unparseable/blank dates. ---------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["date"])
    report["bad_dates_dropped"] = before - len(df)

    # 3) Normalize instance_type strings (strip, lowercase) and validate. ---
    df["instance_type"] = df["instance_type"].astype(str).str.strip().str.lower()
    before = len(df)
    df = df[df["instance_type"].isin(VALID_TYPES)]
    report["unknown_types_dropped"] = before - len(df)

    # 4) Coerce numeric columns; non-numeric -> NaN. ------------------------
    num_cols = ["cpu_avg", "memory_avg", "network_in", "network_out", "cost_usd"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 5) Fix impossible values WITHOUT killing real cost spikes. ------------
    #    - CPU/memory are percentages: anything outside [0, 100] is a bad read.
    #      Set out-of-range to NaN so we can impute, rather than clipping a
    #      130% read down to a plausible-looking 100% (which would hide the
    #      sensor fault).
    for c in ["cpu_avg", "memory_avg"]:
        df.loc[(df[c] < 0) | (df[c] > 100), c] = np.nan
    #    - Network can't be negative. Negatives are corruption -> NaN.
    for c in ["network_in", "network_out"]:
        df.loc[df[c] < 0, c] = np.nan
    #    - cost_usd: negatives are impossible; drop them. But large positive
    #      costs are LEFT ALONE on purpose — those are the anomalies.
    before = len(df)
    df = df[df["cost_usd"].fillna(0) >= 0]
    report["negative_cost_dropped"] = before - len(df)

    # 6) Impute remaining missing numeric values. ---------------------------
    #    Per-instance median is the safest fill: an idle box stays idle, a
    #    busy box stays busy. Falls back to the global median if an instance
    #    has no valid values for that column.
    report["missing_cells_imputed"] = int(df[num_cols].isna().sum().sum())
    for c in num_cols:
        df[c] = df.groupby("instance_id")[c].transform(
            lambda s: s.fillna(s.median())
        )
        df[c] = df[c].fillna(df[c].median())

    # 7) Sort, round, finalize. ---------------------------------------------
    df = df.sort_values(["instance_id", "date"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    for c in num_cols:
        df[c] = df[c].round(4)

    df = df[EXPECTED_COLS]
    report["rows_out"] = len(df)
    return df, report


def sanity_checks(df: pd.DataFrame) -> None:
    """Fail loudly if the processed file is not actually clean."""
    assert df[EXPECTED_COLS].isna().sum().sum() == 0, "NaNs remain after cleaning"
    assert df["cpu_avg"].between(0, 100).all(), "cpu_avg out of [0,100]"
    assert df["memory_avg"].between(0, 100).all(), "memory_avg out of [0,100]"
    assert (df["network_in"] >= 0).all(), "negative network_in remains"
    assert (df["network_out"] >= 0).all(), "negative network_out remains"
    assert (df["cost_usd"] >= 0).all(), "negative cost remains"
    assert df["instance_type"].isin(VALID_TYPES).all(), "invalid instance_type remains"


def main() -> None:
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    raw = load_raw()
    clean_df, report = clean(raw)
    sanity_checks(clean_df)
    clean_df.to_csv(PROCESSED_PATH, index=False)

    print("Cleaning report")
    print("-" * 40)
    for k, v in report.items():
        print(f"  {k:<24} {v:>8,}")
    print("-" * 40)
    print(f"[ok] wrote {len(clean_df):,} clean rows -> {PROCESSED_PATH}")
    print(f"     instances: {clean_df['instance_id'].nunique()}  "
          f"date range: {clean_df['date'].min()} -> {clean_df['date'].max()}")


if __name__ == "__main__":
    main()
