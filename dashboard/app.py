"""
Phase 4 — Streamlit dashboard for the AI-Powered Cloud Cost Optimizer.

Single page, four sections:
  1. Summary KPI cards
  2. Cost forecast chart (actual + walk-forward predicted baseline + 7-day projection)
  3. Waste leaderboard (top 10 by waste score)
  4. Anomaly timeline (flagged spike days highlighted)

It loads the trained models directly (no API server needed) and reuses the SHARED feature
code so there is zero train/serve skew:
  - api.model_store.ModelStore      -> loads the 3 models once
  - common.features                 -> the single source of truth for feature engineering
  - api.features.run_anomaly_detection -> the same anomaly wiring the API uses

Run:  py -m streamlit run dashboard/app.py
"""

import sqlite3
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Make the project root importable regardless of where streamlit is launched from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.model_store import ModelStore  # noqa: E402
from api.features import run_anomaly_detection  # noqa: E402  (same wiring as the API)
from common.features import (  # noqa: E402
    build_forecast_features,
    FORECAST_BASE_FEATURES,
    WASTE_FEATURES,
)

DATA = ROOT / "data" / "processed" / "usage_processed.csv"
GT = ROOT / "data" / "raw" / "ground_truth.csv"
DB = ROOT / "predictions.db"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    return ModelStore().load()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA, parse_dates=["date"])
    gt = pd.read_csv(GT, parse_dates=["date"])
    return df, gt


@st.cache_data
def compute_anomalies(_models, df):
    """Flag anomalies across the whole dataset (per-instance z-score, same as training)."""
    feat = run_anomaly_detection(df[["instance_id", "date", "cost_usd"]], _models.anomaly["model"])
    return feat[["instance_id", "date", "cost_usd", "is_anomaly", "anomaly_score"]]


@st.cache_data
def compute_waste_leaderboard(_models, df):
    """One row per instance: mean utilisation -> waste score + savings estimate."""
    agg = (
        df.groupby(["instance_id", "instance_type"])
        .agg(
            cpu_avg=("cpu_avg", "mean"),
            memory_avg=("memory_avg", "mean"),
            network_in=("network_in", "mean"),
            network_out=("network_out", "mean"),
            daily_cost=("cost_usd", "mean"),
        )
        .reset_index()
    )
    agg["waste_score"] = _models.waste["model"].predict_proba(agg[WASTE_FEATURES])[:, 1]
    agg["monthly_cost"] = agg["daily_cost"] * 30.0

    def recommend(r):
        if r["waste_score"] >= 0.5:
            idle = r["cpu_avg"] < 10 and r["memory_avg"] < 15 and (r["network_in"] + r["network_out"]) < 1.0
            return "Terminate" if idle else "Downsize"
        return "Healthy"

    agg["recommendation"] = agg.apply(recommend, axis=1)
    agg["est_monthly_savings"] = agg.apply(
        lambda r: r["monthly_cost"] if r["recommendation"] == "Terminate"
        else 0.5 * r["monthly_cost"] if r["recommendation"] == "Downsize"
        else 0.0,
        axis=1,
    )
    return agg.sort_values("waste_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Forecast helpers (orchestration only — feature math comes from common.features)
# ---------------------------------------------------------------------------
def _align_X(fdf, feat_list, instance_type):
    """Reindex engineered features to the model's saved column order, set the one-hot type."""
    X = fdf[FORECAST_BASE_FEATURES].reindex(columns=feat_list, fill_value=0)
    col = f"itype_{instance_type}"
    if col in feat_list:
        X[col] = 1
    return X


def walk_forward(inst_df, models):
    """1-step predicted baseline for every day: model predicts next-day cost, which we
    align to the day it refers to. Returns (engineered_df, predicted_df[date, predicted])."""
    feat = run_anomaly_detection(inst_df, models.anomaly["model"])  # adds is_anomaly
    fdf = build_forecast_features(feat)
    itype = str(inst_df["instance_type"].iloc[0])
    X = _align_X(fdf, models.forecast["features"], itype)
    fdf = fdf.assign(pred_next=models.forecast["model"].predict(X))
    pred = pd.DataFrame({
        "date": fdf["date"] + pd.Timedelta(days=1),   # prediction at row d is for day d+1
        "predicted": fdf["pred_next"].values,
    })
    return fdf, pred


def recursive_forecast(inst_df, models, n_days=7):
    """Roll the model forward n_days, feeding each prediction back in as next-day baseline."""
    work = inst_df.sort_values("date").copy()
    itype = str(work["instance_type"].iloc[0])
    feat_list, model = models.forecast["features"], models.forecast["model"]
    last_usage = work.iloc[-1][["cpu_avg", "memory_avg", "network_in", "network_out"]]
    out = []
    for _ in range(n_days):
        feat = run_anomaly_detection(work, models.anomaly["model"])
        fdf = build_forecast_features(feat)
        X = _align_X(fdf.iloc[[-1]], feat_list, itype)
        yhat = float(model.predict(X)[0])
        next_date = work["date"].iloc[-1] + pd.Timedelta(days=1)
        out.append((next_date, yhat))
        work = pd.concat([work, pd.DataFrame([{
            "date": next_date, "instance_id": work["instance_id"].iloc[0],
            "instance_type": itype, "cpu_avg": last_usage["cpu_avg"],
            "memory_avg": last_usage["memory_avg"], "network_in": last_usage["network_in"],
            "network_out": last_usage["network_out"], "cost_usd": yhat,
        }])], ignore_index=True)
    return pd.DataFrame(out, columns=["date", "projected"])


@st.cache_data
def recent_predictions(_db_path):
    if not Path(_db_path).exists():
        return None
    try:
        conn = sqlite3.connect(_db_path)
        rows = conn.execute(
            "SELECT timestamp, endpoint, input_summary FROM predictions ORDER BY id DESC LIMIT 5"
        ).fetchall()
        conn.close()
        return pd.DataFrame(rows, columns=["timestamp", "endpoint", "input_summary"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Cloud Cost Optimizer", layout="wide")
st.title("☁️ AI Cloud Cost Optimizer — Dashboard")

models = load_models()
df, gt = load_data()
leaderboard = compute_waste_leaderboard(models, df)
anomalies = compute_anomalies(models, df)

# ---- Section 1: KPI cards ----
n_instances = df["instance_id"].nunique()
n_wasteful = int((leaderboard["waste_score"] >= 0.5).sum())
total_savings = float(leaderboard["est_monthly_savings"].sum())
n_anomalies = int(anomalies["is_anomaly"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Wasteful Instances", f"{n_wasteful}", help=f"of {n_instances} monitored")
c2.metric("Est. Monthly Savings", f"${total_savings:,.0f}")
c3.metric("Anomalies Found", f"{n_anomalies}")
c4.metric("Instances Monitored", f"{n_instances}")

st.divider()

# ---- Instance selector (drives sections 2 & 4) ----
instance_ids = sorted(df["instance_id"].unique())
selected = st.selectbox("Select an instance", instance_ids)
inst_df = df[df["instance_id"] == selected].sort_values("date").reset_index(drop=True)

# ---- Section 2: Cost forecast ----
st.subheader(f"📈 Cost Forecast — {selected}")
fdf, pred = walk_forward(inst_df, models)
proj = recursive_forecast(inst_df, models, n_days=7)

last30 = inst_df.tail(30)
window_start = last30["date"].min()
pred_win = pred[pred["date"] >= window_start]

chart_rows = []
for d, v in zip(last30["date"], last30["cost_usd"]):
    chart_rows.append({"date": d, "value": float(v), "series": "Actual"})
for d, v in zip(pred_win["date"], pred_win["predicted"]):
    if pd.notna(v) and d <= last30["date"].max():
        chart_rows.append({"date": d, "value": float(v), "series": "Predicted baseline"})
for d, v in zip(proj["date"], proj["projected"]):
    chart_rows.append({"date": d, "value": float(v), "series": "7-day projection"})

chart_df = pd.DataFrame(chart_rows)
forecast_chart = (
    alt.Chart(chart_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Cost (USD/day)"),
        color=alt.Color("series:N", title=""),
        strokeDash=alt.condition(
            alt.datum.series == "7-day projection", alt.value([5, 5]), alt.value([0])
        ),
    )
    .properties(height=340)
)
st.altair_chart(forecast_chart, use_container_width=True)
st.caption(
    "Actual daily cost vs the model's de-spiked baseline (walk-forward 1-step), plus a "
    "recursive 7-day forward projection. Input is de-spiked via the anomaly model first, "
    "exactly as the API does."
)

st.divider()

# ---- Section 3: Waste leaderboard ----
st.subheader("🗑️ Waste Leaderboard — Top 10")
top10 = leaderboard.head(10).copy()
display = pd.DataFrame({
    "instance_id": top10["instance_id"],
    "instance_type": top10["instance_type"],
    "avg CPU %": top10["cpu_avg"].round(1),
    "waste score": top10["waste_score"].round(3),
    "recommendation": top10["recommendation"],
    "est. monthly savings ($)": top10["est_monthly_savings"].round(2),
})
st.dataframe(display, use_container_width=True, hide_index=True)

st.divider()

# ---- Section 4: Anomaly timeline ----
st.subheader(f"🚨 Anomaly Timeline — {selected}")
inst_an = anomalies[anomalies["instance_id"] == selected].sort_values("date").copy()
inst_an["flag"] = inst_an["is_anomaly"].astype(int)

base = alt.Chart(inst_an).encode(x=alt.X("date:T", title="Date"))
line = base.mark_line(color="#4c78a8").encode(y=alt.Y("cost_usd:Q", title="Cost (USD/day)"))
points = (
    base.transform_filter(alt.datum.flag == 1)
    .mark_point(color="#e45756", size=110, filled=True)
    .encode(y="cost_usd:Q", tooltip=["date:T", "cost_usd:Q"])
)
st.altair_chart((line + points).properties(height=320), use_container_width=True)

n_flag = int(inst_an["is_anomaly"].sum())
if n_flag:
    dates = ", ".join(inst_an.loc[inst_an["is_anomaly"], "date"].dt.strftime("%Y-%m-%d"))
    st.caption(f"{n_flag} anomalous day(s) flagged (red): {dates}")
else:
    st.caption("No anomalies flagged for this instance.")

# ---- Optional: recent API predictions from predictions.db ----
with st.expander("Recent API predictions (predictions.db)"):
    rp = recent_predictions(str(DB))
    if rp is None or rp.empty:
        st.write("No predictions logged yet (run the Phase 3 API to populate predictions.db).")
    else:
        st.dataframe(rp, use_container_width=True, hide_index=True)
