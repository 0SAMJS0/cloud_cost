"""
POST /forecast — next-day baseline cost for ONE instance.

Pipeline (Model 3 -> Model 1): the input cost series is first run through the anomaly
model; flagged days are de-spiked to the per-instance median; then lag/rolling features
are built and the forecaster predicts the next-day *baseline* cost.

Sample payload (also runnable via tests/smoke_test.py):
{
  "instance_id": "i-00006704",
  "instance_type": "c5.large",
  "history": [
    {"date": "2024-06-01", "cpu_avg": 56.9, "memory_avg": 62.1,
     "network_in": 6.29, "network_out": 3.18, "cost_usd": 2.33},
    ... (>= 8 days, 30 recommended) ...
  ]
}
"""

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.db import log_prediction
from api.features import forecast_pipeline
from api.schemas import ForecastRequest, ForecastResponse

router = APIRouter(tags=["forecast"])


@router.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest, request: Request):
    store = request.app.state.models

    df = pd.DataFrame([r.model_dump() for r in req.history])
    df["date"] = pd.to_datetime(df["date"])
    df["instance_id"] = req.instance_id or "input"
    df["instance_type"] = req.instance_type or "unknown"

    try:
        out = forecast_pipeline(df, store.anomaly["model"], store.forecast)
    except Exception as exc:  # surface bad input as a 400 instead of a 500
        raise HTTPException(status_code=400, detail=f"Could not forecast: {exc}")

    pred = round(out["prediction"], 4)
    mean = out["recent_mean"]

    if mean and pred > 1.10 * mean:
        rec = (f"Projected next-day cost ${pred:.2f} is ~{(pred / mean - 1) * 100:.0f}% "
               f"above the recent baseline (${mean:.2f}) — review usage.")
    elif mean and pred < 0.90 * mean:
        rec = (f"Projected next-day cost ${pred:.2f} is trending below the recent "
               f"baseline (${mean:.2f}).")
    else:
        rec = f"Projected next-day cost ${pred:.2f}, stable vs recent baseline (${mean:.2f})."

    if out["flagged_dates"]:
        rec += f" {len(out['flagged_dates'])} anomalous day(s) de-spiked before forecasting."

    # Expected reduction vs the most recent (possibly spiked) actual day.
    savings = round(max(0.0, out["last_cost"] - pred), 4)

    resp = ForecastResponse(
        prediction=pred,
        confidence=round(out["confidence"], 3),
        recommendation=rec,
        savings_usd=savings,
        flagged_dates=out["flagged_dates"],
    )

    log_prediction(
        "/forecast",
        {"instance_id": req.instance_id, "instance_type": req.instance_type,
         "n_days": len(req.history), "last_cost": out["last_cost"]},
        resp.model_dump(),
    )
    return resp
