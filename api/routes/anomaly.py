"""
POST /anomaly — flag cost spikes in a daily cost series for ONE instance.

Sample payload:
{
  "instance_id": "i-00006704",
  "series": [
    {"date": "2024-06-01", "cost_usd": 2.33},
    {"date": "2024-06-02", "cost_usd": 2.41},
    ... (>= 5 days) ...
    {"date": "2024-06-15", "cost_usd": 9.80}   <- a spike
  ]
}
"""

import pandas as pd
from fastapi import APIRouter, Request

from api.db import log_prediction
from api.features import run_anomaly_detection
from api.schemas import AnomalyRequest, AnomalyResponse, DateScore

router = APIRouter(tags=["anomaly"])


@router.post("/anomaly", response_model=AnomalyResponse)
def anomaly(req: AnomalyRequest, request: Request):
    store = request.app.state.models
    iso = store.anomaly["model"]

    df = pd.DataFrame([p.model_dump() for p in req.series])
    df["date"] = pd.to_datetime(df["date"])
    df["instance_id"] = req.instance_id or "input"

    feat = run_anomaly_detection(df, iso)

    flagged_dates = feat.loc[feat["is_anomaly"], "date"].dt.strftime("%Y-%m-%d").tolist()
    confidence = [
        DateScore(date=d, score=round(float(s), 4))
        for d, s in zip(feat["date"].dt.strftime("%Y-%m-%d"), feat["anomaly_score"])
    ]

    if flagged_dates:
        recommendation = (f"{len(flagged_dates)} cost spike(s) detected on "
                          f"{', '.join(flagged_dates)} — investigate the underlying usage.")
    else:
        recommendation = "No anomalies detected in the series."

    resp = AnomalyResponse(
        flagged_dates=flagged_dates,
        confidence=confidence,
        recommendation=recommendation,
        savings_usd=None,
    )

    log_prediction(
        "/anomaly",
        {"instance_id": req.instance_id, "n_days": len(req.series),
         "n_flagged": len(flagged_dates)},
        resp.model_dump(),
    )
    return resp
