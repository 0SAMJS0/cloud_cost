"""
POST /waste-check — is this instance under-utilised?

Sample payload:
{
  "cpu_avg": 3.5, "memory_avg": 8.0,
  "network_in": 0.05, "network_out": 0.02,
  "instance_type": "m5.large", "current_monthly_cost": 70.0
}
"""

import pandas as pd
from fastapi import APIRouter, Request

from api.db import log_prediction
from api.features import WASTE_FEATURES
from api.schemas import WasteRequest, WasteResponse

router = APIRouter(tags=["waste"])

# Rough savings model: terminating an idle box recovers ~all of its cost; downsizing a
# under-used one recovers roughly half.
DOWNSIZE_SAVING = 0.50


@router.post("/waste-check", response_model=WasteResponse)
def waste_check(req: WasteRequest, request: Request):
    store = request.app.state.models
    model = store.waste["model"]

    X = pd.DataFrame(
        [[req.cpu_avg, req.memory_avg, req.network_in, req.network_out]],
        columns=WASTE_FEATURES,
    )
    proba = model.predict_proba(X)[0]
    waste_score = float(proba[1])
    confidence = float(max(proba))

    if waste_score >= 0.5:
        idle = req.cpu_avg < 10 and req.memory_avg < 15 and (req.network_in + req.network_out) < 1.0
        recommendation = "Terminate" if idle else "Downsize"
    else:
        recommendation = "Healthy"

    savings = None
    if req.current_monthly_cost is not None:
        if recommendation == "Terminate":
            savings = round(req.current_monthly_cost, 2)
        elif recommendation == "Downsize":
            savings = round(DOWNSIZE_SAVING * req.current_monthly_cost, 2)
        else:
            savings = 0.0

    resp = WasteResponse(
        prediction=round(waste_score, 4),
        confidence=round(confidence, 3),
        recommendation=recommendation,
        savings_usd=savings,
    )

    log_prediction(
        "/waste-check",
        {"cpu_avg": req.cpu_avg, "memory_avg": req.memory_avg,
         "network_in": req.network_in, "network_out": req.network_out,
         "instance_type": req.instance_type,
         "current_monthly_cost": req.current_monthly_cost},
        resp.model_dump(),
    )
    return resp
