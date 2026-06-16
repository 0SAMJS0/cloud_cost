"""
Pydantic request/response models for the Phase 3 API.

Every response carries the same four-ish keys (prediction, confidence, recommendation,
savings_usd); fields that don't apply to an endpoint are set to null rather than omitted,
so the Phase 4 dashboard sees a stable shape.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    models: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# /forecast
# ---------------------------------------------------------------------------
class DailyUsage(BaseModel):
    date: str = Field(..., examples=["2024-06-01"])
    cpu_avg: float
    memory_avg: float
    network_in: float
    network_out: float
    cost_usd: float


class ForecastRequest(BaseModel):
    """Last ~30 days of daily usage for ONE instance."""
    instance_id: Optional[str] = "input"
    instance_type: Optional[str] = None  # optional; enables the one-hot type feature
    history: List[DailyUsage] = Field(..., min_length=8)


class ForecastResponse(BaseModel):
    prediction: float                       # next-day baseline cost_usd
    confidence: float                       # 0-1 (recent-volatility heuristic)
    recommendation: str
    savings_usd: float                      # expected reduction vs last (possibly spiked) day
    flagged_dates: List[str] = Field(default_factory=list)  # days de-spiked before forecasting


# ---------------------------------------------------------------------------
# /waste-check
# ---------------------------------------------------------------------------
class WasteRequest(BaseModel):
    cpu_avg: float
    memory_avg: float
    network_in: float
    network_out: float
    instance_type: Optional[str] = None
    current_monthly_cost: Optional[float] = None  # enables a savings_usd estimate


class WasteResponse(BaseModel):
    prediction: float                       # waste_score 0-1
    confidence: float                       # predict_proba of the chosen class
    recommendation: str                     # Terminate / Downsize / Healthy
    savings_usd: Optional[float] = None      # null when current_monthly_cost is absent


# ---------------------------------------------------------------------------
# /anomaly
# ---------------------------------------------------------------------------
class CostPoint(BaseModel):
    date: str = Field(..., examples=["2024-06-01"])
    cost_usd: float


class AnomalyRequest(BaseModel):
    """A daily cost time series for ONE instance."""
    instance_id: Optional[str] = "input"
    series: List[CostPoint] = Field(..., min_length=5)


class DateScore(BaseModel):
    date: str
    score: float                            # higher = more anomalous


class AnomalyResponse(BaseModel):
    flagged_dates: List[str]
    confidence: List[DateScore]             # per-date anomaly scores
    recommendation: str
    savings_usd: Optional[float] = None      # not applicable -> null
