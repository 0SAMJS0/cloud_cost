"""
Loads the three trained Phase 2 models ONCE and holds them in memory.

Each .joblib is a dict: {"model": <estimator>, "features": [...], ...} as saved by the
training scripts. An instance of ModelStore is created in the FastAPI lifespan and parked
on app.state, so requests never touch disk or re-load a model.
"""

from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "models" / "saved"


class ModelStore:
    def __init__(self):
        self.forecast = None
        self.waste = None
        self.anomaly = None

    def load(self):
        """Load all three model bundles from models/saved/."""
        self.forecast = joblib.load(SAVED / "forecast_model.joblib")
        self.waste = joblib.load(SAVED / "waste_model.joblib")
        self.anomaly = joblib.load(SAVED / "anomaly_model.joblib")
        return self

    @property
    def loaded(self) -> bool:
        return all(b is not None for b in (self.forecast, self.waste, self.anomaly))

    def summary(self) -> dict:
        """Human-readable summary for the /health endpoint."""
        if not self.loaded:
            return {}
        return {
            "forecast": self.forecast.get("name", "forecast"),
            "waste": self.waste.get("name", "waste"),
            "anomaly": "IsolationForest",
        }
