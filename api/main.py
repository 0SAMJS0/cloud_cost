"""
Phase 3 — FastAPI backend for the AI-Powered Cloud Cost Optimizer.

Loads the three trained models ONCE on startup (lifespan), initialises the SQLite
prediction log, and serves four JSON endpoints:
  GET  /health        - liveness + models_loaded
  POST /forecast      - next-day baseline cost (Model 3 -> Model 1 pipeline)
  POST /waste-check   - 0-1 waste score + recommendation
  POST /anomaly       - flagged cost-spike dates

Run:  py -m uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make the project root importable (so `common` and `api` resolve regardless of CWD).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402

from api.db import init_db  # noqa: E402
from api.model_store import ModelStore  # noqa: E402
from api.routes import anomaly, forecast, health, waste  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup: load models once, prepare the log ---
    store = ModelStore().load()
    app.state.models = store
    init_db()
    print(f"[startup] models loaded: {store.loaded} -> {store.summary()}")
    yield
    # --- shutdown: nothing to release (models are GC'd, sqlite opens per-call) ---


app = FastAPI(
    title="AI Cloud Cost Optimizer API",
    description="Serves the three Phase 2 models (forecast, waste, anomaly) as JSON.",
    version="3.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(waste.router)
app.include_router(anomaly.router)


@app.get("/", tags=["root"])
def root():
    return {"service": "AI Cloud Cost Optimizer API", "docs": "/docs", "health": "/health"}
