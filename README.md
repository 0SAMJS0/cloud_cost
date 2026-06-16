# AI-Powered Cloud Cost Optimizer

> Analyzes cloud (AWS EC2) usage data with three ML models to forecast spend, flag idle/over-provisioned instances, and detect cost-spike anomalies — served through a FastAPI backend and a Streamlit dashboard, all runnable with one command.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-gradient%20boosting-006600)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)

---

## What it does

The system turns six months of daily cloud usage into actionable savings using three models:
**cost forecasting** (XGBoost, **13.65% MAPE** next-day baseline), **waste detection** (Random Forest,
0–1 idle/over-provisioning score), and **anomaly detection** (Isolation Forest, **~97% recall at <0.1%
false-positive rate** on planted cost spikes). The models are wired together — the anomaly detector
de-spikes the cost series *before* the forecaster runs — so forecasts track the real baseline instead
of chasing one-off spikes.

---

## Architecture

```
                 +---------------------------+
                 |  Data (synthetic CSV)     |
                 |  20 instances . ~182 days |
                 |  + ground-truth labels    |
                 +-------------+-------------+
                               |  generate_data.py -> process_data.py
                               v
        +---------------------------------------------+
        |  ML Models  (common/features.py = 1 source) |
        |                                             |
        |   Anomaly (IsolationForest)                 |
        |        | de-spike flagged days              |
        |        v                                     |
        |   Forecast (XGBoost)     Waste (RandomForest)|
        +--------------------+------------------------+
                             |  loaded once via api/model_store.py
                             v
        +---------------------------------------------+
        |  FastAPI Backend  (api/)                    |
        |  /health /forecast /waste-check /anomaly    |
        |  logs every prediction -> predictions.db    |
        +--------------------+------------------------+
                             |  same models + features (zero skew)
                             v
        +---------------------------------------------+
        |  Streamlit Dashboard  (dashboard/app.py)    |
        |  KPIs . forecast chart . waste board . alerts|
        +---------------------------------------------+
```

The **anomaly -> forecast** wiring is the core design idea: planted 3–5x cost spikes are unpredictable
by construction, so they are detected and replaced with the per-instance median before the forecaster
builds its lag/rolling features. The exact same `common/features.py` transforms are used at train time
and serve time, so there is **no train/serve skew**.

---

## Key results (Phase 2 scorecard)

| Model | Algorithm | Metric | Result | Target | Status |
|-------|-----------|--------|--------|--------|--------|
| Cost forecast | XGBoost | MAPE (next-day) | **13.65%** | < 15% | PASS |
| Waste detection | Random Forest | Precision (held-out instances) | **100%** | > 80% | PASS |
| Anomaly detection | Isolation Forest | False-positive rate | **0.08%** | < 5% | PASS |
| Anomaly detection | Isolation Forest | Recall (spikes caught) | **96.9%** (63/65) | catch spikes | PASS |

> Note: waste precision is measured on a small grouped hold-out (6 of 20 instances, split by
> `instance_id` to prevent leakage), so the perfect score reflects clean separability, not a
> production-scale guarantee.

---

## How to run

### Option 1 — Docker (recommended)

```bash
docker-compose up
```

Starts both services with no manual steps:
- API docs -> http://localhost:8000/docs
- Dashboard -> http://localhost:8501

### Option 2 — Local (Windows, `py`)

```bash
py demo.py                                # idempotent: generates data + trains models if missing
py -m uvicorn api.main:app --reload       # API  -> http://127.0.0.1:8000/docs
py -m streamlit run dashboard/app.py      # dashboard -> http://127.0.0.1:8501
```

`demo.py` is safe to re-run — it skips any step whose output already exists and never retrains
existing models.

---

## Project structure

```
cloud_ai/
├── data/
│   ├── synthetic/generate_data.py    # Phase 1: generate dirty raw usage + ground-truth labels
│   ├── process_data.py               # Phase 1: clean raw -> processed (keeps planted spikes)
│   ├── raw/usage_2024.csv            # raw daily usage (20 instances, ~182 days)
│   ├── raw/ground_truth.csv          # labels: is_wasteful (per instance), is_anomaly (per row)
│   └── processed/usage_processed.csv # clean, model-ready data
├── common/
│   └── features.py                   # SHARED feature engineering — single source of truth
├── models/
│   ├── train/train_forecast.py       # Phase 2: XGBoost vs RF, temporal split, de-spiked target
│   ├── train/train_waste.py          # Phase 2: LogReg vs RF, grouped split by instance_id
│   ├── train/train_anomaly.py        # Phase 2: IsolationForest on per-instance z-score features
│   ├── saved/*.joblib                # three trained model bundles
│   └── evaluate/report.py            # prints the PASS/FAIL scorecard
├── api/
│   ├── main.py                       # FastAPI app, lifespan loads models once + inits SQLite
│   ├── model_store.py                # loads the 3 .joblib bundles
│   ├── features.py                   # serving helpers over common.features (no skew)
│   ├── schemas.py                    # Pydantic request/response models
│   ├── db.py                         # SQLite prediction log (predictions.db)
│   └── routes/                       # health.py, forecast.py, waste.py, anomaly.py
├── dashboard/
│   └── app.py                        # Streamlit one-page dashboard (KPIs, charts, leaderboard)
├── tests/                            # sample payload generator + endpoint smoke test
├── predictions.db                    # SQLite log of every API prediction
├── demo.py                           # one-command idempotent setup
├── Dockerfile                        # python:3.11-slim image
├── docker-compose.yml                # api + dashboard services
├── .dockerignore
└── requirements.txt
```

---

## API endpoints

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/health` | GET | Liveness check; reports whether all 3 models are loaded |
| `/forecast` | POST | Next-day baseline cost for one instance (anomaly -> de-spike -> forecast pipeline) |
| `/waste-check` | POST | 0–1 waste score + recommendation (Terminate / Downsize / Healthy) + savings estimate |
| `/anomaly` | POST | Flags cost-spike dates in a time series, with per-date anomaly scores |

Every prediction is logged to `predictions.db` (timestamp, endpoint, input summary, output JSON).

---

## Tech stack

| Layer | Tool |
|-------|------|
| Language | Python 3.11 |
| Data wrangling | pandas, NumPy |
| ML models | scikit-learn (Random Forest, Logistic Regression, Isolation Forest), XGBoost |
| Model persistence | joblib |
| Backend API | FastAPI + Uvicorn, Pydantic |
| Prediction log | SQLite |
| Dashboard | Streamlit + Altair |
| Packaging | Docker + docker-compose |

---

## What I'd add next

- **Real AWS integration** — pull live metrics from CloudWatch / Cost & Usage Reports via Boto3 instead of synthetic data.
- **Scheduled retraining** — a pipeline (cron / Airflow) that retrains on fresh data and version-controls model artifacts.
- **Anomaly alerting** — push email/Slack notifications when a cost spike is detected, instead of only surfacing it in the dashboard.
- **Multi-cloud support** — extend the feature layer and connectors to Azure and GCP billing data.
