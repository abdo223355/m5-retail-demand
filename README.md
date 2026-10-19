# M5 Retail Demand Prediction API & Forecasting Pipeline

A deployable, containerized Machine Learning prediction service serving a trained LightGBM regressor for Walmart M5 retail unit sales demand forecasting, backed by a memory-efficient Parquet chunking pipeline.

---

## 🚀 M5 Retail Demand Prediction API (Deployment Layer)

The deployment layer serves the pre-trained LightGBM model (`best_lightgbm_model.pkl`) via a high-performance **FastAPI** service with strict Pydantic schema validation, native categorical handling, non-negative demand clipping, and health diagnostics.

### Deployment Architecture
```text
Client
   ↓
FastAPI (/predict)
   ↓
Request Validation (Pydantic, strict schema, extra="forbid")
   ↓
Feature Contract (re-ordered via feature_names.pkl + native categoricals)
   ↓
Loaded LightGBM Model (singleton in memory)
   ↓
Postprocessing (np.clip(prediction, 0, None))
   ↓
JSON Response (predicted_demand)
```

### Endpoints
* `GET /health` — Health check, verifies model artifacts are in memory and reports feature counts.
* `POST /predict` — Accepts 34 engineered features and returns non-negative predicted unit sales.
* `GET /docs` — Interactive Swagger UI documentation with ready-to-test schema.
* `GET /redoc` — ReDoc alternative API documentation.

---

## 💻 Local Quickstart

### 1. Installation
```bash
git clone <repository-url>
cd gearofproml
python -m venv .venv
.venv\Scripts\activate          # Windows; source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

### 2. Run API Locally
```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive API documentation will be accessible at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 3. Run Automated Tests
```bash
pytest -v
```

---

## 🐳 Docker Deployment

The service is packaged in a lightweight, cloud-ready Docker image based on `python:3.11-slim` with system OpenMP (`libgomp1`) support for LightGBM.

### Build and Run Container
```bash
# 1. Build the Docker image
docker build -t m5-demand-api .

# 2. Run the container
docker run -d -p 8000:8000 --name m5-api m5-demand-api

# 3. Test health check
curl http://localhost:8000/health
```

Swagger UI in Docker will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 📡 API Usage Example

### Request (`POST /predict`)
```json
{
  "item_id": "HOUSEHOLD_1_457",
  "dept_id": "HOUSEHOLD_1",
  "cat_id": "HOUSEHOLD",
  "store_id": "CA_1",
  "state_id": "CA",
  "wm_yr_wk": 11105,
  "weekday": "Saturday",
  "wday": 1,
  "month": 2,
  "year": 2011,
  "event_name_1": "No_Event",
  "event_type_1": "No_Event",
  "event_name_2": "No_Event",
  "event_type_2": "No_Event",
  "snap_CA": 0,
  "snap_TX": 0,
  "snap_WI": 0,
  "sell_price": 9.97,
  "price_lag_1": 9.97,
  "price_diff_1": 0.0,
  "price_pct_change": 0.0,
  "price_max": 10.47,
  "price_relative_to_max": 0.9522,
  "lag_1": 0.0,
  "lag_7": 0.0,
  "lag_14": 0.0,
  "lag_28": 0.0,
  "rolling_mean_7": 0.0,
  "rolling_std_7": 0.0,
  "rolling_mean_14": 0.0,
  "rolling_std_14": 0.0,
  "rolling_mean_28": 0.0,
  "rolling_std_28": 0.0,
  "is_weekend": 0
}
```

### Response (`200 OK`)
```json
{
  "predicted_demand": 0.0287,
  "model_version": "lightgbm_v1"
}
```

### Health Check (`GET /health`)
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "LightGBM Regressor",
  "features_count": 34
}
```

---

## 📁 Repository Structure
```text
gearofproml/
│
├── api/                      # FastAPI deployment layer
│   ├── __init__.py
│   ├── main.py               # Application entrypoint & routes
│   ├── schemas.py            # Pydantic request & response contracts
│   └── predictor.py          # Artifact loading & inference engine
│
├── artifacts/                # Minimal production deployment artifacts (< 4 MB)
│   ├── best_lightgbm_model.pkl
│   ├── feature_names.pkl
│   └── categorical_cols.pkl
│
├── src/                      # Original data & feature pipelines
│   ├── data/                 # Ingestion, preprocessing, chunk validation
│   ├── features/             # Lags, rolling statistics, price features
│   ├── models/               # Baseline train & evaluate scripts
│   └── utils/                # Config & memory helpers
│
├── tests/                    # Pytest test suite (23 unit & integration tests)
│   ├── test_health.py        # API health & loading tests
│   ├── test_predict.py       # Prediction, schema errors & non-negative clipping tests
│   ├── test_features.py      # Feature engineering unit tests
│   └── test_preprocessing.py # Chunk preprocessing unit tests
│
├── configs/                  # Pipeline configurations
├── Dockerfile                # Production container specification
├── .dockerignore             # Context exclusion rules
├── requirements.txt          # Lean production dependencies
└── README.md
```

---

## 🔬 Modeling & Feature Engineering Details

### Feature Engineering
* **Lags:** `lag_1`, `lag_7`, `lag_14`, `lag_28`
* **Rolling:** mean and std over 7, 14, 28 days of `sales.shift(1)`
* **Calendar:** raw weekday, month, year, events, SNAP, plus `is_weekend`
* **Price:** `price_lag_1`, `price_diff_1`, `price_pct_change`, `price_max`, `price_relative_to_max`
* **Total Features:** 34 model features strictly maintained at inference.

### Technical Limitations & Scope
* **Scope:** This service provides fast, stateless single-step prediction for given feature payloads. It does **not** perform 28-day autonomous multi-step recursive forecasting.
* **Feature Generation:** In an end-to-end production setup, feature values (lags, rolling averages) must be supplied by an upstream feature pipeline or database.
* **Known Model Constraints:** The underlying LightGBM model was trained on a series-split subset. See `docs/modeling.md` for full technical evaluation.
