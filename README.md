# 🚀 Automated Stock Prediction System (MLOps Pipeline) — Fixed Edition

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Airflow](https://img.shields.io/badge/Airflow-2.8.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> End-to-end MLOps pipeline for **Indian NSE stock prediction** with scheduled retraining and FastAPI serving.

---

## 🐛 Bugs Fixed (from original)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `src/ingest.py` | `yfinance ≥0.2` returns `MultiIndex` columns → crash | Flatten with `.get_level_values(0)` + `auto_adjust=True` |
| 2 | `src/preprocess.py` | `Date` column not dropped → `MinMaxScaler` crash | Drop `Date` + all non-numeric columns first |
| 3 | `src/preprocess.py` | Scaler never saved → API predictions in wrong scale | `joblib.dump(scaler, scaler_path)` added |
| 4 | `src/train.py` | No `Date` drop → crash; single-threaded on 32 GB box | Drop `Date`, add `n_jobs=-1` |
| 5 | `app/main.py` | Raw OHLCV sent to a model trained on scaled data → garbage predictions | Load scaler, call `scaler.transform()` before `model.predict()` |
| 6 | `docker-compose.yml` | FastAPI service missing entirely | Added `fastapi:` service with shared `models/` volume |
| 7 | `dags/retrain_dag.py` | No `evaluate` or `deploy` tasks; no branching logic | Full DAG: ingest → preprocess → train → evaluate → branch → deploy/skip |
| 8 | `src/evaluate.py` | Empty file | Full RMSE/MAE/R² evaluation + `is_new_model_better()` |
| 9 | `src/deploy.py` | Empty file | Conditional promotion: only deploy if RMSE improves |
| 10 | `scripts/run_pipeline.py` | Empty file | Full local runner with CLI args |

---

## 📂 Project Structure

```
stock-mlops-pipeline/
├── app/
│   └── main.py              # FastAPI service (fixed)
├── dags/
│   └── retrain_dag.py       # Full Airflow DAG (fixed)
├── src/
│   ├── ingest.py            # yfinance fetch (fixed MultiIndex)
│   ├── preprocess.py        # Feature engineering + scaling (fixed)
│   ├── train.py             # RandomForest training (fixed)
│   ├── evaluate.py          # Model evaluation (was empty)
│   ├── deploy.py            # Conditional deployment (was empty)
│   └── utils.py             # Shared helpers (was empty)
├── pipeline/
│   └── retrain_pipeline.py  # Programmatic orchestrator (was empty)
├── scripts/
│   └── run_pipeline.py      # Local CLI runner (was empty)
├── tests/
│   └── test_pipeline.py     # pytest suite
├── data/
│   ├── raw/                 # Downloaded OHLCV CSV
│   └── processed/           # Scaled feature CSV
├── models/                  # .pkl files (gitignored)
├── docker-compose.yml       # Now includes FastAPI service
├── Dockerfile               # FastAPI container
├── Dockerfile.airflow       # Airflow container (PYTHONPATH fixed)
├── requirements.txt
└── requirements-airflow.txt
```

---

## ▶️ Quick Start

### Option A — Local (no Docker, fastest for development)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (fetches RELIANCE.NS by default)
python scripts/run_pipeline.py

# Other Indian tickers:
python scripts/run_pipeline.py --ticker TCS.NS      --start 2020-01-01 --end 2024-12-31
python scripts/run_pipeline.py --ticker INFY.NS
python scripts/run_pipeline.py --ticker HDFCBANK.NS
python scripts/run_pipeline.py --ticker WIPRO.NS

# 4. Start the API
uvicorn app.main:app --reload --port 8000
```

### Option B — Docker Compose (full stack)

```bash
# Build and start all services
docker-compose up --build

# Services:
#   Airflow UI  → http://localhost:8080  (admin / admin)
#   FastAPI     → http://localhost:8000
#   API Docs    → http://localhost:8000/docs
```

---

## 📡 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict (raw OHLCV — scaler is applied automatically)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Open":   2800.0,
    "High":   2850.0,
    "Low":    2760.0,
    "Close":  2820.0,
    "Volume": 5000000
  }'
```

**Response:**
```json
{
  "predicted_close_scaled": 0.7234,
  "note": "Prediction is in MinMaxScaler space [0,1]. Higher = relatively higher next-day close."
}
```

---

## 🔁 Pipeline Workflow

```
Yahoo Finance (NSE) → Raw CSV → Feature Engineering → Scaled CSV
                                                          ↓
                                              RandomForest Training
                                                          ↓
                                              RMSE / MAE / R² Eval
                                                          ↓
                                         (better than deployed?) → Deploy → FastAPI
```

Airflow DAG runs daily and only promotes the model if RMSE improves.

---

## 🧪 Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 📈 Supported Indian NSE Tickers

| Ticker | Company |
|--------|---------|
| `RELIANCE.NS` | Reliance Industries |
| `TCS.NS` | Tata Consultancy Services |
| `INFY.NS` | Infosys |
| `HDFCBANK.NS` | HDFC Bank |
| `WIPRO.NS` | Wipro |
| `ICICIBANK.NS` | ICICI Bank |
| `SBIN.NS` | State Bank of India |
| `BAJFINANCE.NS` | Bajaj Finance |

---

## 👨‍💻 Author

Akash Kundu
