# autoData Backend

FastAPI service for Indian market prediction, JWT authentication, PostgreSQL-backed portfolio tracking, and the existing MLOps pipeline.

## Structure

```text
autoData/
├── alembic/                  # Database migration environment
├── app/
│   ├── api/
│   │   ├── dependencies.py   # DB and current-user dependencies
│   │   └── routes/           # auth, markets, portfolio, predict routers
│   ├── core/                 # config and JWT/password security
│   ├── db/                   # SQLAlchemy models and Pydantic schemas
│   ├── ml/                   # inference engine and ingestion wrapper
│   └── main.py               # FastAPI app, CORS, router registration
├── data/                     # Raw/processed market data
├── models/                   # Existing trained model artifacts
├── src/                      # Training and preprocessing pipeline
├── .env.example
├── alembic.ini
└── requirements.txt
```

## Local Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The included `.env` uses SQLite for local development. For PostgreSQL, copy `.env.example` values into `.env` and set `DATABASE_URL`, for example:

```text
DATABASE_URL=postgresql+psycopg2://autodata:autodata@localhost:5432/autodata
```

## Endpoints

- `GET /health`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/markets/overview`
- `GET /api/portfolio`
- `POST /api/predict`
- `GET /api/model/health`

`POST /predict` is also kept for older clients.

## Notes

Prediction uses the existing ensemble artifacts under `models/prod`. If `torch` or `xgboost` are not installed locally, the API still starts and model health reports the missing runtime package.
