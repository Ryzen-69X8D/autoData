from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.routes import auth, markets, portfolio, predict
from app.core.config import get_settings
from app.db.database import SessionLocal, init_db
from app.db.schemas import HealthResponse
from app.ml import engine


log = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load_models()

    if settings.auto_create_tables:
        try:
            init_db()
        except SQLAlchemyError as exc:
            log.warning("Database initialization failed: %s", exc)

    yield


app = FastAPI(
    title=settings.app_name,
    description="FastAPI service for Indian market prediction, portfolios, and auth.",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(markets.router, prefix=settings.api_prefix)
app.include_router(portfolio.router, prefix=settings.api_prefix)
app.include_router(predict.router, prefix=settings.api_prefix)

# Backward-compatible prediction routes for older clients.
app.include_router(predict.router)


@app.get("/")
def root() -> dict:
    return {
        "status": "online",
        "service": settings.app_name,
        "docs": "/docs",
        "api": settings.api_prefix,
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    database_status = "ok"
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        database_status = "unavailable"

    model_health = engine.model_health()
    return HealthResponse(
        status="ok" if model_health["ready"] and database_status == "ok" else "degraded",
        model_ready=model_health["ready"],
        database=database_status,
        model_version=model_health.get("model_version"),
        errors=model_health.get("errors", []),
    )


@app.get(f"{settings.api_prefix}/health", response_model=HealthResponse)
def api_health() -> HealthResponse:
    return health()
