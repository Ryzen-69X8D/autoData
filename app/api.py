"""
api.py
=======
FastAPI router (v1) that exposes all prediction, health, model-info,
and retrain endpoints.

This module is mounted by app/main.py:
    app.include_router(api_router, prefix="/api/v1")

Separating the router from main.py keeps the entry-point clean and
makes it easy to add a v2 router later.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app import model_loader as ml
from app.schemas import (
    BatchPredictRequest,
    BatchPredictionResponse,
    Currency,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RetrainRequest,
    RetrainResponse,
    StockFeaturesRequest,
    build_error,
)

log = logging.getLogger(__name__)

api_router = APIRouter()


# ── Health ─────────────────────────────────────────────────────────────────────

@api_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["ops"],
)
def health() -> HealthResponse:
    """
    Returns service health.  Returns HTTP 200 even when artefacts are missing
    so that load-balancer probes do not restart the container — the
    *model_loaded* / *scaler_loaded* flags tell you whether predictions work.
    """
    info = ml.model_info()
    return HealthResponse(
        status="ok" if ml.is_ready() else "degraded",
        model_loaded=ml._state.model is not None,
        scaler_loaded=ml._state.scaler is not None,
        model_version=info.get("model_version"),
        uptime_seconds=ml.uptime_seconds(),
    )


# ── Model info ─────────────────────────────────────────────────────────────────

@api_router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    tags=["ops"],
)
def model_info() -> ModelInfoResponse:
    """Returns the currently loaded model's type, feature list, paths, and metrics."""
    info = ml.model_info()
    return ModelInfoResponse(**info)


@api_router.post(
    "/model/reload",
    summary="Hot-reload model artefacts",
    tags=["ops"],
)
def reload_model() -> dict:
    """
    Reloads the model and scaler from disk without restarting the server.
    Useful after a new model has been deployed by the Airflow DAG.
    """
    result = ml.reload()
    if not result["model_loaded"] or not result["scaler_loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Reload completed with errors: {result['errors']}",
        )
    return {"message": "Artefacts reloaded successfully.", **result}


# ── Single-row prediction ──────────────────────────────────────────────────────

@api_router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict next-day close price",
    tags=["prediction"],
)
def predict(features: StockFeaturesRequest) -> PredictionResponse:
    """
    Accepts one day's OHLCV data (+ optional technical indicators) and returns
    the model's predicted next-day return and close price.

    **Scaling is applied automatically** — pass raw INR prices, not normalised values.
    """
    _require_ready()

    try:
        raw_df    = ml.build_feature_frame(
            open_=features.Open,
            high=features.High,
            low=features.Low,
            close=features.Close,
            volume=features.Volume,
            sma_10=features.SMA_10,
            sma_50=features.SMA_50,
            daily_return=features.Daily_Return,
        )
        scaled_df = ml.scale_features(raw_df)
        scaled_pred = ml.predict_return(scaled_df)
        actual_return = ml.inverse_scale_return(scaled_pred)
        predicted_close = round(features.Close * (1 + actual_return), 2)
        band_low, band_high = ml.confidence_band(features.Close, actual_return)
        model_version = ml._state.model_hash[:8] if ml._state.model_hash else "unknown"

        return PredictionResponse(
            predicted_return_pct=round(actual_return * 100, 4),
            predicted_close_price=predicted_close,
            confidence_band_low=band_low,
            confidence_band_high=band_high,
            currency=Currency.INR,
            model_version=model_version,
            note=(
                f"Model predicts a {round(actual_return * 100, 2):+.2f}% move "
                f"from the input Close of ₹{features.Close:,.2f}."
            ),
        )

    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        )


# ── Batch prediction ───────────────────────────────────────────────────────────

@api_router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch predict (up to 100 rows)",
    tags=["prediction"],
)
def predict_batch(request: BatchPredictRequest) -> BatchPredictionResponse:
    """
    Accepts up to 100 feature rows and returns predictions for each.
    Rows are processed sequentially; any single-row error aborts the batch.
    """
    _require_ready()

    results: list[PredictionResponse] = []
    model_version = ml._state.model_hash[:8] if ml._state.model_hash else "unknown"

    for i, row in enumerate(request.rows):
        try:
            raw_df = ml.build_feature_frame(
                open_=row.Open,
                high=row.High,
                low=row.Low,
                close=row.Close,
                volume=row.Volume,
                sma_10=row.SMA_10,
                sma_50=row.SMA_50,
                daily_return=row.Daily_Return,
            )
            scaled_df     = ml.scale_features(raw_df)
            scaled_pred   = ml.predict_return(scaled_df)
            actual_return = ml.inverse_scale_return(scaled_pred)
            predicted_close = round(row.Close * (1 + actual_return), 2)
            band_low, band_high = ml.confidence_band(row.Close, actual_return)

            results.append(PredictionResponse(
                predicted_return_pct=round(actual_return * 100, 4),
                predicted_close_price=predicted_close,
                confidence_band_low=band_low,
                confidence_band_high=band_high,
                currency=Currency.INR,
                model_version=model_version,
                note=f"Row {i}: {round(actual_return * 100, 2):+.2f}% predicted move.",
            ))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing row {i}: {exc}",
            )

    return BatchPredictionResponse(predictions=results, total_rows=len(results))


# ── Data summary ───────────────────────────────────────────────────────────────

@api_router.get(
    "/data/summary",
    summary="Describe the raw stock CSV",
    tags=["data"],
)
def data_summary(
    n_rows: int = Query(5, ge=1, le=50, description="Preview rows"),
) -> dict:
    """
    Returns shape, column list, and head() of the raw stock CSV.
    Handy for debugging ingestion.
    """
    raw_path = os.getenv("RAW_DATA_PATH", "./data/raw/stock_data.csv")
    if not os.path.exists(raw_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raw data not found at {raw_path}. Run the ingest step first.",
        )
    try:
        import pandas as pd
        df = pd.read_csv(raw_path)
        return {
            "path":    raw_path,
            "shape":   list(df.shape),
            "columns": df.columns.tolist(),
            "head":    df.head(n_rows).to_dict(orient="records"),
            "dtypes":  df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ── Metrics ────────────────────────────────────────────────────────────────────

@api_router.get(
    "/metrics",
    summary="Model evaluation metrics",
    tags=["ops"],
)
def metrics() -> dict:
    """
    Returns the deployed and new-model metrics from the models/ directory.
    """
    model_dir   = os.getenv("MODEL_DIR", "./models")
    dep_path    = os.path.join(model_dir, "metrics_deployed.json")
    new_path    = os.path.join(model_dir, "metrics_new.json")

    result: dict[str, Any] = {}

    for label, path in [("deployed", dep_path), ("new", new_path)]:
        if os.path.exists(path):
            import json
            with open(path) as f:
                result[label] = json.load(f)
        else:
            result[label] = None

    return result


# ── Retrain trigger ────────────────────────────────────────────────────────────

@api_router.post(
    "/retrain",
    response_model=RetrainResponse,
    summary="Trigger an ad-hoc pipeline run",
    tags=["ops"],
)
def trigger_retrain(
    body: RetrainRequest,
    background: BackgroundTasks,
) -> RetrainResponse:
    """
    Kicks off the full ingest→preprocess→train→evaluate→deploy pipeline
    in a background thread.  Does **not** block the HTTP response.

    For production use, prefer triggering via the Airflow UI / API.
    """
    ticker     = body.ticker     or "NIFTY_50"
    start_date = body.start_date or "2020-01-01"
    end_date   = body.end_date   or "2024-12-31"

    log.info("Ad-hoc retrain requested: ticker=%s  %s→%s", ticker, start_date, end_date)
    background.add_task(_run_pipeline_bg, ticker, start_date, end_date)

    return RetrainResponse(
        triggered=True,
        message=(
            f"Pipeline triggered for {ticker} ({start_date} → {end_date}). "
            "Check server logs for progress. "
            "Call GET /api/v1/metrics after completion to see new results."
        ),
    )


def _run_pipeline_bg(ticker: str, start: str, end: str) -> None:
    """Background task: runs scripts/run_pipeline.py as a subprocess."""
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_pipeline.py")
    cmd = [
        sys.executable, script,
        "--ticker", ticker,
        "--start",  start,
        "--end",    end,
    ]
    log.info("Running pipeline: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            log.info("Pipeline completed successfully.\n%s", result.stdout[-2000:])
            ml.reload()   # hot-swap the new model into memory
        else:
            log.error("Pipeline failed.\nSTDOUT: %s\nSTDERR: %s",
                      result.stdout[-2000:], result.stderr[-2000:])
    except subprocess.TimeoutExpired:
        log.error("Pipeline timed out after 600 s.")
    except Exception as exc:
        log.exception("Unexpected error in pipeline background task: %s", exc)


# ── Root ───────────────────────────────────────────────────────────────────────

@api_router.get("/", tags=["ops"], include_in_schema=False)
def router_root() -> dict:
    return {
        "message": "Stock Prediction API v1",
        "docs":    "/docs",
        "health":  "/api/v1/health",
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _require_ready() -> None:
    """Raises HTTP 503 if the model or scaler is not loaded."""
    if not ml.is_ready():
        missing = []
        if ml._state.model  is None: missing.append("model")
        if ml._state.scaler is None: missing.append("scaler")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Service not ready — missing artefacts: {', '.join(missing)}. "
                "Run the ML pipeline first, then call POST /api/v1/model/reload."
            ),
        )
