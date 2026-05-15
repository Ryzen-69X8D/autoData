from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.db.schemas import BatchPredictRequest, BatchPredictionResponse, PredictionResponse, StockFeaturesRequest
from app.ml import engine
from app.ml.engine import PredictionUnavailable


router = APIRouter(tags=["prediction"])


@router.get("/model/health")
def model_health() -> dict:
    return engine.model_health()


@router.post("/predict", response_model=PredictionResponse)
def predict_stock(features: StockFeaturesRequest) -> PredictionResponse:
    try:
        return engine.predict(features)
    except PredictionUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictionResponse:
    try:
        predictions = engine.predict_batch(request.rows)
    except PredictionUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        )

    return BatchPredictionResponse(predictions=predictions, total_rows=len(predictions))
