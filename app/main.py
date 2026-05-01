"""
FastAPI Stock Prediction Service
=================================
Loads the trained RandomForest model **and** the MinMaxScaler used during
training, so incoming raw OHLCV values are scaled consistently before
prediction — fixing the silent data-mismatch bug in the original code.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="Stock Prediction API (Indian Market)",
    description="Predicts next-day Close price for Indian NSE stocks.",
    version="2.0.0",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/opt/airflow/models/random_forest.pkl"
SCALER_PATH = "/opt/airflow/models/scaler.pkl"

# Feature order must match preprocess.FEATURE_COLS exactly
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "SMA_10", "SMA_50", "Daily_Return"]

model  = None
scaler = None


# ── Startup: load model + scaler ──────────────────────────────────────────────
@app.on_event("startup")
def load_artifacts():
    global model, scaler

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅  Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️   Model not found at {MODEL_PATH}. Run the pipeline first.")

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"✅  Scaler loaded from {SCALER_PATH}")
    else:
        print(f"⚠️   Scaler not found at {SCALER_PATH}.")


# ── Request schema ────────────────────────────────────────────────────────────
class StockFeatures(BaseModel):
    """
    Raw (un-scaled) OHLCV values for a single trading day.
    SMA_10, SMA_50, Daily_Return are optional; if omitted they are
    approximated from the OHLCV inputs so the API is easy to call.
    """
    Open:         float = Field(..., example=2800.0)
    High:         float = Field(..., example=2850.0)
    Low:          float = Field(..., example=2760.0)
    Close:        float = Field(..., example=2820.0)
    Volume:       float = Field(..., example=5_000_000)
    SMA_10:       float | None = Field(None, example=2790.0)
    SMA_50:       float | None = Field(None, example=2750.0)
    Daily_Return: float | None = Field(None, example=0.005)


class PredictionResponse(BaseModel):
    predicted_close_scaled: float
    note: str


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "model_loaded":  model  is not None,
        "scaler_loaded": scaler is not None,
    }


# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict_stock(features: StockFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is unavailable. Run the training pipeline first.")
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler is unavailable. Run the preprocessing step first.")

    # ── FIX: Fill optional technical indicators if not provided ───────────────
    sma_10       = features.SMA_10       if features.SMA_10       is not None else features.Close
    sma_50       = features.SMA_50       if features.SMA_50       is not None else features.Close
    daily_return = features.Daily_Return if features.Daily_Return is not None else 0.0

    # Build a DataFrame with the same column order as training
    raw_df = pd.DataFrame([{
        "Open":         features.Open,
        "High":         features.High,
        "Low":          features.Low,
        "Close":        features.Close,
        "Volume":       features.Volume,
        "SMA_10":       sma_10,
        "SMA_50":       sma_50,
        "Daily_Return": daily_return,
    }], columns=FEATURE_COLS)

    # ── FIX: Scale inputs using the same scaler used in training ──────────────
    scaled = scaler.transform(raw_df)
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLS)

    prediction = float(model.predict(scaled_df)[0])

    return PredictionResponse(
        predicted_close_scaled=round(prediction, 6),
        note=(
            "Prediction is in MinMaxScaler space [0,1]. "
            "The value represents tomorrow's relative Close price. "
            "Higher = relatively higher next-day close."
        ),
    )
