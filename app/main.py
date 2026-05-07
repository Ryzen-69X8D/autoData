from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="Stock Prediction API (Indian Market)",
    description="Predicts next-day Close price based on Daily Return forecasting.",
    version="2.2.0",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "./models") 
model_path = os.path.join(MODEL_DIR, "random_forest.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "SMA_10", "SMA_50", "Daily_Return"]

model  = None
scaler = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"✅  Model loaded from {model_path}")
    else:
        print(f"⚠️   Model not found at {model_path}. Run the pipeline first.")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✅  Scaler loaded from {scaler_path}")
    else:
        print(f"⚠️   Scaler not found at {scaler_path}.")


class StockFeatures(BaseModel):
    Open:         float = Field(..., example=2800.0)
    High:         float = Field(..., example=2850.0)
    Low:          float = Field(..., example=2760.0)
    Close:        float = Field(..., example=2820.0)
    Volume:       float = Field(..., example=5_000_000)
    SMA_10:       float | None = Field(None, example=2790.0)
    SMA_50:       float | None = Field(None, example=2750.0)
    Daily_Return: float | None = Field(None, example=0.005)


class PredictionResponse(BaseModel):
    predicted_return_percentage: float
    predicted_close_price: float
    currency: str
    note: str


@app.get("/")
def read_root():
    return {
        "status": "online",
        "api_name": "Stock Prediction API (Returns Based)",
        "message": "Welcome! Visit /docs to view the interactive documentation."
    }


@app.get("/health")
def health():
    return {
        "model_loaded":  model  is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_stock(features: StockFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is unavailable.")
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler is unavailable.")

    # Fill optional technical indicators if not provided
    sma_10       = features.SMA_10       if features.SMA_10       is not None else features.Close
    sma_50       = features.SMA_50       if features.SMA_50       is not None else features.Close
    daily_return = features.Daily_Return if features.Daily_Return is not None else 0.0

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

    # Scale inputs
    scaled_data = scaler.transform(raw_df)
    scaled_df = pd.DataFrame(scaled_data, columns=FEATURE_COLS)

    # The prediction is now the scaled "Daily_Return"
    scaled_prediction = float(model.predict(scaled_df)[0])

    # ── Inverse Transform to get the real percentage return ───────────────────
    dummy_array = np.zeros((1, len(FEATURE_COLS)))
    
    # We unscale based on the Daily_Return index, not the Close index
    return_index = FEATURE_COLS.index("Daily_Return")
    dummy_array[0, return_index] = scaled_prediction
    
    inversed_array = scaler.inverse_transform(dummy_array)
    actual_return = float(inversed_array[0, return_index])

    # ── Calculate Actual Future Price ─────────────────────────────────────────
    # Price = Current Close * (1 + predicted percentage change)
    actual_price = features.Close * (1 + actual_return)

    return PredictionResponse(
        predicted_return_percentage=round(actual_return * 100, 4), # e.g. 1.25%
        predicted_close_price=round(actual_price, 2),
        currency="INR",
        note=f"Model predicts a {round(actual_return * 100, 2)}% move from the input Close price."
    )