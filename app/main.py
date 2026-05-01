from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="Stock Prediction API (Indian Market)",
    description="Predicts next-day Close price for Indian NSE stocks.",
    version="2.1.0",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Defaults to the local 'models' folder if the environment variable isn't set
MODEL_DIR = os.getenv("MODEL_DIR", "./models") 

model_path = os.path.join(MODEL_DIR, "random_forest.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

# Feature order must match exactly what the model and scaler were trained on
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "SMA_10", "SMA_50", "Daily_Return"]

model  = None
scaler = None


# ── Startup: load model + scaler ──────────────────────────────────────────────
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


# ── Request schema ────────────────────────────────────────────────────────────
class StockFeatures(BaseModel):
    """
    Raw (un-scaled) OHLCV values for a single trading day.
    SMA_10, SMA_50, Daily_Return are optional; if omitted they are
    approximated from the OHLCV inputs.
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
    predicted_close_price: float
    currency: str
    note: str


# ── Root endpoint ─────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {
        "status": "online",
        "api_name": "Stock Prediction API (Indian Market)",
        "message": "Welcome! The API is running successfully. Visit /docs to view the interactive documentation and test the model."
    }


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
        raise HTTPException(status_code=503, detail="Model is unavailable.")
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler is unavailable.")

    # Fill optional technical indicators if not provided
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

    # Scale inputs
    scaled_data = scaler.transform(raw_df)
    scaled_df = pd.DataFrame(scaled_data, columns=FEATURE_COLS)

    # Get the raw prediction (which is in the scaled 0-1 format)
    scaled_prediction = float(model.predict(scaled_df)[0])

    # ── Inverse Transform to get the real price ───────────────────────────────
    # Create a dummy array with the exact same shape as the scaler expects (1 row, 8 columns)
    dummy_array = np.zeros((1, len(FEATURE_COLS)))
    
    # Find the index of the "Close" column and insert our predicted value there
    close_index = FEATURE_COLS.index("Close")
    dummy_array[0, close_index] = scaled_prediction
    
    # Reverse the scaling for the entire dummy array
    inversed_array = scaler.inverse_transform(dummy_array)
    
    # Extract the actual unscaled price from the "Close" column
    actual_price = float(inversed_array[0, close_index])

    return PredictionResponse(
        predicted_close_scaled=round(scaled_prediction, 6),
        predicted_close_price=round(actual_price, 2),
        currency="INR",
        note="The API now returns both the raw scaled output and the actual predicted price in Indian Rupees."
    )