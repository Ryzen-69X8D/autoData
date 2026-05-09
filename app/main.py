from fastapi import FastAPI, HTTPException
import pandas as pd
import os

from app import model_loader
from app.schemas import StockFeaturesRequest, PredictionResponse

app = FastAPI(
    title="Stock Prediction API - Ensemble Edition",
    description="Predicts next-day Close price using LSTM, XGBoost, and Random Forest.",
    version="3.0.0",
)

@app.on_event("startup")
def load_artifacts():
    model_loader.load_artefacts()

@app.get("/")
def read_root():
    return {"status": "online", "message": "Ensemble API Online. Visit /docs"}

@app.get("/health")
def health():
    return {"ready": model_loader.is_ready()}

@app.post("/predict", response_model=PredictionResponse)
def predict_stock(features: StockFeaturesRequest):
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Models are unavailable. Run the pipeline.")

    # Convert API inputs to DataFrame
    raw_df = model_loader.build_feature_frame(**features.model_dump())
    scaled_df = model_loader.scale_features(raw_df)
    
    # ── Fetch 13-day history so the LSTM has a 14-day sequence to read ──
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed_data.csv")
        history_df = pd.read_csv(csv_path).tail(13)[model_loader.FEATURE_COLS]
        scaled_history = pd.DataFrame(
            model_loader.state.scaler.transform(history_df), 
            columns=model_loader.FEATURE_COLS
        )
        full_seq_df = pd.concat([scaled_history, scaled_df], ignore_index=True)
    except Exception:
        # Fallback if no CSV exists
        full_seq_df = scaled_df 

    # Predict using the 3-model average
    scaled_pred = model_loader.predict_return(full_seq_df)
    actual_return = model_loader.inverse_scale_return(scaled_pred)
    actual_price = features.Close * (1 + actual_return)

    return PredictionResponse(
        predicted_return_pct=round(actual_return * 100, 4),
        predicted_close_price=round(actual_price, 2),
        confidence_band_low=round(actual_price * 0.985, 2),
        confidence_band_high=round(actual_price * 1.015, 2),
        currency="INR",
        model_version="ensemble_v1",
        note=f"Ensemble model predicts a {round(actual_return * 100, 2)}% move."
    )