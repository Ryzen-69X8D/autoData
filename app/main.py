from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import numpy as np

app = FastAPI(title="Stock Prediction API")

MODEL_PATH = "/opt/airflow/models/random_forest.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

class StockFeatures(BaseModel):
    Open: float
    High: float
    Low: float
    Volume: float

@app.post("/predict")
def predict_stock(features: StockFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    
    mock_close = (features.Open + features.High + features.Low) / 3
    
    input_data = pd.DataFrame([{
        'Open': features.Open,
        'High': features.High,
        'Low': features.Low,
        'Close': mock_close,
        'Volume': features.Volume,
        'SMA_10': mock_close,
        'SMA_50': mock_close,
        'Daily_Return': 0.0
    }])
    
    prediction = model.predict(input_data)[0]
    
    return {"prediction": round(prediction, 2)}