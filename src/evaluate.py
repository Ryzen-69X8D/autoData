import pandas as pd
import numpy as np
import torch
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
from train import StockLSTM, create_sequences, SEQ_LENGTH
from preprocess import FEATURE_COLS

def evaluate_ensemble(model_dir: str, data_path: str, metrics_output_path: str):
    df = pd.read_csv(data_path)
    df["Target"] = df["Daily_Return"].shift(-1)
    df.dropna(inplace=True)

    X = df[FEATURE_COLS].values
    y = df["Target"].values

    X_seq, y_seq = create_sequences(X, y, SEQ_LENGTH)
    X_flat, y_flat = X[SEQ_LENGTH:], y[SEQ_LENGTH:]

    split = int(len(X_seq) * 0.8)
    
    # Load LSTM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = StockLSTM(input_size=len(FEATURE_COLS)).to(device)
    lstm_model.load_state_dict(torch.load(os.path.join(model_dir, "lstm_model.pt"), map_location=device))
    lstm_model.eval()

    # Load XGB & RF
    xgb_model = XGBRegressor()
    xgb_model.load_model(os.path.join(model_dir, "xgb_model.json"))
    rf_model = joblib.load(os.path.join(model_dir, "rf_model.pkl"))

    # Predict
    with torch.no_grad():
        lstm_preds = lstm_model(torch.FloatTensor(X_seq[split:]).to(device)).cpu().numpy().flatten()
    
    xgb_preds = xgb_model.predict(X_flat[split:])
    rf_preds = rf_model.predict(X_flat[split:])

    # ENSEMBLE AVERAGE (Equal weights)
    final_preds = (lstm_preds + xgb_preds + rf_preds) / 3.0
    y_test = y_flat[split:]

    rmse = float(np.sqrt(mean_squared_error(y_test, final_preds)))
    mae  = float(mean_absolute_error(y_test, final_preds))
    r2   = float(r2_score(y_test, final_preds))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2, "test_samples": int(len(y_test))}
    print(f"Ensemble Evaluation -> RMSE: {rmse:.6f} | MAE: {mae:.6f} | R2: {r2:.4f}")

    if metrics_output_path:
        with open(metrics_output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics

if __name__ == "__main__":
    evaluate_ensemble(
        os.path.join(os.path.dirname(__file__), "..", "models", "new"),
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed_data.csv"),
        os.path.join(os.path.dirname(__file__), "..", "models", "new", "metrics_new.json")
    )