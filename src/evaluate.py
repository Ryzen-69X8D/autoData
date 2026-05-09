import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os

from preprocess import FEATURE_COLS
from train import StockLSTM, create_sequences  # Imports your new Neural Network architecture

def evaluate_model(
    model_path: str,
    data_path: str,
    metrics_output_path: str = None,
) -> dict:
    
    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    df = df.select_dtypes(include=[np.number])

    df["Target"] = df["Daily_Return"].shift(-1)
    df.dropna(inplace=True)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].values
    y = df["Target"].values

    # Convert to 14-day sequences
    SEQ_LENGTH = 14
    X_seq, y_seq = create_sequences(X, y, SEQ_LENGTH)

    split = int(len(X_seq) * 0.8)
    X_test = X_seq[split:]
    y_test = y_seq[split:]

    # Route to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the PyTorch Model
    model = StockLSTM(input_size=len(available_features)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set to evaluation mode

    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)

    with torch.no_grad():
        y_pred_t = model(X_test_t)
        y_pred = y_pred_t.cpu().numpy()
        y_true = y_test_t.cpu().numpy()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    print(f"Evaluation → RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")

    if metrics_output_path:
        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
        with open(metrics_output_path, "w") as f:
            json.dump({"rmse": rmse, "mae": mae, "r2": r2, "test_samples": int(len(y_test))}, f, indent=2)
        print(f"✅  Eval metrics saved → {metrics_output_path}")

    return {"rmse": rmse, "mae": mae, "r2": r2}

def is_new_model_better(new_metrics_path: str, deployed_metrics_path: str) -> bool:
    if not os.path.exists(deployed_metrics_path):
        return True
    with open(new_metrics_path) as f: new = json.load(f)
    with open(deployed_metrics_path) as f: old = json.load(f)
    
    better = new.get("rmse", float("inf")) < old.get("rmse", float("inf"))
    print(f"New RMSE: {new.get('rmse'):.6f} | Deployed: {old.get('rmse'):.6f} → {'✅ Deploy' if better else '⛔ Keep'}")
    return better