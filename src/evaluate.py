import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os

from preprocess import FEATURE_COLS
from train import create_sequences

# ── SHRUNK ARCHITECTURE TO PREVENT OVERFITTING ──
class StockLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def evaluate_model(model_path: str, data_path: str, metrics_output_path: str = None) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    df = df.select_dtypes(include=[np.number])
    df["Target"] = df["Daily_Return"].shift(-1)
    df.dropna(inplace=True)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].values
    y = df["Target"].values

    SEQ_LENGTH = 14
    X_seq, y_seq = create_sequences(X, y, SEQ_LENGTH)
    split = int(len(X_seq) * 0.8)
    X_test = X_seq[split:]
    y_test = y_seq[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM(input_size=len(available_features)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    X_test_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    metrics = {"rmse": rmse, "mae":  mae, "r2":   r2, "test_samples": int(len(y_test))}
    print(f"LSTM Evaluation → RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")

    if metrics_output_path:
        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
        with open(metrics_output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅  Eval metrics saved → {metrics_output_path}")

    return metrics

def is_new_model_better(new_metrics_path: str, deployed_metrics_path: str) -> bool:
    if not os.path.exists(deployed_metrics_path):
        return True
    with open(new_metrics_path) as f: new = json.load(f)
    with open(deployed_metrics_path) as f: old = json.load(f)
    
    better = new.get("rmse", float("inf")) < old.get("rmse", float("inf"))
    print(f"New RMSE: {new.get('rmse'):.6f} | Deployed: {old.get('rmse'):.6f} → {'✅ Deploy' if better else '⛔ Keep'}")
    return better

if __name__ == "__main__":
    evaluate_model(
        model_path          = os.path.join(os.path.dirname(__file__), "..", "models", "new", "lstm_model.pt"),
        data_path           = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed_data.csv"),
        metrics_output_path = os.path.join(os.path.dirname(__file__), "..", "models", "new", "metrics_new.json"),
    )