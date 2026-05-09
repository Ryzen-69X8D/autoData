import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBRegressor

from src.preprocess import FEATURE_COLS

_PROD_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "prod")
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── SHRUNK ARCHITECTURE TO PREVENT OVERFITTING ──
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class ArtefactState:
    lstm_model = None
    xgb_model = None
    rf_model = None
    scaler = None

state = ArtefactState()

def load_artefacts():
    try:
        state.scaler = joblib.load(os.path.join(_PROD_DIR, "scaler.pkl"))
        
        state.lstm_model = StockLSTM(input_size=len(FEATURE_COLS)).to(_device)
        state.lstm_model.load_state_dict(torch.load(os.path.join(_PROD_DIR, "lstm_model.pt"), map_location=_device, weights_only=True))
        state.lstm_model.eval()

        state.xgb_model = XGBRegressor()
        state.xgb_model.load_model(os.path.join(_PROD_DIR, "xgb_model.json"))

        state.rf_model = joblib.load(os.path.join(_PROD_DIR, "rf_model.pkl"))
        
        print("✅ Production Ensemble Loaded Successfully")
        return {"status": "success"}
    except Exception as e:
        print(f"Error loading models: {e}")
        return {"status": "error", "message": str(e)}

def is_ready():
    return all(m is not None for m in [state.lstm_model, state.xgb_model, state.rf_model, state.scaler])

def build_feature_frame(**kwargs) -> pd.DataFrame:
    """Dynamically builds the frame based on provided kwargs."""
    row = {}
    for col in FEATURE_COLS:
        row[col] = kwargs.get(col) or kwargs.get(col.lower()) or 0.0
    return pd.DataFrame([row], columns=FEATURE_COLS)

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(state.scaler.transform(df[FEATURE_COLS]), columns=FEATURE_COLS)

def predict_return(scaled_history_df: pd.DataFrame) -> float:
    flat_latest = scaled_history_df.iloc[-1:].values
    seq_latest = torch.FloatTensor(scaled_history_df.values).unsqueeze(0).to(_device)

    with torch.no_grad():
        lstm_pred = state.lstm_model(seq_latest).cpu().numpy()[0][0]
    
    xgb_pred = state.xgb_model.predict(flat_latest)[0]
    rf_pred = state.rf_model.predict(flat_latest)[0]

    return float((lstm_pred + xgb_pred + rf_pred) / 3.0)

def inverse_scale_return(scaled_value: float) -> float:
    ret_idx = FEATURE_COLS.index("Daily_Return")
    dummy = np.zeros((1, len(FEATURE_COLS)))
    dummy[0, ret_idx] = scaled_value
    return float(state.scaler.inverse_transform(dummy)[0, ret_idx])