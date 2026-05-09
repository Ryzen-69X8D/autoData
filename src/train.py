import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
import joblib
from preprocess import FEATURE_COLS

SEQ_LENGTH = 14

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(data, target, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(target[i + seq_length])
    return np.array(xs), np.array(ys)

def train_ensemble(input_path: str, model_dir: str, metrics_path: str):
    df = pd.read_csv(input_path)
    df["Target"] = df["Daily_Return"].shift(-1)
    df.dropna(inplace=True)

    X = df[FEATURE_COLS].values
    y = df["Target"].values

    # Sequential data for LSTM
    X_seq, y_seq = create_sequences(X, y, SEQ_LENGTH)
    
    # Flat data for XGB and RF (aligning indices with sequences)
    X_flat = X[SEQ_LENGTH:]
    y_flat = y[SEQ_LENGTH:]

    split = int(len(X_seq) * 0.8)
    
    # 1. Train PyTorch LSTM (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training PyTorch LSTM on: {device}")
    
    X_train_t = torch.FloatTensor(X_seq[:split]).to(device)
    y_train_t = torch.FloatTensor(y_seq[:split]).view(-1, 1).to(device)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=False)

    lstm_model = StockLSTM(input_size=len(FEATURE_COLS)).to(device)
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    lstm_model.train()
    for epoch in range(100):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(lstm_model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    os.makedirs(model_dir, exist_ok=True)
    torch.save(lstm_model.state_dict(), os.path.join(model_dir, "lstm_model.pt"))

    # 2. Train XGBoost (GPU Accelerated)
    print("🚀 Training XGBoost on GPU...")
    xgb_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, 
                             tree_method="hist", device="cuda" if torch.cuda.is_available() else "cpu")
    xgb_model.fit(X_flat[:split], y_flat[:split])
    xgb_model.save_model(os.path.join(model_dir, "xgb_model.json"))

    # 3. Train Random Forest (CPU Multi-core)
    print("🚀 Training Random Forest on CPU (All Cores)...")
    rf_model = RandomForestRegressor(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42)
    rf_model.fit(X_flat[:split], y_flat[:split])
    joblib.dump(rf_model, os.path.join(model_dir, "rf_model.pkl"))

    print("✅ Ensemble training complete. Run evaluate.py for metrics.")
    return True

if __name__ == "__main__":
    train_ensemble(
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed_data.csv"),
        os.path.join(os.path.dirname(__file__), "..", "models", "new"),
        os.path.join(os.path.dirname(__file__), "..", "models", "new", "metrics_new.json")
    )