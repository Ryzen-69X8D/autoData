import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os

from preprocess import FEATURE_COLS

# ── 1. PYTORCH NEURAL NETWORK ARCHITECTURE ────────────────────────────────────
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        
        # The LSTM layer processes the 14-day sequence
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        # The final linear layer outputs the single price prediction
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        # We only care about the prediction at the very last time step of the window
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# ── 2. DATA SEQUENCER ─────────────────────────────────────────────────────────
def create_sequences(data, target, seq_length):
    """Converts flat 2D tabular data into 3D time-series windows."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ── 3. TRAINING LOOP ──────────────────────────────────────────────────────────
def train_model(
    input_path: str,
    model_output_path: str,
    metrics_path: str = None,
) -> float:
    
    df = pd.read_csv(input_path)

    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    df = df.select_dtypes(include=[np.number])

    df["Target"] = df["Daily_Return"].shift(-1)
    df.dropna(inplace=True)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].values
    y = df["Target"].values

    # Convert data into 14-day rolling windows
    SEQ_LENGTH = 14
    X_seq, y_seq = create_sequences(X, y, SEQ_LENGTH)

    # 80/20 Chronological Split
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # 🔥 ROUTE TO GPU (RTX 5050)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Neural Network on: {device}")

    # Convert Numpy arrays to PyTorch Tensors and send to GPU
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)

    # Batching to utilize the 32GB RAM effectively
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = StockLSTM(input_size=len(available_features)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # L2 Regularization built-in

    # Train the network
    EPOCHS = 100
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(train_loader):.6f}")

    # Evaluate the network
    model.eval()
    with torch.no_grad():
        y_pred_t = model(X_test_t)
        # Pull predictions back to CPU for Scikit-Learn metrics
        y_pred = y_pred_t.cpu().numpy()
        y_true = y_test_t.cpu().numpy()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    print(f"Training complete → RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")

    # Save the PyTorch weights (.pt format)
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    pytorch_model_path = model_output_path.replace(".pkl", ".pt")
    torch.save(model.state_dict(), pytorch_model_path)
    print(f"✅  Model saved → {pytorch_model_path}")

    # Save metrics
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": available_features,
        "seq_length": SEQ_LENGTH
    }
    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅  Metrics saved → {metrics_path}")

    return rmse

if __name__ == "__main__":
    INPUT_FILE    = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed_data.csv")
    MODEL_OUTPUT  = os.path.join(os.path.dirname(__file__), "..", "models", "lstm_model.pkl") 
    METRICS_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "metrics_new.json")

    train_model(INPUT_FILE, MODEL_OUTPUT, METRICS_PATH)