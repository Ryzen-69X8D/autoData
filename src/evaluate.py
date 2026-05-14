from __future__ import annotations

import json
import os
import sys
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import FEATURE_COLS


def _register_train_pickle_classes() -> None:
    """
    Airflow runs training as ``python src/train.py``. Older artifacts pickled
    the ensemble classes as __main__.StackedEnsemble instead of train.*.
    Registering the current classes on __main__ lets joblib load both forms.
    """
    try:
        from train import StackedEnsemble, WeightedEnsemble
    except Exception:
        return

    main_module = sys.modules.get("__main__")
    if main_module is not None:
        setattr(main_module, "StackedEnsemble", StackedEnsemble)
        setattr(main_module, "WeightedEnsemble", WeightedEnsemble)


def _create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i : i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.asarray(X_seq), np.asarray(y_seq)


def _load_features_and_target(data_path: str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    df = df.select_dtypes(include=[np.number])
    if "Daily_Return" not in df.columns:
        raise ValueError("Processed data must include Daily_Return to build Target.")

    df["Target"] = df["Daily_Return"].shift(-1)
    df.dropna(inplace=True)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    if not available_features:
        raise ValueError("No expected feature columns were found in processed data.")

    X = df[available_features].values
    y = df["Target"].values
    return X, y, available_features


def _test_split(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    split = int(len(X) * 0.80)
    X_test = X[split:]
    y_test = y[split:]

    if len(X_test) == 0:
        raise ValueError("Not enough rows to create an evaluation test split.")

    return X_test, y_test


def _evaluate_sklearn_model(
    model_path: str,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    _register_train_pickle_classes()

    model = joblib.load(model_path)
    X_test, y_test = _test_split(X, y)
    y_pred = np.asarray(model.predict(X_test)).reshape(-1)
    return y_test, y_pred


def _evaluate_lstm_model(
    model_path: str,
    X: np.ndarray,
    y: np.ndarray,
    n_features: int,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    class StockLSTM(torch.nn.Module):
        def __init__(self, input_size, hidden_size=32, num_layers=1):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    X_seq, y_seq = _create_sequences(X, y, seq_length=14)
    X_test, y_test = _test_split(X_seq, y_seq)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM(input_size=n_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_test_t).cpu().numpy().reshape(-1)

    return y_test, y_pred


def evaluate_model(
    model_path: str,
    data_path: str,
    metrics_output_path: str | None = None,
) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    X, y, available_features = _load_features_and_target(data_path)

    if model_path.endswith(".pt"):
        y_test, y_pred = _evaluate_lstm_model(model_path, X, y, len(available_features))
        model_type = "lstm"
    else:
        y_test, y_pred = _evaluate_sklearn_model(model_path, X, y)
        model_type = "sklearn_ensemble"

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "test_samples": int(len(y_test)),
        "model_type": model_type,
        "features": available_features,
        "n_features": len(available_features),
    }
    print(f"Evaluation -> RMSE: {rmse:.6f} | MAE: {mae:.6f} | R2: {r2:.4f}")

    if metrics_output_path:
        output_dir = os.path.dirname(os.path.abspath(metrics_output_path))
        os.makedirs(output_dir, exist_ok=True)
        with open(metrics_output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Eval metrics saved -> {metrics_output_path}")

    return metrics


def is_new_model_better(new_metrics_path: str, deployed_metrics_path: str) -> bool:
    if not os.path.exists(deployed_metrics_path):
        return True

    with open(new_metrics_path, encoding="utf-8") as f:
        new = json.load(f)
    with open(deployed_metrics_path, encoding="utf-8") as f:
        old = json.load(f)

    better = new.get("rmse", float("inf")) < old.get("rmse", float("inf"))
    print(
        f"New RMSE: {new.get('rmse'):.6f} | "
        f"Deployed: {old.get('rmse'):.6f} -> {'Deploy' if better else 'Keep'}"
    )
    return better


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    evaluate_model(
        model_path=os.path.join(root, "models", "random_forest_new.pkl"),
        data_path=os.path.join(root, "data", "processed", "processed_data.csv"),
        metrics_output_path=os.path.join(root, "models", "eval_metrics.json"),
    )
