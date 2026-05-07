import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os

from preprocess import FEATURE_COLS


def evaluate_model(
    model_path: str,
    data_path: str,
    metrics_output_path: str = None,
) -> dict:
    """
    Evaluates the model on the held-out test split (last 20% of data).
    Returns a metrics dict and optionally writes it to JSON.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    df    = pd.read_csv(data_path)

    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    df = df.select_dtypes(include=[np.number])

    # ── FIX: Evaluate against Daily_Return, matching train.py ─────────────
    df["Target"] = df["Daily_Return"].shift(-1)
    df.dropna(inplace=True)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features]
    y = df["Target"]

    # Keep temporal order for time-series evaluation
    split  = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    y_pred = model.predict(X_test)
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae    = float(mean_absolute_error(y_test, y_pred))
    r2     = float(r2_score(y_test, y_pred))

    metrics = {
        "rmse": rmse,
        "mae":  mae,
        "r2":   r2,
        "test_samples": int(len(y_test)),
    }
    print(f"Evaluation → RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")

    if metrics_output_path:
        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
        with open(metrics_output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅  Eval metrics saved → {metrics_output_path}")

    return metrics


def is_new_model_better(new_metrics_path: str, deployed_metrics_path: str) -> bool:
    """Returns True if the newly trained model has lower RMSE than the deployed one."""
    if not os.path.exists(deployed_metrics_path):
        print("No deployed metrics found → treating new model as better.")
        return True

    with open(new_metrics_path)      as f: new  = json.load(f)
    with open(deployed_metrics_path) as f: old  = json.load(f)

    new_rmse = new.get("rmse", float("inf"))
    old_rmse = old.get("rmse", float("inf"))
    better   = new_rmse < old_rmse

    print(f"New RMSE: {new_rmse:.6f} | Deployed RMSE: {old_rmse:.6f} → "
          f"{'✅ Deploy' if better else '⛔ Keep current'}")
    return better


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate_model(
        model_path          = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest.pkl"),
        data_path           = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed_data.csv"),
        metrics_output_path = os.path.join(os.path.dirname(__file__), "..", "models", "eval_metrics.json"),
    )