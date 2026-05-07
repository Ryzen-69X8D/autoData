import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os

from preprocess import FEATURE_COLS   # reuse the single source-of-truth list


def train_model(
    input_path: str,
    model_output_path: str,
    metrics_path: str = None,
) -> float:
    """
    Trains a RandomForestRegressor to predict the *next day's scaled Daily Return*.
    """
    df = pd.read_csv(input_path)

    # ── FIX 1: Remove any non-feature columns ────────────────────────────────
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    df = df.select_dtypes(include=[np.number])

    # ── NEW TARGET: Predict percentage change instead of absolute price ──────
    df["Target"] = df["Daily_Return"].shift(-1)
    df.dropna(inplace=True)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False   # time-series order
    )

    # ── FIX 2: n_jobs=-1 for multi-core system ───────────────────────────────
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,         
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    print(f"Training complete → RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"✅  Model saved → {model_output_path}")

    # ── FIX 3: Persist metrics ────────────────────────────────────────────────
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features": available_features,
    }
    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅  Metrics saved → {metrics_path}")

    return rmse


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_FILE    = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed_data.csv")
    MODEL_OUTPUT  = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest.pkl")
    METRICS_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")

    train_model(INPUT_FILE, MODEL_OUTPUT, METRICS_PATH)