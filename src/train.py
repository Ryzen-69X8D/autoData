import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor  
import joblib
import json
import os

from preprocess import FEATURE_COLS   


def train_model(
    input_path: str,
    model_output_path: str,
    metrics_path: str = None,
) -> float:
    """
    Trains an XGBoost model to predict the next day's scaled Daily Return.
    GPU Accelerated via CUDA.
    """
    df = pd.read_csv(input_path)

    # ── Remove any non-feature columns
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    df = df.select_dtypes(include=[np.number])

    # ── Predict percentage change
    df["Target"] = df["Daily_Return"].shift(-1)
    df.dropna(inplace=True)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # ── TUNED FOR FINANCIAL DATA (GPU Acceleration + Prevents Overfitting) ──
    model = XGBRegressor(
        n_estimators=150,        # Reduced so it doesn't memorize data
        max_depth=4,             # Shallower trees force the AI to look at broader trends
        learning_rate=0.05,      # Slightly faster learning rate
        subsample=0.8,           # Uses 80% of data per tree 
        colsample_bytree=0.8,    # Uses 80% of features per tree
        reg_lambda=1.5,          # L2 Regularization (penalizes extreme price guesses)
        reg_alpha=0.1,           # L1 Regularization (ignores useless features)
        random_state=42,
        tree_method="hist",      
        device="cuda",           # 🔥 Routes math to RTX 5050 GPU
        n_jobs=-1                # 🔥 Routes data processing to Ryzen 7 CPU
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    print(f"Training complete → RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")

    # Note: We still save it as 'random_forest.pkl' so we don't have to rewrite API/Airflow code!
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"✅  Model saved → {model_output_path}")

    # ── Persist metrics
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
    METRICS_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "metrics_new.json") 

    train_model(INPUT_FILE, MODEL_OUTPUT, METRICS_PATH)