import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "SMA_10", "SMA_50", "Daily_Return"]


def preprocess_data(
    input_path: str,
    output_path: str,
    scaler_path: str = None,
) -> pd.DataFrame:
    """
    Reads raw OHLCV CSV, engineers features, scales them, and saves the result.

    FIX 1 – 'Date' column is dropped before any numeric operation.
    FIX 2 – MinMaxScaler is fit only on FEATURE_COLS (not on the future 'Target')
             so there is no target-leakage.
    FIX 3 – Scaler is persisted to disk so the FastAPI app can inverse-transform
             or scale incoming prediction requests consistently.
    """
    df = pd.read_csv(input_path)

    # ── FIX 1: Drop non-numeric / index columns ───────────────────────────────
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    # Drop any other object-dtype columns that might have slipped through
    df = df.select_dtypes(include=[np.number])
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("DataFrame is empty after removing NaN rows. "
                         "Check the raw data file.")

    # ── Feature engineering ───────────────────────────────────────────────────
    df["SMA_10"]       = df["Close"].rolling(window=10).mean()
    df["SMA_50"]       = df["Close"].rolling(window=50).mean()
    df["Daily_Return"] = df["Close"].pct_change()

    df.dropna(inplace=True)

    # ── FIX 2: Scale only input features, NOT the future target ──────────────
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    if not available_features:
        raise ValueError(f"None of the expected feature columns found. "
                         f"Got: {df.columns.tolist()}")

    scaler = MinMaxScaler()
    df[available_features] = scaler.fit_transform(df[available_features])

    # ── FIX 3: Persist scaler so the API can use it ───────────────────────────
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"✅  Scaler saved → {scaler_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅  Processed data saved → {output_path}  ({len(df)} rows, "
          f"{len(available_features)} features)")

    return df


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_FILE  = "/opt/airflow/data/raw/stock_data.csv"
    OUTPUT_FILE = "/opt/airflow/data/processed/processed_data.csv"
    SCALER_PATH = "/opt/airflow/models/scaler.pkl"

    preprocess_data(INPUT_FILE, OUTPUT_FILE, SCALER_PATH)
