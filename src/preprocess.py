import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ── Added MACD and RSI to the feature list ────────────────────────────────────
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume", 
    "SMA_10", "SMA_50", "Daily_Return",
    "MACD", "MACD_Signal", "RSI_14"  
]

def preprocess_data(
    input_path: str,
    output_path: str,
    scaler_path: str = None,
) -> pd.DataFrame:
    """
    Reads raw OHLCV CSV, engineers advanced momentum features, scales them, 
    and saves the result.
    """
    df = pd.read_csv(input_path)

    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    df = df.select_dtypes(include=[np.number])
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("DataFrame is empty. Check the raw data file.")

    # ── Basic Feature engineering
    df["SMA_10"]       = df["Close"].rolling(window=10).mean()
    df["SMA_50"]       = df["Close"].rolling(window=50).mean()
    df["Daily_Return"] = df["Close"].pct_change()

    # ── ADVANCED FEATURE 1: MACD ──────────────────────────────────────────────
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ── ADVANCED FEATURE 2: RSI (14-Day) ──────────────────────────────────────
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Drop the NaN rows created by the rolling 50-day windows and EMA math
    df.dropna(inplace=True)

    # ── Scale features
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    if not available_features:
        raise ValueError("None of the expected feature columns found.")

    scaler = MinMaxScaler()
    df[available_features] = scaler.fit_transform(df[available_features])

    # ── Persist scaler
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"✅  Scaler saved → {scaler_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅  Processed data saved → {output_path}  ({len(df)} rows, {len(available_features)} features)")

    return df

# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_FILE  = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "stock_data.csv")
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed_data.csv")
    SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")

    preprocess_data(INPUT_FILE, OUTPUT_FILE, SCALER_PATH)