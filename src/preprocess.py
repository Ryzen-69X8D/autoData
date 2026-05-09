import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume", 
    "SMA_10", "SMA_50", "Daily_Return",
    "MACD", "MACD_Signal", "RSI_14", "BB_Upper", "BB_Lower"
]

def preprocess_data(input_path: str, output_path: str, scaler_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    df = df.select_dtypes(include=[np.number]).dropna()

    # Base Features
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["Daily_Return"] = df["Close"].pct_change()

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

    df.dropna(inplace=True)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    
    scaler = MinMaxScaler()
    df[available_features] = scaler.fit_transform(df[available_features])

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved -> {output_path} ({len(df)} rows, {len(available_features)} features)")
    return df

if __name__ == "__main__":
    preprocess_data(
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "stock_data.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed_data.csv"),
        os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
    )