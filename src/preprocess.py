"""
preprocess.py
==============
Advanced feature engineering for NSE NIFTY 50 stock prediction.
Produces 30 scaled features covering momentum, volatility, trend,
volume, and calendar effects.

Compatible with train.py → XGBoost/LightGBM/RF ensemble.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS  (must match across preprocess / train / evaluate / API)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # ── Raw OHLCV ──────────────────────────────────────────────────────────
    "Open", "High", "Low", "Close", "Volume",

    # ── Trend: Simple & Exponential Moving Averages ────────────────────────
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "EMA_9", "EMA_21",

    # ── Momentum: Returns at multiple horizons ─────────────────────────────
    "Daily_Return", "Return_2d", "Return_5d", "Return_10d",

    # ── MACD family ────────────────────────────────────────────────────────
    "MACD", "MACD_Signal", "MACD_Hist",

    # ── RSI ────────────────────────────────────────────────────────────────
    "RSI_14",

    # ── Bollinger Bands ────────────────────────────────────────────────────
    "BB_width", "BB_pct",

    # ── Volatility ─────────────────────────────────────────────────────────
    "ATR_14", "Volatility_5", "Volatility_10",

    # ── Volume ─────────────────────────────────────────────────────────────
    "Volume_SMA_10", "Volume_ratio",

    # ── Price structure ────────────────────────────────────────────────────
    "High_Low_ratio", "Open_Close_ratio",

    # ── Calendar ───────────────────────────────────────────────────────────
    "Day_of_week", "Month",
]
# Total: 30 features


def preprocess_data(
    input_path: str,
    output_path: str,
    scaler_path: str = None,
) -> pd.DataFrame:
    """
    Reads raw OHLCV CSV  →  engineers 30 features  →  MinMaxScales  →  saves.

    Returns the scaled DataFrame (Date column retained for temporal ordering
    in train.py but NOT included in FEATURE_COLS / scaler).
    """
    df = pd.read_csv(input_path)

    # ── Parse & sort by Date ────────────────────────────────────────────────
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    # ── Coerce OHLCV to numeric ─────────────────────────────────────────────
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError("DataFrame is empty after cleaning raw OHLCV data.")

    # ── Moving Averages ─────────────────────────────────────────────────────
    df["SMA_5"]  = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_9"]  = df["Close"].ewm(span=9,  adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()

    # ── Multi-horizon Returns ───────────────────────────────────────────────
    df["Daily_Return"] = df["Close"].pct_change()
    df["Return_2d"]    = df["Close"].pct_change(2)
    df["Return_5d"]    = df["Close"].pct_change(5)
    df["Return_10d"]   = df["Close"].pct_change(10)

    # ── MACD (12/26/9) ──────────────────────────────────────────────────────
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── RSI 14 ──────────────────────────────────────────────────────────────
    delta    = df["Close"].diff()
    up       = delta.clip(lower=0)
    down     = (-delta).clip(lower=0)
    ema_up   = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands (20-day, 2σ) ────────────────────────────────────────
    bb_sma   = df["Close"].rolling(20).mean()
    bb_std   = df["Close"].rolling(20).std()
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std
    denom    = (bb_upper - bb_lower).replace(0, np.nan)
    df["BB_width"] = denom / bb_sma
    df["BB_pct"]   = (df["Close"] - bb_lower) / denom

    # ── Average True Range 14 ───────────────────────────────────────────────
    hl   = df["High"] - df["Low"]
    hpc  = (df["High"]  - df["Close"].shift()).abs()
    lpc  = (df["Low"]   - df["Close"].shift()).abs()
    tr   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # ── Volatility ──────────────────────────────────────────────────────────
    df["Volatility_5"]  = df["Daily_Return"].rolling(5).std()
    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()

    # ── Volume features ─────────────────────────────────────────────────────
    df["Volume_SMA_10"] = df["Volume"].rolling(10).mean()
    df["Volume_ratio"]  = df["Volume"] / (df["Volume_SMA_10"].replace(0, np.nan))

    # ── Price structure ─────────────────────────────────────────────────────
    df["High_Low_ratio"]   = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
    df["Open_Close_ratio"] = (df["Open"] - df["Close"]) / df["Close"].replace(0, np.nan)

    # ── Calendar ────────────────────────────────────────────────────────────
    if "Date" in df.columns:
        df["Day_of_week"] = df["Date"].dt.dayofweek.astype(float)
        df["Month"]       = df["Date"].dt.month.astype(float)
    else:
        df["Day_of_week"] = 2.0   # Wednesday as neutral default
        df["Month"]       = 6.0

    # ── Drop NaN rows created by rolling windows ─────────────────────────────
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError("DataFrame empty after feature engineering. Need more data.")

    # ── Scale features (Date is NOT scaled) ─────────────────────────────────
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"⚠️  Missing features (will be skipped): {missing}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    df[available] = scaler.fit_transform(df[available])

    # ── Persist scaler ───────────────────────────────────────────────────────
    if scaler_path:
        os.makedirs(os.path.dirname(os.path.abspath(scaler_path)), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"✅  Scaler saved → {scaler_path}")

    # ── Save processed CSV (Date column kept for temporal ordering) ──────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(
        f"✅  Processed data saved → {output_path}  "
        f"({len(df)} rows, {len(available)} features)"
    )

    return df


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    _root       = os.path.join(os.path.dirname(__file__), "..")
    INPUT_FILE  = os.path.join(_root, "data", "raw",       "stock_data.csv")
    OUTPUT_FILE = os.path.join(_root, "data", "processed", "processed_data.csv")
    SCALER_PATH = os.path.join(_root, "models",             "scaler.pkl")

    preprocess_data(INPUT_FILE, OUTPUT_FILE, SCALER_PATH)
