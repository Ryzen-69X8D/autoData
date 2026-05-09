"""
utils.py
=========
Shared helper utilities used across the MLOps pipeline.
Covers logging setup, path resolution, JSON I/O, metric comparison,
data validation, and simple notification hooks.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a module-level logger with a consistent format.
    All pipeline modules should call this instead of print() for production use.

    Example:
        log = get_logger(__name__)
        log.info("Starting training...")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = get_logger(__name__)


# ── Path helpers ──────────────────────────────────────────────────────────────

def project_root() -> Path:
    """Returns the repository root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def resolve_path(*parts: str) -> str:
    """
    Builds an absolute path relative to the project root.

    Example:
        resolve_path("models", "random_forest.pkl")
        # → /path/to/project/models/random_forest.pkl
    """
    return str(project_root().joinpath(*parts))


def ensure_dir(path: str) -> str:
    """Creates all intermediate directories for *path* and returns it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


# ── JSON helpers ──────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    """Loads a JSON file and returns the parsed dict.  Raises if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str, indent: int = 2) -> None:
    """Serialises *data* to *path*, creating parent dirs as needed."""
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=_json_serialiser)
    log.info("JSON saved → %s", path)


def _json_serialiser(obj: Any) -> Any:
    """Handles numpy scalars so json.dump never raises TypeError."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ── Metric helpers ────────────────────────────────────────────────────────────

def compare_metrics(
    new_metrics: dict,
    deployed_metrics: dict,
    primary_key: str = "rmse",
    lower_is_better: bool = True,
) -> bool:
    """
    Returns True if *new_metrics* is an improvement over *deployed_metrics*.

    Args:
        new_metrics      : Metrics dict from the freshly trained model.
        deployed_metrics : Metrics dict currently in production.
        primary_key      : Metric to compare (default: 'rmse').
        lower_is_better  : If True, a lower value is considered better.

    Returns:
        bool — True means "deploy the new model".
    """
    new_val = new_metrics.get(primary_key, float("inf"))
    dep_val = deployed_metrics.get(primary_key, float("inf"))

    if lower_is_better:
        result = new_val < dep_val
    else:
        result = new_val > dep_val

    direction = "↓" if lower_is_better else "↑"
    status = "✅ BETTER" if result else "⛔ WORSE/EQUAL"
    log.info(
        "%s comparison — new: %.6f | deployed: %.6f %s → %s",
        primary_key.upper(), new_val, dep_val, direction, status,
    )
    return result


def metrics_summary(metrics: dict) -> str:
    """Returns a one-line human-readable summary of a metrics dict."""
    parts = [f"{k.upper()}={v:.6f}" if isinstance(v, float) else f"{k.upper()}={v}"
             for k, v in metrics.items()]
    return " | ".join(parts)


# ── Data validation ───────────────────────────────────────────────────────────

def validate_ohlcv(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Checks that a DataFrame has valid OHLCV data.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: list[str] = []
    required = ["Open", "High", "Low", "Close", "Volume"]

    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return False, errors

    # High must be ≥ Low
    bad_hl = (df["High"] < df["Low"]).sum()
    if bad_hl:
        errors.append(f"{bad_hl} rows where High < Low")

    # No negative prices
    for col in ["Open", "High", "Low", "Close"]:
        neg = (df[col] <= 0).sum()
        if neg:
            errors.append(f"{neg} non-positive values in {col}")

    # No negative volume
    if (df["Volume"] < 0).sum():
        errors.append("Negative Volume values found")

    # Check for NaNs
    nan_counts = df[required].isnull().sum()
    for col, count in nan_counts.items():
        if count:
            errors.append(f"{count} NaN values in {col}")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_dataframe_not_empty(df: pd.DataFrame, context: str = "") -> None:
    """Raises ValueError if *df* is empty."""
    if df.empty:
        msg = f"DataFrame is empty{' — ' + context if context else ''}."
        log.error(msg)
        raise ValueError(msg)


# ── File helpers ──────────────────────────────────────────────────────────────

def safe_copy(src: str, dst: str) -> None:
    """Copies *src* → *dst*, creating parent directories as needed."""
    ensure_dir(dst)
    shutil.copy2(src, dst)
    log.info("Copied %s → %s", src, dst)


def file_age_seconds(path: str) -> float:
    """Returns how many seconds ago *path* was last modified (mtime)."""
    if not os.path.exists(path):
        return float("inf")
    return time.time() - os.path.getmtime(path)


def file_age_hours(path: str) -> float:
    return file_age_seconds(path) / 3600


# ── Date helpers ──────────────────────────────────────────────────────────────

def today_str(fmt: str = "%Y-%m-%d") -> str:
    """Returns today's date as a formatted string."""
    return datetime.today().strftime(fmt)


def date_range_str(start: str, end: str) -> str:
    return f"{start} → {end}"


# ── Feature helpers ───────────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds SMA_10, SMA_50, Daily_Return, and optionally RSI_14
    to any DataFrame that has a 'Close' column.
    Returns a new DataFrame (does not mutate the original).
    """
    df = df.copy()

    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["Daily_Return"] = df["Close"].pct_change()

    # RSI-14
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20-period, 2σ)
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma20

    # Volatility proxy (10-day rolling std of daily returns)
    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()

    return df


def compute_stats(series: pd.Series) -> dict:
    """Returns basic descriptive statistics for a numeric Series."""
    return {
        "count": int(series.count()),
        "mean":  float(series.mean()),
        "std":   float(series.std()),
        "min":   float(series.min()),
        "p25":   float(series.quantile(0.25)),
        "p50":   float(series.median()),
        "p75":   float(series.quantile(0.75)),
        "max":   float(series.max()),
    }


# ── Notification stub ─────────────────────────────────────────────────────────

def notify(message: str, channel: str = "log") -> None:
    """
    Dispatches a notification.
    Currently only logs; extend to Slack / email / webhook as needed.

    Args:
        message : Human-readable notification text.
        channel : 'log' (default) | 'slack' | 'email'
    """
    if channel == "log":
        log.info("[NOTIFY] %s", message)
    elif channel == "slack":
        # Extend: post to Slack webhook URL stored in env var SLACK_WEBHOOK_URL
        webhook = os.getenv("SLACK_WEBHOOK_URL")
        if webhook:
            try:
                import urllib.request
                payload = json.dumps({"text": message}).encode()
                req = urllib.request.Request(
                    webhook,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=5)
                log.info("[NOTIFY] Slack message sent.")
            except Exception as exc:
                log.warning("[NOTIFY] Slack failed: %s", exc)
        else:
            log.warning("[NOTIFY] SLACK_WEBHOOK_URL not set; falling back to log.")
            log.info("[NOTIFY] %s", message)
    else:
        log.info("[NOTIFY][%s] %s", channel.upper(), message)


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("project_root = %s", project_root())
    log.info("today        = %s", today_str())

    # Validate a tiny synthetic OHLCV frame
    sample = pd.DataFrame({
        "Open":   [100.0, 101.0],
        "High":   [105.0, 106.0],
        "Low":    [99.0,  100.0],
        "Close":  [103.0, 104.0],
        "Volume": [1_000_000, 1_200_000],
    })
    ok, errs = validate_ohlcv(sample)
    log.info("OHLCV valid=%s  errors=%s", ok, errs)

    # Metric comparison
    new_m = {"rmse": 0.038, "mae": 0.027, "r2": 0.12}
    dep_m = {"rmse": 0.040, "mae": 0.029, "r2": 0.09}
    deploy = compare_metrics(new_m, dep_m)
    log.info("Should deploy? %s", deploy)

    notify("Pipeline self-test complete.")
