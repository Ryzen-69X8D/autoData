from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from app.core.config import get_settings
from app.db.schemas import Currency, PredictionResponse, StockFeaturesRequest

try:
    from src.preprocess import FEATURE_COLS
except Exception:
    FEATURE_COLS = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA_5",
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "EMA_9",
        "EMA_21",
        "Daily_Return",
        "Return_2d",
        "Return_5d",
        "Return_10d",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "RSI_14",
        "BB_width",
        "BB_pct",
        "ATR_14",
        "Volatility_5",
        "Volatility_10",
        "Volume_SMA_10",
        "Volume_ratio",
        "High_Low_ratio",
        "Open_Close_ratio",
        "Day_of_week",
        "Month",
    ]

try:
    from app import model_loader
except Exception as exc:
    model_loader = None
    MODEL_LOADER_ERROR: Exception | None = exc
else:
    MODEL_LOADER_ERROR = None


class PredictionUnavailable(RuntimeError):
    pass


@dataclass
class RuntimeState:
    loaded_at: datetime | None = None
    errors: list[str] = field(default_factory=list)
    model_version: str | None = None


runtime = RuntimeState()


def load_models() -> dict:
    runtime.errors.clear()
    if model_loader is None:
        runtime.loaded_at = datetime.utcnow()
        runtime.model_version = _model_hash()
        runtime.errors.append(f"Model loader unavailable: {MODEL_LOADER_ERROR}")
        return {
            "ready": False,
            "model_version": runtime.model_version,
            "errors": runtime.errors,
        }

    result = model_loader.load_artefacts()
    runtime.loaded_at = datetime.utcnow()
    runtime.model_version = _model_hash()

    if result.get("status") != "success":
        runtime.errors.append(result.get("message", "Model artefacts could not be loaded"))

    return {
        "ready": is_ready(),
        "model_version": runtime.model_version,
        "errors": runtime.errors,
    }


def is_ready() -> bool:
    if model_loader is None:
        return False
    return model_loader.is_ready()


def model_health() -> dict:
    return {
        "ready": is_ready(),
        "loaded_at": runtime.loaded_at.isoformat() if runtime.loaded_at else None,
        "model_version": runtime.model_version,
        "feature_count": len(FEATURE_COLS),
        "features": FEATURE_COLS,
        "errors": runtime.errors,
    }


def predict(features: StockFeaturesRequest) -> PredictionResponse:
    if model_loader is None:
        raise PredictionUnavailable(f"Model loader unavailable: {MODEL_LOADER_ERROR}")

    if not is_ready():
        raise PredictionUnavailable("Model artefacts are not loaded. Run the ML pipeline first.")

    raw_df = pd.DataFrame([_feature_values(features)], columns=FEATURE_COLS)
    scaled_df = model_loader.scale_features(raw_df)

    history = _scaled_history()
    if history is not None and not history.empty:
        scaled_df = pd.concat([history.tail(13), scaled_df], ignore_index=True)

    scaled_pred = model_loader.predict_return(scaled_df)
    actual_return = model_loader.inverse_scale_return(scaled_pred)
    predicted_close = features.Close * (1 + actual_return)
    band_low = predicted_close * 0.985
    band_high = predicted_close * 1.015

    return PredictionResponse(
        predicted_return_pct=round(actual_return * 100, 4),
        predicted_close_price=round(predicted_close, 2),
        confidence_band_low=round(band_low, 2),
        confidence_band_high=round(band_high, 2),
        currency=Currency.INR,
        model_version=runtime.model_version or "ensemble",
        note=f"Ensemble model predicts a {actual_return * 100:+.2f}% next-session move.",
    )


def predict_batch(rows: list[StockFeaturesRequest]) -> list[PredictionResponse]:
    return [predict(row) for row in rows]


def _feature_values(features: StockFeaturesRequest) -> dict[str, float]:
    payload = features.model_dump()
    daily_return = payload.get("Daily_Return")
    if daily_return is None:
        daily_return = (features.Close - features.Open) / features.Open

    high_low_ratio = payload.get("High_Low_ratio")
    if high_low_ratio is None:
        high_low_ratio = (features.High - features.Low) / features.Close

    open_close_ratio = payload.get("Open_Close_ratio")
    if open_close_ratio is None:
        open_close_ratio = (features.Open - features.Close) / features.Close

    trading_date = _parse_date(features.date)

    defaults = {
        "SMA_5": features.Close,
        "SMA_10": features.Close,
        "SMA_20": features.Close,
        "SMA_50": features.Close,
        "EMA_9": features.Close,
        "EMA_21": features.Close,
        "Daily_Return": daily_return,
        "Return_2d": daily_return,
        "Return_5d": daily_return,
        "Return_10d": daily_return,
        "MACD": 0.0,
        "MACD_Signal": 0.0,
        "MACD_Hist": 0.0,
        "RSI_14": 50.0,
        "BB_width": max((features.High - features.Low) / features.Close, 0.0001),
        "BB_pct": 0.5,
        "ATR_14": max(features.High - features.Low, 0.0001),
        "Volatility_5": abs(daily_return),
        "Volatility_10": abs(daily_return),
        "Volume_SMA_10": features.Volume,
        "Volume_ratio": 1.0,
        "High_Low_ratio": high_low_ratio,
        "Open_Close_ratio": open_close_ratio,
        "Day_of_week": float(trading_date.weekday()),
        "Month": float(trading_date.month),
    }

    values: dict[str, float] = {}
    for column in FEATURE_COLS:
        value = payload.get(column)
        if value is None:
            value = defaults.get(column, 0.0)
        values[column] = float(value)
    return values


def _parse_date(value: str | None) -> date:
    if not value:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError:
        return date.today()


def _scaled_history() -> pd.DataFrame | None:
    settings = get_settings()
    path = Path(settings.processed_data_path)
    if not path.exists():
        return None

    try:
        history = pd.read_csv(path)
        history = history[[column for column in FEATURE_COLS if column in history.columns]]
        if list(history.columns) != FEATURE_COLS:
            return None

        numeric = history[FEATURE_COLS].astype(float)
        if numeric.min().min() < -0.05 or numeric.max().max() > 1.05:
            if model_loader is None:
                return None
            numeric = model_loader.scale_features(numeric)
        return numeric.tail(13)
    except Exception as exc:
        runtime.errors.append(f"History load failed: {exc}")
        return None


def _model_hash() -> str | None:
    settings = get_settings()
    digest = hashlib.sha256()
    paths = [
        Path(settings.model_dir) / "lstm_model.pt",
        Path(settings.model_dir) / "xgb_model.json",
        Path(settings.model_dir) / "rf_model.pkl",
        Path(settings.model_dir) / "scaler.pkl",
    ]

    existing = [path for path in paths if path.exists()]
    if not existing:
        return None

    for path in existing:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]
