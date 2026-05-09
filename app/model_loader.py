"""
model_loader.py
================
Thread-safe, hot-reloadable artefact loader for the FastAPI service.

Responsibilities:
  • Load RandomForest model (.pkl) and MinMaxScaler (.pkl) from disk
  • Cache them in module-level singletons so uvicorn workers share state
  • Expose a reload() function for hot-swapping without restarting the server
  • Provide model metadata (version, size, feature list, metrics)
  • Raise clean HTTPException with 503 when artefacts are missing
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Default paths (overridden by env vars or explicit call to configure()) ─────
_DEFAULT_MODEL_DIR  = os.getenv("MODEL_DIR", "./models")
_MODEL_FILENAME     = "random_forest.pkl"
_SCALER_FILENAME    = "scaler.pkl"
_METRICS_FILENAME   = "metrics_deployed.json"

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_50", "Daily_Return",
]


# ── Internal state ─────────────────────────────────────────────────────────────

@dataclass
class _ArtefactState:
    model:          Any            = None
    scaler:         Any            = None
    model_path:     str            = ""
    scaler_path:    str            = ""
    metrics_path:   str            = ""
    model_hash:     Optional[str]  = None
    loaded_at:      Optional[float] = None   # epoch seconds
    load_errors:    list[str]      = field(default_factory=list)


_state = _ArtefactState()
_lock  = Lock()
_start_time = time.time()


# ── Public API ─────────────────────────────────────────────────────────────────

def configure(model_dir: str | None = None) -> None:
    """
    (Optional) Set the model directory before the first load.
    If not called, the loader uses MODEL_DIR env var or ./models.
    """
    global _state
    base = model_dir or _DEFAULT_MODEL_DIR
    with _lock:
        _state.model_path   = os.path.join(base, _MODEL_FILENAME)
        _state.scaler_path  = os.path.join(base, _SCALER_FILENAME)
        _state.metrics_path = os.path.join(base, _METRICS_FILENAME)


def load_artefacts(model_dir: str | None = None) -> dict:
    """
    Loads (or reloads) the model and scaler from disk.
    Thread-safe.  Called at startup and by reload().

    Returns a status dict with keys:
        model_loaded, scaler_loaded, model_version, errors
    """
    configure(model_dir)
    errors: list[str] = []

    with _lock:
        # ── Model ──────────────────────────────────────────────────────────────
        if os.path.exists(_state.model_path):
            try:
                _state.model       = joblib.load(_state.model_path)
                _state.model_hash  = _file_md5(_state.model_path)
                _state.loaded_at   = time.time()
                log.info("✅  Model loaded from %s  (md5=%s)",
                         _state.model_path, _state.model_hash[:8])
            except Exception as exc:
                msg = f"Failed to load model: {exc}"
                errors.append(msg)
                log.error(msg)
        else:
            msg = f"Model not found at {_state.model_path}. Run the pipeline first."
            errors.append(msg)
            log.warning("⚠️  %s", msg)

        # ── Scaler ─────────────────────────────────────────────────────────────
        if os.path.exists(_state.scaler_path):
            try:
                _state.scaler = joblib.load(_state.scaler_path)
                log.info("✅  Scaler loaded from %s", _state.scaler_path)
            except Exception as exc:
                msg = f"Failed to load scaler: {exc}"
                errors.append(msg)
                log.error(msg)
        else:
            msg = f"Scaler not found at {_state.scaler_path}. Run the pipeline first."
            errors.append(msg)
            log.warning("⚠️  %s", msg)

        _state.load_errors = errors

    return {
        "model_loaded":  _state.model  is not None,
        "scaler_loaded": _state.scaler is not None,
        "model_version": _state.model_hash[:8] if _state.model_hash else None,
        "errors":        errors,
    }


def reload() -> dict:
    """Hot-reload artefacts from disk without restarting the server."""
    log.info("🔄  Hot-reloading model artefacts…")
    return load_artefacts()


# ── Getters (raise 503 if artefact missing) ───────────────────────────────────

def get_model() -> Any:
    """Returns the loaded model or raises RuntimeError if unavailable."""
    if _state.model is None:
        raise RuntimeError(
            "Model is not loaded. Call load_artefacts() first, "
            "or run the pipeline to generate a model file."
        )
    return _state.model


def get_scaler() -> Any:
    """Returns the loaded scaler or raises RuntimeError if unavailable."""
    if _state.scaler is None:
        raise RuntimeError(
            "Scaler is not loaded. Call load_artefacts() first, "
            "or run the pipeline to generate a scaler file."
        )
    return _state.scaler


def is_ready() -> bool:
    """True only when both model and scaler are in memory."""
    return _state.model is not None and _state.scaler is not None


# ── Inference helpers ──────────────────────────────────────────────────────────

def build_feature_frame(
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    sma_10: float | None = None,
    sma_50: float | None = None,
    daily_return: float | None = None,
) -> pd.DataFrame:
    """
    Assembles a single-row DataFrame with the correct column order
    expected by the trained model.

    Falls back to Close for SMA values and 0.0 for Daily_Return when
    optional fields are omitted.
    """
    return pd.DataFrame(
        [{
            "Open":         open_,
            "High":         high,
            "Low":          low,
            "Close":        close,
            "Volume":       volume,
            "SMA_10":       sma_10       if sma_10       is not None else close,
            "SMA_50":       sma_50       if sma_50       is not None else close,
            "Daily_Return": daily_return if daily_return is not None else 0.0,
        }],
        columns=FEATURE_COLS,
    )


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the fitted MinMaxScaler to *df* (must have FEATURE_COLS columns)."""
    scaler = get_scaler()
    scaled = scaler.transform(df[FEATURE_COLS])
    return pd.DataFrame(scaled, columns=FEATURE_COLS)


def predict_return(scaled_df: pd.DataFrame) -> float:
    """
    Runs inference and returns the *scaled* Daily_Return prediction.
    Use inverse_scale_return() to convert back to a decimal percentage.
    """
    model = get_model()
    return float(model.predict(scaled_df)[0])


def inverse_scale_return(scaled_value: float) -> float:
    """
    Inverse-transforms a scaled Daily_Return prediction back to a
    real-world decimal (e.g. 0.0125 = +1.25%).
    """
    scaler     = get_scaler()
    ret_idx    = FEATURE_COLS.index("Daily_Return")
    dummy      = np.zeros((1, len(FEATURE_COLS)))
    dummy[0, ret_idx] = scaled_value
    inversed   = scaler.inverse_transform(dummy)
    return float(inversed[0, ret_idx])


def confidence_band(
    close: float,
    predicted_return: float,
    half_width_pct: float = 0.015,
) -> tuple[float, float]:
    """
    Returns a simple ±half_width_pct symmetric confidence band around
    the predicted close price.

    The default ±1.5% is a rough empirical estimate for daily NSE moves;
    replace with a proper quantile regression or conformal interval in prod.
    """
    predicted_close = close * (1 + predicted_return)
    low  = predicted_close * (1 - half_width_pct)
    high = predicted_close * (1 + half_width_pct)
    return round(low, 2), round(high, 2)


# ── Model metadata ────────────────────────────────────────────────────────────

def model_info() -> dict:
    """
    Returns a dict with model type, size, feature list, and any persisted metrics.
    Used by GET /model/info.
    """
    info: dict = {
        "model_type":      type(_state.model).__name__ if _state.model else "not_loaded",
        "feature_columns": FEATURE_COLS,
        "model_path":      _state.model_path,
        "scaler_path":     _state.scaler_path,
        "model_version":   _state.model_hash[:8] if _state.model_hash else None,
        "loaded_at":       _state.loaded_at,
        "model_size_kb":   None,
        "training_metrics": None,
        "deployed_metrics": None,
    }

    if os.path.exists(_state.model_path):
        info["model_size_kb"] = round(os.path.getsize(_state.model_path) / 1024, 1)

    if os.path.exists(_state.metrics_path):
        try:
            with open(_state.metrics_path) as f:
                info["deployed_metrics"] = json.load(f)
        except Exception:
            pass

    new_metrics_path = _state.metrics_path.replace("deployed", "new")
    if os.path.exists(new_metrics_path):
        try:
            with open(new_metrics_path) as f:
                info["training_metrics"] = json.load(f)
        except Exception:
            pass

    return info


def uptime_seconds() -> float:
    return round(time.time() - _start_time, 1)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _file_md5(path: str, chunk: int = 65_536) -> str:
    """Computes the MD5 hex-digest of *path* for change detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()
