"""
schemas.py
===========
Pydantic v2-compatible request and response schemas for the
Stock Prediction FastAPI service.

Separating schemas from main.py keeps the API contract explicit,
enables easy versioning, and makes unit-testing schemas trivial.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class Currency(str, Enum):
    INR = "INR"
    USD = "USD"


class Ticker(str, Enum):
    RELIANCE   = "RELIANCE.NS"
    TCS        = "TCS.NS"
    INFY       = "INFY.NS"
    HDFCBANK   = "HDFCBANK.NS"
    WIPRO      = "WIPRO.NS"
    ICICIBANK  = "ICICIBANK.NS"
    SBIN       = "SBIN.NS"
    BAJFINANCE = "BAJFINANCE.NS"
    NIFTY_50   = "NIFTY_50"
    UNKNOWN    = "UNKNOWN"


# ── Request schemas ───────────────────────────────────────────────────────────

class StockFeaturesRequest(BaseModel):
    """
    OHLCV + optional technical indicators for a single trading day.
    Used as the POST /predict request body.
    """

    Open:   float = Field(..., gt=0, description="Opening price in INR", example=2800.0)
    High:   float = Field(..., gt=0, description="Intraday high in INR",  example=2850.0)
    Low:    float = Field(..., gt=0, description="Intraday low in INR",   example=2760.0)
    Close:  float = Field(..., gt=0, description="Closing price in INR",  example=2820.0)
    Volume: float = Field(..., gt=0, description="Shares traded",         example=5_000_000)

    # Optional — model will fall back to Close-based defaults if omitted
    SMA_10:       Optional[float] = Field(None, ge=0, description="10-day simple moving average",  example=2790.0)
    SMA_50:       Optional[float] = Field(None, ge=0, description="50-day simple moving average",  example=2750.0)
    Daily_Return: Optional[float] = Field(None,        description="Today's % change (decimal)",   example=0.005)

    # Optional metadata
    ticker: Optional[str] = Field(None, description="NSE ticker symbol (informational)", example="RELIANCE.NS")
    date:   Optional[str] = Field(None, description="Trading date ISO-8601 (YYYY-MM-DD)", example="2024-12-31")

    @field_validator("High")
    @classmethod
    def high_gte_low(cls, v: float, info) -> float:  # noqa: N805
        # At this point 'Low' may not yet be validated; skip cross-field here
        return v

    @model_validator(mode="after")
    def cross_field_checks(self) -> "StockFeaturesRequest":
        if self.High < self.Low:
            raise ValueError(f"High ({self.High}) must be ≥ Low ({self.Low})")
        if self.High < self.Open:
            raise ValueError(f"High ({self.High}) must be ≥ Open ({self.Open})")
        if self.Low  > self.Open:
            raise ValueError(f"Low ({self.Low}) must be ≤ Open ({self.Open})")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "Open":   2800.0,
                "High":   2850.0,
                "Low":    2760.0,
                "Close":  2820.0,
                "Volume": 5_000_000,
                "SMA_10": 2790.0,
                "SMA_50": 2750.0,
                "Daily_Return": 0.005,
                "ticker": "RELIANCE.NS",
                "date":   "2024-12-31",
            }
        }
    }


class BatchPredictRequest(BaseModel):
    """
    A list of StockFeaturesRequest objects for bulk prediction.
    Used as the POST /predict/batch request body.
    Max 100 rows per call to protect latency.
    """
    rows: list[StockFeaturesRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Between 1 and 100 feature rows",
    )


# ── Response schemas ──────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Single-row prediction response returned by POST /predict."""

    predicted_return_pct:  float  = Field(..., description="Predicted next-day return as a percentage, e.g. 1.25")
    predicted_close_price: float  = Field(..., description="Predicted next-day close price in INR")
    confidence_band_low:   float  = Field(..., description="Lower bound of the 80% confidence band")
    confidence_band_high:  float  = Field(..., description="Upper bound of the 80% confidence band")
    currency:              Currency = Field(Currency.INR, description="Currency of the price fields")
    model_version:         str    = Field(..., description="Model artefact identifier")
    note:                  str    = Field(..., description="Human-readable interpretation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_return_pct":  1.25,
                "predicted_close_price": 2855.25,
                "confidence_band_low":   2810.00,
                "confidence_band_high":  2900.50,
                "currency":              "INR",
                "model_version":         "random_forest_v1",
                "note":                  "Model predicts a 1.25% move from the input Close price.",
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """Multi-row prediction response returned by POST /predict/batch."""
    predictions: list[PredictionResponse]
    total_rows:  int = Field(..., description="Number of rows processed")


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status:        str  = Field(..., description="'ok' when fully operational")
    model_loaded:  bool
    scaler_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: Optional[float] = None


class ModelInfoResponse(BaseModel):
    """Detailed model information returned by GET /model/info."""
    model_type:       str
    feature_columns:  list[str]
    training_metrics: Optional[dict] = None
    deployed_metrics: Optional[dict] = None
    model_path:       str
    scaler_path:      str
    model_size_kb:    Optional[float] = None


class ErrorResponse(BaseModel):
    """Standard error envelope used for 4xx / 5xx responses."""
    error:   str = Field(..., description="Short machine-readable error code")
    detail:  str = Field(..., description="Human-readable description")
    status_code: int


class RetrainRequest(BaseModel):
    """
    Optional body for POST /retrain — triggers an ad-hoc pipeline run.
    All fields are optional; defaults are read from config.yaml.
    """
    ticker:     Optional[str] = Field(None, example="RELIANCE.NS")
    start_date: Optional[str] = Field(None, example="2020-01-01")
    end_date:   Optional[str] = Field(None, example="2024-12-31")
    force:      bool = Field(False, description="Deploy even if new model is not better")


class RetrainResponse(BaseModel):
    """Response body for POST /retrain."""
    triggered:    bool
    message:      str
    new_metrics:  Optional[dict] = None
    deployed:     Optional[bool] = None


# ── Utility functions ─────────────────────────────────────────────────────────

def build_error(code: str, detail: str, status: int = 400) -> ErrorResponse:
    return ErrorResponse(error=code, detail=detail, status_code=status)
