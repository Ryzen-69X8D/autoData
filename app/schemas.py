from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator

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

class StockFeaturesRequest(BaseModel):
    Open:   float = Field(..., gt=0, description="Opening price in INR", example=2800.0)
    High:   float = Field(..., gt=0, description="Intraday high in INR",  example=2850.0)
    Low:    float = Field(..., gt=0, description="Intraday low in INR",   example=2760.0)
    Close:  float = Field(..., gt=0, description="Closing price in INR",  example=2820.0)
    Volume: float = Field(..., gt=0, description="Shares traded",         example=5_000_000)

    SMA_10:       Optional[float] = Field(None, ge=0, description="10-day simple moving average")
    SMA_50:       Optional[float] = Field(None, ge=0, description="50-day simple moving average")
    Daily_Return: Optional[float] = Field(None, description="Today's % change")

    # ── New Momentum Indicators ──────────────────────────────────────────────
    MACD:         Optional[float] = Field(None, description="Moving Average Convergence Divergence")
    MACD_Signal:  Optional[float] = Field(None, description="MACD Signal Line")
    RSI_14:       Optional[float] = Field(None, description="Relative Strength Index (14-day)")
    BB_Upper:     Optional[float] = Field(None, description="Bollinger Band Upper")
    BB_Lower:     Optional[float] = Field(None, description="Bollinger Band Lower")

    ticker: Optional[str] = Field(None, description="NSE ticker symbol (informational)", example="RELIANCE.NS")
    date:   Optional[str] = Field(None, description="Trading date ISO-8601", example="2024-12-31")

    @field_validator("High")
    @classmethod
    def high_gte_low(cls, v: float, info) -> float:
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

class PredictionResponse(BaseModel):
    predicted_return_pct:  float  = Field(..., description="Predicted next-day return as a percentage, e.g. 1.25")
    predicted_close_price: float  = Field(..., description="Predicted next-day close price in INR")
    confidence_band_low:   float  = Field(..., description="Lower bound of the 80% confidence band")
    confidence_band_high:  float  = Field(..., description="Upper bound of the 80% confidence band")
    currency:              Currency = Field(Currency.INR, description="Currency of the price fields")
    model_version:         str    = Field(..., description="Model artefact identifier")
    note:                  str    = Field(..., description="Human-readable interpretation")