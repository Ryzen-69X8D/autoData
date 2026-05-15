from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class Currency(str, Enum):
    INR = "INR"
    USD = "USD"


class UserCreate(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=160)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: EmailStr
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


class HoldingBase(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=40, examples=["RELIANCE.NS"])
    shares: Decimal = Field(..., gt=0)
    average_buy_price: Decimal = Field(..., gt=0)


class HoldingCreate(HoldingBase):
    pass


class HoldingUpdate(BaseModel):
    shares: Optional[Decimal] = Field(None, gt=0)
    average_buy_price: Optional[Decimal] = Field(None, gt=0)


class HoldingRead(HoldingBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    portfolio_id: int


class PortfolioRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    total_value: Decimal
    cash_balance: Decimal
    updated_at: datetime
    holdings: list[HoldingRead] = []


class PortfolioCashUpdate(BaseModel):
    cash_balance: Decimal = Field(..., ge=0)


class WatchlistRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    ticker_symbols: list[str]


class WatchlistUpdate(BaseModel):
    ticker_symbols: list[str] = Field(default_factory=list, max_length=100)


class StockFeaturesRequest(BaseModel):
    Open: float = Field(..., gt=0, description="Opening price in INR", examples=[22500.0])
    High: float = Field(..., gt=0, description="Intraday high in INR", examples=[22640.0])
    Low: float = Field(..., gt=0, description="Intraday low in INR", examples=[22420.0])
    Close: float = Field(..., gt=0, description="Closing price in INR", examples=[22580.0])
    Volume: float = Field(..., gt=0, description="Shares traded", examples=[240000000])

    SMA_5: Optional[float] = Field(None, ge=0)
    SMA_10: Optional[float] = Field(None, ge=0)
    SMA_20: Optional[float] = Field(None, ge=0)
    SMA_50: Optional[float] = Field(None, ge=0)
    EMA_9: Optional[float] = Field(None, ge=0)
    EMA_21: Optional[float] = Field(None, ge=0)
    Daily_Return: Optional[float] = None
    Return_2d: Optional[float] = None
    Return_5d: Optional[float] = None
    Return_10d: Optional[float] = None
    MACD: Optional[float] = None
    MACD_Signal: Optional[float] = None
    MACD_Hist: Optional[float] = None
    RSI_14: Optional[float] = Field(None, ge=0, le=100)
    BB_width: Optional[float] = Field(None, ge=0)
    BB_pct: Optional[float] = None
    ATR_14: Optional[float] = Field(None, ge=0)
    Volatility_5: Optional[float] = Field(None, ge=0)
    Volatility_10: Optional[float] = Field(None, ge=0)
    Volume_SMA_10: Optional[float] = Field(None, ge=0)
    Volume_ratio: Optional[float] = Field(None, ge=0)
    High_Low_ratio: Optional[float] = None
    Open_Close_ratio: Optional[float] = None
    Day_of_week: Optional[float] = Field(None, ge=0, le=6)
    Month: Optional[float] = Field(None, ge=1, le=12)

    ticker: Optional[str] = Field(None, examples=["RELIANCE.NS"])
    date: Optional[str] = Field(None, examples=["2026-05-15"])

    @model_validator(mode="after")
    def validate_ohlc(self) -> "StockFeaturesRequest":
        if self.High < self.Low:
            raise ValueError("High must be greater than or equal to Low")
        if self.High < max(self.Open, self.Close):
            raise ValueError("High must be greater than or equal to Open and Close")
        if self.Low > min(self.Open, self.Close):
            raise ValueError("Low must be less than or equal to Open and Close")
        return self


class BatchPredictRequest(BaseModel):
    rows: list[StockFeaturesRequest] = Field(..., min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    predicted_return_pct: float
    predicted_close_price: float
    confidence_band_low: float
    confidence_band_high: float
    currency: Currency = Currency.INR
    model_version: str
    note: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_rows: int


class MarketTicker(BaseModel):
    ticker: str
    name: str
    price: float
    change_pct: float
    volume: int
    sparkline: list[float]


class SectorPerformance(BaseModel):
    sector: str
    change_pct: float


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    database: str
    model_version: str | None = None
    errors: list[str] = []
