from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db.schemas import MarketTicker, SectorPerformance


router = APIRouter(prefix="/markets", tags=["markets"])


MARKET_TICKERS = [
    MarketTicker(
        ticker="NIFTY_50",
        name="Nifty 50",
        price=22580.35,
        change_pct=0.74,
        volume=245000000,
        sparkline=[22410, 22480, 22452, 22505, 22530, 22580],
    ),
    MarketTicker(
        ticker="RELIANCE.NS",
        name="Reliance Industries",
        price=2875.4,
        change_pct=1.18,
        volume=12600000,
        sparkline=[2830, 2842, 2858, 2850, 2869, 2875],
    ),
    MarketTicker(
        ticker="TCS.NS",
        name="Tata Consultancy Services",
        price=3924.65,
        change_pct=-0.32,
        volume=3120000,
        sparkline=[3944, 3938, 3949, 3932, 3928, 3924],
    ),
    MarketTicker(
        ticker="INFY.NS",
        name="Infosys",
        price=1498.2,
        change_pct=0.41,
        volume=8750000,
        sparkline=[1482, 1489, 1492, 1491, 1496, 1498],
    ),
    MarketTicker(
        ticker="HDFCBANK.NS",
        name="HDFC Bank",
        price=1548.75,
        change_pct=-0.58,
        volume=18400000,
        sparkline=[1565, 1558, 1550, 1554, 1549, 1548],
    ),
]

SECTORS = [
    SectorPerformance(sector="Banking", change_pct=-0.21),
    SectorPerformance(sector="Energy", change_pct=1.08),
    SectorPerformance(sector="Information Technology", change_pct=0.35),
    SectorPerformance(sector="FMCG", change_pct=0.12),
    SectorPerformance(sector="Auto", change_pct=0.87),
    SectorPerformance(sector="Pharma", change_pct=-0.44),
]


@router.get("/overview", response_model=list[MarketTicker])
def market_overview() -> list[MarketTicker]:
    return MARKET_TICKERS


@router.get("/ticker/{ticker}", response_model=MarketTicker)
def ticker_quote(ticker: str) -> MarketTicker:
    symbol = ticker.upper()
    for item in MARKET_TICKERS:
        if item.ticker.upper() == symbol:
            return item
    raise HTTPException(status_code=404, detail=f"Ticker {ticker} is not available")


@router.get("/sectors", response_model=list[SectorPerformance])
def sector_performance() -> list[SectorPerformance]:
    return SECTORS


@router.get("/movers")
def market_movers() -> dict[str, list[MarketTicker]]:
    ranked = sorted(MARKET_TICKERS, key=lambda item: item.change_pct, reverse=True)
    return {"gainers": ranked[:3], "losers": ranked[-3:]}
