from __future__ import annotations

import pandas as pd

from src.ingest import fetch_stock_data


def load_historical_market_data(
    ticker: str = "NIFTY_50",
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    output_path: str | None = None,
) -> pd.DataFrame:
    return fetch_stock_data(ticker, start_date, end_date, output_path)
