import yfinance as yf
import pandas as pd
import os


def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    output_path: str = None
) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance for Indian or global stocks.

    Args:
        ticker      : Yahoo Finance ticker (e.g. 'RELIANCE.NS', 'TCS.NS', 'INFY.NS').
        start_date  : Start date as 'YYYY-MM-DD'.
        end_date    : End date as 'YYYY-MM-DD'.
        output_path : Optional CSV path to save the data.

    Returns:
        pd.DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")

    try:
        # auto_adjust=True → gives adjusted OHLC, removes 'Adj Close'
        # progress=False   → suppresses tqdm bar (cleaner logs in Airflow)
        stock_data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

        if stock_data.empty:
            print(f"⚠️  No data found for {ticker}. Check ticker symbol or dates.")
            return stock_data

        # ── FIX 1: yfinance ≥0.2 returns MultiIndex columns ──────────────────
        # e.g. ('Open', 'RELIANCE.NS') → 'Open'
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)

        # ── FIX 2: promote Date from index to a regular column ────────────────
        stock_data.reset_index(inplace=True)

        # Keep only the columns we need (some tickers have extra columns)
        keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"]
                if c in stock_data.columns]
        stock_data = stock_data[keep]

        # Drop any rows where essential OHLCV data is missing
        stock_data.dropna(subset=["Open", "High", "Low", "Close", "Volume"],
                          inplace=True)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            stock_data.to_csv(output_path, index=False)
            print(f"✅  Data saved → {output_path}  ({len(stock_data)} rows)")

        return stock_data

    except Exception as e:
        print(f"❌  Error while fetching {ticker}: {e}")
        return pd.DataFrame()


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Popular NSE tickers: RELIANCE.NS  TCS.NS  INFY.NS  HDFCBANK.NS  WIPRO.NS
    TICKER     = "RELIANCE.NS"
    START_DATE = "2020-01-01"
    END_DATE   = "2024-12-31"
    OUTPUT_FILE = "/opt/airflow/data/raw/stock_data.csv"

    df = fetch_stock_data(TICKER, START_DATE, END_DATE, OUTPUT_FILE)

    if not df.empty:
        print("\nFirst 5 rows:")
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
