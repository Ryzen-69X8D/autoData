import pandas as pd
import os
import glob

def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    output_path: str = None
) -> pd.DataFrame:
    """
    Loads historical stock data from local NIFTY 50 CSV files.
    Replaces the yfinance API call to avoid rate limits.

    Args:
        ticker      : Ignored (kept for compatibility with existing pipeline).
        start_date  : Start date as 'YYYY-MM-DD'.
        end_date    : End date as 'YYYY-MM-DD'.
        output_path : Optional CSV path to save the data.

    Returns:
        pd.DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    print(f"Fetching local NIFTY 50 data from {start_date} to {end_date}...")

    try:
        # 1. Point this to where you stored your CSV files
        # This relative path resolves to autoData/data/nifty_csvs
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "nifty_csvs")
        
        # Grab all the NIFTY 50 CSV files in that directory
        csv_files = glob.glob(f"{data_dir}/NIFTY 50-*.csv")
        
        if not csv_files:
            print(f"⚠️  No CSV files found in {data_dir}.")
            # Fallback in case files are placed in the root execution directory
            csv_files = glob.glob("*NIFTY 50-*.csv")
            if not csv_files:
                raise FileNotFoundError("Could not locate the NIFTY 50 CSV files.")

        # 2. Read and combine all CSVs
        dfs = [pd.read_csv(f) for f in csv_files]
        stock_data = pd.concat(dfs, ignore_index=True)

        # 3. Clean and format to exactly match the old yfinance output
        stock_data.columns = stock_data.columns.str.strip()
        stock_data.rename(columns={'Shares Traded': 'Volume'}, inplace=True)
        
        # 4. Convert Date and sort chronologically
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)

        # 5. Filter by the requested date range (Mimics yfinance start/end dates)
        mask = (stock_data['Date'] >= pd.to_datetime(start_date)) & (stock_data['Date'] <= pd.to_datetime(end_date))
        stock_data = stock_data.loc[mask].copy()

        if stock_data.empty:
            print(f"⚠️  No data found between {start_date} and {end_date}.")
            return stock_data

        # Keep only the essential columns (Drops the Turnover column)
        keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"]
                if c in stock_data.columns]
        stock_data = stock_data[keep]

        # Drop any rows where essential OHLCV data is missing
        stock_data.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

        # 6. Save the output for the Preprocess task
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            stock_data.to_csv(output_path, index=False)
            print(f"✅  Local data saved → {output_path}  ({len(stock_data)} rows)")

        return stock_data

    except Exception as e:
        print(f"❌  Error while processing local CSVs: {e}")
        return pd.DataFrame()


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # TICKER is now ignored, but kept so you don't have to change other scripts
    TICKER     = "NIFTY_50_LOCAL" 
    START_DATE = "2020-01-01"
    END_DATE   = "2024-12-31"
    
    # Resolving path to work locally without Docker
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "stock_data.csv")

    df = fetch_stock_data(TICKER, START_DATE, END_DATE, OUTPUT_FILE)

    if not df.empty:
        print("\nFirst 5 rows:")
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")