import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker: str, start_date: str, end_date: str, output_path: str = None) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        output_path (str, optional): The file path to save the CSV. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the stock data.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if stock_data.empty:
            print(f"⚠️ Warning: No data found for {ticker}. Check the ticker symbol or dates.")
            return stock_data
            
        stock_data.reset_index(inplace=True)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            stock_data.to_csv(output_path, index=False)
            print(f"✅ Data successfully saved to {output_path}")
            
        return stock_data
        
    except Exception as e:
        print(f"❌ An error occurred while fetching data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    TICKER = "AAPL"
    START_DATE = "2020-01-01"
    END_DATE = "2024-01-01"
    
    OUTPUT_FILE = "../data/raw/stock_data.csv"
    
    df = fetch_stock_data(TICKER, START_DATE, END_DATE, OUTPUT_FILE)
    
    if not df.empty:
        print("\nFirst 5 rows of fetched data:")
        print(df.head())