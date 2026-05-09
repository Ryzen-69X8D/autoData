import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest     import fetch_stock_data
from preprocess import preprocess_data
from train      import train_ensemble
from evaluate   import evaluate_model
from deploy     import deploy_model

RAW_DATA_PATH       = "data/raw/stock_data.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
MODEL_NEW_DIR       = "models/new"
MODEL_PROD_DIR      = "models/prod"
SCALER_PATH         = "models/scaler.pkl"
METRICS_NEW_PATH    = "models/new/metrics_new.json"
METRICS_PROD_PATH   = "models/prod/metrics_deployed.json"

def run(ticker: str, start: str, end: str):
    print("\n" + "="*60)
    print(f"  Stock MLOps Pipeline  |  {ticker}  |  {start} → {end}")
    print("="*60 + "\n")

    print("── Step 1: Ingesting data...")
    df = fetch_stock_data(ticker, start, end, RAW_DATA_PATH)
    if df.empty:
        print("❌  Ingestion failed. Aborting.")
        sys.exit(1)

    print("\n── Step 2: Preprocessing...")
    preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH, SCALER_PATH)

    print("\n── Step 3: Training Ensemble...")
    train_ensemble(PROCESSED_DATA_PATH, MODEL_NEW_DIR, METRICS_NEW_PATH)

    print("\n── Step 4: Evaluating LSTM Model...")
    evaluate_model(os.path.join(MODEL_NEW_DIR, "lstm_model.pt"), PROCESSED_DATA_PATH, METRICS_NEW_PATH)

    print("\n── Step 5: Deploying Ensemble...")
    deployed = deploy_model(
        new_model_dir         = MODEL_NEW_DIR,
        deployed_model_dir    = MODEL_PROD_DIR,
        new_metrics_path      = METRICS_NEW_PATH,
        deployed_metrics_path = METRICS_PROD_PATH,
    )

    print("\n" + "="*60)
    print(f"  Pipeline complete.  Model {'DEPLOYED ✅' if deployed else 'NOT changed ⛔'}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the stock prediction pipeline locally.")
    parser.add_argument("--ticker", default="NIFTY_50", help="The index or ticker being predicted")
    parser.add_argument("--start",  default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default="2024-12-31", help="End date   YYYY-MM-DD")
    args = parser.parse_args()

    run(args.ticker, args.start, args.end)