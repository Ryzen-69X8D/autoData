"""
run_pipeline.py
================
Runs the complete ML pipeline locally (without Airflow / Docker).
Useful for quick local testing and development.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --ticker INFY.NS --start 2021-01-01 --end 2024-12-31
"""

import argparse
import sys
import os

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest     import fetch_stock_data
from preprocess import preprocess_data
from train      import train_model
from evaluate   import evaluate_model, is_new_model_better
from deploy     import deploy_model


# ── Default paths (local layout, not Docker paths) ────────────────────────────
RAW_DATA_PATH       = "data/raw/stock_data.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
MODEL_NEW_PATH      = "models/lstm_model_new.pt"
MODEL_PROD_PATH     = "models/lstm_model.pt"
SCALER_PATH         = "models/scaler.pkl"
METRICS_NEW_PATH    = "models/metrics_new.json"
METRICS_PROD_PATH   = "models/metrics_deployed.json"


def run(ticker: str, start: str, end: str):
    print("\n" + "="*60)
    print(f"  Stock MLOps Pipeline  |  {ticker}  |  {start} → {end}")
    print("="*60 + "\n")

    # Step 1 ── Ingest
    print("── Step 1: Ingesting data...")
    df = fetch_stock_data(ticker, start, end, RAW_DATA_PATH)
    if df.empty:
        print("❌  Ingestion failed. Aborting.")
        sys.exit(1)

    # Step 2 ── Preprocess
    print("\n── Step 2: Preprocessing...")
    preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH, SCALER_PATH)

    # Step 3 ── Train
    print("\n── Step 3: Training model...")
    train_model(PROCESSED_DATA_PATH, MODEL_NEW_PATH, METRICS_NEW_PATH)

    # Step 4 ── Evaluate
    print("\n── Step 4: Evaluating model...")
    evaluate_model(MODEL_NEW_PATH, PROCESSED_DATA_PATH, METRICS_NEW_PATH)

    # Step 5 ── Deploy (conditional)
    print("\n── Step 5: Deploying model...")
    deployed = deploy_model(
        new_model_path        = MODEL_NEW_PATH,
        deployed_model_path   = MODEL_PROD_PATH,
        new_metrics_path      = METRICS_NEW_PATH,
        deployed_metrics_path = METRICS_PROD_PATH,
    )

    print("\n" + "="*60)
    print(f"  Pipeline complete.  Model {'DEPLOYED ✅' if deployed else 'NOT changed ⛔'}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the stock prediction pipeline locally.")
    parser.add_argument("--ticker", default="NIFTY_50",
                        help="The index or ticker being predicted")
    parser.add_argument("--start",  default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default="2024-12-31", help="End date   YYYY-MM-DD")
    args = parser.parse_args()

    run(args.ticker, args.start, args.end)
