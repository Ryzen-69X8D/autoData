import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    
    df.dropna(inplace=True)
    
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    
    df.dropna(inplace=True)
    
    features_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'Daily_Return']
    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    INPUT_FILE = "/opt/airflow/data/raw/stock_data.csv"
    OUTPUT_FILE = "/opt/airflow/data/processed/processed_data.csv"
    preprocess_data(INPUT_FILE, OUTPUT_FILE)