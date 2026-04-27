import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model(input_path: str, model_output_path: str):
    df = pd.read_csv(input_path)
    
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    INPUT_FILE = "/opt/airflow/data/processed/processed_data.csv"
    MODEL_OUTPUT = "/opt/airflow/models/random_forest.pkl"
    train_model(INPUT_FILE, MODEL_OUTPUT)