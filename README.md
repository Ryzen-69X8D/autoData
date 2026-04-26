# 🚀 Automated Stock Prediction System (MLOps Pipeline)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Airflow](https://img.shields.io/badge/Airflow-Orchestration-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> ⚡ End-to-end MLOps pipeline that automates stock prediction with scheduled retraining, evaluation, and deployment.

---

## 📸 Demo

### 🔹 Airflow DAG
![Airflow UI](assets/airflow_ui.png)

### 🔹 API Response
![API Response](assets/api_response.png)

---

## 🧠 Features

- 📥 Automated data ingestion from stock APIs  
- 🧹 Data preprocessing & feature engineering  
- 🤖 ML model training (RandomForest / XGBoost)  
- 📊 Model evaluation (RMSE-based comparison)  
- 🔁 Scheduled retraining with Airflow  
- 🚀 Conditional model deployment  
- 🌐 FastAPI-based prediction service  
- 🐳 Fully Dockerized setup  

---

## 🏗️ Architecture

Data Source → Preprocessing → Training → Evaluation → Deployment → API

🔁 Orchestrated using Apache Airflow

---

## 🧰 Tech Stack

- Python  
- FastAPI  
- Docker  
- Apache Airflow  
- scikit-learn  
- yfinance  

---

## 📂 Project Structure

stock-mlops-pipeline/
│
├── airflow/        # DAGs
├── app/            # FastAPI app
├── src/            # ML pipeline
├── pipeline/       # Orchestration logic
├── scripts/        # Runner scripts
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md

---

## ▶️ Getting Started

### 1️⃣ Clone the repo

git clone https://github.com/your-username/stock-mlops-pipeline.git  
cd stock-mlops-pipeline  

---

### 2️⃣ Run with Docker

docker-compose up --build  

---

### 3️⃣ Open Airflow UI

http://localhost:8080  

Login:  
username: admin  
password: admin  

---

## 📡 API Usage

Endpoint:  
POST /predict  

Example Request:

{
  "Open": 150,
  "High": 155,
  "Low": 148,
  "Volume": 1000000
}

Example Response:

{
  "prediction": 152.34
}

---

## 🔁 Pipeline Workflow

1. Fetch stock data  
2. Preprocess dataset  
3. Train ML model  
4. Evaluate performance  
5. Compare with previous model  
6. Deploy if improved  
7. Serve predictions via API  

---

## 📈 Future Improvements

- Add MLflow for experiment tracking  
- Build Streamlit dashboard  
- Real-time prediction system  
- Model drift detection  
- Cloud deployment (AWS/GCP)  

---

## 👨‍💻 Author

Akash Kundu

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
