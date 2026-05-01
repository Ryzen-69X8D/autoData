"""
stock_prediction_pipeline DAG
==============================
Full end-to-end retraining pipeline:
  ingest → preprocess → train → evaluate → deploy

Runs daily. On first run, deploys unconditionally.
On subsequent runs, only promotes the model if RMSE improves.
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import json
import os


# ── DAG default arguments ─────────────────────────────────────────────────────
default_args = {
    "owner":             "airflow",
    "depends_on_past":   False,
    "email_on_failure":  False,
    "email_on_retry":    False,
    "retries":           1,
    "retry_delay":       timedelta(minutes=5),
}

dag = DAG(
    "stock_prediction_pipeline",
    default_args     = default_args,
    description      = "Automated ML pipeline for Indian stock prediction",
    schedule_interval= timedelta(days=1),
    start_date       = datetime(2026, 4, 26),
    catchup          = False,
    tags             = ["mlops", "stocks", "india"],
)

# ── Task 1: Ingest raw data from Yahoo Finance ────────────────────────────────
ingest_task = BashOperator(
    task_id      = "ingest_data",
    bash_command = "cd /opt/airflow && python src/ingest.py",
    dag          = dag,
)

# ── Task 2: Preprocess + feature engineering + scale ─────────────────────────
preprocess_task = BashOperator(
    task_id      = "preprocess_data",
    bash_command = "cd /opt/airflow && python src/preprocess.py",
    dag          = dag,
)

# ── Task 3: Train model ───────────────────────────────────────────────────────
train_task = BashOperator(
    task_id      = "train_model",
    bash_command = "cd /opt/airflow && python src/train.py",
    dag          = dag,
)

# ── Task 4: Evaluate model ────────────────────────────────────────────────────
evaluate_task = BashOperator(
    task_id      = "evaluate_model",
    bash_command = "cd /opt/airflow && python src/evaluate.py",
    dag          = dag,
)

# ── Task 5: Conditional deploy ────────────────────────────────────────────────
def _should_deploy(**context):
    """Branch: returns 'deploy_model' if new model is better, else 'skip_deploy'."""
    new_metrics_path      = "/opt/airflow/models/metrics.json"
    deployed_metrics_path = "/opt/airflow/models/metrics_deployed.json"

    if not os.path.exists(deployed_metrics_path):
        return "deploy_model"

    with open(new_metrics_path)      as f: new  = json.load(f)
    with open(deployed_metrics_path) as f: dep  = json.load(f)

    return "deploy_model" if new.get("rmse", 1e9) < dep.get("rmse", 1e9) \
           else "skip_deploy"


branch_task = BranchPythonOperator(
    task_id         = "check_model_performance",
    python_callable = _should_deploy,
    dag             = dag,
)

deploy_task = BashOperator(
    task_id      = "deploy_model",
    bash_command = "cd /opt/airflow && python src/deploy.py",
    dag          = dag,
)

skip_task = EmptyOperator(
    task_id = "skip_deploy",
    dag     = dag,
)

# ── DAG task chain ────────────────────────────────────────────────────────────
ingest_task >> preprocess_task >> train_task >> evaluate_task >> branch_task
branch_task >> [deploy_task, skip_task]
