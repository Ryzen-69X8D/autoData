from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_prediction_pipeline',
    default_args=default_args,
    description='Automated ML pipeline for stock prediction',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2026, 4, 26),
    catchup=False,
)

ingest_task = BashOperator(
    task_id='ingest_data',
    bash_command='python /opt/airflow/src/ingest.py',
    dag=dag,
)

preprocess_task = BashOperator(
    task_id='preprocess_data',
    bash_command='python /opt/airflow/src/preprocess.py',
    dag=dag,
)

train_task = BashOperator(
    task_id='train_model',
    bash_command='python /opt/airflow/src/train.py',
    dag=dag,
)

ingest_task >> preprocess_task >> train_task