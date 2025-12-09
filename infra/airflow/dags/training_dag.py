from airflow import DAG
from docker.types import Mount
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator

from datetime import timedelta


default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG("nova_training", default_args=default_args)

perhaps_populate_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    image="curlimages/curl:8.4.0",
    command=(
        "sh -c 'curl --fail --silent --show-error -X POST http://init:5001/populate'"
    ),
    network_mode="wisdm_default",
    task_id="perhaps_populate",
    dag=dag,
)

model_training_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main_training.py train",
    image="wisdm:latest",
    network_mode="wisdm_default",
    task_id="training",
    dag=dag,
)
