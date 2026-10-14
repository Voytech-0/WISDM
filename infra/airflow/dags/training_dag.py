import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import timedelta, datetime

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "email_on_failure": False,
    "email_on_retry": False,
}

dag = DAG(
    "wisdm_training",
    default_args=default_args,
    description="Daily training pipeline",
    schedule="59 23 * * *",
    start_date=datetime(2023, 1, 1),
    catchup=False
)

perhaps_populate_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    image="curlimages/curl:8.4.0",
    command=(
        "sh -c 'set -eu; "
        "code=$(curl -sS -o /dev/null -w \"%{http_code}\" http://db-init:5001/populate || true); "
        "if [ \"$code\" = \"200\" ] || [ \"$code\" = \"409\" ]; then "
        "echo \"populate returned $code\"; exit 0; "
        "else echo \"Unexpected HTTP code: $code\"; exit 2; fi'"
    ),
    network_mode="infra_default",
    mount_tmp_dir=False,
    task_id="perhaps_populate",
    dag=dag,
)

model_training_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main_training.py",
    image="wisdm:latest",
    network_mode="infra_default",
    mount_tmp_dir=False,
    environment={
        "MYSQL_HOST": "mysql",
        "MYSQL_PORT": "3306",
        "MYSQL_USER": "user",
        "MYSQL_PASSWORD": "password",
        "MYSQL_DATABASE": "wisdm",
        "MAX_ROWS": os.getenv("MAX_ROWS", ""),  # note: set e.g. MAX_ROWS=10000 in .env for a quick run on a subset
    },
    mounts=[Mount(target="/app/models", source="infra_models", type="volume")],
    task_id="training",
    dag=dag,
)

trigger_reload_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    image="curlimages/curl:8.4.0",
    command="curl -X POST -sS --fail http://app:8000/reload",
    network_mode="infra_default",
    mount_tmp_dir=False,
    task_id="trigger_model_reload",
    dag=dag,
)

perhaps_populate_task >> model_training_task >> trigger_reload_task