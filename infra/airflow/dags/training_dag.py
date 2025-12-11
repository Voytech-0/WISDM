from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from datetime import timedelta

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "schedule": "59 23 * * *"
}

dag = DAG("wisdm_training", default_args=default_args)

perhaps_populate_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    image="curlimages/curl:8.4.0",
    command=(
        "sh -c 'set -euo pipefail; code=$(curl -sS -o /dev/null -w \"%{http_code}\" http://db-init:5001/populate || true); "
        "if [ \"$code\" = \"202\" ] || [ \"$code\" = \"409\" ]; then echo \"populate returned $code\"; exit 0; else echo \"Unexpected HTTP code: $code\"; exit 2; fi'"
    ),
    network_mode="infra_default",
    mount_tmp_dir=False,
    task_id="perhaps_populate",
    dag=dag,
)

model_training_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main_training.py train",
    image="wisdm:latest",
    network_mode="infra_default",
    mount_tmp_dir=False,
    environment={
        "MYSQL_HOST": "mysql",
        "MYSQL_PORT": "3306",
        "MYSQL_USER": "user",
        "MYSQL_PASSWORD": "password",
        "MYSQL_DATABASE": "wisdm",
    },
    # Use docker.types.Mount to mount the named volume `app` into the container at /app/models
    mounts=[Mount(target="/app", source="infra_app", type="volume")],
    task_id="training",
    dag=dag,
)

restart_app_task = BashOperator(
    bash_command=(
        "docker compose -f infra/docker-compose.yaml restart app "
    ),
    task_id="restart_app",
    dag=dag,
)

perhaps_populate_task >> model_training_task >> restart_app_task
