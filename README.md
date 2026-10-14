# WISDM Activity Recognition

Activity recognition on the [WISDM smartphone and smartwatch dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset).
A small MLP is trained daily by an **Airflow** DAG, and a **Flask** API serves predictions and picks up each newly trained model without a restart.
Everything runs locally with Docker Compose, with Prometheus and Grafana for monitoring.

## Architecture

```
                 ┌──────────────── Airflow DAG `wisdm_training` (daily 23:59, or via POST /train) ──────────────┐
                 │                                                                                              │
                 │  perhaps_populate ───────▶ training ─────────────────────────▶ trigger_model_reload          │
                 │  GET db-init:5001/populate  wisdm:latest container:             POST app:8000/reload         │
                 │  (idempotent)               reads MySQL, writes `models` volume                              │
                 └──────────────────────────────────────────────────────────────────────────────────────────────┘

 data/wisdm-dataset ──▶ db-init ──▶ MySQL ──▶ training container ──▶ models volume ──▶ app (Flask) ──▶ /predict
                                                                                            │
                                                                       Prometheus ◀── /metrics ──▶ Grafana
```

| Service | Role | Host port |
| :--- | :--- | :--- |
| **app** | Flask API: inference, model reload, training trigger, `/metrics` | `8000` |
| **airflow** | Orchestration (`airflow standalone`, UI + API) | `4242` |
| **db-init** | Flask service that loads the raw dataset files into MySQL | `5001` |
| **mysql** | Raw sensor data (`wisdm_raw` table) | `3306` |
| **prometheus** | Metrics store | `9090` |
| **push_gateway** | Prometheus Pushgateway | `9091` |
| **grafana** | Dashboards (anonymous admin, local use only) | `3000` |

The training task runs the `wisdm:latest` image as a sibling container, started by Airflow through the Docker socket.
It shares the named volume `models` with the `app` service: training writes `models/mlp.ckpt` and `models/processed/z_score_params.json`, and `/reload` reads them.

## Requirements

- Docker with Compose v2 (Docker Desktop is fine)
- The dataset on disk (about 300 MB zipped, 31M raw rows)
- Python 3.11 and [uv](https://docs.astral.sh/uv/) only if you want to run the tests locally

## Setup

### 1. Get the dataset

Download the zip from the UCI page above and extract it so the raw files end up under `data/wisdm-dataset/raw/` (that folder holds `phone/` and `watch/`).
The archive contains a nested `wisdm-dataset.zip`, so extract twice.

```bash
mkdir -p data && cd data
unzip wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip
unzip wisdm-dataset.zip
cd ..
```

`data/` is git-ignored.

### 2. Create `.env`

The repo does not ship an example file. Create `.env` in the repository root:

```bash
MYSQL_HOST=mysql
MYSQL_PORT=3306
MYSQL_USER=user
MYSQL_PASSWORD=password
MYSQL_ROOT_PASSWORD=rootpassword
MYSQL_DATABASE=wisdm

# Optional: cap the rows used for training (see "Dataset size and memory")
MAX_ROWS=10000
```

> The DAG's training task currently hardcodes `user` / `password` / `wisdm` for MySQL in `infra/airflow/dags/training_dag.py`.
> If you change the credentials above, change them there too.

### 3. Start the stack

```bash
docker compose -f infra/docker-compose.yaml up -d --build
```

This builds `wisdm:latest` (from `infra/build/Dockerfile`) and `db-init:latest`.
The first start takes a few minutes. `GET /health` on the app returns `503` until the first model has been trained, which is expected.

## Training

The DAG `wisdm_training` is unpaused at creation and runs daily at 23:59.
To run it right away, either call the app:

```bash
curl -X POST http://localhost:8000/train
```

or trigger `wisdm_training` in the Airflow UI at <http://localhost:4242> (authentication is disabled, local use only).

The first run populates MySQL from the dataset files, which takes a while. Later runs skip this step (`/populate` answers `already_populated`).
When the run finishes, the app has the new model loaded:

```bash
curl http://localhost:8000/health        # OK
```

### Dataset size and memory

Training currently loads the whole `wisdm_raw` table into memory.
With all 31M rows this needs roughly **18 GiB** (measured on a 3M-row sample, scaling linearly), which does not fit Docker Desktop's default VM or a 16 GB laptop.
The training container is then killed with exit code `137`.

Until training is made incremental, cap the data with `MAX_ROWS` in `.env`. The DAG passes it to the training task, which adds a `LIMIT` to its MySQL query.
`MAX_ROWS=10000` is enough to check that the pipeline works end to end, but it yields only a few dozen windows from a single subject, so it is not a useful model.
Leave `MAX_ROWS` unset to use all rows on a machine with enough memory.

### Training settings

Read from environment variables on the training task (defaults in `configs/configs.py`):
`SEED`, `MAX_EPOCHS`, `BATCH_SIZE`, `LR`, `HIDDEN_SIZE`, `NUM_LAYERS`, `NUM_CLASSES`, `MODEL_PATH`, `STATS_PATH`, `MAX_ROWS`.

## API

### Main application (`src/main_api.py`, port 8000)

| Endpoint | Description |
| :--- | :--- |
| `GET /health` | `200 OK` when a model is loaded, `503` otherwise |
| `POST /predict` | Classify one window or a list of windows (see below) |
| `POST /reload` | Reload the model and z-score stats from the `models` volume (Airflow calls this after training) |
| `POST /train`, `GET /train` | Trigger the `wisdm_training` DAG via the Airflow API; `202` on success, `502` if Airflow rejects the request |
| `GET /metrics` | Prometheus metrics |

`/predict` takes one JSON object or a list of them. Each has `device` (`phone`/`watch`), `sensor` (`accel`/`gyro`) and equal-length `x`, `y`, `z` series (a 10 second window is 200 samples at 20 Hz):

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "device": "phone", "sensor": "accel",
  "x": [0.1, 0.2, ...], "y": [...], "z": [...]
}'
# [{"pred": 3, "probs": [0.01, 0.02, ...]}]
```

`pred` is the class index and `probs` are the softmax probabilities.

### Database initialization service (`db_init/populate_db.py`, port 5001)

| Endpoint | Description |
| :--- | :--- |
| `GET /status` | Whether the database has been populated |
| `GET /populate` | Load the dataset files into MySQL (blocking). `200` when done or already populated, `503` if MySQL is unavailable, `500` on failure |

### Simulated client

`scripts/fake_client.py` posts synthetic windows to `/predict` at random intervals so the dashboards have traffic:

```bash
python scripts/fake_client.py                     # synthetic series
python scripts/fake_client.py --use-local         # real windows from data/ (needs the dataset)
python scripts/fake_client.py --count 20 --verbose
```

Run `python scripts/fake_client.py --help` for all options.

## Monitoring

- **Grafana:** <http://localhost:3000> with the dashboards `wisdm-api-dashboard` and `observability-dashboard` provisioned from `infra/telemetry/`.
- **Prometheus:** <http://localhost:9090>, scraping the app every 2 seconds.
- `infra/telemetry/alerts.yml` is loaded by Prometheus but has no rules yet.

## Tests

```bash
uv venv --python 3.11 .venv
uv pip install -r requirements.txt pytest
.venv/bin/python -m pytest -q
```

The tests do not need Docker, MySQL or the dataset.

## Project layout

```
configs/            dataclass configs (paths, data, MySQL, training), read from .env / environment
db_init/            Flask service + loader that fills MySQL from the raw files
infra/
  airflow/dags/     training_dag.py (the wisdm_training DAG)
  build/            Dockerfile for the wisdm image (API + training)
  db_init/          Dockerfile for the db-init image
  telemetry/        Prometheus config, alert rules, Grafana provisioning and dashboards
  docker-compose.yaml
scripts/            fake_client.py
src/
  main_api.py       Flask API, ModelLoader, InputPreparer
  main_training.py  dataset, MLP, training loop
  preprocessing/    windowing, z-score normalization, feature expansion
tests/
```

## Troubleshooting

- **Training task fails with `StatusCode: 137`:** out of memory, see [Dataset size and memory](#dataset-size-and-memory).
- **`/health` stays `503`:** no model has been trained yet. Trigger a run and check the `training` task log in the Airflow UI. If training succeeded but the app still has no model, call `POST /reload`.
- **`docker compose` fails with `docker-credential-desktop` not found:** Docker Desktop's CLI directory is not on your `PATH`. On macOS add `/Applications/Docker.app/Contents/Resources/bin`.
- **Prometheus exits right after start:** `infra/telemetry/alerts.yml` must be a file. If Docker created a directory with that name, delete it and restore the file from git.
- **Start from scratch:** `docker compose -f infra/docker-compose.yaml down -v` also removes the MySQL data and trained models.

## Notes

- Airflow runs with authentication disabled (`SIMPLE_AUTH_MANAGER_ALL_ADMINS`) and has access to the Docker socket, and Grafana allows anonymous admin access. This setup is for local development only.
- The DAG is mounted from `infra/airflow/dags/` for convenience. In a real deployment, ship DAGs through CI or a bucket the scheduler reads from.
