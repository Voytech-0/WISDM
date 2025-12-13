
I was aiming for the starts and did not realise I have fallen off a cliff. I wanted to use ZenML instead of Airflow to show some ambition the task has devoured me.

# WISDM Activity Recognition System

This project implements an activity recognition system using the WISDM smartphone and smartwatch dataset, containerized with Docker and orchestrated via Docker Compose.

## 1\. Setup & Installation

### Step 1: Preparation

Before building the containers, you must download the dataset.

1.  Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset).
2.  Extract the contents into the `data/` folder in the root directory.

Run the following commands to build the database initialization image and the main application image.

```bash
# Build the DB Initialization image
docker build -f infra/db_init/Dockerfile . -t db-init:latest

# Build the Main Application image
docker build -f infra/build/Dockerfile . -t wisdm:latest
```

To connect to MySQL, '.env' file is used to store the credentials.
Create a `.env` file in the root directory. An example file is provided for reference.:

```bash
MYSQL_HOST=mysql
MYSQL_PORT=3306
MYSQL_USER=user
MYSQL_PASSWORD=password
MYSQL_ROOT_PASSWORD=rootpassword
MYSQL_DATABASE=wisdm
```

I did not manage to set up ZenML automatically, so you need to set it up manually before launching the services.
Open localhost:8081 and create an admin user with your desired credentials.

Then, log in to ZenML from your terminal:

```bash
zenml login http://localhost:8081
```
Now you will be abe to locally run `python src/main_training.py` and ` python src/main_api.py serve`


zenml login http://localhost:8081

### Step 3: Launch Services

Start the full stack using Docker Compose:

```bash
docker compose -f infra/docker-compose.yaml up -d
```

-----

## 2\. Database Initialization
Before using the main application, you need to populate the MySQL database with the WISDM dataset.
This step is not done automatically, as typically a server with data would be separate from the model serving infrastructure.

To initialize the database, use the DB Initialization Service API:
```bash
curl -X GET http://localhost:5001/populate
```

-----

## 3\. Simulated Usage

> **Important:** Before simulating data insertion, ensure you have initialized the database and triggered the initial model training via the API (see endpoints below).

To simulate a client inserting data, run the fake client script in a separate terminal:

```bash
python scripts/fake_client.py
```

-----

## 4\. Service Overview

Below is a quick reference for the services running in the stack, their internal container ports, and the exposed host ports.

| Service Name | Description | Host Port | Base URL |
| :--- | :--- |:----------| :--- |
| **app** | Main Flask API (Inference/Training) | `8000`    | `http://localhost:8000` |
| **db-init** | DB Population Service | `5001`    | `http://localhost:5001` |
| **grafana** | UI Dashboards | `3000`    | `http://localhost:3000` |
| **prometheus** | Metrics Server | `9090`    | `http://localhost:9090` |
| **push\_gateway**| Prometheus Pushgateway | `9091`    | `http://localhost:9091` |
| **zenml** | ZenML Server | `8081`    | `http://localhost:8081` |
| **mysql** | Database | `3306`    | N/A |

-----

## 4\. API Reference

### Main Application

**File:** `src/main_api.py`

  * **`GET /health`**

      * **Purpose:** Health check. Returns `200` when the model is loaded, `503` otherwise.
      * *Example:* `http://localhost:8000/health`

  * **`POST /reload`**

      * **Purpose:** Forces a reload of the model from ZenML.
      * *Example:* `http://localhost:8000/reload`

  * **`POST /predict`**

      * **Purpose:** Inference endpoint. Accepts raw or pre-expanded features.
      * *Example:* `http://localhost:8000/predict`

  * **`GET /train`**

      * **Purpose:** Triggers the training pipeline (runs in the background).
      * *Example:* `http://localhost:8000/train`

  * **`GET /run_daily`**

      * **Purpose:** Manually triggers the daily maintenance pipeline (runs in the background).
      * *Example:* `http://localhost:8000/run_daily`

  * **`GET /metrics`**

      * **Purpose:** Exposes Prometheus metrics (registered by `prometheus_flask_exporter`).
      * *Example:* `http://localhost:8000/metrics`

### Database Initialization Service

**File:** `db_init/populate_db.py`

  * **`GET /status`**

      * **Purpose:** Checks if the database is already populated.
      * *Example:* `http://localhost:5001/status`

  * **`GET /populate`**

      * **Purpose:** Kicks off population of the DB from local data files (Note: this is a blocking operation).
      * *Example:* `http://localhost:5001/populate`

### Infrastructure & Monitoring Endpoints

  * **Prometheus:** `GET /metrics` @ `http://localhost:9090/metrics`
  * **Pushgateway:** `GET /metrics` @ `http://localhost:9091/metrics`
  * **Grafana:** `GET /api/health` (Used for health checks)
  * **ZenML:** `GET /api/v1/health` @ `http://localhost:8081/api/v1/health`
      * *Note:* The `main_api.ModelLoader` connects to ZenML using the internal container hostname.

-----