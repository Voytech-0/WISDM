import requests
import logging
from zenml import step, pipeline
from configs.configs import TrainingConfig
from main_training import run_training


@step
def trigger_populate_db() -> bool:
    url = "http://db-init:5001/populate"
    try:
        logging.info(f"Triggering DB population at {url}...")
        response = requests.post(url)
        response.raise_for_status()
        logging.info("DB Population successful.")
        return True
    except Exception as e:
        logging.error(f"Failed to populate DB: {e}")
        raise e


@step
def trigger_model_reload() -> bool:
    url = "http://localhost:8000/reload"
    try:
        logging.info(f"Triggering Model Reload at {url}...")
        response = requests.post(url)
        response.raise_for_status()
        logging.info("Model reload successful.")
        return True
    except Exception as e:
        logging.error(f"Failed to reload model: {e}")
        raise e


# The Full Pipeline
@pipeline
def daily_maintenance_pipeline():
    trigger_populate_db()
    run_training()
    trigger_model_reload()