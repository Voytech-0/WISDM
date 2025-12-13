import requests
import logging
from typing import Any
from zenml import step, pipeline
from main_training import run_training


@step(enable_cache=False)
def trigger_populate_db() -> bool:
    url = "http://db-init:5001/populate"
    try:
        logging.info(f"Triggering DB population at {url}...")
        response = requests.get(url)
        response.raise_for_status()
        logging.info("DB Population successful.")
        return True
    except Exception as e:
        logging.error(f"Failed to populate DB: {e}")
        raise e


@step(enable_cache=False)
def trigger_model_reload(_dep: Any = None) -> bool:
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


@pipeline
def daily_maintenance_pipeline():
    logging.info("Starting Daily Maintenance Pipeline in daily_pipeline.py")
    db_ok = trigger_populate_db()
    model, stats = run_training(db_ok)
    trigger_model_reload(model)
