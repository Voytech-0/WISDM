import logging
import os
import json
import threading
from http import HTTPStatus
from time import perf_counter
from typing import Any, List, Optional, Tuple, Dict

import click
import numpy as np
import torch
from flask import request, jsonify, Flask, g
from torch import nn
import requests

from configs.configs import TrainingConfig
from main_training import LittleMLP
from preprocessing.preprocess import expand_raw_items_to_expanded_df
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram

logging.basicConfig(level=logging.INFO)

AIRFLOW_URL = os.getenv('AIRFLOW_URL', 'http://airflow:8080')
AIRFLOW_DAG_ID = 'wisdm_training'




class _MetricsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return '/metrics' not in record.getMessage()


logging.getLogger('werkzeug').addFilter(_MetricsFilter())

request_counter = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])
predict_counter = Counter('predict_requests_total', 'Total /predict requests', ['status'])
predicted_class_counter = Counter('predicted_class_total', 'Total predicted classes', ['predicted_class'])
request_latency_histogram = Histogram('request_latency_seconds', 'Request latency', ['endpoint', 'method', 'status'])


class ModelLoader:
    
    def __init__(self):
        self._lock = threading.RLock()
        self._cached_model: Optional[nn.Module] = None
        self._cached_stats: Optional[List[Dict]] = None
        self._last_loaded_time = 0

    def load(self, force_reload: bool = False) -> Tuple[nn.Module, List[Dict]]:
        with self._lock:
            if not os.path.exists(TrainingConfig.model_path):
                raise FileNotFoundError(f"Model file not found at {TrainingConfig.model_path}. Run the training DAG first.")

            if self._cached_model is not None and not force_reload:
                return self._cached_model, self._cached_stats or []

            logging.info(f"Loading model from disk: {TrainingConfig.model_path}")

            try:
                self._cached_model = torch.load(TrainingConfig.model_path, map_location=torch.device('cpu'), weights_only=False)
                self._cached_model.eval()
            except Exception as e:
                logging.error(f"Failed to load model file: {e}")
                raise

            if os.path.exists(TrainingConfig.stats_path):
                try:
                    with open(TrainingConfig.stats_path, 'r') as f:
                        self._cached_stats = json.load(f)
                except Exception as e:
                    logging.warning(f"Failed to load stats file: {e}. Defaulting to empty.")
                    self._cached_stats = []
            else:
                logging.warning("Stats file not found. Defaulting to empty.")
                self._cached_stats = []

            logging.info("Model and stats loaded successfully.")
            return self._cached_model, self._cached_stats


class InputPreparer:
    def __init__(self, z_score_stats: List[Dict] = None):
        self._set_stats_internal(z_score_stats)

    def set_stats(self, z_score_stats: List[Dict]):
        self._set_stats_internal(z_score_stats)

    def _set_stats_internal(self, stats_list: List[Dict]):
        stats_list = stats_list or []
        self.z_score_dict = {}
        for item in stats_list:
            if 'device' in item and 'sensor' in item:
                key = (item['device'], item['sensor'])
                stats_vals = {k: v for k, v in item.items() if k not in ['device', 'sensor']}
                self.z_score_dict[key] = stats_vals

    def prepare(self, data_json: Any, model: Optional[torch.nn.Module] = None) -> torch.Tensor:
        items: List[dict] = data_json if isinstance(data_json, list) else [data_json]
        if not items: raise ValueError("Empty input payload")

        df_expanded = expand_raw_items_to_expanded_df(items, z_score=self.z_score_dict)

        all_cols = list(df_expanded.columns)
        label_cols = [c for c in all_cols if c.startswith('label')]
        feature_cols = [c for c in all_cols if c not in label_cols]
        X_np = df_expanded.select(feature_cols).to_numpy()
        return torch.from_numpy(X_np).float()


def create_app(model_loader: ModelLoader, preparer: InputPreparer) -> Flask:
    app = Flask(__name__)
    metrics = PrometheusMetrics(app)
    app_context = {"model": None}

    try:
        model, stats = model_loader.load()
        app_context["model"] = model
        preparer.set_stats(stats)
    except Exception as e:
        logging.warning(f"Startup load failed: {e}")

    @app.before_request
    def before_request_timing():
        if request.path == "/metrics": return
        g.request_start_time = perf_counter()

    @app.after_request
    def after_request_timing(response):
        if request.path == "/metrics": return response
        start = getattr(g, 'request_start_time', None)
        if start:
            elapsed = perf_counter() - start
            request_latency_histogram.labels(request.endpoint, request.method, str(response.status_code)).observe(
                elapsed)
            request_counter.labels(request.endpoint, request.method, str(response.status_code)).inc()
        return response

    @app.route("/health", methods=["GET"])
    @metrics.do_not_track()
    def health():
        if app_context["model"] is None:
            return "Model not loaded", HTTPStatus.SERVICE_UNAVAILABLE
        return "OK", HTTPStatus.OK

    @app.route("/reload", methods=["POST"])
    def reload_model():
        try:
            model, stats = model_loader.load(force_reload=True)
            app_context["model"] = model
            preparer.set_stats(stats)
            return jsonify({"status": "reloaded"}), HTTPStatus.OK
        except Exception as e:
            return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

    @app.route("/predict", methods=["POST"])
    def predict():
        model = app_context["model"]
        if model is None:
            try:
                model, stats = model_loader.load()
                app_context["model"] = model
                preparer.set_stats(stats)
            except Exception:
                return jsonify({"error": "Model not available"}), HTTPStatus.INTERNAL_SERVER_ERROR

        try:
            payload = request.get_json(force=True)
            x_tensor = preparer.prepare(payload, model=model)
            with torch.no_grad():
                logits = model(x_tensor)
                probs = nn.functional.softmax(logits, dim=1).cpu().numpy().tolist()
                preds = np.argmax(probs, axis=1).tolist()
            out = [{"pred": int(p), "probs": pr} for p, pr in zip(preds, probs)]

            predict_counter.labels(status='success').inc()
            for p in preds:
                predicted_class_counter.labels(predicted_class=str(int(p))).inc()
            return jsonify(out)
        except Exception as e:
            logging.error(f"Predict error: {e}")
            predict_counter.labels(status='error').inc()
            return jsonify({"error": str(e)}), 400

    @app.route("/train", methods=["POST", "GET"])
    def trigger_training():
        try:
            token = requests.get(f"{AIRFLOW_URL}/auth/token", timeout=10).json()["access_token"]
            response = requests.post(
                f"{AIRFLOW_URL}/api/v2/dags/{AIRFLOW_DAG_ID}/dagRuns",
                json={"logical_date": None},
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            if response.ok:
                return jsonify({"status": "Training triggered via Airflow",
                                "airflow_response": response.json()}), HTTPStatus.ACCEPTED
            return jsonify({"error": "Failed to trigger Airflow", "details": response.text}), HTTPStatus.BAD_GATEWAY
        except Exception as e:
            logging.error(f"Failed to reach Airflow: {e}")
            return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

    return app


@click.command()
@click.option("--host", default="0.0.0.0", help="Host")
@click.option("--port", default=8000, type=int)
def serve(host, port):
    loader = ModelLoader()
    preparer = InputPreparer()
    app = create_app(loader, preparer)
    app.run(host=host, port=port)


if __name__ == "__main__":
    serve()