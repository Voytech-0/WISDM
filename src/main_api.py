import logging
from http import HTTPStatus
from time import perf_counter
from typing import Any, List, Optional, Tuple, Dict
import threading

import click
import numpy as np
import torch
from flask import request, jsonify, Flask, g
from torch import nn
import polars as pl

# ZenML Imports
from zenml.client import Client
from zenml.exceptions import DoesNotExistException

# App Imports
from main_training import LittleMLP  # Class def needed for unpickling
from preprocessing.preprocess import expand_raw_items_to_expanded_df
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit

# Logging Setup
logging.basicConfig(level=logging.INFO)

# Metrics
request_counter = Counter(
    'api_requests_total', 'Total API requests', ['endpoint', 'method', 'status']
)
predict_counter = Counter('predict_requests_total', 'Total /predict requests', ['status'])
predicted_class_counter = Counter(
    'predicted_class_total', 'Total predicted classes', ['predicted_class']
)
request_latency_histogram = Histogram(
    'request_latency_seconds', 'Request latency histogram in seconds', ['endpoint', 'method', 'status']
)


class ModelLoader:
    """
    Fetches the latest model and statistics from the ZenML Artifact Store.
    """

    def __init__(self, pipeline_name: str = "training_pipeline", step_name: str = "run_training"):
        self.pipeline_name = pipeline_name
        self.step_name = step_name
        self._lock = threading.RLock()

        # Cache to hold the loaded artifacts in memory
        self._cached_model: Optional[nn.Module] = None
        self._cached_stats: Optional[List[Dict]] = None

    def load(self, force_reload: bool = False) -> Tuple[nn.Module, List[Dict]]:
        """
        Connects to ZenML, finds the last successful run, and loads the artifacts.
        Returns: (model, z_score_stats)
        """
        with self._lock:
            if self._cached_model is not None and self._cached_stats is not None and not force_reload:
                return self._cached_model, self._cached_stats

            logging.info(f"Connecting to ZenML Client to fetch pipeline: {self.pipeline_name}...")
            client = Client()

            try:
                pipeline = client.get_pipeline(self.pipeline_name)
                last_run = pipeline.last_successful_run

                if not last_run:
                    raise RuntimeError(f"No successful runs found for pipeline '{self.pipeline_name}'")

                logging.info(f"Loading artifacts from Run ID: {last_run.id} (Step: {self.step_name})")

                step_outputs = last_run.steps[self.step_name].outputs

                if "trained_model" not in step_outputs:
                    raise KeyError("Artifact 'trained_model' not found in step outputs.")
                self._cached_model = step_outputs["trained_model"].load()
                self._cached_model.eval()

                if "z_score_stats" in step_outputs:
                    self._cached_stats = step_outputs["z_score_stats"].load()
                else:
                    logging.warning("Artifact 'z_score_stats' not found. Defaulting to empty list.")
                    self._cached_stats = []

                logging.info("Successfully loaded model and stats from ZenML.")
                return self._cached_model, self._cached_stats

            except DoesNotExistException:
                logging.error(f"Pipeline '{self.pipeline_name}' not found in ZenML store.")
                raise
            except Exception as e:
                logging.error(f"Error loading artifacts from ZenML: {e}")
                raise


class InputPreparer:
    """
    Prepare input tensors. 
    Stores z_score_stats to apply normalization if needed.
    """

    def __init__(self, z_score_stats: List[Dict] = None):
        self.z_score_stats = z_score_stats or []

    def set_stats(self, z_score_stats: List[Dict]):
        self.z_score_stats = z_score_stats or []

    @staticmethod
    def _is_raw(obj: dict) -> bool:
        return (
                isinstance(obj, dict)
                and 'x' in obj and 'y' in obj and 'z' in obj
                and hasattr(obj['x'], '__iter__') and hasattr(obj['y'], '__iter__') and hasattr(obj['z'], '__iter__')
        )

    def prepare(self, data_json: Any, model: Optional[torch.nn.Module] = None) -> torch.Tensor:
        items: List[dict] = data_json if isinstance(data_json, list) else [data_json]
        if not items:
            raise ValueError("Empty input payload")

        if all(self._is_raw(it) for it in items):
            df_expanded = expand_raw_items_to_expanded_df(items, z_score=self.z_score_stats)
        else:
            # Fallback for pre-expanded data
            if not all(isinstance(it, dict) for it in items):
                raise ValueError("Each input must be an object with feature keys or x/y/z lists")
            df_expanded = pl.DataFrame(items)

        all_cols = list(df_expanded.columns)
        label_cols = [c for c in all_cols if c.startswith('label')]
        feature_cols = [c for c in all_cols if c not in label_cols]

        if not feature_cols:
            raise ValueError("No feature columns found after preprocessing")

        X_np = df_expanded.select(feature_cols).to_numpy()

        expected = None
        if model is not None:
            if hasattr(model, 'input_size'):
                expected = int(model.input_size)
            elif hasattr(model, 'hparams') and isinstance(model.hparams, dict):
                expected = int(model.hparams.get('input_size', 0))

        if expected is not None and 0 < expected != X_np.shape[1]:
            raise ValueError(
                f"Input has {X_np.shape[1]} features but model expects {expected}."
            )

        return torch.from_numpy(X_np).float()


def create_app(model_loader: ModelLoader, preparer: InputPreparer) -> Flask:
    app = Flask(__name__)
    metrics = PrometheusMetrics(app)

    app_context = {
        "model": None
    }

    try:
        model, stats = model_loader.load()
        app_context["model"] = model
        preparer.set_stats(stats)
        logging.info("Initial model load successful.")
    except Exception as e:
        logging.warning(f"Failed to load model during startup: {e}")

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
            endpoint = request.endpoint or request.path
            status = str(response.status_code)
            request_latency_histogram.labels(endpoint, request.method, status).observe(elapsed)
            request_counter.labels(endpoint, request.method, status).inc()
        return response

    def scheduled_job():
        """Function to be run by the scheduler."""
        logging.info("Scheduler waking up.")
        try:
            # Run the ZenML pipeline directly
            daily_maintenance_pipeline()
            logging.info("Daily Pipeline finished successfully.")
        except Exception as e:
            logging.error(f"Daily Pipeline failed: {e}")

    scheduler = BackgroundScheduler()

    scheduler.add_job(
        func=scheduled_job,
        trigger=CronTrigger(hour=0, minute=0),
        id='daily_pipeline_job',
        name='Daily ZenML Pipeline',
        replace_existing=True
    )

    scheduler.start()
    logging.info("Scheduler started: Daily pipeline set for 00:00.")

    atexit.register(lambda: scheduler.shutdown())

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
            logging.exception("Failed to reload model")
            return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

    @app.route("/predict", methods=["POST"])
    def predict():
        model = app_context["model"]
        if model is None:
            try:
                model, stats = model_loader.load()
                app_context["model"] = model
                preparer.set_stats(stats)
            except Exception as e:
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
            logging.error("Error during /predict: %s", e, exc_info=True)
            predict_counter.labels(status='error').inc()
            return jsonify({"error": str(e)}), 400

    @app.route("/train")
    def trigger_training():
        """
        Triggers the ZenML training pipeline in a background thread.
        """
        def run_pipeline_task():
            try:
                logging.info("Starting training pipeline via API trigger...")
                training_pipeline()
                logging.info("Training pipeline finished.")
            except Exception as e:
                logging.error(f"Training pipeline failed: {e}")

        thread = threading.Thread(target=run_pipeline_task)
        thread.start()

        return jsonify({
            "status": "Training started",
            "message": "Check container logs or Aim dashboard for progress."
        }), HTTPStatus.ACCEPTED

    return app


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind the Flask server")
@click.option("--port", default=8000, type=int)
@click.option("--pipeline", default="training_pipeline", help="ZenML pipeline name to fetch model from")
def serve(host, port, pipeline):
    """Start the inference server fetching artifacts from ZenML."""
    loader = ModelLoader(pipeline_name=pipeline)
    preparer = InputPreparer()
    app = create_app(loader, preparer)
    logging.info(f"Starting ZenML-backed inference server on http://{host}:{port}")
    app.run(host=host, port=port)


cli = click.Group()
cli.add_command(serve)

if __name__ == "__main__":
    cli()