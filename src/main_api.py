import logging
from http import HTTPStatus
from pathlib import Path
from time import perf_counter
from typing import Any, List, Optional
from functools import cache

import click
import numpy as np
import torch
from flask import request, jsonify, Flask, g
from torch import nn
import polars as pl

from configs.configs import TrainingConfig
from main_training import LittleMLP
import threading

from preprocessing.preprocess import expand_raw_items_to_expanded_df
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram

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
    """Encapsulates model checkpoint loading.

    Usage:
        loader = ModelLoader(path_or_none)
        model = loader.load()
    """

    def __init__(self, checkpoint_path: Optional[str] = None, model_cls=LittleMLP):
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else Path(TrainingConfig.model_path)
        )
        self.model_cls = model_cls
        # synchronization for cache operations
        self._lock = threading.RLock()

    @staticmethod
    @cache
    def _load_cached(checkpoint_path_str: str, model_cls):

        # Load checkpoint file
        ckpt = torch.load(checkpoint_path_str, map_location='cpu')

        hparams = ckpt['hparams']
        state_dict = ckpt['model_state_dict']

        hparams = dict(hparams)
        # Validate hparams contains required constructor args for LittleMLP
        required_keys = {'input_size', 'hidden_size', 'num_layers', 'num_classes', 'lr'}
        if not required_keys.issubset(set(hparams.keys())):
            missing = required_keys - set(hparams.keys())
            raise RuntimeError(f"Checkpoint 'hparams' missing required keys: {sorted(missing)}")

        model = model_cls(
            input_size=hparams['input_size'],
            hidden_size=hparams['hidden_size'],
            num_layers=hparams['num_layers'],
            num_classes=hparams['num_classes'],
            lr=hparams.get('lr', TrainingConfig.lr)
        )

        model.load_state_dict(state_dict)
        model.hparams = hparams
        model.eval()
        return model


    def load(self, force_reload: bool = False) -> torch.nn.Module:
        """Load the model from checkpoint and return it in eval mode.

        Uses a module-level functools.cache to memoize the loaded model.
        Passing `force_reload=True` clears the cache before loading.
        """
        with self._lock:
            if force_reload:
                # clear the cached loader to force a fresh load
                try:
                    # class-scoped cached function
                    self._load_cached.cache_clear()
                except Exception:
                    pass

            # ensure the checkpoint is present (raises FileNotFoundError if missing)
            model = self._load_cached(str(self.checkpoint_path), self.model_cls)
            return model


class InputPreparer:
    """Prepare input tensors for the model from JSON payloads or expanded feature dicts.

    The class mirrors the previous `prepare_input_from_json` logic but packages it
    so it's testable and reusable.
    """

    def __init__(self):
        pass

    @staticmethod
    def _is_raw(obj: dict) -> bool:
        # Raw mode if object contains keys 'x','y','z' with list-like values
        return (
            isinstance(obj, dict)
            and 'x' in obj and 'y' in obj and 'z' in obj
            and hasattr(obj['x'], '__iter__') and hasattr(obj['y'], '__iter__') and hasattr(obj['z'], '__iter__')
        )

    def prepare(self, data_json: Any, model: Optional[torch.nn.Module] = None) -> torch.Tensor:
        """Convert incoming JSON into a torch tensor matching training features.

        Accepts either raw time-series objects with 'x','y','z' lists, or already-expanded
        feature dicts. Returns a FloatTensor shaped (batch, n_features).
        """

        items: List[dict] = data_json if isinstance(data_json, list) else [data_json]
        if not items:
            raise ValueError("Empty input payload")

        # Branch: raw timeseries mode -> delegate to preprocess helpers
        if all(self._is_raw(it) for it in items):
            df_expanded = expand_raw_items_to_expanded_df(items)
        else:
            # Assume items already contain flattened/expanded features. Convert to polars DataFrame directly.
            if not all(isinstance(it, dict) for it in items):
                raise ValueError("Each input must be an object with feature keys or x/y/z lists")
            df_expanded = pl.DataFrame(items)

        all_cols = list(df_expanded.columns)
        label_cols = [c for c in all_cols if c.startswith('label')]
        feature_cols = [c for c in all_cols if c not in label_cols]
        logging.info("Prepared input with %d items, %d features", df_expanded.height, len(feature_cols))
        if not feature_cols:
            logging.error("No feature columns found after preprocessing")
            raise ValueError("No feature columns found after preprocessing")

        X_np = df_expanded.select(feature_cols).to_numpy()

        expected = None
        if model is not None and hasattr(model, 'hparams'):
            # model.hparams may be an argparse Namespace-like or dict-like object
            if hasattr(model.hparams, 'input_size'):
                expected = int(model.hparams.input_size)
        if expected is not None and X_np.shape[1] != expected:
            raise ValueError(
                f"Preprocessed input has {X_np.shape[1]} features but model expects {expected} features. Ensure the incoming JSON can be converted to the same expanded feature layout used during training."
            )

        return torch.from_numpy(X_np).float()


def create_app(model_loader: ModelLoader, preparer: InputPreparer) -> Flask:
    """Factory that creates and configures the Flask app with `/predict` route.

    Loads the model once during app creation and reuses it for incoming requests.
    """
    app = Flask(__name__)

    # load model using checkpoint's saved hparams
    try:
        model = model_loader.load()
        logging.info("Model loaded from checkpoint")
    except Exception as e:
        model = None
        logging.warning("Failed to load model during app startup")

    metrics = PrometheusMetrics(app)


    @app.before_request
    def before_request_timing():
        # prevent recursion for metrics endpoint
        if request.path == "/metrics":
            return
        g.request_start_time = perf_counter()

    @app.after_request
    def after_request_timing(response):
        if request.path == "/metrics":
            return response
        start = getattr(g, 'request_start_time', None)
        endpoint_label = request.endpoint or request.path
        status_label = str(response.status_code)
        elapsed = perf_counter() - start
        request_latency_histogram.labels(endpoint=endpoint_label, method=request.method,
                                         status=status_label).observe(elapsed)
        request_counter.labels(endpoint=endpoint_label, method=request.method, status=status_label).inc()
        return response

    @app.route("/health", methods=["GET"])
    @metrics.do_not_track()
    def health():
        logging.debug("Health endpoint pinged.")
        return "OK", HTTPStatus.OK

    @app.route("/reload", methods=["POST"])
    def reload_model():
        """Reload the cached model from disk and replace the in-memory model used by handlers."""
        nonlocal model  # assign to outer-scope model variable
        try:
            new_model = model_loader.load(force_reload=True)
            model = new_model
            logging.info("Model reloaded from checkpoint: %s",
                         getattr(model_loader.checkpoint_path, "name", str(model_loader.checkpoint_path)))
            return jsonify({"status": "reloaded"}), HTTPStatus.OK
        except Exception as e:
            logging.exception("Failed to reload model")
            return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

    @app.route("/predict", methods=["POST"])
    def predict():
        nonlocal model
        if model is None:
            try:
                nonlocal model
                model = model_loader.load()
                logging.info("Model loaded from checkpoint on /predict request")
            except Exception as e:
                logging.error("Model not loaded and failed to load on /predict: %s", e, exc_info=True)
                return jsonify({"error": "Model not loaded"}), HTTPStatus.INTERNAL_SERVER_ERROR

        try:
            payload = request.get_json(force=True)
            x_tensor = preparer.prepare(payload, model=model)
            with torch.no_grad():
                logits = model(x_tensor.float())
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

    return app


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind the Flask server")
@click.argument("port", default=8000, required=False, type=int)
def serve(host, port):
    """Start a Flask server exposing /predict"""
    loader = ModelLoader()
    preparer = InputPreparer()
    app = create_app(loader, preparer)
    logging.info(f"Serving model {TrainingConfig.model_path} on http://{host}:{port}")
    app.run(host=host, port=int(port))


# create a CLI group and register both commands into a single entrypoint
cli = click.Group()

cli.add_command(serve)


if __name__ == "__main__":
    cli()