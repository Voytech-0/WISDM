import json
import logging
from pathlib import Path

import click
import numpy as np
import torch
from flask import request, jsonify, Flask
from torch import nn
import polars as pl
from torch.utils.data import DataLoader

from configs.configs import TrainingConfig
from preprocessing.preprocess import expand_raw_items_to_expanded_df
from main_training import LittleMLP, SequenceDataset


@click.command()
@click.option("--checkpoint", default=None, help="Path to checkpoint (.ckpt). If omitted, uses TrainingConfig.model_path")
@click.option("--input", "input_csv", required=True, help="CSV with a single row or multiple rows to infer")
def infer(checkpoint, input_csv):
    """Run inference on a CSV and print predictions as JSON"""
    if checkpoint is None:
        checkpoint_path = Path(TrainingConfig.model_path)
    else:
        checkpoint_path = Path(checkpoint)

    input_csv = Path(input_csv)
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        raise click.ClickException(f"Checkpoint not found: {checkpoint_path}")
    if not input_csv.exists():
        logging.error(f"Input CSV not found: {input_csv}")
        raise click.ClickException(f"Input CSV not found: {input_csv}")

    # load model; rely on checkpoint/hparams to reconstruct model
    model = LittleMLP.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    # load CSV with polars and build dataset
    df = pl.read_csv(str(input_csv))
    ds = SequenceDataset(df)
    loader = DataLoader(ds, batch_size=TrainingConfig.batch_size)
    results = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, tuple):
                x, _ = batch
            else:
                x = batch
            logits = model(x.float())
            probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1).tolist()
            for p, pr in zip(preds, probs.tolist()):
                results.append({"pred": int(p), "probs": pr})
    logging.info(json.dumps(results))


def build_flask_app():
    app = Flask(__name__)
    checkpoint_path = TrainingConfig.model_path
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # load model using checkpoint's saved hparams
    model = LittleMLP.load_from_checkpoint(str(checkpoint_path))
    model.eval()

    def prepare_input_from_json(data_json):
        """Convert incoming JSON into a torch tensor matching training features.

        Accepts either:
        - A single object or list of objects with raw time-series fields: {"x": [...], "y": [...], "z": [...], "device": "phone", "sensor": "accel"}
          The lists can be shorter/longer than the model's fixed length; padding/truncation and z-score normalization will be applied using the project's preprocessing utilities.
        - A single object or list of objects where each object already contains expanded feature columns (e.g. x_0, x_1, ..., y_0, ..., z_0, ... and/or dummy columns for device/sensor/label). In that case the object is converted directly to a DataFrame and used as-is.

        Returns a torch.FloatTensor shaped (batch, n_features) suitable to pass into the model.
        """

        def is_raw(obj: dict) -> bool:
            # Raw mode if object contains keys 'x','y','z' with list-like values
            return (
                isinstance(obj, dict)
                and 'x' in obj and 'y' in obj and 'z' in obj
                and hasattr(obj['x'], '__iter__') and hasattr(obj['y'], '__iter__') and hasattr(obj['z'], '__iter__')
            )

        # normalize to list of objects
        items = data_json if isinstance(data_json, list) else [data_json]
        if not items:
            raise ValueError("Empty input payload")

        # Branch: raw timeseries mode -> delegate to preprocess helpers
        if all(is_raw(it) for it in items):
            df_expanded = expand_raw_items_to_expanded_df(items)
        else:
            # Assume items already contain flattened/expanded features. Convert to polars DataFrame directly.
            if not all(isinstance(it, dict) for it in items):
                raise ValueError("Each input must be an object with feature keys or x/y/z lists")
            df_expanded = pl.DataFrame(items)

        all_cols = list(df_expanded.columns)
        label_cols = [c for c in all_cols if c.startswith('label')]
        feature_cols = [c for c in all_cols if c not in label_cols]
        if not feature_cols:
            logging.error("No feature columns found after preprocessing")
            raise ValueError("No feature columns found after preprocessing")

        X_np = df_expanded.select(feature_cols).to_numpy()
        # If the model saved expected input size in hparams, validate it matches the preprocessed features
        try:
            expected = None
            if hasattr(model, 'hparams') and hasattr(model.hparams, 'input_size'):
                expected = int(model.hparams.input_size)
            elif hasattr(model, 'hparams') and 'input_size' in dict(model.hparams):
                expected = int(dict(model.hparams)['input_size'])
            if expected is not None:
                if X_np.shape[1] != expected:
                    raise ValueError(f"Preprocessed input has {X_np.shape[1]} features but model expects {expected} features. Ensure the incoming JSON can be converted to the same expanded feature layout used during training.")
        except Exception:
            # If model hparams can't be introspected, continue but the caller should ensure proper shape
            logging.debug("Could not validate feature vector against model.hparams; skipping strict check")

        return torch.from_numpy(X_np).float()

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            payload = request.get_json(force=True)
            x_tensor = prepare_input_from_json(payload)
            with torch.no_grad():
                logits = model(x_tensor.float())
                probs = nn.functional.softmax(logits, dim=1).cpu().numpy().tolist()
                preds = np.argmax(probs, axis=1).tolist()
            out = [{"pred": int(p), "probs": pr} for p, pr in zip(preds, probs)]
            return jsonify(out)
        except Exception as e:
            logging.error("Error during /predict: %s", e, exc_info=True)
            return jsonify({"error": str(e)}), 400

    return app


# add click decorators to serve and register commands into a group
@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind the Flask server")
@click.argument("port", default=8000, required=False, type=int)
def serve(host, port):
    """Start a Flask server exposing /predict"""
    app = build_flask_app()
    logging.info(f"Serving model {TrainingConfig.model_path} on http://{host}:{port}")
    app.run(host=host, port=int(port))


# create a CLI group and register both commands so both 'infer' and 'serve' are available
@click.group()
def cli():
    pass

cli.add_command(infer)
cli.add_command(serve)

if __name__ == "__main__":
    cli()
