import json
import logging
from pathlib import Path

import click
import numpy as np
import torch
from flask import cli, request, jsonify, Flask
from torch import nn
import polars as pl
from torch.utils.data import DataLoader

from configs.configs import TrainingConfig
from main_training import LittleMLP, SequenceDataset


@cli.command()
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


def build_flask_app(checkpoint_path=None):
    app = Flask(__name__)
    if checkpoint_path is None:
        checkpoint_path = TrainingConfig.model_path
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # load model using checkpoint's saved hparams
    model = LittleMLP.load_from_checkpoint(str(checkpoint_path))
    model.eval()

    def prepare_input_from_json(data_json):
        # Accept { "x": [...], "y": [...], "z":[...] } or arrays of such objects
        def one(obj):
            x = obj.get("x")
            y = obj.get("y")
            z = obj.get("z")
            if x is None or y is None or z is None:
                raise ValueError("Each input must contain x,y,z arrays")
            arr = np.stack([np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), np.array(z, dtype=np.float32)], axis=-1)
            # arr shape (seq_len, 3) -> expand to (1, seq_len, 3)
            return arr
        if isinstance(data_json, list):
            arrs = [one(o) for o in data_json]
            return torch.from_numpy(np.stack(arrs, axis=0))
        else:
            return torch.from_numpy(np.expand_dims(one(data_json), axis=0))

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


@cli.command()
@click.option("--checkpoint", default=None, help="Path to checkpoint (.ckpt). If omitted, uses TrainingConfig.model_path")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=5000, type=int)
def serve(checkpoint, host, port):
    """Start a Flask server exposing /predict"""
    app = build_flask_app(checkpoint)
    logging.info(f"Serving model {checkpoint or TrainingConfig.model_path} on http://{host}:{port}")
    app.run(host=host, port=port)
