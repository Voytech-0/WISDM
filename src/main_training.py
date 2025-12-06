import json
from pathlib import Path
import click
import logging
from flask import Flask, request, jsonify

import random
import numpy as np
import pandas as pd
import torch
import polars as pl
from typing import Tuple
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as torchl
from pytorch_lightning.callbacks import ModelCheckpoint
from configs.configs import TrainingConfig
from preprocessing.preprocess import preprocessing_pipeline_for_wisdm_raw_data

# configure basic logging for CLI/info output
logging.basicConfig(level=logging.INFO)


# seed RNGs using config
try:
    seed_val = TrainingConfig.seed
except Exception:
    # fall back to a simple default if CFG doesn't provide seed
    logging.warning("TrainingConfig.seed not found, using default seed 25")
    seed_val = 25

torchl.seed_everything(seed_val)
torch.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)



class SequenceDataset(Dataset):
    """
    A PyTorch Dataset class that takes either a path to a CSV or a Polars DataFrame,
    splits it into features (X) and labels (y), and converts all data to torch tensors.

    Features are all columns *not* starting with "label".
    Labels are all columns *starting* with "label".
    """

    def __init__(self, df: pl.DataFrame):
        """
        Initializes the dataset by splitting the DataFrame and converting to tensors.

        Args:
            df: Polars DataFrame
        """

        label_cols = [col for col in df.columns if col.startswith("label")]
        feature_cols = [col for col in df.columns if col not in label_cols]

        if not label_cols:
            logging.error("No columns starting with 'label' found in the DataFrame to serve as targets (y).")
            raise ValueError("No columns starting with 'label' found in the DataFrame to serve as targets (y).")

        X_df = df.select(feature_cols)
        y_df = df.select(label_cols)

        X_np = X_df.to_numpy()
        y_np = y_df.to_numpy()

        self.X = torch.from_numpy(X_np).float()
        self.y = torch.from_numpy(y_np).float()

        logging.info(f"Dataset initialized with {len(self.X)} samples.")
        logging.debug(f"X shape: {self.X.shape}")
        logging.debug(f"y shape: {self.y.shape}")

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves one sample (feature-label pair) from the dataset.
        """
        # Return the X and y tensors for the given index
        return self.X[idx], self.y[idx]


class LittleLSTM(torchl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        # Replace LSTM with a simple MLP. Keep constructor signature for compatibility
        # We'll lazily build the MLP on first forward pass since the flattened input dim
        # depends on whether the dataset provides (batch, seq_len, feat) or (batch, feat).
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.net = None
        self.dropout = nn.Dropout(p=0.3)
        self.criterion = nn.CrossEntropyLoss()

    def _build_mlp(self, input_dim: int):
        """Create an MLP dynamically given the flattened input dimension."""
        layers = []
        in_dim = input_dim
        # create `num_layers` hidden layers using hidden_size
        for i in range(max(1, self.num_layers)):
            layers.append(nn.Linear(in_dim, self.hidden_size))
            layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(self.dropout)
            in_dim = self.hidden_size
        # final classifier
        layers.append(nn.Linear(in_dim, self.num_classes))
        self.net = nn.Sequential(*layers)
        # register the module so it's saved/loaded with checkpoints
        self.add_module("net", self.net)

    def forward(self, x):
        # Accept either (batch, seq_len, input_size) or (batch, features)
        if x.dim() == 3:
            b, seq_len, feat = x.shape
            x = x.view(b, seq_len * feat)
        elif x.dim() == 1:
            # single sample, make it a batch
            x = x.unsqueeze(0)
        elif x.dim() != 2:
            raise ValueError(f"Unexpected input tensor shape: {x.shape}")

        input_dim = x.size(1)
        if self.net is None:
            self._build_mlp(input_dim)

        logits = self.net(x.float())
        return logits

    def _label_to_indices(self, y: torch.Tensor) -> torch.Tensor:
        # Accept one-hot, (N,1), or (N,) integer labels
        if y.dim() > 1 and y.size(1) > 1:
            return y.argmax(dim=1).long()
        return y.view(-1).long()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x.float())
        y_idx = self._label_to_indices(y)
        loss = self.criterion(logits, y_idx)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_idx).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x.float())
        y_idx = self._label_to_indices(y)
        loss = self.criterion(logits, y_idx)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_idx).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def make_dataloaders(data, batch_size, val_split=0.1, test_split=0.1):
    ds = SequenceDataset(data)
    n = len(ds)
    val_n = int(n * val_split)
    test_n = int(n * test_split)
    train_n = n - val_n - test_n
    if train_n <= 0:
        logging.error("Not enough samples for train/val/test splits: n=%s val_n=%s test_n=%s", n, val_n, test_n)
        raise ValueError("Not enough samples for splits")
    train_ds, val_ds, test_ds = random_split(ds, [train_n, val_n, test_n])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader


@click.group()
def cli():
    pass


@cli.command()
@click.option("--epochs", default=TrainingConfig.max_epochs, type=int)
@click.option("--batch-size", default=TrainingConfig.batch_size, type=int)
@click.option("--lr", default=TrainingConfig.lr, type=float)
@click.option("--checkpoint", default=None, help="Optional path to resume checkpoint")
def train(epochs, batch_size, lr, checkpoint):
    """Train the LSTM model"""
    data = preprocessing_pipeline_for_wisdm_raw_data()

    train_loader, val_loader, _ = make_dataloaders(data, batch_size, val_split=0.1, test_split=0.1)

    input_size = data.width - len([col for col in data.columns if col.startswith("label")])

    model = LittleLSTM(input_size=input_size, hidden_size=TrainingConfig.hidden_size,
                       num_layers=TrainingConfig.num_layers, num_classes=TrainingConfig.num_classes, lr=lr)

    checkpoint_cb = ModelCheckpoint(dirpath=str(TrainingConfig.model_dir), filename="lstm", save_top_k=1, monitor="val_loss", mode="min")
    trainer = torchl.Trainer(max_epochs=epochs, callbacks=[checkpoint_cb], logger=True, enable_checkpointing=True)
    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint)
    # save final checkpoint path to config path
    ckpt = checkpoint_cb.best_model_path or ""
    if ckpt:
        logging.info(f"Best checkpoint saved at: {ckpt}")
    else:
        # if no checkpoint callback for some reason, save manually
        trainer.save_checkpoint(str(TrainingConfig.model_path))
        logging.info(f"Checkpoint saved at: {TrainingConfig.model_path}")


@cli.command()
@click.option("--checkpoint", required=True, help="Path to checkpoint (.ckpt)")
@click.option("--input", "input_csv", required=True, help="CSV with a single row or multiple rows to infer")
def infer(checkpoint, input_csv):
    """Run inference on a CSV and print predictions as JSON"""
    checkpoint = Path(checkpoint)
    input_csv = Path(input_csv)
    if not checkpoint.exists():
        logging.error(f"Checkpoint not found: {checkpoint}")
        raise click.ClickException(f"Checkpoint not found: {checkpoint}")
    if not input_csv.exists():
        logging.error(f"Input CSV not found: {input_csv}")
        raise click.ClickException(f"Input CSV not found: {input_csv}")

    # load model
    model = LittleLSTM.load_from_checkpoint(str(checkpoint), input_size=TrainingConfig.input_size,
                                            hidden_size=TrainingConfig.hidden_size, num_layers=TrainingConfig.num_layers,
                                            num_classes=TrainingConfig.num_classes, lr=TrainingConfig.lr)
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


def build_flask_app(checkpoint_path):
    app = Flask(__name__)
    if not Path(checkpoint_path).exists():
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = LittleLSTM.load_from_checkpoint(str(checkpoint_path), input_size=TrainingConfig.input_size,
                                            hidden_size=TrainingConfig.hidden_size, num_layers=TrainingConfig.num_layers,
                                            num_classes=TrainingConfig.num_classes, lr=TrainingConfig.lr)
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
@click.option("--checkpoint", required=True, help="Path to checkpoint (.ckpt)")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=5000, type=int)
def serve(checkpoint, host, port):
    """Start a Flask server exposing /predict"""
    app = build_flask_app(checkpoint)
    logging.info(f"Serving model {checkpoint} on http://{host}:{port}")
    app.run(host=host, port=port)


if __name__ == "__main__":
    cli()
