import json
from pathlib import Path
import click
import logging
from flask import Flask, request, jsonify

import random
import numpy as np
import torch
import polars as pl
from typing import Tuple
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as torchl
from pytorch_lightning.callbacks import ModelCheckpoint
from configs.configs import TrainingConfig
from preprocessing.preprocess import preprocessing_pipeline_for_wisdm_raw_data
import shutil

# configure basic logging for CLI/info output
logging.basicConfig(level=logging.INFO)


# seed RNGs using config
try:
    seed_val = TrainingConfig.seed
except Exception:
    # fall back to a simple default if CFG doesn't provide seed
    logging.warning("TrainingConfig.seed not found, using default seed 44")
    seed_val = 44

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


class LittleMLP(torchl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.net = None
        self.dropout = nn.Dropout(p=0.3)
        self.criterion = nn.CrossEntropyLoss()

        layers = []
        # ensure hyperparams are normal python ints (helps static checkers and downstream ops)
        num_layers = int(self.num_layers)
        hidden_size = int(self.hidden_size)
        num_layers = max(1, num_layers)
        # create `num_layers` hidden layers using hidden_size
        in_dim = input_size
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(self.dropout)
            in_dim = hidden_size
        # final classifier
        layers.append(nn.Linear(in_dim, int(self.num_classes)))
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
    """Train the model"""
    data = preprocessing_pipeline_for_wisdm_raw_data()

    train_loader, val_loader, _ = make_dataloaders(data, batch_size, val_split=0.1, test_split=0.1)

    input_size = data.width - len([col for col in data.columns if col.startswith("label")])

    model = LittleMLP(input_size=input_size, hidden_size=TrainingConfig.hidden_size,
                      num_layers=TrainingConfig.num_layers, num_classes=TrainingConfig.num_classes, lr=lr)

    # ensure model dir exists
    model_dir_path = Path(TrainingConfig.model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    # Keep only the best checkpoint (save_top_k=1). We'll copy that best checkpoint to TrainingConfig.model_path
    checkpoint_cb = ModelCheckpoint(dirpath=str(model_dir_path), filename="lstm", save_top_k=1, monitor="val_loss", mode="min")
    trainer = torchl.Trainer(max_epochs=epochs, callbacks=[checkpoint_cb], logger=True, enable_checkpointing=True)
    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint)

    # copy the best checkpoint (if available) to deterministic path from config
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        try:
            dest = Path(TrainingConfig.model_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            best_path = Path(best_ckpt)
            if best_path.resolve(strict=False) == dest.resolve(strict=False):
                logging.info(f"Best checkpoint already at configured model_path: {dest}")
            else:
                shutil.copyfile(str(best_path), str(dest))
                logging.info(f"Best checkpoint copied to configured model_path: {dest}")
        except Exception as e:
            logging.error("Failed to copy best checkpoint to configured model_path: %s", e, exc_info=True)
            # fallback: still try to save trainer checkpoint to configured path
            try:
                trainer.save_checkpoint(str(TrainingConfig.model_path))
                logging.info(f"Fallback checkpoint saved at: {TrainingConfig.model_path}")
            except Exception as e2:
                logging.error("Fallback save also failed: %s", e2, exc_info=True)
    else:
        # No best model found (maybe training didn't run any validation); save last checkpoint to configured path
        try:
            trainer.save_checkpoint(str(TrainingConfig.model_path))
            logging.info(f"Checkpoint saved at: {TrainingConfig.model_path}")
        except Exception as e:
            logging.error("Failed to save checkpoint to configured model_path: %s", e, exc_info=True)



if __name__ == "__main__":
    cli()
