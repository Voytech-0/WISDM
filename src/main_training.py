import logging
from pathlib import Path
import click

import random
import numpy as np
import torch
import polars as pl
from typing import Tuple
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import time
from configs.configs import TrainingConfig
from preprocessing.preprocess import preprocessing_pipeline_for_wisdm_raw_data

# configure basic logging for CLI/info output
logging.basicConfig(level=logging.INFO)


# seed RNGs using config
try:
    seed_val = TrainingConfig.seed
except Exception:
    # fall back to a simple default if CFG doesn't provide seed
    logging.warning("TrainingConfig.seed not found, using default seed 44")
    seed_val = 44

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


class LittleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lr):
        super().__init__()
        # save hyperparams for later introspection
        self.hparams = {
            'input_size': int(input_size),
            'hidden_size': int(hidden_size),
            'num_layers': int(num_layers),
            'num_classes': int(num_classes),
            'lr': float(lr),
        }
        self.lr = float(lr)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_classes = int(num_classes)

        self.dropout = nn.Dropout(p=0.3)

        layers = []
        num_layers = max(1, self.num_layers)
        in_dim = self.input_size
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_size))
            layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(self.dropout)
            in_dim = self.hidden_size
        layers.append(nn.Linear(in_dim, int(self.num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either (batch, seq_len, input_size) or (batch, features)
        if x.dim() == 3:
            b, seq_len, feat = x.shape
            x = x.view(b, seq_len * feat)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() != 2:
            raise ValueError(f"Unexpected input tensor shape: {x.shape}")
        return self.net(x.float())

    @staticmethod
    def _label_to_indices(y: torch.Tensor) -> torch.Tensor:
        if y.dim() > 1 and y.size(1) > 1:
            return y.argmax(dim=1).long()
        return y.view(-1).long()


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
    """Train the model using plain PyTorch (no Lightning)."""
    data = preprocessing_pipeline_for_wisdm_raw_data()

    train_loader, val_loader, _ = make_dataloaders(data, batch_size, val_split=0.1, test_split=0.1)

    input_size = data.width - len([col for col in data.columns if col.startswith("label")])

    model = LittleMLP(input_size=input_size, hidden_size=TrainingConfig.hidden_size,
                      num_layers=TrainingConfig.num_layers, num_classes=TrainingConfig.num_classes, lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare model dir
    model_dir_path = Path(TrainingConfig.model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    start_time = time.time()

    # Optionally resume
    if checkpoint:
        try:
            ckpt = torch.load(checkpoint, map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            logging.info(f"Resumed model weights from checkpoint: {checkpoint}")
        except Exception as e:
            logging.warning(f"Failed to load checkpoint {checkpoint}: {e}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_accs = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            y_idx = LittleMLP._label_to_indices(y_batch)
            loss = criterion(logits, y_idx)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_idx).float().mean().item()
            train_losses.append(loss.item())
            train_accs.append(acc)

        # validation
        model.eval()
        val_losses = []
        val_accs = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                y_idx = LittleMLP._label_to_indices(y_batch)
                loss = criterion(logits, y_idx)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y_idx).float().mean().item()
                val_losses.append(loss.item())
                val_accs.append(acc)

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_acc = float(np.mean(train_accs)) if train_accs else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
        val_acc = float(np.mean(val_accs)) if val_accs else 0.0

        logging.info(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                dest = Path(TrainingConfig.model_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'model_state_dict': model.state_dict(), 'hparams': model.hparams}, str(dest))
                logging.info(f"Saved new best model to: {dest}")
            except Exception as e:
                logging.error("Failed to save best model: %s", e, exc_info=True)

    total_time = time.time() - start_time
    logging.info(f"Training finished in {total_time:.1f}s. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    cli()
