import logging
import random
import numpy as np
import torch
import polars as pl
from typing import Tuple, Dict, List
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from zenml import step, pipeline
from typing_extensions import Annotated

from configs.configs import TrainingConfig
from preprocessing.preprocess import preprocessing_pipeline_for_wisdm_raw_data, get_z_score

logging.basicConfig(level=logging.INFO)

seed_val = TrainingConfig.seed

torch.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)



class SequenceDataset(Dataset):
    """
    A PyTorch Dataset class that takes Polars DataFrame,
    splits it into features (X) and labels (y), and converts all data to torch tensors.

    Features are all columns *not* starting with "label".
    Labels are all columns *starting* with "label".
    """

    def __init__(self, df: pl.DataFrame):
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
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LittleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lr):
        super().__init__()
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
        """Reversing one-hot encoding to class indices."""
        if y.dim() > 1 and y.size(1) > 1:
            return y.argmax(dim=1).long()
        return y.view(-1).long()


def make_dataloaders(data, batch_size, val_split=0.1, test_split=0.1):
    ds = SequenceDataset(data)
    n = len(ds)
    val_n = int(n * val_split)
    test_n = int(n * test_split)
    train_n = n - val_n - test_n
    if train_n <= 0 or val_n <= 0 or test_n <= 0:
        logging.error("Not enough samples for train/val/test splits: n=%s val_n=%s test_n=%s", n, val_n, test_n)
        raise ValueError("Not enough samples for splits")
    train_ds, val_ds, test_ds = random_split(ds, [train_n, val_n, test_n])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader

@step
def run_training(
        epochs: int = TrainingConfig.max_epochs,
        batch_size: int = TrainingConfig.batch_size,
        lr: float = TrainingConfig.lr
) -> Tuple[
    Annotated[nn.Module, "trained_model"],
    Annotated[List[Dict], "z_score_stats"]
]:
    """
    Run training and return the model and stats as ZenML artifacts.
    """
    logging.info("Running training with epochs=%d, batch_size=%d, lr=%.6f", epochs, batch_size, lr)
    data = preprocessing_pipeline_for_wisdm_raw_data()

    stats_dict = []
    try:
        stats_lf = get_z_score()
        stats_dict = stats_lf.collect().to_dicts()
    except Exception as e:
        logging.warning("Failed to load or save z-score stats: %s", e)

    train_loader, val_loader, _ = make_dataloaders(data, batch_size)
    input_size = data.width - len([col for col in data.columns if col.startswith("label")])

    model = LittleMLP(input_size=input_size, hidden_size=TrainingConfig.hidden_size,
                      num_layers=TrainingConfig.num_layers, num_classes=TrainingConfig.num_classes, lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            y_idx = LittleMLP._label_to_indices(y_batch)
            loss = criterion(logits, y_idx)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss_avg = float(np.mean(train_losses))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                y_idx = LittleMLP._label_to_indices(y_batch)
                loss = criterion(logits, y_idx)
                val_losses.append(loss.item())

        val_loss_avg = float(np.mean(val_losses)) if val_losses else float('inf')

        logging.info(f"Epoch {epoch}/{epochs} train_loss={train_loss_avg:.4f} val_loss={val_loss_avg:.4f}")


        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg

    model.to('cpu')

    return model, stats_dict


@pipeline
def training_pipeline():
    logging.info("Starting Training Pipeline in main_training.py")
    model, stats_dict = run_training()


if __name__ == "__main__":
    training_pipeline()