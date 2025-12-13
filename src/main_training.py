import logging
import random
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import polars as pl
from typing import Tuple, Dict, List, Any

from configs.configs import TrainingConfig
from preprocessing.preprocess import preprocessing_pipeline_for_wisdm_raw_data, get_z_score

logging.basicConfig(level=logging.INFO)

MODEL_PATH = TrainingConfig.TrainingConfig
STATS_PATH = TrainingConfig.stats_path

seed_val = TrainingConfig.seed
torch.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)



class SequenceDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        label_cols = [col for col in df.columns if col.startswith("label")]
        feature_cols = [col for col in df.columns if col not in label_cols]

        if not label_cols:
            raise ValueError("No columns starting with 'label' found.")

        X_df = df.select(feature_cols)
        y_df = df.select(label_cols)

        self.X = torch.from_numpy(X_df.to_numpy()).float()
        self.y = torch.from_numpy(y_df.to_numpy()).float()

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
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_classes = int(num_classes)
        self.dropout = nn.Dropout(p=0.3)

        layers = []
        in_dim = self.input_size
        for i in range(max(1, self.num_layers)):
            layers.append(nn.Linear(in_dim, self.hidden_size))
            layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(self.dropout)
            in_dim = self.hidden_size
        layers.append(nn.Linear(in_dim, int(self.num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            b, seq_len, feat = x.shape
            x = x.view(b, seq_len * feat)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
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
    train_ds, val_ds, test_ds = random_split(ds, [train_n, val_n, test_n])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size)
    )



def execute_training_logic():
    epochs = TrainingConfig.max_epochs
    batch_size = TrainingConfig.batch_size
    lr = TrainingConfig.lr

    logging.info("Running training logic...")

    data = preprocessing_pipeline_for_wisdm_raw_data()
    raw_stats_dict = get_z_score()

    serializable_stats = []
    if raw_stats_dict:
        for key_tuple, stats in raw_stats_dict.items():
            device, sensor = (key_tuple[0], key_tuple[1]) if isinstance(key_tuple, tuple) else (str(key_tuple),
                                                                                                "unknown")
            serializable_stats.append({"device": device, "sensor": sensor, **stats})

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)

    with open(STATS_PATH, 'w') as f:
        json.dump(serializable_stats, f)
    logging.info(f"Stats saved to {STATS_PATH}")

    train_loader, val_loader, _ = make_dataloaders(data, batch_size)
    input_size = data.width - len([col for col in data.columns if col.startswith("label")])

    model = LittleMLP(input_size=input_size,
                      hidden_size=TrainingConfig.hidden_size,
                      num_layers=TrainingConfig.num_layers,
                      num_classes=TrainingConfig.num_classes,
                      lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            y_idx = LittleMLP._label_to_indices(y_batch)
            loss = criterion(logits, y_idx)
            loss.backward()
            optimizer.step()

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
        logging.info(f"Epoch {epoch} Val Loss: {val_loss_avg:.4f}")

    model.to('cpu')
    torch.save(model, MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    execute_training_logic()