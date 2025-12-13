import pytest
import torch
import polars as pl
import numpy as np
from main_training import make_dataloaders



@pytest.fixture
def integration_data():
    data = {
        "feature_a": np.random.randn(50),
        "feature_b": np.random.randn(50),
        "feature_c": np.random.randn(50),
        "feature_d": np.random.randn(50),
        # One-hot encoded labels for 2 classes
        "label_0": [1, 0] * 25,
        "label_1": [0, 1] * 25
    }
    return pl.DataFrame(data)


def test_dataloader_dataset_integration(integration_data):
    batch_size = 5

    train_loader, val_loader, test_loader = make_dataloaders(
        integration_data,
        batch_size=batch_size,
        val_split=0.2,
        test_split=0.2
    )

    total_len = len(integration_data)
    train_len = len(train_loader.dataset)
    val_len = len(val_loader.dataset)
    test_len = len(test_loader.dataset)

    assert train_len + val_len + test_len == total_len
    assert train_len > 0 and val_len > 0

    x_batch, y_batch = next(iter(train_loader))

    assert isinstance(x_batch, torch.Tensor)
    assert isinstance(y_batch, torch.Tensor)

    assert x_batch.shape[0] == batch_size
    assert x_batch.shape[1] == 4