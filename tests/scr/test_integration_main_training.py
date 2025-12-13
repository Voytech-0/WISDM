import pytest
import torch
import polars as pl
import numpy as np
from unittest.mock import MagicMock, patch

# Import the module to be tested
import main_training
from main_training import run_training, make_dataloaders, SequenceDataset


# --- Fixtures for Integration ---

@pytest.fixture
def integration_data():
    """
    Creates a small but valid dataset to verify the flow from
    Data -> Loader -> Model -> Loss.
    """
    # 50 samples, 4 features, 2 classes
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


# --- Integration Tests ---

@patch("main_training.get_z_score")
@patch("main_training.preprocessing_pipeline_for_wisdm_raw_data")
def test_full_training_pipeline_integration(mock_prep, mock_stats, integration_data):

    mock_prep.return_value = integration_data
    mock_stats.return_value = pl.LazyFrame({"col": [], "mean": [], "std": []})

    model, stats = run_training(epochs=2, batch_size=10, lr=0.01)

    assert model.hparams['input_size'] == 4
    assert model.hparams['num_classes'] == main_training.TrainingConfig.num_classes

    assert isinstance(model, torch.nn.Module)
    param = next(model.parameters())
    assert param.device.type == 'cpu'

    assert isinstance(stats, list)


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