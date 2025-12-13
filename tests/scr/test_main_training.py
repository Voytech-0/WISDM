import pytest
import torch
import polars as pl
import numpy as np
from unittest.mock import MagicMock, patch, ANY
from click.testing import CliRunner

import main_training
from main_training import (
    SequenceDataset,
    LittleMLP,
    make_dataloaders,
    cli
)


# --- Fixtures ---

@pytest.fixture
def sample_dataframe():
    data = {
        "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "feature_2": [1.1, 1.2, 1.3, 1.4, 1.5],
        "label_0": [1, 0, 0, 0, 1],
        "label_1": [0, 1, 0, 0, 0],
        "label_2": [0, 0, 1, 1, 0],
    }
    return pl.DataFrame(data)


@pytest.fixture
def mock_config():
    with patch("main_training.TrainingConfig") as MockConfig:
        MockConfig.seed = 44
        MockConfig.max_epochs = 1
        MockConfig.batch_size = 2
        MockConfig.lr = 0.001
        MockConfig.hidden_size = 8
        MockConfig.num_layers = 1
        MockConfig.num_classes = 3
        MockConfig.model_dir = "/tmp/models"
        MockConfig.model_path = "/tmp/models/best_model.pt"
        yield MockConfig


def test_sequence_dataset_init_success(sample_dataframe):
    ds = SequenceDataset(sample_dataframe)

    assert len(ds) == 5
    assert ds.X.shape == (5, 2)
    assert ds.y.shape == (5, 3)
    assert ds.X.dtype == torch.float32
    assert ds.y.dtype == torch.float32


def test_sequence_dataset_init_no_labels():
    df = pl.DataFrame({"feature_1": [1, 2], "feature_2": [3, 4]})
    with pytest.raises(ValueError, match="No columns starting with 'label'"):
        SequenceDataset(df)


def test_sequence_dataset_getitem(sample_dataframe):

    ds = SequenceDataset(sample_dataframe)
    x, y = ds[0]

    assert torch.is_tensor(x)
    assert torch.is_tensor(y)
    assert x.shape == (2,)
    assert y.shape == (3,)


def test_little_mlp_init():
    input_size = 10
    hidden_size = 16
    num_layers = 2
    num_classes = 3
    lr = 0.01

    model = LittleMLP(input_size, hidden_size, num_layers, num_classes, lr)

    assert model.input_size == input_size
    assert model.hidden_size == hidden_size
    assert model.net is not None
    expected_hparams = {
        'lr': lr,
        'num_classes': num_classes,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
    }
    for key, expected in expected_hparams.items():
        actual = model.hparams.get(key)
        assert actual == expected, f"hparam `{key}` mismatch: expected {expected}, got {actual}"


def test_little_mlp_forward_2d():
    model = LittleMLP(input_size=4, hidden_size=8, num_layers=1, num_classes=2, lr=0.01)
    x = torch.randn(5, 4)  # Batch of 5, 4 features

    out = model(x)
    assert out.shape == (5, 2)


def test_little_mlp_forward_3d():
    model = LittleMLP(input_size=12, hidden_size=8, num_layers=1, num_classes=2, lr=0.01)
    # Input: Batch=2, Seq=3, Feat=4
    x = torch.randn(2, 3, 4)

    out = model(x)
    assert out.shape == (2, 2)


def test_little_mlp_label_to_indices():
    cases = [
        {
            "data": torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
            "expected": torch.tensor([1, 2, 0]),
        },
        {
            "data": torch.tensor([1, 2, 0]),
            "expected": torch.tensor([1, 2, 0]),
        },
    ]

    for case in cases:
        indices = LittleMLP._label_to_indices(case["data"])
        assert torch.equal(indices, case["expected"])



def test_make_dataloaders(sample_dataframe):

    data = {
        "f1": np.random.rand(100),
        "label_1": np.random.randint(0, 2, 100)
    }
    df = pl.DataFrame(data)

    train_dl, val_dl, test_dl = make_dataloaders(df, batch_size=10, val_split=0.2, test_split=0.2)

    total = 100
    val_len = int(100 * 0.2)  # 20
    test_len = int(100 * 0.2)  # 20
    train_len = total - val_len - test_len  # 60

    assert len(train_dl.dataset) == train_len
    assert len(val_dl.dataset) == val_len
    assert len(test_dl.dataset) == test_len


def test_make_dataloaders_insufficient_data():
    """Test ValueError when data is too small for splits."""
    df = pl.DataFrame({"f1": [1], "label_1": [1]})

    with pytest.raises(ValueError, match="Not enough samples"):
        make_dataloaders(df, batch_size=1, val_split=0.5, test_split=0.5)


# --- CLI / Training Loop Tests ---

@patch("main_training.preprocessing_pipeline_for_wisdm_raw_data")
@patch("main_training.torch.save")  # Prevent writing files
@patch("pathlib.Path.mkdir")  # Prevent directory creation
def test_train_command(mock_mkdir, mock_torch_save, mock_pipeline, mock_config, sample_dataframe):
    """
    Test the CLI train command.
    """
    runner = CliRunner()
    large_df = pl.concat([sample_dataframe] * 4)
    mock_pipeline.return_value = large_df

    result = runner.invoke(cli, ['train', '--epochs', '1', '--batch-size', '4'])

    if result.exit_code != 0:
        print(result.output)
    assert result.exit_code == 0
    mock_pipeline.assert_called_once()
    mock_mkdir.assert_called()
    assert mock_torch_save.call_count >= 1
    saved_args = mock_torch_save.call_args[0][0]
    assert 'model_state_dict' in saved_args
    assert 'hparams' in saved_args


@patch("main_training.preprocessing_pipeline_for_wisdm_raw_data")
@patch("main_training.torch.load")  # Mock loading
@patch("pathlib.Path.mkdir")
def test_train_command_resume_checkpoint(mock_mkdir, mock_torch_load, mock_pipeline, mock_config, sample_dataframe):
    """Test the CLI train command when resuming from a checkpoint."""
    runner = CliRunner()
    large_df = pl.concat([sample_dataframe] * 4)
    mock_pipeline.return_value = large_df
    mock_ckpt = {
        'model_state_dict': {},
        'hparams': {}
    }
    mock_torch_load.return_value = mock_ckpt
    result = runner.invoke(cli, ['train', '--epochs', '1', '--checkpoint', 'fake_ckpt.pt'])
    assert result.exit_code == 0
    mock_torch_load.assert_called_with('fake_ckpt.pt', map_location=ANY)
