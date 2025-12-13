import sys
import pytest
import torch
import polars as pl
import numpy as np
from unittest.mock import MagicMock


# --- Fixtures ---

@pytest.fixture
def mock_zenml(monkeypatch):

    def identity_decorator(*args, **kwargs):
        def wrapper(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrapper

    mock_zenml = MagicMock()
    mock_zenml.step = identity_decorator
    mock_zenml.pipeline = identity_decorator

    monkeypatch.setitem(sys.modules, "zenml", mock_zenml)


    return mock_zenml


@pytest.fixture
def main_training_module(mock_zenml):

    if "main_training" in sys.modules:
        import main_training
        import importlib
        importlib.reload(main_training)
        return main_training
    else:
        import main_training
        return main_training


@pytest.fixture
def sample_dataframe():
    """Creates a small dummy Polars DataFrame for testing."""
    data = {
        "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "feature_2": [1.1, 1.2, 1.3, 1.4, 1.5],
        "label_0": [1, 0, 0, 0, 1],
        "label_1": [0, 1, 0, 0, 0],
        "label_2": [0, 0, 1, 1, 0],
    }
    return pl.DataFrame(data)


@pytest.fixture
def mock_config(monkeypatch, main_training_module):

    class MockConfig:
        seed = 44
        max_epochs = 1
        batch_size = 2
        lr = 0.001
        hidden_size = 8
        num_layers = 1
        num_classes = 3

    monkeypatch.setattr(main_training_module, "TrainingConfig", MockConfig)
    return MockConfig



def test_sequence_dataset_init_success(main_training_module, sample_dataframe):
    ds = main_training_module.SequenceDataset(sample_dataframe)

    assert len(ds) == 5
    assert ds.X.shape == (5, 2)
    assert ds.y.shape == (5, 3)
    assert ds.X.dtype == torch.float32
    assert ds.y.dtype == torch.float32


def test_sequence_dataset_init_no_labels(main_training_module):
    df = pl.DataFrame({"feature_1": [1, 2], "feature_2": [3, 4]})
    with pytest.raises(ValueError, match="No columns starting with 'label'"):
        main_training_module.SequenceDataset(df)


def test_little_mlp_init(main_training_module):
    model = main_training_module.LittleMLP(
        input_size=10, hidden_size=16, num_layers=2, num_classes=3, lr=0.01
    )

    assert model.input_size == 10
    assert model.hparams['lr'] == 0.01
    assert model.net is not None


def test_little_mlp_forward(main_training_module):
    model = main_training_module.LittleMLP(
        input_size=4, hidden_size=8, num_layers=1, num_classes=2, lr=0.01
    )
    x = torch.randn(5, 4)
    out = model(x)
    assert out.shape == (5, 2)


def test_make_dataloaders(main_training_module):
    data = {
        "f1": np.random.rand(100),
        "label_1": np.random.randint(0, 2, 100)
    }
    df = pl.DataFrame(data)

    train_dl, val_dl, test_dl = main_training_module.make_dataloaders(
        df, batch_size=10, val_split=0.2, test_split=0.2
    )

    # 100 samples: 60 train, 20 val, 20 test
    assert len(train_dl.dataset) == 60
    assert len(val_dl.dataset) == 20
    assert len(test_dl.dataset) == 20


def test_run_training_step_success(
        monkeypatch,
        main_training_module,
        mock_config,
        sample_dataframe
):

    large_df = pl.concat([sample_dataframe] * 10)
    mock_pipeline = MagicMock(return_value=large_df)
    monkeypatch.setattr(main_training_module, "preprocessing_pipeline_for_wisdm_raw_data", mock_pipeline)

    mock_lf = MagicMock()
    mock_lf.collect.return_value.to_dicts.return_value = [{"mean": 0.5}]
    mock_get_z_score = MagicMock(return_value=mock_lf)
    monkeypatch.setattr(main_training_module, "get_z_score", mock_get_z_score)

    model, stats = main_training_module.run_training(epochs=1, batch_size=4, lr=0.01)

    assert isinstance(model, main_training_module.LittleMLP)
    assert stats == [{"mean": 0.5}]


def test_run_training_step_failure(
        monkeypatch,
        main_training_module,
        mock_zenml,
        sample_dataframe
):

    large_df = pl.concat([sample_dataframe] * 10)
    mock_pipeline = MagicMock(return_value=large_df)
    monkeypatch.setattr(main_training_module, "preprocessing_pipeline_for_wisdm_raw_data", mock_pipeline)

    # Should not raise exception
    model, stats = main_training_module.run_training(epochs=1, batch_size=4)

    assert isinstance(model, main_training_module.LittleMLP)
    mock_pipeline.assert_called_once()