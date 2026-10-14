import json
import pytest
import torch
import polars as pl
import numpy as np
from unittest.mock import MagicMock


# --- Fixtures ---

@pytest.fixture
def main_training_module():
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


def test_execute_training_logic_saves_model_and_stats(monkeypatch, tmp_path, main_training_module, sample_dataframe):
    mt = main_training_module
    model_path = tmp_path / "models" / "mlp.ckpt"
    stats_path = tmp_path / "models" / "processed" / "z.json"

    monkeypatch.setattr(mt, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(mt, "STATS_PATH", str(stats_path))
    for k, v in dict(max_epochs=1, batch_size=4, lr=0.01, hidden_size=8, num_layers=1, num_classes=3).items():
        monkeypatch.setattr(mt.TrainingConfig, k, v)
    monkeypatch.setattr(mt, "preprocessing_pipeline_for_wisdm_raw_data",
                        MagicMock(return_value=pl.concat([sample_dataframe] * 10)))
    monkeypatch.setattr(mt, "get_z_score",
                        MagicMock(return_value={("phone", "accel"): {"mean": 0.5, "std": 1.0}}))

    mt.execute_training_logic()

    assert json.loads(stats_path.read_text()) == [{"device": "phone", "sensor": "accel", "mean": 0.5, "std": 1.0}]
    model = torch.load(model_path, weights_only=False)
    assert isinstance(model, mt.LittleMLP)
