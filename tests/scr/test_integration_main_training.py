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

@patch("main_training.aim.Run")
@patch("main_training.get_z_score")
@patch("main_training.preprocessing_pipeline_for_wisdm_raw_data")
def test_full_training_pipeline_integration(mock_prep, mock_stats, mock_aim, integration_data):
    """
    INTEGRATION TEST: Verifies the entire training loop orchestration.

    This tests the integration of:
    1. preprocessing_pipeline (Mocked input)
    2. make_dataloaders (Splitting logic)
    3. SequenceDataset (Tensor conversion)
    4. LittleMLP (Model initialization and forward pass)
    5. Optimizer & Loss (Backprop flow)
    6. Aim Logging (Metadata tracking)
    """
    # 1. Setup the interface between Preprocessing and Training
    mock_prep.return_value = integration_data
    mock_stats.return_value = pl.LazyFrame({"col": [], "mean": [], "std": []})

    # Mock Aim to prevent actual network calls, but verify it receives data
    mock_run_inst = MagicMock()
    mock_aim.return_value = mock_run_inst

    # 2. Run the pipeline (Reduced epochs for speed)
    model, stats = run_training(epochs=2, batch_size=10, lr=0.01)

    # 3. VERIFY INTEGRATION POINTS

    # A. Verify Data->Model Integration
    # The model input size should have automatically adjusted to 4 features (from integration_data)
    # If the integration failed, this would mismatch or crash.
    assert model.hparams['input_size'] == 4
    assert model.hparams['num_classes'] == main_training.TrainingConfig.num_classes

    # B. Verify Model->Output Integration
    # We expect a valid model object that has been moved back to CPU
    assert isinstance(model, torch.nn.Module)
    param = next(model.parameters())
    assert param.device.type == 'cpu'

    # C. Verify Training Loop Integration (Aim Tracking)
    # The loop should have called track() for train and val loss
    # 2 epochs * 2 metrics (train/val) = 4 minimum track calls
    assert mock_run_inst.track.call_count >= 4

    # D. Verify Artifact Integration
    # Ensure stats were collected and returned
    assert isinstance(stats, list)


def test_dataloader_dataset_integration(integration_data):
    """
    INTEGRATION TEST: Verifies the flow between DataFrames, Dataset Class, and DataLoader.

    Ensures that:
    1. Polars DataFrame converts correctly to PyTorch Dataset.
    2. RandomSplit correctly partitions data.
    3. DataLoader correctly batches tensors.
    """
    batch_size = 5

    # Call the function that integrates these three components
    train_loader, val_loader, test_loader = make_dataloaders(
        integration_data,
        batch_size=batch_size,
        val_split=0.2,
        test_split=0.2
    )

    # 1. Verify Split Integration
    total_len = len(integration_data)
    train_len = len(train_loader.dataset)
    val_len = len(val_loader.dataset)
    test_len = len(test_loader.dataset)

    assert train_len + val_len + test_len == total_len
    assert train_len > 0 and val_len > 0

    # 2. Verify Batching Integration
    # Fetch a batch to ensure the Dataset __getitem__ works with the DataLoader collation
    x_batch, y_batch = next(iter(train_loader))

    # Should get tensors back
    assert isinstance(x_batch, torch.Tensor)
    assert isinstance(y_batch, torch.Tensor)

    # Shapes should match batch size and feature counts
    assert x_batch.shape[0] == batch_size
    assert x_batch.shape[1] == 4  # 4 features in integration_data