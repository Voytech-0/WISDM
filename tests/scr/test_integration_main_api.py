import pytest
import json
import torch
import polars as pl
from unittest.mock import MagicMock, patch
from src.main_api import create_app, ModelLoader, InputPreparer
from src.main_training import LittleMLP



@pytest.fixture
def real_model_instance():
    return LittleMLP(input_size=6, hidden_size=8, num_layers=2, num_classes=2, lr=0.01)


@pytest.fixture
def mock_zenml_client(real_model_instance):
    mock_client = MagicMock()

    mock_pipeline = MagicMock()
    mock_run = MagicMock()
    mock_run.id = "test_run_123"

    mock_step_outputs = {
        "trained_model": MagicMock(),
        "z_score_stats": MagicMock()
    }

    mock_step_outputs["trained_model"].load.return_value = real_model_instance
    mock_step_outputs["z_score_stats"].load.return_value = [
        {"column": "feature_1", "mean": 0.0, "std": 1.0}
    ]

    mock_run.steps = {"run_training": MagicMock(outputs=mock_step_outputs)}
    mock_pipeline.last_successful_run = mock_run
    mock_client.get_pipeline.return_value = mock_pipeline

    return mock_client


@pytest.fixture
def api_client(mock_zenml_client):

    with patch("main_api.Client", return_value=mock_zenml_client):
        loader = ModelLoader(pipeline_name="test_pipe", step_name="run_training")
        preparer = InputPreparer()

        app = create_app(loader, preparer)
        app.config["TESTING"] = True

        with app.test_client() as client:
            yield client


def test_app_startup_integration(api_client, mock_zenml_client):

    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.data.decode() == "OK"

    mock_zenml_client.get_pipeline.assert_called_with("test_pipe")


def test_predict_endpoint_flow(api_client):

    payload = {
        "user_id": 1,
        "activity": "A",
        "timestamp": 1600000000,
        "x": [0.1, 0.2, 0.3],
        "y": [0.1, 0.2, 0.3],
        "z": [0.1, 0.2, 0.3]
    }


    pre_calculated_payload = [
        {"feature_1": 1.0, "feature_2": 0.5, "feature_3": 0.1,
         "feature_4": 0.1, "feature_5": 0.1, "feature_6": 0.1}
    ]

    response = api_client.post(
        "/predict",
        data=json.dumps(pre_calculated_payload),
        content_type="application/json"
    )

    assert response.status_code == 200

    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert "pred" in data[0]
    assert "probs" in data[0]

    assert data[0]["pred"] in [0, 1]


def test_reload_endpoint_integration(api_client, mock_zenml_client):

    initial_count = mock_zenml_client.get_pipeline.call_count

    response = api_client.post("/reload")

    assert response.status_code == 200
    assert response.get_json() == {"status": "reloaded"}

    assert mock_zenml_client.get_pipeline.call_count == initial_count + 1


@patch("main_api.training_pipeline")
def test_train_trigger_integration(mock_training_pipeline, api_client):

    response = api_client.get("/train")

    assert response.status_code == 202
    assert "Training started" in response.get_json()["status"]

    import time
    time.sleep(0.1)

    mock_training_pipeline.assert_called_once()


def test_input_dimension_mismatch(api_client):

    bad_payload = [{"f1": 1.0, "f2": 2.0}]

    response = api_client.post(
        "/predict",
        data=json.dumps(bad_payload),
        content_type="application/json"
    )

    assert response.status_code == 400
    assert "error" in response.get_json()
    assert "model expects 6" in response.get_json()["error"]