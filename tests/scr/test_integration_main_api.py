import sys
from unittest.mock import MagicMock, patch

mock_zenml_client_module = MagicMock()
sys.modules["zenml.client"] = mock_zenml_client_module

mock_zenml_exceptions = MagicMock()


class MockDoesNotExistException(Exception): pass


mock_zenml_exceptions.DoesNotExistException = MockDoesNotExistException
sys.modules["zenml.exceptions"] = mock_zenml_exceptions

sys.modules["daily_pipeline"] = MagicMock()
sys.modules["main_training"] = MagicMock()

from main_api import create_app, ModelLoader, InputPreparer
import pytest
import json
import torch
import torch.nn as nn


class MockModel(nn.Module):
    def __init__(self, input_size=6, num_classes=2):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def real_model_instance():
    return MockModel(input_size=6)


@pytest.fixture
def mock_zenml_client(real_model_instance):
    mock_client_instance = MagicMock()
    mock_pipeline = MagicMock()
    mock_run = MagicMock()
    mock_run.id = "test_run_123"

    mock_step_outputs = {
        "trained_model": MagicMock(),
        "z_score_dict": MagicMock()
    }

    mock_step_outputs["trained_model"].load.return_value = real_model_instance

    mock_step_outputs["z_score_dict"].load.return_value = [
        {
            "device": "device_1",
            "sensor": "accel",
            "feature_1_mean": 0.0,
            "feature_1_std": 1.0
        }
    ]

    mock_run.steps = {"run_training": MagicMock(outputs=mock_step_outputs)}
    mock_pipeline.last_successful_run = mock_run
    mock_client_instance.get_pipeline.return_value = mock_pipeline

    return mock_client_instance


@pytest.fixture
def api_client(mock_zenml_client):
    with patch("main_api.Client", return_value=mock_zenml_client), \
            patch("main_api.BackgroundScheduler"), \
            patch("main_api.PrometheusMetrics") as MockMetrics:
        mock_metrics_instance = MagicMock()
        MockMetrics.return_value = mock_metrics_instance
        mock_metrics_instance.do_not_track.return_value = lambda func: func

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
    pre_calculated_payload = [
        {
            "feature_1": 1.0, "feature_2": 0.5, "feature_3": 0.1,
            "feature_4": 0.1, "feature_5": 0.1, "feature_6": 0.1
        }
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

    from main_api import training_pipeline
    training_pipeline.assert_called()


def test_input_dimension_mismatch(api_client):
    bad_payload = [{"f1": 1.0, "f2": 2.0}]

    response = api_client.post(
        "/predict",
        data=json.dumps(bad_payload),
        content_type="application/json"
    )

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data