import pytest
import json
import torch
import polars as pl
from unittest.mock import MagicMock, patch
from main_api import create_app, ModelLoader, InputPreparer, LittleMLP


# --- Fixtures ---

@pytest.fixture
def real_model_instance():
    """
    Creates a real instance of LittleMLP to ensure shape compatibility
    during integration tests.
    """
    # Matches the default config usually expected by the app
    return LittleMLP(input_size=6, hidden_size=8, num_layers=2, num_classes=2, lr=0.01)


@pytest.fixture
def mock_zenml_client(real_model_instance):
    """
    Mocks the ZenML Client to return a real Model object and stats
    without connecting to a real ZenML server.
    """
    mock_client = MagicMock()

    # Mock the pipeline structure
    mock_pipeline = MagicMock()
    mock_run = MagicMock()
    mock_run.id = "test_run_123"

    # Mock Step Outputs
    mock_step_outputs = {
        "trained_model": MagicMock(),
        "z_score_stats": MagicMock()
    }

    # Define what .load() returns
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
    """
    Creates the Flask Test Client with the integrated ModelLoader and InputPreparer.
    """
    # Patch the Client inside main_api to return our mock
    with patch("main_api.Client", return_value=mock_zenml_client):
        # We use the real ModelLoader and InputPreparer to test their integration
        loader = ModelLoader(pipeline_name="test_pipe", step_name="run_training")
        preparer = InputPreparer()

        # Initialize app (this triggers the initial load() call)
        app = create_app(loader, preparer)
        app.config["TESTING"] = True

        with app.test_client() as client:
            yield client


# --- Integration Tests ---

def test_app_startup_integration(api_client, mock_zenml_client):
    """
    Verifies that create_app correctly calls ModelLoader, which calls ZenML,
    and the app starts up healthy.
    """
    # Check /health endpoint
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.data.decode() == "OK"

    # Verify the integration: Did the app actually call the ZenML client on startup?
    mock_zenml_client.get_pipeline.assert_called_with("test_pipe")


def test_predict_endpoint_flow(api_client):
    """
    End-to-End Test for /predict.
    Verifies: Request -> InputPreparer -> Polars -> Tensor -> Model -> Response
    """
    # Raw input data (simulating accelerometer data)
    payload = {
        "user_id": 1,
        "activity": "walking",
        "timestamp": 1600000000,
        "x": [0.1, 0.2, 0.3],
        "y": [0.1, 0.2, 0.3],
        "z": [0.1, 0.2, 0.3]
    }

    # The 'real_model_instance' fixture expects input_size=6.
    # The 'expand_raw_items_to_expanded_df' (inside InputPreparer) usually produces
    # features based on aggregations (mean, std, etc.).
    # We mock the expansion simply by sending pre-calculated features that match the model
    # to avoid depending on the complex preprocessing logic in this specific test.

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

    # Verify predictions are valid class indices (0 or 1 for the 2-class model)
    assert data[0]["pred"] in [0, 1]


def test_reload_endpoint_integration(api_client, mock_zenml_client):
    """
    Verifies the /reload endpoint triggers a fresh fetch from ZenML
    and updates the app context.
    """
    # Initial call count (from startup)
    initial_count = mock_zenml_client.get_pipeline.call_count

    response = api_client.post("/reload")

    assert response.status_code == 200
    assert response.get_json() == {"status": "reloaded"}

    # Verify ModelLoader called ZenML again
    assert mock_zenml_client.get_pipeline.call_count == initial_count + 1


@patch("main_api.training_pipeline")
def test_train_trigger_integration(mock_training_pipeline, api_client):
    """
    Verifies /train endpoint successfully spawns a background thread
    to run the pipeline.
    """
    response = api_client.get("/train")

    assert response.status_code == 202
    assert "Training started" in response.get_json()["status"]

    # Wait briefly to ensure the thread has time to start and call the mock
    # (Since it's a thread, we can't deterministic wait, but usually it's instant in tests)
    import time
    time.sleep(0.1)

    mock_training_pipeline.assert_called_once()


def test_input_dimension_mismatch(api_client):
    """
    Integration test for error handling:
    Ensures that if input data doesn't match Model requirements,
    the API returns a 400 error (not 500).
    """
    # Sending 2 features, but model expects 6
    bad_payload = [{"f1": 1.0, "f2": 2.0}]

    response = api_client.post(
        "/predict",
        data=json.dumps(bad_payload),
        content_type="application/json"
    )

    # Should be caught by InputPreparer validation
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert "model expects 6" in response.get_json()["error"]