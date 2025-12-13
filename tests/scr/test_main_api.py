import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
from click.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock
import src.main_api as main_api
from types import SimpleNamespace

from src.main_api import create_app, InputPreparer, ModelLoader


class DummyModel(torch.nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        # mimic saved hparams used by InputPreparer
        self.hparams = {'input_size': 2}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # simple deterministic rule: class = int(sum(features)) % num_classes
        sums = x.sum(dim=1).to(torch.int64)
        preds = (sums % self.num_classes).tolist()
        # create logits such that argmax is the chosen class
        logits = torch.full((x.shape[0], self.num_classes), -10.0)
        for i, p in enumerate(preds):
            logits[i, int(p)] = 10.0
        return logits


class StubLoader:

    def __init__(self, model: DummyModel, stats=None):
        self._model = model
        self.z_score_dict = stats or []

    def load(self, force_reload: bool = False):
        # The new interface returns a tuple
        return self._model, self.z_score_dict


def test_create_app_request_timing():
    loader = StubLoader(DummyModel())
    preparer = InputPreparer()
    app = create_app(loader, preparer)
    client = app.test_client()

    # Health endpoint should return OK and 200
    r = client.get('/health')
    assert r.status_code == 200
    assert r.get_data(as_text=True) == 'OK'


def test_create_app_reload_model():
    # Initial setup with 3 classes
    loader = StubLoader(DummyModel(num_classes=3))
    preparer = InputPreparer()
    app = create_app(loader, preparer)
    client = app.test_client()

    # initial predict with one item -> sum of features 1+2 = 3 -> 3 % 3 == 0
    payload = [{"f0": 1.0, "f1": 2.0}]
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    out = r.get_json()
    assert isinstance(out, list) and len(out) == 1
    assert out[0]['pred'] == 0
    assert len(out[0]['probs']) == 3

    # change underlying model to different num_classes and reload
    # We simulate an update in the loader
    loader._model = DummyModel(num_classes=2)

    r = client.post('/reload')
    assert r.status_code == 200
    assert r.get_json() == {"status": "reloaded"}

    # now /predict should use new model behavior (2 classes)
    r2 = client.post('/predict', json=payload)
    assert r2.status_code == 200
    out2 = r2.get_json()
    assert out2[0]['pred'] in (0, 1)
    assert len(out2[0]['probs']) == 2


def test_create_app_predict():
    loader = StubLoader(DummyModel(num_classes=3))
    preparer = InputPreparer()
    app = create_app(loader, preparer)
    client = app.test_client()

    payload = [{"a": 1.0, "b": 2.0}, {"a": 0.0, "b": 1.0}]
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    out = r.get_json()
    assert isinstance(out, list) and len(out) == 2
    for item in out:
        assert 'pred' in item and 'probs' in item
        assert isinstance(item['pred'], int)
        assert isinstance(item['probs'], list)


def test_serve():
    runner = CliRunner()
    loader_instance = Mock(name='loader')
    # loader.load() must return a tuple (model, stats) now
    loader_instance.load.return_value = (Mock(), [])

    preparer_instance = Mock(name='preparer')
    app_mock = Mock(name='flask_app')
    app_mock.run = Mock(name='run')

    with patch('src.main_api.ModelLoader') as MockLoaderCls, \
            patch('src.main_api.InputPreparer') as MockPreparerCls, \
            patch('src.main_api.create_app') as mock_create_app:
        MockLoaderCls.return_value = loader_instance
        MockPreparerCls.return_value = preparer_instance
        mock_create_app.return_value = app_mock

        host = '127.0.0.1'
        port = '9010'
        pipeline_name = "my_training_pipeline"

        # Test that CLI accepts the new --pipeline argument
        result = runner.invoke(main_api.cli, ['serve', '--host', host, '--port', port, '--pipeline', pipeline_name])

    assert result.exit_code == 0

    # Check that ModelLoader was initialized with the pipeline name
    MockLoaderCls.assert_called_once_with(pipeline_name=pipeline_name)
    assert MockPreparerCls.called, "InputPreparer should be instantiated"

    mock_create_app.assert_called_once_with(loader_instance, preparer_instance)
    app_mock.run.assert_called_once_with(host=host, port=int(port))


# --- New ModelLoader Tests (Mocking ZenML) ---

@patch('src.main_api.Client')
def test_modelLoader_load_success(MockClient):
    """
    Test that ModelLoader correctly interacts with the ZenML Client
    to fetch the pipeline, run, and artifacts.
    """
    # 1. Setup the Mock Chain
    mock_client_instance = MockClient.return_value
    mock_pipeline = MagicMock()
    mock_run = MagicMock()
    mock_step_outputs = MagicMock()

    # Artifact Mocks
    mock_model_artifact = MagicMock()
    mock_stats_artifact = MagicMock()

    # The actual python objects returned by .load()
    real_model_obj = DummyModel(num_classes=3)
    real_stats_obj = [{"mean": 0.5, "std": 1.0}]

    mock_model_artifact.load.return_value = real_model_obj
    mock_stats_artifact.load.return_value = real_stats_obj

    # Chain them together
    mock_client_instance.get_pipeline.return_value = mock_pipeline
    mock_pipeline.last_successful_run = mock_run

    # Simulate the dictionary of outputs from the step
    # Keys must match what is in main_api.py (run_training step)
    mock_run.steps.__getitem__.return_value.outputs = {
        "trained_model": mock_model_artifact,
        "z_score_dict": mock_stats_artifact
    }

    # 2. Execute
    loader = ModelLoader(pipeline_name="test_pipe", step_name="test_step")
    loaded_model, loaded_stats = loader.load()

    # 3. Assertions
    assert isinstance(loaded_model, DummyModel)
    assert loaded_stats == real_stats_obj

    mock_client_instance.get_pipeline.assert_called_with("test_pipe")
    mock_model_artifact.load.assert_called_once()
    mock_stats_artifact.load.assert_called_once()


@patch('src.main_api.Client')
def test_modelLoader_load_caching(MockClient):
    """
    Test that the loader caches the model and doesn't hit ZenML again
    unless force_reload is True.
    """
    mock_client_instance = MockClient.return_value
    mock_pipeline = MagicMock()
    mock_run = MagicMock()

    mock_client_instance.get_pipeline.return_value = mock_pipeline
    mock_pipeline.last_successful_run = mock_run

    # Mock return values
    mock_model_artifact = MagicMock()
    mock_model_artifact.load.return_value = DummyModel()
    mock_run.steps.__getitem__.return_value.outputs = {
        "trained_model": mock_model_artifact,
        "z_score_dict": MagicMock()  # stats
    }

    loader = ModelLoader()

    # First load
    m1, s1 = loader.load()
    assert MockClient.call_count == 1

    # Second load (should be cached)
    m2, s2 = loader.load()
    assert MockClient.call_count == 1  # Still 1
    assert m1 is m2

    # Force reload
    m3, s3 = loader.load(force_reload=True)
    assert MockClient.call_count == 2  # Increased


def test_input_preparer_stats():
    """Test that InputPreparer can hold and update stats."""
    prep = InputPreparer()
    assert prep.z_score_dict == {}

    new_stats = [
        {"device": "dev1", "sensor": "acc", "col": "x", "mean": 10}
    ]
    prep.set_stats(new_stats)
    expected = {("dev1", "acc"): {"col": "x", "mean": 10}}
    assert prep.z_score_dict == expected


def test_input_preparer_prepare():
    prep = InputPreparer()

    # empty payload -> error
    with pytest.raises(ValueError):
        prep.prepare([])

    # expanded features -> returns tensor
    data = {"f0": 1.0, "f1": 2.0}
    t = prep.prepare(data)
    assert isinstance(t, torch.Tensor)
    assert t.shape[0] == 1
    assert t.shape[1] == 2

    # Test dimension validation
    model = SimpleNamespace()
    # production code expects hparams as a dict; set it accordingly in the test
    model.hparams = {'input_size': 1}  # Model expects 1 feature

    # Input has 2 features -> Should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        prep.prepare(data, model=model)
    assert "features but model expects 1" in str(excinfo.value)


def test_predict_when_model_none_initially(monkeypatch):
    """If the app starts with no model available, /predict should attempt to load it and still succeed when loader becomes available."""
    # loader whose first call raises, second call returns a model
    class FlakyLoader:
        def __init__(self, model):
            self.model = model
            self.called = 0

        def load(self, force_reload: bool = False):
            self.called += 1
            if self.called == 1:
                raise RuntimeError("no model yet")
            return self.model, []

    # Avoid starting the real scheduler inside create_app during tests
    class DummyScheduler:
        def __init__(self):
            pass
        def add_job(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def shutdown(self):
            pass

    monkeypatch.setattr('src.main_api.BackgroundScheduler', DummyScheduler)

    loader = FlakyLoader(DummyModel())
    preparer = InputPreparer()
    app = create_app(loader, preparer)
    client = app.test_client()

    # At startup loader.load() should have been called and failed, leaving model None
    # Now /predict should trigger another load and succeed
    payload = [{"a": 1.0, "b": 2.0}]
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    out = r.get_json()
    assert isinstance(out, list) and len(out) == 1


def test_train_endpoint_triggers_pipeline(monkeypatch):
    """/train should return 202 and trigger the training pipeline in a background thread."""
    # Avoid starting the real scheduler inside create_app during tests
    class DummyScheduler:
        def __init__(self):
            pass
        def add_job(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def shutdown(self):
            pass

    monkeypatch.setattr('src.main_api.BackgroundScheduler', DummyScheduler)

    loader = StubLoader(DummyModel())
    preparer = InputPreparer()
    app = create_app(loader, preparer)
    client = app.test_client()

    called = {"flag": False}

    def fake_training_pipeline():
        called["flag"] = True

    # Patch the training_pipeline function so the background thread calls our fake
    monkeypatch.setattr('src.main_api.training_pipeline', fake_training_pipeline)

    r = client.get('/train')
    assert r.status_code == 202
    j = r.get_json()
    assert j["status"] == "Training started"

    # wait briefly for the background thread to run the fake
    import time
    timeout = time.time() + 2.0
    while not called["flag"] and time.time() < timeout:
        time.sleep(0.01)

    assert called["flag"] is True

