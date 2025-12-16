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
        self.hparams = {'input_size': 2}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sums = x.sum(dim=1).to(torch.int64)
        preds = (sums % self.num_classes).tolist()
        logits = torch.full((x.shape[0], self.num_classes), -10.0)
        for i, p in enumerate(preds):
            logits[i, int(p)] = 10.0
        return logits


class StubLoader:

    def __init__(self, model: DummyModel, stats=None):
        self._model = model
        self.z_score_dict = stats or []

    def load(self, force_reload: bool = False):
        return self._model, self.z_score_dict


def test_create_app_request_timing():
    loader = StubLoader(DummyModel())
    preparer = InputPreparer()
    app = create_app(loader, preparer)
    client = app.test_client()

    r = client.get('/health')
    assert r.status_code == 200
    assert r.get_data(as_text=True) == 'OK'


def test_create_app_reload_model():
    loader = StubLoader(DummyModel(num_classes=3))
    preparer = InputPreparer()
    app = create_app(loader, preparer)
    client = app.test_client()

    payload = [{"f0": 1.0, "f1": 2.0}]
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    out = r.get_json()
    assert isinstance(out, list) and len(out) == 1
    assert out[0]['pred'] == 0
    assert len(out[0]['probs']) == 3

    loader._model = DummyModel(num_classes=2)

    r = client.post('/reload')
    assert r.status_code == 200
    assert r.get_json() == {"status": "reloaded"}

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

        result = runner.invoke(main_api.cli, ['serve', '--host', host, '--port', port, '--pipeline', pipeline_name])

    assert result.exit_code == 0

    MockLoaderCls.assert_called_once_with(pipeline_name=pipeline_name)
    assert MockPreparerCls.called, "InputPreparer should be instantiated"

    mock_create_app.assert_called_once_with(loader_instance, preparer_instance)
    app_mock.run.assert_called_once_with(host=host, port=int(port))


@patch('src.main_api.Client')
def test_modelLoader_load_success(MockClient):
    mock_client_instance = MockClient.return_value
    mock_pipeline = MagicMock()
    mock_run = MagicMock()
    mock_step_outputs = MagicMock()

    mock_model_artifact = MagicMock()
    mock_stats_artifact = MagicMock()

    real_model_obj = DummyModel(num_classes=3)
    real_stats_obj = [{"mean": 0.5, "std": 1.0}]

    mock_model_artifact.load.return_value = real_model_obj
    mock_stats_artifact.load.return_value = real_stats_obj

    mock_client_instance.get_pipeline.return_value = mock_pipeline
    mock_pipeline.last_successful_run = mock_run

    mock_run.steps.__getitem__.return_value.outputs = {
        "trained_model": mock_model_artifact,
        "z_score_dict": mock_stats_artifact
    }

    loader = ModelLoader(pipeline_name="test_pipe", step_name="test_step")
    loaded_model, loaded_stats = loader.load()

    assert isinstance(loaded_model, DummyModel)
    assert loaded_stats == real_stats_obj

    mock_client_instance.get_pipeline.assert_called_with("test_pipe")
    mock_model_artifact.load.assert_called_once()
    mock_stats_artifact.load.assert_called_once()


@patch('src.main_api.Client')
def test_modelLoader_load_caching(MockClient):
    mock_client_instance = MockClient.return_value
    mock_pipeline = MagicMock()
    mock_run = MagicMock()

    mock_client_instance.get_pipeline.return_value = mock_pipeline
    mock_pipeline.last_successful_run = mock_run

    mock_model_artifact = MagicMock()
    mock_model_artifact.load.return_value = DummyModel()
    mock_run.steps.__getitem__.return_value.outputs = {
        "trained_model": mock_model_artifact,
        "z_score_dict": MagicMock()
    }

    loader = ModelLoader()

    m1, s1 = loader.load()
    assert MockClient.call_count == 1

    m2, s2 = loader.load()
    assert MockClient.call_count == 1
    assert m1 is m2

    m3, s3 = loader.load(force_reload=True)
    assert MockClient.call_count == 2


def test_input_preparer_stats():
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

    with pytest.raises(ValueError):
        prep.prepare([])

    data = {"f0": 1.0, "f1": 2.0}
    t = prep.prepare(data)
    assert isinstance(t, torch.Tensor)
    assert t.shape[0] == 1
    assert t.shape[1] == 2

    model = SimpleNamespace()
    model.hparams = {'input_size': 1}

    with pytest.raises(ValueError) as excinfo:
        prep.prepare(data, model=model)
    assert "features but model expects 1" in str(excinfo.value)


def test_predict_when_model_none_initially(monkeypatch):
    class FlakyLoader:
        def __init__(self, model):
            self.model = model
            self.called = 0

        def load(self, force_reload: bool = False):
            self.called += 1
            if self.called == 1:
                raise RuntimeError("no model yet")
            return self.model, []

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

    payload = [{"a": 1.0, "b": 2.0}]
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    out = r.get_json()
    assert isinstance(out, list) and len(out) == 1


def test_train_endpoint_triggers_pipeline(monkeypatch):
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

    monkeypatch.setattr('src.main_api.training_pipeline', fake_training_pipeline)

    r = client.get('/train')
    assert r.status_code == 202
    j = r.get_json()
    assert j["status"] == "Training started"

    import time
    timeout = time.time() + 2.0
    while not called["flag"] and time.time() < timeout:
        time.sleep(0.01)

    assert called["flag"] is True

