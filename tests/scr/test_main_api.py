import json
import pytest
import torch
from click.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock
import main_api
from types import SimpleNamespace

from main_api import create_app, InputPreparer, ModelLoader


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
        self.z_score_stats = stats or []

    def load(self, force_reload: bool = False):
        # The new interface returns a tuple
        return self._model, self.z_score_stats


class StubPreparer(InputPreparer):
    """Turns {"f0": 1.0, ...} dicts into a feature tensor, skipping raw-sensor preprocessing."""

    def prepare(self, data_json, model=None):
        items = data_json if isinstance(data_json, list) else [data_json]
        return torch.tensor([list(i.values()) for i in items], dtype=torch.float32)


def test_create_app_request_timing():
    loader = StubLoader(DummyModel())
    preparer = StubPreparer()
    app = create_app(loader, preparer)
    client = app.test_client()

    # Health endpoint should return OK and 200
    r = client.get('/health')
    assert r.status_code == 200
    assert r.get_data(as_text=True) == 'OK'


def test_create_app_reload_model():
    # Initial setup with 3 classes
    loader = StubLoader(DummyModel(num_classes=3))
    preparer = StubPreparer()
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
    preparer = StubPreparer()
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

    with patch('main_api.ModelLoader') as MockLoaderCls, \
            patch('main_api.InputPreparer') as MockPreparerCls, \
            patch('main_api.create_app') as mock_create_app:
        MockLoaderCls.return_value = loader_instance
        MockPreparerCls.return_value = preparer_instance
        mock_create_app.return_value = app_mock

        host = '127.0.0.1'
        port = '9010'

        result = runner.invoke(main_api.serve, ['--host', host, '--port', port])

    assert result.exit_code == 0

    MockLoaderCls.assert_called_once_with()
    assert MockPreparerCls.called, "InputPreparer should be instantiated"

    mock_create_app.assert_called_once_with(loader_instance, preparer_instance)
    app_mock.run.assert_called_once_with(host=host, port=int(port))


# --- ModelLoader (file based) ---

@pytest.fixture
def model_files(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "mlp.ckpt"
    stats_path = tmp_path / "models" / "processed" / "z.json"
    monkeypatch.setattr(main_api.TrainingConfig, "model_path", str(model_path), raising=False)
    monkeypatch.setattr(main_api.TrainingConfig, "stats_path", str(stats_path), raising=False)
    return model_path, stats_path


def _save(model_path, stats_path, stats=None):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.nn.Linear(2, 2), model_path)
    if stats is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats))


def test_modelloader_missing_file(model_files):
    with pytest.raises(FileNotFoundError):
        ModelLoader().load()


def test_modelloader_load_success(model_files):
    stats = [{"device": "phone", "sensor": "accel", "mean": 0.5}]
    _save(*model_files, stats=stats)
    model, loaded_stats = ModelLoader().load()
    assert isinstance(model, torch.nn.Linear)
    assert loaded_stats == stats


def test_modelloader_missing_stats_defaults_empty(model_files):
    _save(*model_files)
    _, stats = ModelLoader().load()
    assert stats == []


def test_modelloader_caching_and_force_reload(model_files):
    _save(*model_files)
    loader = ModelLoader()
    m1, _ = loader.load()
    m2, _ = loader.load()
    assert m1 is m2
    m3, _ = loader.load(force_reload=True)
    assert m3 is not m1


def test_input_preparer_stats():
    """Test that InputPreparer can hold and update stats."""
    prep = InputPreparer()
    assert prep.z_score_dict == {}

    prep.set_stats([{"device": "phone", "sensor": "accel", "mean": 10}])
    assert prep.z_score_dict == {("phone", "accel"): {"mean": 10}}


def test_input_preparer_prepare(monkeypatch):
    import polars as pl
    prep = InputPreparer([{"device": "phone", "sensor": "accel", "mean": 0.0}])

    with pytest.raises(ValueError):
        prep.prepare([])

    seen = {}

    def fake_expand(items, z_score):
        seen["items"], seen["z"] = items, z_score
        return pl.DataFrame({"x_0": [1.0], "y_0": [2.0], "label_A": [1]})

    monkeypatch.setattr(main_api, "expand_raw_items_to_expanded_df", fake_expand)
    t = prep.prepare({"device": "phone"})

    assert t.shape == (1, 2)  # label columns dropped
    assert seen["items"] == [{"device": "phone"}]  # single dict wrapped in list
    assert seen["z"] == prep.z_score_dict


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

    loader = FlakyLoader(DummyModel())
    preparer = StubPreparer()
    app = create_app(loader, preparer)
    client = app.test_client()

    # At startup loader.load() should have been called and failed, leaving model None
    # Now /predict should trigger another load and succeed
    payload = [{"a": 1.0, "b": 2.0}]
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    out = r.get_json()
    assert isinstance(out, list) and len(out) == 1


def _train_client():
    return create_app(StubLoader(DummyModel()), InputPreparer()).test_client()


def test_train_triggers_airflow_dag():
    token_resp = Mock(json=Mock(return_value={"access_token": "tok"}))
    run_resp = Mock(ok=True, json=Mock(return_value={"dag_run_id": "r1"}))
    with patch("main_api.requests.get", return_value=token_resp) as get, \
            patch("main_api.requests.post", return_value=run_resp) as post:
        r = _train_client().post("/train")

    assert r.status_code == 202
    assert r.get_json()["airflow_response"] == {"dag_run_id": "r1"}
    assert get.call_args.args[0].endswith("/auth/token")
    url = post.call_args.args[0]
    assert url.endswith("/api/v2/dags/wisdm_training/dagRuns")
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer tok"}


def test_train_airflow_rejects():
    token_resp = Mock(json=Mock(return_value={"access_token": "tok"}))
    run_resp = Mock(ok=False, text="boom")
    with patch("main_api.requests.get", return_value=token_resp), \
            patch("main_api.requests.post", return_value=run_resp):
        r = _train_client().post("/train")
    assert r.status_code == 502
    assert r.get_json()["details"] == "boom"


def test_train_airflow_unreachable():
    with patch("main_api.requests.get", side_effect=ConnectionError("down")):
        r = _train_client().post("/train")
    assert r.status_code == 500
    assert "down" in r.get_json()["error"]
