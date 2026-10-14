import pytest
import torch

import main_api
from main_api import create_app, ModelLoader
from test_main_api import StubPreparer
from main_training import LittleMLP


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main_api.TrainingConfig, "model_path", str(tmp_path / "mlp.ckpt"), raising=False)
    monkeypatch.setattr(main_api.TrainingConfig, "stats_path", str(tmp_path / "z.json"), raising=False)
    return create_app(ModelLoader(), StubPreparer()).test_client()


def _save_model(path, num_classes=3):
    torch.save(LittleMLP(input_size=2, hidden_size=4, num_layers=1, num_classes=num_classes, lr=0.01).eval(), path)


def test_health_unavailable_until_model_exists(client):
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=[{"a": 1.0, "b": 2.0}]).status_code == 500


def test_reload_picks_up_model_written_by_training(client, tmp_path):
    assert client.post("/reload").status_code == 500  # nothing trained yet

    _save_model(tmp_path / "mlp.ckpt")
    assert client.post("/reload").get_json() == {"status": "reloaded"}
    assert client.get("/health").status_code == 200

    r = client.post("/predict", json=[{"a": 1.0, "b": 2.0}])
    assert r.status_code == 200
    assert len(r.get_json()[0]["probs"]) == 3


def test_reload_swaps_model(client, tmp_path):
    _save_model(tmp_path / "mlp.ckpt", num_classes=3)
    client.post("/reload")
    _save_model(tmp_path / "mlp.ckpt", num_classes=2)
    client.post("/reload")
    r = client.post("/predict", json=[{"a": 1.0, "b": 2.0}])
    assert len(r.get_json()[0]["probs"]) == 2


def test_input_dimension_mismatch(client, tmp_path):
    _save_model(tmp_path / "mlp.ckpt")
    client.post("/reload")
    r = client.post("/predict", json=[{"f1": 1.0, "f2": 2.0, "f3": 3.0}])
    assert r.status_code == 400
    assert "error" in r.get_json()
