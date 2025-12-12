# ensure the project root is on sys.path so `import src...` works when pytest runs
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
from click.testing import CliRunner
from unittest.mock import Mock, patch
import src.main_api as main_api
from types import SimpleNamespace

from src.main_api import create_app, InputPreparer, ModelLoader


class DummyModel(torch.nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        # mimic saved hparams used by InputPreparer
        self.hparams = {'input_size': None}

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
    def __init__(self, model: DummyModel):
        self._model = model
        # provide a checkpoint_path attribute to satisfy logging in reload endpoint
        self.checkpoint_path = "stub"

    def load(self, force_reload: bool = False):
        return self._model


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
        result = runner.invoke(main_api.cli, ['serve', '--host', host, port])

    assert result.exit_code == 0
    assert MockLoaderCls.called, "ModelLoader should be instantiated"
    assert MockPreparerCls.called, "InputPreparer should be instantiated"
    mock_create_app.assert_called_once_with(loader_instance, preparer_instance)
    app_mock.run.assert_called_once_with(host=host, port=int(port))

class FakeLittleMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lr=None):
        super().__init__()
        # minimal layers so state_dict() has content
        self.fc = torch.nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

def test_modelLoader_load_cached(tmp_path):


    hparams = {
        'input_size': 2,
        'hidden_size': 4,
        'num_layers': 1,
        'num_classes': 3,
        'lr': 0.01,
    }
    model = FakeLittleMLP(
        input_size=hparams['input_size'],
        hidden_size=hparams['hidden_size'],
        num_layers=hparams['num_layers'],
        num_classes=hparams['num_classes'],
        lr=hparams['lr'],
    )

    ckpt = {'model_state_dict': model.state_dict(), 'hparams': hparams}
    ckpt_path = tmp_path / 'ckpt.pth'
    torch.save(ckpt, str(ckpt_path))

    loaded = ModelLoader._load_cached(str(ckpt_path), FakeLittleMLP)
    assert hasattr(loaded, 'hparams')
    assert int(loaded.hparams['input_size']) == hparams['input_size']
    assert isinstance(loaded, FakeLittleMLP)


def test_modelLoader_load(tmp_path):

    hparams = {
        'input_size': 2,
        'hidden_size': 4,
        'num_layers': 1,
        'num_classes': 3,
        'lr': 0.01,
    }
    model = FakeLittleMLP(
        input_size=hparams['input_size'],
        hidden_size=hparams['hidden_size'],
        num_layers=hparams['num_layers'],
        num_classes=hparams['num_classes'],
        lr=hparams['lr'],
    )

    ckpt = {'model_state_dict': model.state_dict(), 'hparams': hparams}
    ckpt_path = tmp_path / 'ckpt2.pth'
    torch.save(ckpt, str(ckpt_path))

    loader = ModelLoader(checkpoint_path=str(ckpt_path), model_cls=model.__class__)
    m1 = loader.load()
    assert isinstance(m1, model.__class__)

    m2 = loader.load(force_reload=True)
    assert isinstance(m2, model.__class__)
    assert m1 is not m2  # should be different instances


def test_input_preparer_is_raw():
    prep = InputPreparer()
    assert prep._is_raw({'x': [1], 'y': [2], 'z': [3]})
    assert not prep._is_raw({'a': 1, 'b': 2})
    assert not prep._is_raw(None)


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

    model = SimpleNamespace()
    model.hparams = SimpleNamespace(input_size=1)
    with pytest.raises(ValueError):
        prep.prepare(data, model=model)
