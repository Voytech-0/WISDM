import sys
import time
import pytest
import requests
import argparse
from unittest.mock import MagicMock

import scripts.fake_client as fake_client
from scripts.fake_client import (
    generate_series,
    make_item,
    send_payload,
    parse_args,
    main
)


# --- Helper Classes ---

class MockResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json



@pytest.fixture
def mock_preprocess(monkeypatch):
    mp = MagicMock()
    mp_loader = MagicMock()
    mp_df = MagicMock()

    dummy_data = [{
        "device": "phone", "sensor": "accel",
        "x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0]
    }]
    mp_df.to_dicts.return_value = dummy_data
    mp_loader.collect.return_value = mp_df

    mp.load_raw_data_from_local.return_value = mp_loader
    mp.raw_data_into_xyz_series.return_value = mp_loader
    mp.convert_data_to_fixed_length_intervals.return_value = mp_loader

    monkeypatch.setitem(sys.modules, "src.preprocessing.preprocess", mp)
    monkeypatch.setattr(fake_client, "preprocess", mp, raising=False)
    return mp


# --- Tests ---

def test_generate_series():
    """Test that series generation produces lists of floats of correct length."""
    length = 50
    series = generate_series(length, base_freq=1.0)
    assert len(series) == length
    assert isinstance(series, list)
    assert all(isinstance(x, float) for x in series)


def test_make_item():
    """Test payload creation structure."""
    item = make_item("phone", "accel", 100)

    assert item["device"] == "phone"
    assert item["sensor"] == "accel"
    assert "x" in item
    assert "y" in item
    assert "z" in item

    assert len(item["x"]) == 100
    assert len(item["y"]) == 100
    assert len(item["z"]) == 100


def test_send_payload_success(monkeypatch):
    """Test successful POST request."""

    def mock_post(url, json, timeout):
        assert url == "http://localhost:8000/predict"
        assert "device" in json
        return MockResponse(200, {"status": "success"})

    monkeypatch.setattr(requests, "post", mock_post)

    resp = send_payload("http://localhost:8000/predict", {"device": "test"})
    assert resp.status_code == 200


def test_send_payload_retry_logic(monkeypatch):
    """Test that the client retries on 5xx errors and eventually succeeds."""
    attempts = 0

    def mock_post(url, json, timeout):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return MockResponse(502)  # Bad Gateway
        return MockResponse(200, {"status": "ok"})

    # Mock sleep to speed up test
    monkeypatch.setattr(time, "sleep", lambda x: None)
    monkeypatch.setattr(requests, "post", mock_post)

    resp = send_payload("http://dummy", {}, max_retries=3)
    assert resp.status_code == 200
    assert attempts == 3


def test_send_payload_failure(monkeypatch):
    """Test that the client raises exception after max retries."""

    def mock_post(*args, **kwargs):
        raise requests.RequestException("Connection refused")

    monkeypatch.setattr(time, "sleep", lambda x: None)
    monkeypatch.setattr(requests, "post", mock_post)

    with pytest.raises(requests.RequestException):
        send_payload("http://dummy", {}, max_retries=2)


def test_parse_args(monkeypatch):
    """Test argument parsing defaults and overrides."""
    # Test Defaults
    monkeypatch.setattr(sys, "argv", ["fake_client.py"])
    args = parse_args()
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.series_len == 200

    # Test Overrides
    monkeypatch.setattr(sys, "argv", ["fake_client.py", "--port", "9000", "--series-len", "50", "--use-local"])
    args = parse_args()
    assert args.port == 9000
    assert args.series_len == 50
    assert args.use_local is True


def test_main_synthetic_flow(monkeypatch):
    """
    Test the main loop with synthetic data generation.
    Mocks network and time to run a single iteration.
    """
    # 1. Mock Args
    mock_args = argparse.Namespace(
        host="localhost", port=8000,
        interval_min=0.1, interval_max=0.2,
        batch_size=1, series_len=10,
        device="phone", sensor="accel",
        count=1, seed=42,
        verbose=False, use_local=False
    )
    monkeypatch.setattr(fake_client, "parse_args", lambda: mock_args)

    # 2. Mock Network
    posted_payloads = []

    def mock_send(url, payload, **kwargs):
        posted_payloads.append(payload)
        return MockResponse(200, {"prediction": "Walk"})

    monkeypatch.setattr(fake_client, "send_payload", mock_send)

    # 3. Mock Sleep
    monkeypatch.setattr(time, "sleep", lambda x: None)

    # 4. Run Main
    main()

    # 5. Assertions
    assert len(posted_payloads) == 1
    item = posted_payloads[0]
    assert item["device"] == "phone"
    assert len(item["x"]) == 10


def test_main_local_data_flow(monkeypatch, mock_preprocess):
    """
    Test for --use-local is True.
    """
    mock_args = argparse.Namespace(
        host="localhost", port=8000,
        interval_min=0.1, interval_max=0.1,
        batch_size=1, series_len=10,
        device="phone", sensor="accel",
        count=1, seed=None,
        verbose=False, use_local=True
    )
    monkeypatch.setattr(fake_client, "parse_args", lambda: mock_args)

    posted_payloads = []

    def mock_send(url, payload, **kwargs):
        posted_payloads.append(payload)
        return MockResponse(200, {"prediction": "Run"})

    monkeypatch.setattr(fake_client, "send_payload", mock_send)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    monkeypatch.setattr(sys, "argv", ["fake_client.py", "--use-local"])
    main()
    assert mock_preprocess.load_raw_data_from_local.called
    assert len(posted_payloads) == 1
    assert posted_payloads[0]["x"] == [1.0, 2.0]
    assert posted_payloads[0]["device"] == "phone"
