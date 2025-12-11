import pytest
import torch
import polars as pl

from src.main_api import InputPreparer


def test_raw_x_y_z_input(monkeypatch):
    preparer = InputPreparer()
    raw = [
        {"x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0]},
        {"x": [7.0, 8.0], "y": [9.0, 10.0], "z": [11.0, 12.0]},
    ]

    def fake_expand(items):
        # return a deterministic expanded DataFrame with 6 feature columns
        return pl.DataFrame([
            {"x_0": 1, "x_1": 2, "y_0": 3, "y_1": 4, "z_0": 5, "z_1": 6},
            {"x_0": 7, "x_1": 8, "y_0": 9, "y_1": 10, "z_0": 11, "z_1": 12},
        ])

    import src.main_api as main_api
    monkeypatch.setattr(main_api, "expand_raw_items_to_expanded_df", fake_expand)

    t = preparer.prepare(raw)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (2, 6)
    # spot check ordering and values
    assert t[0, 0].item() == 1
    assert t[0, 5].item() == 6
    assert t[1, 0].item() == 7
    assert t[1, 5].item() == 12


def test_empty_payload_raises():
    preparer = InputPreparer()
    with pytest.raises(ValueError) as exc:
        preparer.prepare([])
    assert "Empty input payload" in str(exc.value)


def test_non_dict_item_raises():
    preparer = InputPreparer()
    with pytest.raises(ValueError) as exc:
        preparer.prepare([123])
    assert "Each input must be an object with feature keys or x/y/z lists" in str(exc.value)




