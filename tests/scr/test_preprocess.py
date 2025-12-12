import pickle
import numpy as np
import pandas as pd
import polars as pl
import pytest
from statistics import stdev
from pathlib import Path

import preprocessing.preprocess as preprocess
from configs.configs import DataConfig, PathsConfig
from preprocessing.preprocess import (
    load_raw_data_from_local,
    raw_data_into_xyz_series,
    convert_data_to_10_seconds_intervals,
    apply_z_score_normalization,
    _source_target_split,
    preprocessing_pipeline_for_wisdm_raw_data,
    build_raw_lazyframe_from_items,
    lazyframe_to_single_10s,
    collect_and_dummify_drop_subject,
    expand_lists,
    generate_splits,
    expand_raw_items_to_expanded_df,
    set_z_score_stats
)


def test_wait_for_mysql(monkeypatch):
    """
    Test wait_for_mysql with a mocked engine and connection.
    Cases:
    1. Immediate success.
    2. Success after failure (retries).
    3. Timeout failure.
    """
    from sqlalchemy.exc import OperationalError
    import time

    # Mock Engine and Connection
    class MockConnection:
        def __init__(self, succeed_on_attempt=1):
            self.attempt = 0
            self.succeed_on_attempt = succeed_on_attempt

        def execute(self, statement):
            self.attempt += 1
            if self.attempt < self.succeed_on_attempt:
                raise OperationalError("Mock failure", None, None)
            return "Success"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockEngine:
        def __init__(self, succeed_on_attempt=1):
            self.conn = MockConnection(succeed_on_attempt)

        def connect(self):
            return self.conn

    # Case 1: Immediate Success
    engine_success = MockEngine(succeed_on_attempt=1)
    preprocess.wait_for_mysql(engine_success, timeout=5, interval=0.1)

    # Case 2: Success after 2 failures (succeeds on 3rd)
    engine_retry = MockEngine(succeed_on_attempt=3)
    start_time = time.time()
    preprocess.wait_for_mysql(engine_retry, timeout=5, interval=0.1)
    # Ensure it actually waited (approx 0.1s + 0.2s = 0.3s)
    assert time.time() - start_time >= 0.2

    # Case 3: Timeout
    engine_fail = MockEngine(succeed_on_attempt=99)
    with pytest.raises(OperationalError):
        preprocess.wait_for_mysql(engine_fail, timeout=0.5, interval=0.1)


def test_load_raw_data_from_mysql(monkeypatch):
    """
    Test loading data from MySQL using mocked SQLAlchemy engine/connection.
    """
    # 1. Mock Configs
    monkeypatch.setattr(preprocess.MySQLConfig, "user", "test_user")
    monkeypatch.setattr(preprocess.MySQLConfig, "host", "localhost")
    monkeypatch.setattr(preprocess.MySQLConfig, "port", 3306)
    monkeypatch.setattr(preprocess.MySQLConfig, "db", "test_db")
    monkeypatch.setattr(preprocess.DataConfig, "db_connect_timeout", 1)
    monkeypatch.setattr(preprocess.DataConfig, "db_read_chunk_size", 2)

    class MockResult:
        def __init__(self, data, columns):
            self.data = data
            self.keys_val = columns
            self.iter_data = iter([data[i:i + 2] for i in range(0, len(data), 2)])  # chunk size 2

        def keys(self):
            return self.keys_val

        def fetchmany(self, size=None):
            try:
                return next(self.iter_data)
            except StopIteration:
                return []

    class MockConnection:
        def execute(self, *args, **kwargs):
            return MockResult(
                data=[
                    ("1", "Walk", 100, 0.1, 0.1, 0.1, "accel", "phone"),
                    ("1", "Walk", 101, 0.2, 0.2, 0.2, "accel", "phone"),
                    ("\n2", "Run", 200, 0.5, 0.5, 0.5, "gyro", "watch")
                ],
                columns=["subject_id", "label", "timestamp", "x", "y", "z", "sensor", "device"]
            )

        def execution_options(self, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockEngine:
        def connect(self):
            return MockConnection()

    monkeypatch.setattr(preprocess, "create_engine", lambda url: MockEngine())
    monkeypatch.setattr(preprocess, "wait_for_mysql", lambda engine, timeout: None)

    from preprocessing.preprocess import load_raw_data_from_mysql
    lf = load_raw_data_from_mysql()
    df = lf.collect()

    assert df.height == 3
    assert df["subject_id"][2] == "2"
    assert df["device"][0] == "phone"
    assert df["x"][0] == 0.1


def test_load_raw_data_from_local(tmp_path, monkeypatch):
    """
    Tests loading from disk separately from processing.
    """
    phone_dir = tmp_path / "phone" / "accel"
    phone_dir.mkdir(parents=True)

    f1 = phone_dir / "data_1_1_accel_phone.txt"
    f1.write_text("1,Walking,100,0.1,0.2,0.3;\n")

    monkeypatch.setattr(DataConfig, "data_path_prefix", str(tmp_path))

    lf = load_raw_data_from_local()
    assert lf is not None

    df = lf.collect()
    assert df.height == 1
    assert "x" in df.columns
    assert df["sensor"][0] == "accel"
    assert df["device"][0] == "phone"


def test_raw_data_into_xyz_series():
    """
    Test aggregation of raw rows into lists grouped by subject/device/sensor/label.
    Input must be a LazyFrame.
    """
    cases = [
        {
            "name": "basic_aggregation",
            "input_lf": pl.LazyFrame({
                "subject_id": ["1", "1"],
                "device": ["phone", "phone"],
                "sensor": ["accel", "accel"],
                "label": ["Walk", "Walk"],
                "timestamp": [100, 101],
                "x": [0.1, 0.2],
                "y": [0.2, 0.3],
                "z": [0.3, 0.4]
            }),
            "expected_data": {
                "subject_id": ["1"],
                "x": [[0.1, 0.2]],
                "y": [[0.2, 0.3]],
                "z": [[0.3, 0.4]]
            }
        },
        {
            "name": "sorting_by_timestamp",
            "input_lf": pl.LazyFrame({
                "subject_id": ["1", "1"],
                "device": ["phone", "phone"],
                "sensor": ["accel", "accel"],
                "label": ["Walk", "Walk"],
                "timestamp": [200, 100],
                "x": [0.2, 0.1],
                "y": [0.2, 0.3],
                "z": [0.2, 0.2]
            }),
            "expected_data": {
                "x": [[0.1, 0.2]],
                "y": [[0.3, 0.2]],

            }
        }
    ]

    for case in cases:
        lf_out = raw_data_into_xyz_series(case["input_lf"])
        df_out = lf_out.collect()

        for col, expected_vals in case["expected_data"].items():
            actual_vals = df_out.get_column(col).to_list()
            if isinstance(actual_vals[0], list):
                for i, row_list in enumerate(actual_vals):
                    np.testing.assert_allclose(row_list, expected_vals[i],
                                               err_msg=f"Case {case['name']} failed on {col}")
            else:
                assert actual_vals == expected_vals


def test_convert_data_to_10_seconds_intervals(monkeypatch):
    monkeypatch.setattr(DataConfig, "feature_length_in_seconds", 10)
    monkeypatch.setattr(DataConfig, "frequency", 1)

    cases = [
        {
            "data": pl.LazyFrame(
                {
                    "subject_id": ["1"],
                    "device": ["phone"],
                    "sensor": ["accel"],
                    "label": ["A"],
                    "x": [[float(i) for i in range(25)]],
                    "y": [[float(i + 0.1) for i in range(25)]],
                    "z": [[float(i + 0.2) for i in range(25)]],
                }
            ),
            "expected": pl.LazyFrame(
                {
                    "subject_id": ["1", "1"],
                    "device": ["phone", "phone"],
                    "sensor": ["accel", "accel"],
                    "label": ["A", "A"],
                    "x": [[float(i) for i in range(10)], [float(i) for i in range(10, 20)]],
                    "y": [[float(i + 0.1) for i in range(10)], [float(i + 0.1) for i in range(10, 20)]],
                    "z": [[float(i + 0.2) for i in range(10)], [float(i + 0.2) for i in range(10, 20)]],
                }
            )
        }
    ]
    for case in cases:
        out = convert_data_to_10_seconds_intervals(case["data"])
        pd.testing.assert_frame_equal(
            out.collect().to_pandas(),
            case["expected"].collect().to_pandas(),
            check_like=True
        )


def test_apply_z_score_normalization(monkeypatch, tmp_path):
    stats_path = tmp_path / "z_score_stats.pkl"
    monkeypatch.setattr(PathsConfig, "z_score_params_path", str(stats_path))

    input_lf = pl.LazyFrame({
        "subject_id": ["1", "2"],
        "device": ["devA", "devB"],
        "sensor": ["sens1", "sens1"],
        "label": ["L", "L"],
        "x": [[0.0, 10.0, 20.0], [100.0, 100.0, 100.0]],
        "y": [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]],
        "z": [[2.0, 2.0, 2.0], [6.0, 6.0, 6.0]],
    })

    cases = [
        {
            "mode": "train",
            "train_flag": True,
            "input": input_lf,
            "expected_x_first_vals": [-1.0, 0.0, 1.0],
            "check_file_exists": True
        },
        {
            "mode": "inference",
            "train_flag": False,
            "input": pl.LazyFrame({
                "subject_id": ["3"],
                "device": ["devA"],
                "sensor": ["sens1"],
                "label": ["L"],
                "x": [[10.0, 10.0, 10.0]],
                "y": [[1.0, 1.0, 1.0]],
                "z": [[2.0, 2.0, 2.0]]
            }),
            "expected_x_first_vals": [0.0, 0.0, 0.0],
            "check_file_exists": True
        }
    ]

    # Run TRAIN case first to generate the file
    train_case = cases[0]
    out_train = apply_z_score_normalization(train_case["input"], train=True)
    df_train = out_train.collect()

    assert stats_path.exists()

    # Verify Train output (DevA is row 0)
    row_0_x = df_train.filter(pl.col("device") == "devA").get_column("x")[0]
    np.testing.assert_allclose(row_0_x, train_case["expected_x_first_vals"])

    # Run INFERENCE case
    infer_case = cases[1]
    out_infer = apply_z_score_normalization(infer_case["input"], train=False)
    df_infer = out_infer.collect()

    row_infer_x = df_infer.get_column("x")[0]
    np.testing.assert_allclose(row_infer_x, infer_case["expected_x_first_vals"])


def test_expand_lists(monkeypatch):
    monkeypatch.setattr(DataConfig, "feature_length_in_seconds", 3)
    monkeypatch.setattr(DataConfig, "frequency", 1)

    cases = [
        {
            "data": pl.DataFrame({
                "x": [[1, 2, 3]],
                "y": [[10, 20, 30]]
            }),
            "cols_to_expand": ["x", "y"],
            "expected_columns": ["x_0", "x_1", "x_2", "y_0", "y_1", "y_2"],
            "expected_val_x_2": 3
        }
    ]

    for case in cases:
        out = expand_lists(case["data"], case["cols_to_expand"])
        assert set(case["expected_columns"]).issubset(set(out.columns))
        assert "x" not in out.columns
        assert out["x_2"][0] == case["expected_val_x_2"]


def test_source_target_split():
    cases = [
        {
            "data": pl.LazyFrame(
                {
                    "feature1": [1, 2],
                    "feature2": [3, 4],
                    "label_A": [0, 1],
                    "label_B": [1, 0],
                }
            ),
            "expected_features": ["feature1", "feature2"],
            "expected_labels": ["label_A", "label_B"],
        }
    ]
    for case in cases:
        df = case["data"].collect()
        X, y = _source_target_split(df, target_prefix="label")
        assert sorted(X.columns) == sorted(case["expected_features"])
        assert sorted(y.columns) == sorted(case["expected_labels"])


def test_generate_splits(monkeypatch):
    monkeypatch.setattr(DataConfig, "train_split", 0.5)
    monkeypatch.setattr(DataConfig, "validation_split", 0.3)

    # 10 rows. Train=5, Val=3, Test=2
    df = pl.DataFrame({"f": range(10), "label_x": range(10)})

    splits = generate_splits(df)

    assert splits["train"]["X"].height == 5
    assert splits["validation"]["X"].height == 3
    assert splits["test"]["X"].height == 2


def test_lazyframe_to_single_10s(monkeypatch):
    monkeypatch.setattr(DataConfig, "feature_length_in_seconds", 5)
    monkeypatch.setattr(DataConfig, "frequency", 1)

    cases = [
        {
            "input": pl.LazyFrame({"x": [[1., 2.]], "y": [[1.]], "z": [[1.]]}),
            "expected_len": 5,
            "expected_x_tail": 0.0
        },
        {
            "input": pl.LazyFrame({"x": [[1., 2., 3., 4., 5., 6.]], "y": [[1.]], "z": [[1.]]}),
            "expected_len": 5,
            "expected_x_last": 5.0
        }
    ]

    for case in cases:
        out = lazyframe_to_single_10s(case["input"]).collect()
        x_list = out["x"][0]
        assert len(x_list) == case["expected_len"]

        if len(case["input"].collect()["x"][0]) < 5:
            assert x_list[-1] == case["expected_x_tail"]
        else:
            assert x_list[-1] == case["expected_x_last"]


def test_expand_raw_items_to_expanded_df(monkeypatch, tmp_path):

    monkeypatch.setattr(PathsConfig, "z_score_params_path", str(tmp_path / "stats.pkl"))
    stats_df = pl.DataFrame({
        "device": ["phone"], "sensor": ["accel"],
        "x_mean": [0.0], "y_mean": [0.0], "z_mean": [0.0],
        "x_std": [1.0], "y_std": [1.0], "z_std": [1.0],
    })
    set_z_score_stats(stats_df.lazy())

    monkeypatch.setattr(DataConfig, "feature_length_in_seconds", 2)
    monkeypatch.setattr(DataConfig, "frequency", 1)
    monkeypatch.setattr(DataConfig, "possible_devices", ["phone"])
    monkeypatch.setattr(DataConfig, "possible_sensors", ["accel"])
    monkeypatch.setattr(DataConfig, "possible_labels", [])

    items = [{
        "x": [10., 10., 10.],
        "y": [10., 10., 10.],
        "z": [10., 10., 10.],
        "device": "phone", "sensor": "accel"
    }]

    df_out = expand_raw_items_to_expanded_df(items)

    assert df_out.height == 1
    assert "x_0" in df_out.columns
    assert "x_1" in df_out.columns
    assert "x_2" not in df_out.columns
    assert df_out["x_0"][0] == 10.0
    assert df_out["device_phone"][0] == 1