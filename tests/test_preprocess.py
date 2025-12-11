from statistics import stdev

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sqlalchemy.util import monkeypatch_proxied_specials

import preprocessing.preprocess as preprocess
from configs.configs import DataConfig
from preprocessing.preprocess import (
    raw_data_into_xyz_series,
    convert_data_to_10_seconds_intervals,
    apply_z_score_normalization,
    _source_target_split,
    preprocessing_pipeline_for_wisdm_raw_data,
    build_raw_lazyframe_from_items,
    lazyframe_to_single_10s,
    collect_and_dummify_drop_subject
)


def test_load_data_raw_with_row_per_sensor_device_activity_subject(tmp_path):
    phone_dir = tmp_path / "phone"
    watch_dir = tmp_path / "watch"
    (phone_dir / "accel").mkdir(parents=True)
    (phone_dir / "gyro").mkdir(parents=True)
    (watch_dir / "accel").mkdir(parents=True)
    (watch_dir / "gyro").mkdir(parents=True)

    f1 = phone_dir / "accel" / "data_1_accel_phone.txt"
    f1.write_text(
        "1,Walking,100,0.1,0.2,0.3;\n"
        "1,Walking,101,0.2,0.3,0.4;\n"
    )

    f2 = watch_dir / "gyro" / "data_2_gyro_watch.txt"
    f2.write_text(
        "2,Jogging,200,1.0,1.1,1.2;\n"
        "2,Jogging,201,1.3,1.4,1.5;\n"
    )

    lf = raw_data_into_xyz_series(str(tmp_path))
    assert lf is not None

    df = lf.collect()

    expected_cols = {"label", "x", "y", "z", "sensor", "device", "subject_id"}
    assert expected_cols.issubset(set(df.columns))
    assert df.height == 2

    sensors = df.get_column("sensor").to_list()
    devices = df.get_column("device").to_list()
    subjects = df.get_column("subject_id").to_list()

    # Validate the accel/phone row
    accel_idx = sensors.index("accel")
    assert devices[accel_idx] == "phone"
    assert str(subjects[accel_idx]) == "1"

    x_accel = df.get_column("x")[accel_idx]
    y_accel = df.get_column("y")[accel_idx]
    z_accel = df.get_column("z")[accel_idx]

    np.testing.assert_allclose([float(v) for v in list(x_accel)], [0.1, 0.2])
    np.testing.assert_allclose([float(v) for v in list(y_accel)], [0.2, 0.3])
    np.testing.assert_allclose([float(v) for v in list(z_accel)], [0.3, 0.4])

    gyro_idx = sensors.index("gyro")
    assert devices[gyro_idx] == "watch"
    assert str(subjects[gyro_idx]) == "2"

    np.testing.assert_allclose([float(v) for v in list(df.get_column("x")[gyro_idx])], [1.0, 1.3])
    np.testing.assert_allclose([float(v) for v in list(df.get_column("y")[gyro_idx])], [1.1, 1.4])
    np.testing.assert_allclose([float(v) for v in list(df.get_column("z")[gyro_idx])], [1.2, 1.5])


def test_convert_data_to_10_seconds_intervals(monkeypatch):
    # mock DataConfig.feature_length_in_seconds and DataConfig.frequency
    import preprocessing.preprocess as preprocess

    monkeypatch.setattr(preprocess.DataConfig, "feature_length_in_seconds", 10)
    monkeypatch.setattr(preprocess.DataConfig, "frequency", 1)
    cases = [
        {
            "data": pl.LazyFrame(
                {
                    "subject_id": ["1"],
                    "device": ["phone"],
                    "sensor": ["accel"],
                    "label": ["A"],
                    "x": [[float(i) for i in range(20)]],
                    "y": [[float(i + 0.1) for i in range(20)]],
                    "z": [[float(i + 0.2) for i in range(20)]],
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

def test_apply_z_score_normalization():
    cases = [
        {
            "data": pl.LazyFrame(
                {
                    "subject_id": ["1", "1"],
                    "device": ["phone", "phone"],
                    "sensor": ["accel", "accel"],
                    "label": ["A", "A"],
                    "x": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    "y": [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
                    "z": [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                }
            )
        }
    ]
    for case in cases:
        out = apply_z_score_normalization(case["data"])
        df = out.collect().to_pandas()
        assert stdev(df['x'].explode()) - 1.0 < 1e-6
        assert stdev(df['y'].explode()) - 1.0 < 1e-6
        assert stdev(df['z'].explode()) - 1.0 < 1e-6
        assert abs(df['x'].explode().mean()) < 1e-6
        assert abs(df['y'].explode().mean()) < 1e-6
        assert abs(df['z'].explode().mean()) < 1e-6


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
            "expected_features": pl.LazyFrame(
                {
                    "feature1": [1, 2],
                    "feature2": [3, 4],
                }
            ),
            "expected_labels": pl.LazyFrame(
                {
                    "label_A": [0, 1],
                    "label_B": [1, 0],
                }
            ),
        }
    ]
    for case in cases:
        features, labels = _source_target_split(case["data"], target_prefix="label")
        pd.testing.assert_frame_equal(
            features.collect().to_pandas(),
            case["expected_features"].collect().to_pandas(),
            check_like=True
        )
        pd.testing.assert_frame_equal(
            labels.collect().to_pandas(),
            case["expected_labels"].collect().to_pandas(),
            check_like=True
        )

def test_build_raw_lazyframe_from_items():
    cases = [
        {
            "data": [
                {"x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0], "label": "A"},
                {"x": [7.0, 8.0], "y": [9.0, 10.0], "z": [11.0, 12.0], "label": "B"},
            ],
            "expected": pl.LazyFrame(
                {
                    "x": [[1.0, 2.0], [7.0, 8.0]],
                    "y": [[3.0, 4.0], [9.0, 10.0]],
                    "z": [[5.0, 6.0], [11.0, 12.0]],
                    "label": ["A", "B"],
                    "subject_id": ["unknown", "unknown"],
                    "device": [None, None],
                    "sensor": [None, None],
                }
            )
        },
        {
            "data": [
                {"x": [0.0, 1.0], "y": [1.0, 2.0], "z": [2.0, 3.0]},
            ],
            "expected": pl.LazyFrame(
                {
                    "x": [[0.0, 1.0]],
                    "y": [[1.0, 2.0]],
                    "z": [[2.0, 3.0]],
                    "label": [None],
                    "subject_id": ["unknown"],
                    "device": [None],
                    "sensor": [None],
                }
            )
        }
    ]
    for case in cases:
        out = build_raw_lazyframe_from_items(case["data"])
        pd.testing.assert_frame_equal(
            out.collect().to_pandas(),
            case["expected"].collect().to_pandas(),
            check_like=True
        )

def test_lazyframe_to_single_10s(monkeypatch):
    cases = [
        {
            "data": pl.LazyFrame(
            {
                "subject_id": ["1"],
                "device": ["phone"],
                "sensor": ["accel"],
                "label": ["A"],
                "x": [[float(i) for i in range(20)]],
                "y": [[float(i + 0.1) for i in range(20)]],
                "z": [[float(i + 0.2) for i in range(20)]],
            }
            ),
            "expected": pl.LazyFrame(
                {
                    "subject_id": ["1"],
                    "device": ["phone"],
                    "sensor": ["accel"],
                    "label": ["A"],
                    "x": [[float(i) for i in range(10)]],
                    "y": [[float(i + 0.1) for i in range(10)]],
                    "z": [[float(i + 0.2) for i in range(10)]],
                }
            )
        },
        {
            "data": pl.LazyFrame(
                {
                    "subject_id": ["2"],
                    "device": ["watch"],
                    "sensor": ["gyro"],
                    "label": ["B"],
                    "x": [[float(i) for i in range(5)]],
                    "y": [[float(i + 0.1) for i in range(5)]],
                    "z": [[float(i + 0.2) for i in range(5)]],
                }
            ),
            "expected": pl.LazyFrame(
                {
                    "subject_id": ["2"],
                    "device": ["watch"],
                    "sensor": ["gyro"],
                    "label": ["B"],
                    "x": [[0, 1, 2, 3, 4, 0, 0, 0, 0, 0]],
                    "y": [[0.1, 1.1, 2.1, 3.1, 4.1, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    "z": [[0.2, 1.2, 2.2, 3.2, 4.2, 0.0, 0.0, 0.0, 0.0, 0.0]],
                }
            )
        }
    ]

    # patch the DataConfig object used in the preprocessing module
    monkeypatch.setattr(preprocess.DataConfig, "feature_length_in_seconds", 10)
    monkeypatch.setattr(preprocess.DataConfig, "frequency", 1)

    for case in cases:
        out = lazyframe_to_single_10s(case["data"])
        print("OUT done", flush=True)
        df = out.collect()
        print("DF done", flush=True)
        pd.testing.assert_frame_equal(
            df.to_pandas(),
            case["expected"].collect().to_pandas(),
            check_like=True
        )


def test_collect_and_dummify_drop_subject(monkeypatch):
    cases = [
        {
            "data": pl.LazyFrame(
                {
                    "subject_id": ["1", "2"],
                    "device": ["phone", "watch"],
                    "sensor": ["accel", "gyro"],
                    "label": ["A", "A"],
                    "x": [[1.0, 2.0], [3.0, 4.0]],
                    "y": [[2.0, 3.0], [4.0, 5.0]],
                    "z": [[3.0, 4.0], [5.0, 6.0]],
                }
            ),
            "expected_columns": {"device_phone", "device_watch", "sensor_accel", "sensor_gyro", "label_A", "label_B", "x", "y", "z"},
            "expected_height": 2,
            "expected_dummies_with_1": [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0]]
        },
        {
            "data": pl.LazyFrame(
                {
                    "subject_id": ["3"],
                    "device": ["phone"],
                    "sensor": ["accel"],
                    "label": ["B"],
                    "x": [[5.0, 6.0]],
                    "y": [[6.0, 7.0]],
                    "z": [[7.0, 8.0]],
                }
            ),
            "expected_columns": {"device_phone", "device_watch", "sensor_accel", "sensor_gyro", "label_A", "label_B", "x", "y", "z"},
            "expected_height": 1,
            "expected_dummies_with_1": [[1, 0, 1, 0, 0, 1]]
        }
    ]
    monkeypatch.setattr(preprocess.DataConfig, "possible_devices", ["phone", "watch"])
    monkeypatch.setattr(preprocess.DataConfig, "possible_sensors", ["accel", "gyro"])
    monkeypatch.setattr(preprocess.DataConfig, "possible_labels", ["A", "B"])

    for case in cases:
        df = collect_and_dummify_drop_subject(case["data"])
        assert set(df.columns) == set(case["expected_columns"])
        assert df.height == case["expected_height"]
        assert "subject_id" not in df.columns
        for i, row in enumerate(df.to_dicts()):
            dummies = [
                row["device_phone"],
                row["device_watch"],
                row["sensor_accel"],
                row["sensor_gyro"],
                row["label_A"],
                row["label_B"],
            ]
            assert dummies == case["expected_dummies_with_1"][i]