from statistics import stdev

import numpy as np
import pandas as pd
import polars as pl
import pytest
from preprocessing.preprocess import (
    raw_data_into_xyz_series,
    convert_data_to_10_seconds_intervals,
    apply_z_score_normalization,
    _source_target_split,
    preprocessing_pipeline_for_wisdm_raw_data
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
    case_1_length = 20
    cases = [
        {
            "data": pl.LazyFrame(
                {
                    "subject_id": ["1"],
                    "device": ["phone"],
                    "sensor": ["accel"],
                    "label": ["A"],
                    "x": [[float(i) for i in range(case_1_length)]],
                    "y": [[float(i + 0.1) for i in range(case_1_length)]],
                    "z": [[float(i + 0.2) for i in range(case_1_length)]],
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
        