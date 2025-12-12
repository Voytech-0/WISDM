import pickle
from statistics import stdev
import numpy as np
import pandas as pd
import polars as pl
import pytest
from pathlib import Path

# Import the module under test
import preprocessing.preprocess as preprocess
from preprocessing.preprocess import (
    load_raw_data_from_local,
    raw_data_into_xyz_series,
    convert_data_to_10_seconds_intervals,
    apply_z_score_normalization,
    _source_target_split,
    build_raw_lazyframe_from_items,
    lazyframe_to_single_10s,
    collect_and_dummify_drop_subject,
    expand_lists,
    generate_splits,
    expand_raw_items_to_expanded_df,
    set_z_score_stats
)
from configs.configs import DataConfig, PathsConfig


def test_load_and_process_raw_data(tmp_path, monkeypatch):
    """
    Tests the chain: load_raw_data_from_local -> raw_data_into_xyz_series
    """
    # 1. Setup mock file system
    # The loader expects filenames like: data_{anything}_{subject}_{sensor}_{device}.txt
    # Example parsing logic in code: str(p).removesuffix(".txt").split('_')[-3:]

    phone_dir = tmp_path / "phone" / "accel"
    phone_dir.mkdir(parents=True)
    watch_dir = tmp_path / "watch" / "gyro"
    watch_dir.mkdir(parents=True)

    # Subject 1, Accel, Phone
    f1 = phone_dir / "data_1_1_accel_phone.txt"
    f1.write_text(
        "1,Walking,100,0.1,0.2,0.3;\n"
        "1,Walking,101,0.2,0.3,0.4;\n"
    )

    # Subject 2, Gyro, Watch
    f2 = watch_dir / "data_2_2_gyro_watch.txt"
    f2.write_text(
        "2,Jogging,200,1.0,1.1,1.2;\n"
        "2,Jogging,201,1.3,1.4,1.5;\n"
    )

    # 2. Monkeypatch DataConfig to point to tmp_path
    monkeypatch.setattr(DataConfig, "data_path_prefix", str(tmp_path))

    # 3. Test Loading
    lf_raw = load_raw_data_from_local()
    assert lf_raw is not None

    # 4. Test Series Aggregation
    lf_grouped = raw_data_into_xyz_series(lf_raw)
    df = lf_grouped.collect()

    expected_cols = {"label", "x", "y", "z", "sensor", "device", "subject_id"}
    assert expected_cols.issubset(set(df.columns))
    assert df.height == 2

    # Verify content
    # Sort to ensure order: usually phone/accel comes before watch/gyro alphabetically or by chance
    df = df.sort(["subject_id"])

    # Row 0: Subject 1
    row_1 = df.filter(pl.col("subject_id") == "1").row(0, named=True)
    assert row_1["device"] == "phone"
    assert row_1["sensor"] == "accel"
    np.testing.assert_allclose(row_1["x"], [0.1, 0.2])
    np.testing.assert_allclose(row_1["y"], [0.2, 0.3])

    # Row 1: Subject 2
    row_2 = df.filter(pl.col("subject_id") == "2").row(0, named=True)
    assert row_2["device"] == "watch"
    assert row_2["sensor"] == "gyro"
    np.testing.assert_allclose(row_2["x"], [1.0, 1.3])


def test_convert_data_to_10_seconds_intervals(monkeypatch):
    # Mock DataConfig
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
            out.collect().to_pandas().sort_values("x"),  # Sort to ensure deterministic comparison
            case["expected"].collect().to_pandas().sort_values("x"),
            check_like=True
        )


def test_apply_z_score_normalization(monkeypatch, tmp_path):
    """
    Test normalization.
    1. Test training mode (calculating stats).
    2. Test inference mode (loading stats).
    """
    stats_file = tmp_path / "z_score_stats.pkl"
    monkeypatch.setattr(PathsConfig, "z_score_params_path", str(stats_file))

    # Data: Two groups (phone/accel) and (watch/gyro)
    # Group 1: x=[1,2,3], mean=2, std=1
    # Group 2: x=[10, 20, 30], mean=20, std=10
    lf_input = pl.LazyFrame({
        "subject_id": ["1", "2"],
        "device": ["phone", "watch"],
        "sensor": ["accel", "gyro"],
        "label": ["A", "B"],
        "x": [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
        "y": [[2.0, 3.0, 4.0], [10.0, 20.0, 30.0]],  # Just dummy values
        "z": [[3.0, 4.0, 5.0], [10.0, 20.0, 30.0]],
    })

    # --- Test Train Mode ---
    lf_train_out = apply_z_score_normalization(lf_input, train=True)
    df_train = lf_train_out.collect()

    # Check if stats file was created
    assert stats_file.exists()

    # Check normalization for phone/accel (Row 0)
    # x was [1, 2, 3] -> mean=2, std=1 -> normalized: [-1, 0, 1]
    row0 = df_train.filter(pl.col("device") == "phone").row(0, named=True)
    np.testing.assert_allclose(row0["x"], [-1.0, 0.0, 1.0])

    # Check normalization for watch/gyro (Row 1)
    # x was [10, 20, 30] -> mean=20, std=10 -> normalized: [-1, 0, 1]
    row1 = df_train.filter(pl.col("device") == "watch").row(0, named=True)
    np.testing.assert_allclose(row1["x"], [-1.0, 0.0, 1.0])

    # --- Test Inference Mode ---
    # Create new data that should use the *same* stats as above.
    # New data for phone/accel: [2, 2, 2].
    # Using saved mean=2, std=1 -> (2-2)/1 = 0
    lf_test = pl.LazyFrame({
        "subject_id": ["3"],
        "device": ["phone"],
        "sensor": ["accel"],
        "label": ["A"],
        "x": [[2.0, 2.0, 2.0]],
        "y": [[2.0, 2.0, 2.0]],
        "z": [[2.0, 2.0, 2.0]],
    })

    lf_infer_out = apply_z_score_normalization(lf_test, train=False)
    df_infer = lf_infer_out.collect()

    row_infer = df_infer.row(0, named=True)
    np.testing.assert_allclose(row_infer["x"], [0.0, 0.0, 0.0])


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
        # Collect first because _source_target_split expects DataFrame, not LazyFrame
        df_in = case["data"].collect()
        features, labels = _source_target_split(df_in, target_prefix="label")

        pd.testing.assert_frame_equal(
            features.to_pandas(),
            case["expected_features"].collect().to_pandas(),
            check_like=True
        )
        pd.testing.assert_frame_equal(
            labels.to_pandas(),
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
                    "subject_id": ["unknown", "unknown"],
                    "device": [None, None],
                    "sensor": [None, None],
                    "label": ["A", "B"],
                    "x": [[1.0, 2.0], [7.0, 8.0]],
                    "y": [[3.0, 4.0], [9.0, 10.0]],
                    "z": [[5.0, 6.0], [11.0, 12.0]],
                }
            )
        }
    ]
    for case in cases:
        out = build_raw_lazyframe_from_items(case["data"])
        # Compare sets of columns or reorder to ensure match
        df_out = out.collect()
        df_exp = case["expected"].collect()

        # Select matching column order
        cols = sorted(df_exp.columns)
        pd.testing.assert_frame_equal(
            df_out.select(cols).to_pandas(),
            df_exp.select(cols).to_pandas(),
            check_like=True
        )


def test_lazyframe_to_single_10s(monkeypatch):
    # Mock Config: Length 5, Freq 1 => Desired length 5
    monkeypatch.setattr(DataConfig, "feature_length_in_seconds", 5)
    monkeypatch.setattr(DataConfig, "frequency", 1)

    cases = [
        # Case 1: Exact length -> No change
        {
            "data": pl.LazyFrame({"x": [[1., 2., 3., 4., 5.]], "y": [[1.] * 5], "z": [[1.] * 5]}),
            "expected_x": [1., 2., 3., 4., 5.]
        },
        # Case 2: Short length -> Pad with 0
        {
            "data": pl.LazyFrame({"x": [[1., 2.]], "y": [[1.] * 2], "z": [[1.] * 2]}),
            "expected_x": [1., 2., 0., 0., 0.]
        },
        # Case 3: Long length -> Truncate
        {
            "data": pl.LazyFrame({"x": [[1., 2., 3., 4., 5., 6.]], "y": [[1.] * 6], "z": [[1.] * 6]}),
            "expected_x": [1., 2., 3., 4., 5.]
        }
    ]

    for case in cases:
        out = lazyframe_to_single_10s(case["data"]).collect()
        x_val = out.get_column("x").to_list()[0]
        # Depending on impl, padding might be int or float, strict comparison handles this?
        # The impl uses `pad_x = [0] * len`, so integers. Input is float.
        # Polars might cast up to Float if the list contains floats.
        assert len(x_val) == 5
        np.testing.assert_allclose(x_val, case["expected_x"])


def test_collect_and_dummify_drop_subject(monkeypatch):
    monkeypatch.setattr(DataConfig, "possible_devices", ["phone", "watch"])
    monkeypatch.setattr(DataConfig, "possible_sensors", ["accel", "gyro"])
    monkeypatch.setattr(DataConfig, "possible_labels", ["A", "B"])

    data = pl.LazyFrame({
        "subject_id": ["1"],
        "device": ["phone"],
        "sensor": ["accel"],
        "label": ["A"],
        "x": [[1.0]], "y": [[1.0]], "z": [[1.0]]
    })

    df = collect_and_dummify_drop_subject(data)

    assert "subject_id" not in df.columns
    # Check that dummies are created
    assert "device_phone" in df.columns
    assert "device_watch" in df.columns
    assert df["device_phone"][0] == 1
    assert df["device_watch"][0] == 0


def test_expand_lists(monkeypatch):
    # Mock Config: Length 3, Freq 1 -> Total 3 columns per feature
    monkeypatch.setattr(DataConfig, "feature_length_in_seconds", 3)
    monkeypatch.setattr(DataConfig, "frequency", 1)

    df = pl.DataFrame({
        "x": [[1, 2, 3], [4, 5, 6]],
        "y": [[10, 20, 30], [40, 50, 60]],
        "other": ["row1", "row2"]
    })

    df_expanded = expand_lists(df, ["x", "y"])

    # Expect columns x_0, x_1, x_2, y_0, y_1, y_2, other
    # Original list columns should be gone
    assert "x" not in df_expanded.columns
    assert "x_0" in df_expanded.columns
    assert "x_2" in df_expanded.columns
    assert "y_0" in df_expanded.columns

    # Check values
    # Row 1, x_0 should be 1
    assert df_expanded["x_0"][0] == 1
    assert df_expanded["x_2"][0] == 3
    assert df_expanded["y_1"][1] == 50


def test_generate_splits(monkeypatch):
    monkeypatch.setattr(DataConfig, "train_split", 0.6)
    monkeypatch.setattr(DataConfig, "validation_split", 0.2)
    # Test split will be remaining 0.2

    # Create 10 rows
    df = pl.DataFrame({
        "feature_1": range(10),
        "label_A": range(10)
    })

    splits = generate_splits(df)

    # Train: 60% of 10 = 6 rows
    assert splits["train"]["X"].height == 6
    assert splits["train"]["y"].height == 6

    # Val: 20% of 10 = 2 rows
    assert splits["validation"]["X"].height == 2

    # Test: Remaining 2 rows
    assert splits["test"]["X"].height == 2

    # Verify content strictness (slicing)
    assert splits["train"]["X"]["feature_1"][0] == 0
    assert splits["test"]["X"]["feature_1"][-1] == 9


def test_expand_raw_items_to_expanded_df(monkeypatch, tmp_path):
    """
    Integration test for the helper function used in inference.
    Steps:
    1. Create dummy z-score stats.
    2. Mock Config.
    3. Run pipeline.
    """
    # 1. Setup Z-score stats
    monkeypatch.setattr(PathsConfig, "z_score_params_path", str(tmp_path / "stats.pkl"))

    # Create stats manually
    stats_df = pl.DataFrame({
        "device": ["phone"],
        "sensor": ["accel"],
        "x_mean": [0.0], "y_mean": [0.0], "z_mean": [0.0],
        "x_std": [1.0], "y_std": [1.0], "z_std": [1.0],
    })
    set_z_score_stats(stats_df.lazy())

    # 2. Mock Config
    monkeypatch.setattr(DataConfig, "feature_length_in_seconds", 2)
    monkeypatch.setattr(DataConfig, "frequency", 1)
    monkeypatch.setattr(DataConfig, "possible_devices", ["phone"])
    monkeypatch.setattr(DataConfig, "possible_sensors", ["accel"])
    monkeypatch.setattr(DataConfig, "possible_labels", [])

    # 3. Input Items
    # Length 3 list, but config says length 2 -> should be truncated to 2
    items = [
        {
            "x": [10.0, 20.0, 30.0],
            "y": [10.0, 20.0, 30.0],
            "z": [10.0, 20.0, 30.0],
            "device": "phone",
            "sensor": "accel",
            "subject_id": "101"
        }
    ]

    # 4. Execute
    df_out = expand_raw_items_to_expanded_df(items)

    # 5. Assertions
    # Should have x_0, x_1 (length 2)
    assert "x_0" in df_out.columns
    assert "x_1" in df_out.columns
    assert "x_2" not in df_out.columns  # Truncated

    # Z-score: value 10, mean 0, std 1 -> 10
    # The stats created above had mean 0, std 1.
    assert df_out["x_0"][0] == 10.0
    assert df_out["x_1"][0] == 20.0

    # Dummies
    assert "device_phone" in df_out.columns
    assert df_out["device_phone"][0] == 1