import numpy as np
import pandas as pd
from pathlib import Path
import sys
import polars as pl

# When tests import this file directly, the repo root may not be on sys.path.
# Prepend the repository root so `configs.configs` can be resolved.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from configs.configs import DataConfig


def load_data_raw_with_row_per_sensor_device_activity_subject(prefix_path: str) -> pl.LazyFrame:
    file_header = ['subject_id', 'label', 'timestamp', 'x', 'y', 'z']

    paths = list(Path(prefix_path).rglob("data_*.txt"))

    lf_all = []
    for p in paths:
        lf = pl.scan_csv(str(p), has_header=False, new_columns=file_header, eol_char=";")
        subject_id, sensor, device = str(p).removesuffix(".txt").split('_')[-3:]
        lf = (
            lf
            .drop(["subject_id", "timestamp"])
            .group_by("label", maintain_order=True).all()
            .with_columns(
                (pl.lit(sensor).alias("sensor")),
                (pl.lit(device).alias("device")),
                (pl.lit(subject_id).alias("subject_id")),
            )
        )
        lf_all.append(lf)

    lf_all = pl.concat(lf_all).drop_nulls()

    return lf_all


def convert_data_to_10_seconds_intervals(lf: pl.LazyFrame) -> pl.LazyFrame:
    # for each 'subject_id', 'device', 'sensor', 'label', create a dataframe of 'timestamp', 'x', 'y', 'z'
    assert lf is not None, "timeseries cannot be computed for None or empty"
    names = set(lf.collect_schema().names())
    required = {'subject_id', 'device', 'sensor', 'label', 'x', 'y', 'z'}
    assert required.issubset(names), "DataFrame for timeseries must contain required columns"

    chunk_size = DataConfig.feature_length_in_seconds * DataConfig.frequency
    group_columns = ['subject_id', 'device', 'sensor', 'label']

    result_lf = (
        lf
        .explode(["x", "y", "z"])
        .with_columns(
            pl.int_range(pl.len()).over(group_columns).alias("series_idx")
        )
        .with_columns(
            (pl.col("series_idx") // chunk_size).alias("chunk_idx"),
        )
        .group_by(
            group_columns + ["chunk_idx"], maintain_order=True,
        )
        .all()
        .filter(
            (pl.col("chunk_idx") + 1) * chunk_size == (pl.col("series_idx").list.last() + 1)
        )
        .drop(['chunk_idx', 'series_idx'])
    )
    return result_lf


def apply_z_score_normalization(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Normalize each axis (x,y,z) per device, sensor, participant using z-score.
    """
    assert lf is not None

    group_columns = ['subject_id', 'device', 'sensor']

    stats = (
        lf
        .select(group_columns + ['x', 'y', 'z'])
        .explode(['x', 'y', 'z'])
        .group_by(group_columns)
        .agg([
            pl.col('x').mean().alias('x_mean'),
            pl.col('y').mean().alias('y_mean'),
            pl.col('z').mean().alias('z_mean'),
            pl.col('x').std().alias('x_std'),
            pl.col('y').std().alias('y_std'),
            pl.col('z').std().alias('z_std'),
        ])
    )

    lf_norm = (
        lf
        .with_columns(pl.arange(0, pl.len()).alias('row_idx'))
        .join(stats, on=group_columns, how='left')
        .with_columns([
            pl.when(pl.col('x_std').is_null() | (pl.col('x_std') == 0)).then(1.0).otherwise(pl.col('x_std')).alias(
                'x_std_safe'),
            pl.when(pl.col('y_std').is_null() | (pl.col('y_std') == 0)).then(1.0).otherwise(pl.col('y_std')).alias(
                'y_std_safe'),
            pl.when(pl.col('z_std').is_null() | (pl.col('z_std') == 0)).then(1.0).otherwise(pl.col('z_std')).alias(
                'z_std_safe'),
        ])
        .explode(['x', 'y', 'z'])
        .with_columns([
            ((pl.col('x').cast(pl.Float64) - pl.col('x_mean')) / pl.col('x_std_safe')).alias('x'),
            ((pl.col('y').cast(pl.Float64) - pl.col('y_mean')) / pl.col('y_std_safe')).alias('y'),
            ((pl.col('z').cast(pl.Float64) - pl.col('z_mean')) / pl.col('z_std_safe')).alias('z'),
        ])
        .group_by('row_idx')
        .agg(
            pl.first('subject_id').alias('subject_id'),
            pl.first('device').alias('device'),
            pl.first('sensor').alias('sensor'),
            pl.first('label').alias('label'),
            pl.col('x').alias('x'),
            pl.col('y').alias('y'),
            pl.col('z').alias('z'),
        )
        .drop('row_idx')
    )

    return lf_norm

def source_target_split(lf: pl.LazyFrame, target_prefix = "label"):
    all_column_names = lf.collect_schema().names()
    label_cols = [col for col in all_column_names if col.startswith(target_prefix)]
    feature_cols = [col for col in all_column_names if col not in label_cols]
    return lf.select(feature_cols), lf.select(label_cols)


def preprocessing_pipeline_for_wisdm_raw_data(data_path_to_raw_folder):
    """

    :param data_path_to_raw_folder:
    :return:
    """

    lf = load_data_raw_with_row_per_sensor_device_activity_subject(data_path_to_raw_folder)
    # print(data.collect())
    # print(lf.collect())
    lf = convert_data_to_10_seconds_intervals(lf)
    # print(lf.collect())
    lf = apply_z_score_normalization(lf)
    # lf = one_hot_encode(lf, ["device", "sensor", "label"])
    # DataFrame from LazyFrame
    df = lf.collect()
    df = df.to_dummies(columns=["device", "sensor", "label"])

    return df


if __name__ == "__main__":
    data_path_prefix = "/Users/wojtek/Library/Mobile Documents/com~apple~CloudDocs/Documents/EPRS/WISDM/data/wisdm-dataset/raw"
    df = preprocessing_pipeline_for_wisdm_raw_data(data_path_prefix)
    print(df)