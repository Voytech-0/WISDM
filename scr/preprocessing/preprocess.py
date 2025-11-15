import os
from typing import List

import numpy as np
import pandas as pd
from pathlib import Path
import polars as pl
from setuptools.command.alias import alias

from configs.configs import DataConfig


def _extract_axis_array_from_ts(ts_df: pd.DataFrame, axis: str) -> np.ndarray:
    """
    Return a 1-D numpy array of values for `axis` from a timeseries DataFrame.
    """
    if not isinstance(ts_df, pd.DataFrame):
        return np.array([])
    if axis not in ts_df.columns:
        return np.array([])
    if len(ts_df) == 0:
        return np.array([])

    array = np.asarray(ts_df[axis])
    return array.reshape(-1, )


def load_data_raw_with_row_per_sensor_device_activity_subject(prefix_path: str) -> pl.LazyFrame:
    file_header = ['subject_id', 'label', 'timestamp', 'x', 'y', 'z']

    paths = list(Path(prefix_path).rglob("data_*.txt"))

    lf_all = []
    for p in paths:
        lf = pl.scan_csv(str(p), has_header=False, new_columns=file_header)
        subject_id, sensor, device = str(p).removesuffix(".txt").split('_')[-3:]
        lf = lf.drop(["subject_id", "timestamp"])
        lf = lf.group_by("label", maintain_order=True).all()
        lf = lf.with_columns(pl.lit(sensor).alias("sensor"))
        lf = lf.with_columns(pl.lit(device).alias("device"))
        lf = lf.with_columns(pl.lit(subject_id).alias("subject_id"))
        lf_all.append(lf)

    lf_all = pl.concat(lf_all)
    return lf_all


def convert_data_to_10_seconds_intervals(lf: pl.LazyFrame) -> pl.LazyFrame:
    # for each 'subject_id', 'device', 'sensor', 'label', create a dataframe of 'timestamp', 'x', 'y', 'z'
    assert lf is not None, "timeseries cannot be computed for None or empty"
    assert set(lf.collect_schema().names()) == {'subject_id', 'device', 'sensor', 'label', 'x', 'y', 'z'}, \
        "DataFrame for timeseries must contain required columns"

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
            group_columns + ["chunk_idx"], maintain_order=True
        )
        .all()
        .drop(['chunk_idx', 'series_idx'])
    )

    return result_lf


def apply_z_score_normalization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each axis (x,y,z) per device and sensor using z-score.
    """
    assert data is not None and not data.empty, "z-score cannot be computed for None or empty"
    assert 'timeseries' in data.columns, "DataFrame for z-score must contain 'timeseries' column"
    data = data.copy()

    devices = data['device'].unique()
    sensors = data['sensor'].unique()

    for device in devices:
        for sensor in sensors:
            mask = (data['device'] == device) & (data['sensor'] == sensor)
            subset = data.loc[mask]

            if subset.empty:
                continue

            axis_stats = {}
            for axis in ['x', 'y', 'z']:
                # collect arrays for this axis from each timeseries DataFrame using the helper
                axis_arrays = subset['timeseries'].apply(lambda ts: _extract_axis_array_from_ts(ts, axis))

                non_empty = [a for a in axis_arrays.to_list() if a.size > 0]
                if len(non_empty) == 0:
                    continue
                all_values = np.concatenate(non_empty, axis=0)

                if all_values.size == 0:
                    mean = 0.0
                    std = 1.0
                else:
                    mean = float(np.mean(all_values))
                    std = float(np.std(all_values))
                    if std == 0.0:
                        std = 1.0

                axis_stats[axis] = (mean, std)

            if not axis_stats:
                continue

            # Apply normalization to each timeseries dataframe in the subset
            def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
                df = df.copy()
                for axis, (mean, std) in axis_stats.items():
                    if axis in df.columns:
                        df[axis] = (df[axis] - mean) / std
                return df

            normalized_series = subset['timeseries'].apply(normalize_df)
            data.loc[mask, 'timeseries'] = normalized_series.values

    return data


def convert_timeseries_to_features(data: pd.DataFrame) -> pd.DataFrame:
    assert data is not None and not data.empty, "features cannot be computed for None or empty"

    if 'subject_id' in data.columns:
        data = data.drop(columns=['subject_id'])

    data = apply_z_score_normalization(data)

    return data


if __name__ == "__main__":
    data_path_prefix = "/Users/wojtek/Library/Mobile Documents/com~apple~CloudDocs/Documents/EPRS/WISDM/data/wisdm-dataset/raw"

    data = load_data_raw_with_row_per_sensor_device_activity_subject(data_path_prefix)
    # print(data.collect())
    data = convert_data_to_10_seconds_intervals(data)
    exit()
    data = convert_timeseries_to_features(data)
    print(data.head())
