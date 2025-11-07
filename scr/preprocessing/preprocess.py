import os
from typing import List

import numpy as np
import pandas as pd

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
    return array.reshape(-1,)


def load_data_raw(prefix_path: str) -> pd.DataFrame:
    df_columns = ['subject_id', 'device', 'sensor', 'label', 'timestamp', 'x', 'y', 'z']
    file_header = ['subject_id', 'label', 'timestamp', 'x', 'y', 'z']

    seperate_df: List[pd.DataFrame] = []
    total_files = 0

    for device in ('phone', 'watch'):
        for sensor in ('accel', 'gyro'):
            dirpath = os.path.join(prefix_path, device, sensor)
            if not os.path.isdir(dirpath):
                print(f"warning: directory not found: {dirpath}")
                continue

            with os.scandir(dirpath) as iterator_for_directory:
                entries = [entry for entry in iterator_for_directory if entry.is_file()]

            for entry in entries[:2]:
                entry_path = entry.path
                try:
                    # low_memory=False helps pandas parse mixed types faster (fewer dtype passes)
                    file_dataframe = pd.read_csv(entry_path, header=None, names=file_header, low_memory=False)
                except Exception as e:
                    print(f"failed to read {entry_path}: {e}")
                    continue

                file_dataframe['device'] = device
                file_dataframe['sensor'] = sensor
                seperate_df.append(file_dataframe)
                total_files += 1

    if not seperate_df:
        print("warning: unable to read any data")
        return pd.DataFrame(columns=df_columns)

    result = pd.concat(seperate_df, ignore_index=True)

    print(f"Loaded {total_files} files -> {len(result):,} rows")
    return result


def convert_raw_data_to_10_second_timeseries(data: pd.DataFrame) -> pd.DataFrame:
    # for each 'subject_id', 'device', 'sensor', 'label', create a dataframe of 'timestamp', 'x', 'y', 'z'
    assert data is not None and not data.empty, "timeseries cannot be computed for None or empty"
    assert all(col in data.columns for col in
        ['subject_id', 'device', 'sensor', 'label', 'timestamp', 'x', 'y', 'z']), \
        "DataFrame for timeseries must contain required columns"
    data = data.copy()
    separate_df = []
    group_columns = ['subject_id', 'device', 'sensor', 'label']
    data_grouped = data.groupby(group_columns)
    for group_keys, group_data in data_grouped:
        group_data_sorted = group_data.sort_values(by='timestamp')
        inner_df = group_data_sorted[['timestamp', 'x', 'y', 'z']].reset_index(drop=True)

        segment_size = DataConfig.feature_length_in_seconds * DataConfig.frequency
        for start in range(0, len(inner_df), segment_size):
            end = start + segment_size
            segment = inner_df.iloc[start:end].reset_index(drop=True)
            outer_df = pd.DataFrame(
                {'subject_id': [group_keys[0]], 'device': [group_keys[1]], 'sensor': [group_keys[2]],
                    'label': [group_keys[3]], 'timeseries': [segment]})
            separate_df.append(outer_df)
    result = pd.concat(separate_df, ignore_index=True)
    print(f"Converted to timeseries format -> {len(result):,} rows")
    return result


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

    data = load_data_raw(data_path_prefix)
    data = convert_raw_data_to_10_second_timeseries(data)
    data = convert_timeseries_to_features(data)
    print(data.head())
