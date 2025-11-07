import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# helper to import the preprocess module by file path
def load_preprocess_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / 'scr' / 'preprocessing' / 'preprocess.py'
    spec = importlib.util.spec_from_file_location('preprocess_mod', str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_data_raw_reads_csvs(tmp_path):
    mod = load_preprocess_module()

    prefix = tmp_path / 'raw'
    accel_dir = prefix / 'phone' / 'accel'
    gyro_dir = prefix / 'watch' / 'gyro'
    accel_dir.mkdir(parents=True)
    gyro_dir.mkdir(parents=True)


    accel_file = accel_dir / 'data_000_accel_phone.csv'
    accel_file.write_text('\n'.join([
        '1,a,0,0.1,0.2,0.3',
        '1,a,1,0.2,0.3,0.4',
    ]))
    gyro_file = gyro_dir / 'data_000_gyro_watch.csv'
    gyro_file.write_text('\n'.join([
        '2,b,0,10.0,11.0,12.0',
        '2,b,1,11.0,12.0,13.0',
    ]))

    df = mod.load_data_raw(str(prefix))


    assert not df.empty
    assert set(['device', 'sensor', 'subject_id', 'label', 'timestamp', 'x', 'y', 'z']).issubset(df.columns)

    assert len(df) == 4

    assert set(df['device'].unique()) == {'phone', 'watch'}
    assert set(df['sensor'].unique()) == {'accel', 'gyro'}


def test_convert_raw_data_to_10_second_timeseries_splits_into_equal_segments():
    mod = load_preprocess_module()

    segment_size = mod.DataConfig.feature_length_in_seconds * mod.DataConfig.frequency
    total = segment_size * 2
    timestamps = list(range(total))
    df = pd.DataFrame({
        'subject_id': [1] * total,
        'device': ['phone'] * total,
        'sensor': ['accel'] * total,
        'label': ['a'] * total,
        'timestamp': timestamps,
        'x': np.arange(total),
        'y': np.arange(total) + 0.5,
        'z': np.arange(total) + 1.0,
    })

    ts_df = mod.convert_raw_data_to_10_second_timeseries(df)
    assert len(ts_df) == 2

    for idx in range(2):
        series = ts_df['timeseries'].iloc[idx]
        assert isinstance(series, pd.DataFrame)
        assert len(series) == segment_size
        assert set(series.columns) == {'timestamp', 'x', 'y', 'z'}


def test_apply_z_score_normalization_per_device_sensor_axis():
    mod = load_preprocess_module()
    # create two groups with distinct scales
    seg1 = pd.DataFrame({'timestamp': [0, 1, 2], 'x': [1.0, 2.0, 3.0], 'y': [2.0, 3.0, 4.0], 'z': [3.0, 4.0, 5.0]})
    seg2 = pd.DataFrame({'timestamp': [0, 1, 2], 'x': [2.0, 3.0, 4.0], 'y': [3.0, 4.0, 5.0], 'z': [4.0, 5.0, 6.0]})
    seg3 = pd.DataFrame({'timestamp': [0, 1, 2], 'x': [10.0, 11.0, 12.0], 'y': [20.0, 21.0, 22.0], 'z': [30.0, 31.0, 32.0]})
    seg4 = pd.DataFrame({'timestamp': [0, 1, 2], 'x': [11.0, 12.0, 13.0], 'y': [21.0, 22.0, 23.0], 'z': [31.0, 32.0, 33.0]})

    rows = [
        {'subject_id': 1, 'device': 'phone', 'sensor': 'accel', 'label': 'a', 'timeseries': seg1},
        {'subject_id': 1, 'device': 'phone', 'sensor': 'accel', 'label': 'a', 'timeseries': seg2},
        {'subject_id': 2, 'device': 'watch', 'sensor': 'gyro', 'label': 'b', 'timeseries': seg3},
        {'subject_id': 2, 'device': 'watch', 'sensor': 'gyro', 'label': 'b', 'timeseries': seg4},
    ]
    df = pd.DataFrame(rows)

    def flat_stats(subdf):
        arrs = {axis: np.concatenate(subdf['timeseries'].apply(lambda d: np.asarray(d[axis]).reshape(-1,)).to_list(), axis=0)
                for axis in ['x', 'y', 'z']}
        return {k: (v.mean(), v.std()) for k, v in arrs.items()}

    before_phone = flat_stats(df[df['device'] == 'phone'])
    before_watch = flat_stats(df[df['device'] == 'watch'])
    assert before_phone['x'][0] != 0
    assert before_watch['x'][0] != 0

    out = mod.apply_z_score_normalization(df)

    after_phone = flat_stats(out[out['device'] == 'phone'])
    after_watch = flat_stats(out[out['device'] == 'watch'])

    for mean, std in after_phone.values():
        assert abs(mean) < 1e-8
        assert pytest.approx(std, rel=1e-6) == 1.0
    for mean, std in after_watch.values():
        assert abs(mean) < 1e-8
        assert pytest.approx(std, rel=1e-6) == 1.0


def test_convert_timeseries_to_features_drops_subject_and_normalizes():
    mod = load_preprocess_module()
    seg1 = pd.DataFrame({'timestamp': [0, 1, 2], 'x': [1.0, 2.0, 3.0], 'y': [2.0, 3.0, 4.0], 'z': [3.0, 4.0, 5.0]})
    seg2 = pd.DataFrame({'timestamp': [0, 1, 2], 'x': [2.0, 3.0, 4.0], 'y': [3.0, 4.0, 5.0], 'z': [4.0, 5.0, 6.0]})
    df = pd.DataFrame([
        {'subject_id': 1, 'device': 'phone', 'sensor': 'accel', 'label': 'a', 'timeseries': seg1},
        {'subject_id': 1, 'device': 'phone', 'sensor': 'accel', 'label': 'a', 'timeseries': seg2},
    ])

    out = mod.convert_timeseries_to_features(df)

    assert 'subject_id' not in out.columns

    arr_x = np.concatenate(out['timeseries'].apply(lambda d: np.asarray(d['x']).reshape(-1,)).to_list(), axis=0)
    assert abs(arr_x.mean()) < 1e-8
    assert pytest.approx(arr_x.std(), rel=1e-6) == 1.0

