from pathlib import Path
import sys
import polars as pl
from sqlalchemy import create_engine, text

# When tests import this file directly, the repo root may not be on sys.path.
# Prepend the repository root so `configs.configs` can be resolved.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from configs.configs import DataConfig, PathsConfig, MySQLConfig


def load_data_raw_with_row_per_sensor_device_activity_subject() -> pl.LazyFrame:
    """
    Returns a polars LazyFrame where each row corresponds to one (subject_id, device, sensor, label)
    and the columns x,y,z are lists of values ordered by timestamp.
    """
    engine = create_engine(
        f"mysql+pymysql://{MySQLConfig.user}:{MySQLConfig.password}@{MySQLConfig.host}:{MySQLConfig.port}/{MySQLConfig.db}"
    )
    query = "SELECT subject_id, label, timestamp, x, y, z, sensor, device FROM wisdm_raw"

    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        cols = result.keys()

    if not rows:
        raise RuntimeError("Query returned no rows")

    df = pl.DataFrame(rows, schema=list(cols))

    # Ensure numeric types for timestamp,x,y,z and drop incomplete rows
    df = df.with_columns([
        pl.col('timestamp').cast(pl.Float64).alias('timestamp'),
        pl.col('x').cast(pl.Float64).alias('x'),
        pl.col('y').cast(pl.Float64).alias('y'),
        pl.col('z').cast(pl.Float64).alias('z'),
    ]).filter(pl.col('z').is_not_null())

    # sort by timestamp to preserve ordering within groups
    df = df.sort(['subject_id', 'device', 'sensor', 'label', 'timestamp'])

    # group and aggregate x,y,z into lists (maintain order)
    grouped = (
        df
        .groupby(['subject_id', 'device', 'sensor', 'label'], maintain_order=True)
        .agg([
            pl.col('x').alias('x'),
            pl.col('y').alias('y'),
            pl.col('z').alias('z'),
        ])
    )

    return grouped.lazy()



def convert_data_to_10_seconds_intervals(lf: pl.LazyFrame) -> pl.LazyFrame:
    # for each 'subject_id', 'device' 'sensor', 'label', create lists of', 'x', 'y', 'z'
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


def get_z_score_stats() -> pl.LazyFrame:
    file_path = PathsConfig.z_score_params_path
    if Path(file_path).exists():
        stats = pl.read_pickle(file_path)
        return stats
    else:
        raise FileNotFoundError(f"Z-score stats file not found at {file_path}")


def set_z_score_stats(stats: pl.LazyFrame):
    file_path = PathsConfig.z_score_params_path
    stats.collect().write_csv(file_path)


def apply_z_score_normalization(lf: pl.LazyFrame, train: bool = False, group_columns=None) -> pl.LazyFrame:
    """
    Normalize each axis (x,y,z) per device, sensor, participant using z-score.
    """
    assert lf is not None

    if group_columns is None:
        group_columns = ['device', 'sensor']

    if train:
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
        set_z_score_stats(stats)
    else:
        stats = get_z_score_stats()


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

def source_target_split(df: pl.DataFrame, target_prefix ="label"):
    all_column_names = df.collect_schema().names()
    label_cols = [col for col in all_column_names if col.startswith(target_prefix)]
    feature_cols = [col for col in all_column_names if col not in label_cols]
    return df.select(feature_cols), df.select(label_cols)

def generate_splits(df: pl.DataFrame):
    # split using DataConfig.train_split, DataConfig.validation_split and DataConfig.test_split
    n = df.height
    train_end = int(n * DataConfig.train_split)
    val_end = train_end + int(n * DataConfig.validation_split)
    df_train = df.slice(0, train_end)
    df_val = df.slice(train_end, val_end - train_end)
    df_test = df.slice(val_end, n - val_end)
    output_dict = {}
    for split in ['train', 'validation', 'test']:
        if split == 'train':
            df_split = df_train
        elif split == 'validation':
            df_split = df_val
        else:
            df_split = df_test
        X, y = source_target_split(df_split)
        output_dict[split] = {'X': X, 'y': y}
    return output_dict


def expand_lists(df, list_cols):

    max_len = DataConfig.feature_length_in_seconds * DataConfig.frequency
    new_cols = []
    for i in range(max_len):
        for c in list_cols:
            new_cols.append(pl.col(c).arr.get(i).alias(f"{c}_{i}"))

    return df.with_columns(new_cols).drop(list_cols)


def preprocessing_pipeline_for_wisdm_raw_data(data_path_to_raw_folder):

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
    df = expand_lists(df, ["x", "y", "z"])
    return df


if __name__ == "__main__":
    df = preprocessing_pipeline_for_wisdm_raw_data()
    print(df)