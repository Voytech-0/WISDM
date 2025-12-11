from pathlib import Path
import sys
from typing import List, Dict, Any, Optional

import numpy as np
import polars as pl
from sqlalchemy import create_engine, text
import time
from sqlalchemy.exc import OperationalError, SQLAlchemyError
import pickle
import logging

logging.basicConfig(level=logging.INFO)


# When tests import this file directly, the repo root may not be on sys.path.
# Prepend the repository root so `configs.configs` can be resolved.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from configs.configs import DataConfig, PathsConfig, MySQLConfig


def wait_for_mysql(engine, timeout: int = 30, interval: float = 1.0):
    """Wait until a connection to the database can be established.

    Tries to connect repeatedly until success or until timeout seconds have elapsed.
    Prints progress and raises the last exception on timeout.
    """
    start = time.time()
    attempt = 0
    last_exc = None
    while True:
        attempt += 1
        try:
            with engine.connect() as conn:
                # simple validation: run a lightweight query
                conn.execute(text("SELECT 1"))
            logging.info("MySQL available after %d attempt(s)", attempt)
            return
        except (OperationalError, SQLAlchemyError) as exc:
            last_exc = exc
            elapsed = time.time() - start
            if elapsed >= timeout:
                logging.error("Timed out after %.1fs waiting for MySQL (attempts=%d)", elapsed, attempt)
                raise
            # backoff strategy: sleep for interval * min(2**attempt, 16)
            sleep_for = interval * min(2 ** (attempt - 1), 16)
            logging.warning("MySQL not ready (attempt=%d), sleeping %.1fs: %s", attempt, sleep_for, exc)
            time.sleep(sleep_for)

def load_raw_data_from_mysql() -> pl.LazyFrame:

    logging.info("user %s connecting to %s:%s database %s", MySQLConfig.user, MySQLConfig.host, MySQLConfig.port,
                 MySQLConfig.db)
    engine = create_engine(
        f"mysql+pymysql://{MySQLConfig.user}:{MySQLConfig.password}@{MySQLConfig.host}:{MySQLConfig.port}/{MySQLConfig.db}"
    )
    # wait for database to accept connections before executing the query
    wait_for_mysql(engine, timeout=DataConfig.db_connect_timeout if hasattr(DataConfig, 'db_connect_timeout') else 30)
    query = "SELECT subject_id, label, timestamp, x, y, z, sensor, device FROM wisdm_raw"

    chunk_size = int(getattr(DataConfig, 'db_read_chunk_size', 10000))

    frames = []
    cols = None

    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(text(query))
        cols = list(result.keys())
        while True:
            batch = result.fetchmany(size=chunk_size)
            if not batch:
                break
            rows = [tuple(r) for r in batch]
            df_chunk = pl.DataFrame(rows, schema=cols, orient="row")
            frames.append(df_chunk)
    logging.info("Finished reading data from MySQL in %d chunks of up to %d rows", len(frames), chunk_size)

    if not frames:
        logging.error("No data found in wisdm_raw table.")
        df = pl.DataFrame([], schema=cols if cols is not None else [])
        return df.lazy()

    # Concatenate all chunk DataFrames
    df = pl.concat(frames, how='vertical')

    # drop prefix '\n' from subject_id if present
    # should not be necessary if data ingestion is correct, but added because of prior error
    df = df.with_columns([
        pl.col('subject_id').str.replace_all(r'^\n', '').alias('subject_id')
    ])
    logging.info("Loaded raw data from MySQL: %d rows, %d columns", df.height, len(df.columns))
    return df.lazy()

def raw_data_into_xyz_series(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Returns a polars LazyFrame where each row corresponds to one (subject_id, device, sensor, label)
    and the columns x,y,z are lists of values ordered by timestamp.
    """

    # Ensure numeric types for timestamp,x,y,z and drop incomplete rows
    lf = lf.with_columns([
        pl.col('timestamp').cast(pl.Float64).alias('timestamp'),
        pl.col('x').cast(pl.Float64).alias('x'),
        pl.col('y').cast(pl.Float64).alias('y'),
        pl.col('z').cast(pl.Float64).alias('z'),
    ]).filter(pl.col('z').is_not_null())

    # sort by timestamp to preserve ordering within groups
    lf = lf.sort(['subject_id', 'device', 'sensor', 'label', 'timestamp'])

    # group and aggregate x,y,z into lists (maintain order)
    grouped = (
        lf
        .group_by(['subject_id', 'device', 'sensor', 'label'], maintain_order=True)
        .agg([
            pl.col('x').alias('x'),
            pl.col('y').alias('y'),
            pl.col('z').alias('z'),
        ])
    )

    return grouped



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
        with open(file_path, 'rb') as f:
            stats_df = pickle.load(f)
        return pl.LazyFrame(stats_df)
    else:
        logging.error("Z-score stats file not found at %s", file_path)
        raise FileNotFoundError(f"Z-score stats file not found at {file_path}")


def set_z_score_stats(stats: pl.LazyFrame):
    file_path = PathsConfig.z_score_params_path
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = stats.collect()
    with open(path, "wb") as f:
        pickle.dump(df, f)


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

def _source_target_split(df: pl.DataFrame, target_prefix ="label"):
    all_column_names = df.collect_schema().names()
    label_cols = [col for col in all_column_names if col.startswith(target_prefix)]
    feature_cols = [col for col in all_column_names if col not in label_cols]
    return df.select(feature_cols), df.select(label_cols)

def generate_splits(df: pl.DataFrame):
    """
    
    :param df:
    :return:
    """
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
        X, y = _source_target_split(df_split)
        output_dict[split] = {'X': X, 'y': y}
    return output_dict


def expand_lists(df: pl.DataFrame, list_cols: List[str]) -> pl.DataFrame:
    """Expand list-valued columns in `df` into separate scalar columns using Polars.

    For each column name in `list_cols` the function creates columns
    {col}_0 ... {col}_{L-1} where L is the maximum list length defined in conf.
    Shorter lists produce nulls in the corresponding positions.

    Returns a new DataFrame with the original list columns dropped.
    """
    # 1. Determine the maximum length (L)
    max_length = DataConfig.feature_length_in_seconds * DataConfig.frequency

    # List to hold the expressions for the new columns
    expansion_expressions = []

    for col in list_cols:
        # 2. Create the Polars expression for padding/truncating
        # This expression ensures all lists are exactly 'max_length' long.
        list_expression = (
            # 2a. Pad the list to at least 'max_length' with nulls.
            pl.col(col).list.to_array(max_length).cast(pl.List)
            # 2b. Truncate (slice) the list to exactly 'max_length'.
            # Note: The combination of pl.list.to_array() and pl.list.slice()
            # or pl.list.to_array().cast(pl.List) handles both padding and truncation implicitly
            # and efficiently. pl.list.to_array() handles the max length and padding with nulls.
        )

        # 3. Convert the padded/truncated list into a Struct
        # This allows us to flatten it into multiple columns in the next step.
        struct_expression = list_expression.list.to_struct(
            # Generate field names: {col}_0, {col}_1, ...
            fields=[f'{col}_{i}' for i in range(max_length)],
            # The 'n_field_strategy' here is essential for padding shorter lists
            # and is handled by pl.list.to_array().
        ).alias(f'{col}_struct')

        expansion_expressions.append(struct_expression)

    # 4. Add the new Struct columns and unnest them
    df_expanded = (
        df.with_columns(expansion_expressions)
        .unnest([f'{col}_struct' for col in list_cols])
        .drop(list_cols)
    )

    return df_expanded


def preprocessing_pipeline_for_wisdm_raw_data():
     lf = load_raw_data_from_mysql()
     lf = raw_data_into_xyz_series(lf)
     lf = convert_data_to_10_seconds_intervals(lf)
     lf = apply_z_score_normalization(lf, train=True)
     # lf = one_hot_encode(lf, ["device", "sensor", "label"])
     # DataFrame from LazyFrame
     df = collect_and_dummify_drop_subject(lf)
     df = expand_lists(df, ["x", "y", "z"])
     logging.info("Preprocessing produced DataFrame with %d rows and %d columns", df.height, len(df.columns))
     return df


def build_raw_lazyframe_from_items(items: List[Dict[str, Any]]) -> pl.LazyFrame:
    """Build a Polars LazyFrame from a list of raw input dicts containing x,y,z lists and metadata.

    Each item is expected to contain at least keys: 'x','y','z'. Optional metadata: 'device','sensor','subject_id','label'.
    """
    rows = []
    for it in items:
        device = it.get('device') or None
        sensor = it.get('sensor') or None
        subject_id = it.get('subject_id') or 'unknown'
        label = it.get('label') if 'label' in it else None
        # coerce numeric values - allow callers to raise if conversion fails
        x = list(map(float, it['x']))
        y = list(map(float, it['y']))
        z = list(map(float, it['z']))
        rows.append({'subject_id': subject_id, 'device': device, 'sensor': sensor, 'label': label, 'x': x, 'y': y, 'z': z})
    df = pl.DataFrame(rows)
    return df.lazy()


def lazyframe_to_single_10s(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Ensure each row has x,y,z lists exactly `feature_length_in_seconds * frequency` long.
    Long lists are truncated, short lists are padded with nulls. Returns a LazyFrame.
    """
    desired_length = int(DataConfig.feature_length_in_seconds * DataConfig.frequency)
    if lf is None:
        raise ValueError("Input LazyFrame cannot be None")

    # Ensure required columns exist
    names = set(lf.collect_schema().names())
    required = {'x', 'y', 'z'}
    if not required.issubset(names):
        raise AssertionError(f"LazyFrame must contain columns {required}, got {names}")

    # Pad/truncate lists to the desired length. We use concat_list + slice so we can
    # pad with zeros (matching test expectations) and avoid `.list.to_array` width errors.
    pad_x = [0] * desired_length
    pad_y = [0.0] * desired_length
    pad_z = [0.0] * desired_length

    lf_fixed = lf.with_columns([
        pl.concat_list([pl.col('x'), pl.lit(pad_x)]).list.slice(0, desired_length).alias('x'),
        pl.concat_list([pl.col('y'), pl.lit(pad_y)]).list.slice(0, desired_length).alias('y'),
        pl.concat_list([pl.col('z'), pl.lit(pad_z)]).list.slice(0, desired_length).alias('z'),
    ])

    return lf_fixed


def collect_and_dummify_drop_subject(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Collect lazyframe, dummify device/sensor/label efficiently, ensure
    all specified dummy columns exist, and drop subject_id/original columns.
    """
    df = lf.collect()
    columns_to_dummify = ['device', 'sensor', 'label']

    present_cols = [col for col in columns_to_dummify if col in df.columns]

    if present_cols:
        df = df.to_dummies(present_cols)
    new_cols_expressions = []

    for col in columns_to_dummify:
        possible_values: List[str] = getattr(DataConfig, f'possible_{col}s', [])

        for val in possible_values:
            colname = f"{col}_{str(val)}"

            if colname not in df.columns:
                new_cols_expressions.append(pl.lit(0).alias(colname))

    if new_cols_expressions:
        df = df.with_columns(new_cols_expressions)

    if 'subject_id' in df.columns:
        df = df.drop('subject_id')

    return df


def expand_raw_items_to_expanded_df(items: List[Dict[str, Any]]) -> pl.DataFrame:
    """High-level helper: run raw items through chunking, normalization, collecting/dummify, and expansion.

    Returns a Polars DataFrame with expanded scalar feature columns (x_0.., y_0.., z_0..)
    """
    lf = build_raw_lazyframe_from_items(items)
    lf = lazyframe_to_single_10s(lf)
    lf_norm = apply_z_score_normalization(lf, train=False, group_columns=['device', 'sensor'])
    df_norm = collect_and_dummify_drop_subject(lf_norm)
    df_expanded = expand_lists(df_norm, ['x', 'y', 'z'])
    return df_expanded


if __name__ == "__main__":
    df = preprocessing_pipeline_for_wisdm_raw_data()
    logging.info("Preprocessing finished. DataFrame summary:\n%s", df)
