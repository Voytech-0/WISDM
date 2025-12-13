import os
import tempfile
from pathlib import Path
import time
import logging
import polars as pl
import pymysql
from sqlalchemy import create_engine, text
from flask import Flask, jsonify
from http import HTTPStatus

from configs.configs import MySQLConfig, DataConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

FILE_HEADER = ['subject_id', 'label', 'timestamp', 'x', 'y', 'z']

ENGINE = None


def get_connected_engine(timeout_s: int = 60):
    global ENGINE
    if ENGINE is None:
        url = f"mysql+pymysql://{MySQLConfig.user}:{MySQLConfig.password}@{MySQLConfig.host}:{MySQLConfig.port}/{MySQLConfig.db}"
        ENGINE = create_engine(url, connect_args={"local_infile": 1})

    if not wait_for_db(ENGINE, timeout_s=timeout_s):
        raise RuntimeError("Database not available")

    return ENGINE


def wait_for_db(engine, timeout_s=60):
    start = time.time()
    while True:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logging.info("Database is available")
            return True
        except Exception as e:
            if time.time() - start > timeout_s:
                logging.error(f"Timed out waiting for DB: {e}")
                return False
            logging.info("Waiting for DB to be available...")
            time.sleep(3)


def create_table(engine):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS wisdm_raw (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        subject_id VARCHAR(255),
        label VARCHAR(255),
        timestamp BIGINT,
        x DOUBLE,
        y DOUBLE,
        z DOUBLE,
        sensor VARCHAR(10),
        device VARCHAR(10)
    ) ENGINE=InnoDB;
    """
    with engine.connect() as conn:
        conn.execute(text(create_table_query))
        conn.commit()
    logging.info("Ensured table wisdm_raw exists.")


def create_status_table(engine):
    q = """
    CREATE TABLE IF NOT EXISTS init_status (
        `key` VARCHAR(100) PRIMARY KEY,
        `value` VARCHAR(100),
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """
    with engine.connect() as conn:
        conn.execute(text(q))
        conn.commit()


def set_status_in_db(key: str = None, value: str = None):
    if key is None:
        raise ValueError("key is required")

    engine = get_connected_engine(timeout_s=10)

    query = """
    INSERT INTO init_status (`key`, `value`) VALUES (:k, :v)
    ON DUPLICATE KEY UPDATE `value` = :v, updated_at = CURRENT_TIMESTAMP;
    """
    with engine.connect() as conn:
        conn.execute(text(query), {"k": key, "v": value})
        conn.commit()
        logging.info("Set init_status key=%s to value=%s", key, value)


def get_status_from_db(key: str = None) -> str | None:
    if key is None:
        raise ValueError("key is required")

    try:
        engine = get_connected_engine(timeout_s=5)
    except Exception:
        logging.debug("DB not available when reading status")
        return None

    q = "SELECT `value` FROM init_status WHERE `key` = :k LIMIT 1"
    try:
        with engine.connect() as conn:
            r = conn.execute(text(q), {"k": key}).fetchone()
            if r:
                return r[0]
            logging.info("No entry found in init_status for key=%s", key)
            logging.info("r: %s", r)
    except Exception:
        logging.debug("Could not read init_status from DB (it may not be ready)")
    return None


def load_data_raw_with_row_per_sensor_device_activity_subject(prefix_path: str) -> pl.LazyFrame:
    file_header = ['subject_id', 'label', 'timestamp', 'x', 'y', 'z']

    paths = list(Path(prefix_path).rglob("data_*.txt"))

    lf_all = []
    for p in paths:
        lf = pl.scan_csv(str(p), has_header=False, new_columns=file_header, eol_char=';\n')
        _, sensor, device = str(p).removesuffix('.txt').split('_')[-3:]
        lf = (
            lf
            .with_columns(
                pl.lit(sensor).alias('sensor'),
                pl.lit(device).alias('device'),
            )
        )
        lf_all.append(lf)

    lf_all = pl.concat(lf_all).drop_nulls()

    return lf_all


def parse_and_insert(prefix_path: str):
    if not Path(prefix_path).exists():
        raise FileNotFoundError(f"Raw data folder not found at {prefix_path}")

    logging.info("Building lazy frame for all raw files...")
    lf = load_data_raw_with_row_per_sensor_device_activity_subject(prefix_path)


    try:
        logging.info("Collecting lazy frame into memory (Polars DataFrame)")
        df_pl = lf.collect()
    except Exception as e:
        logging.exception(f"Failed to collect LazyFrame: {e}")
        raise

    expected_cols = ['subject_id', 'label', 'timestamp', 'x', 'y', 'z', 'sensor', 'device']
    missing = [c for c in expected_cols if c not in df_pl.columns]
    if missing:
        logging.warning(f"Missing expected columns: {missing}. Filling with nulls.")
        df_pl = df_pl.with_columns([pl.lit(None).alias(c) for c in missing])

    df_pl = df_pl.with_columns([
        pl.col('timestamp').cast(pl.Int64, strict=False),
        pl.col('x').cast(pl.Float64, strict=False),
        pl.col('y').cast(pl.Float64, strict=False),
        pl.col('z').cast(pl.Float64, strict=False),
    ])

    df_pl = df_pl.select(expected_cols)

    try:
        insert_using_load_data(df_pl, table_name='wisdm_raw')
    except Exception as e:
        logging.exception(f"Failed to append aggregated data into DB: {e}")
        raise


def insert_using_load_data(df: pl.DataFrame, table_name: str = 'wisdm_raw') -> bool:
    if df.height == 0:
        logging.info("DataFrame is empty, nothing to insert")
        return True

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    tmp_name = tmp.name
    tmp.close()

    try:
        logging.info("Writing Polars DataFrame to CSV...")
        df.write_csv(
            tmp_name,
            include_header=True,
            separator=',',
            quote_style='always',
            null_value='\\N'
        )

        path_for_sql = tmp_name.replace('\\', '\\\\')
        cols = df.columns
        cols_sql = ','.join(f'`{c}`' for c in cols)

        conn = None
        cursor = None
        try:
            conn = pymysql.connect(
                host=MySQLConfig.host,
                user=MySQLConfig.user,
                password=MySQLConfig.password,
                database=MySQLConfig.db,
                port=int(MySQLConfig.port),
                local_infile=True,
                autocommit=False,
            )
            cursor = conn.cursor()

            try:
                cursor.execute("SET SESSION local_infile = 1;")
            except Exception as exc:
                logging.debug(f"Could not enable session local_infile: {exc}")

            sql = (
                "LOAD DATA LOCAL INFILE '{}' INTO TABLE {} "
                "FIELDS TERMINATED BY ',' ENCLOSED BY '\"' "
                "LINES TERMINATED BY '\\n' IGNORE 1 LINES "
                "({});"
            ).format(path_for_sql, table_name, cols_sql)

            logging.info(f"Running bulk LOAD DATA LOCAL INFILE for {df.height} rows")
            cursor.execute(sql)
            conn.commit()
            return True
        finally:
            if cursor: cursor.close()
            if conn: conn.close()
    finally:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
    return False


def is_populated():
    try:
        val = get_status_from_db('is_populated')
        logging.info("is_populated status from DB: %s", val)
        return val in ('1', 'true', 'True')
    except Exception:
        logging.debug('Could not determine persisted populate status')
        return False


def populate(prefix_path: str = None) -> tuple[bool, str]:
    if prefix_path is None:
        prefix_path = DataConfig.data_path_prefix

    try:
        engine = get_connected_engine(timeout_s=120)
    except Exception:
        return (False, "db_unavailable")

    try:
        create_status_table(engine)
    except Exception:
        logging.debug('Could not create init_status table before reading status')

    try:
        create_table(engine)
        parse_and_insert(prefix_path)
        try:
            logging.info("Putting is_populated flag to DB")
            set_status_in_db('is_populated', '1')
        except Exception:
            logging.exception('Failed to persist is_populated flag to DB')

    except Exception as exc:
        logging.exception('Error during populate')
        return (False, f'error:{exc}')

    logging.info("Finished populating database")
    return (True, 'populated')


logger.info("Starting flask app.")
app = Flask(__name__)


@app.route('/status', methods=['GET'])
def status_endpoint():
    return jsonify({'is_populated': is_populated()}), HTTPStatus.OK


@app.route('/populate', methods=['GET'])
def populate_endpoint():

    try:
        if is_populated():
            return jsonify({'started': False, 'populated': True, 'reason': 'already_populated'}), HTTPStatus.OK
        else:
            logging.info('Database not populated at start of /populate.')
    except Exception:
        logging.info('Population status unknown at start of /populate')

    success, reason = populate()
    if success:
        return jsonify({'started': True, 'populated': True, 'reason': reason}), HTTPStatus.OK

    if reason == 'db_unavailable':
        return jsonify({'error': 'Database unavailable during populate attempt'}), HTTPStatus.SERVICE_UNAVAILABLE

    logging.error('Populate failed: %s', reason)
    return jsonify({'error': f'Populate failed: {reason}'}), HTTPStatus.INTERNAL_SERVER_ERROR