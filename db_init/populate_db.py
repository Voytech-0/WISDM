from pathlib import Path
import time
import logging
import pandas as pd
import polars as pl
from sqlalchemy import create_engine, text
from flask import Flask, jsonify
from http import HTTPStatus
import threading

from configs.configs import MySQLConfig, DataConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

FILE_HEADER = ['subject_id', 'label', 'timestamp', 'x', 'y', 'z']

def get_engine():
    url = f"mysql+pymysql://{MySQLConfig.user}:{MySQLConfig.password}@{MySQLConfig.host}:{MySQLConfig.port}/{MySQLConfig.db}"
    # enable local_infile so we can use LOAD DATA LOCAL INFILE for much faster bulk loads when available
    return create_engine(url, connect_args={"local_infile": 1})


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
    logging.info("Ensured table wisdm_raw exists.")

def create_status_table(engine):
    """Create a small key/value table to persist init status across container restarts."""
    q = """
    CREATE TABLE IF NOT EXISTS init_status (
        `key` VARCHAR(100) PRIMARY KEY,
        `value` VARCHAR(100),
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """
    with engine.connect() as conn:
        conn.execute(text(q))

def set_status_in_db(engine, key: str, value: str):
    # Use INSERT ... ON DUPLICATE KEY UPDATE to upsert
    q = """
    INSERT INTO init_status (`key`, `value`) VALUES (:k, :v)
    ON DUPLICATE KEY UPDATE `value` = :v, updated_at = CURRENT_TIMESTAMP;
    """
    with engine.connect() as conn:
        conn.execute(text(q), {"k": key, "v": value})

def get_status_from_db(engine, key: str) -> str | None:
    q = "SELECT `value` FROM init_status WHERE `key` = :k LIMIT 1"
    try:
        with engine.connect() as conn:
            r = conn.execute(text(q), {"k": key}).fetchone()
            if r:
                return r[0]
    except Exception:
        logging.debug("Could not read init_status from DB (it may not be ready)")
    return None

def load_data_raw_with_row_per_sensor_device_activity_subject(prefix_path: str) -> pl.LazyFrame:
    file_header = ['subject_id', 'label', 'timestamp', 'x', 'y', 'z']

    paths = list(Path(prefix_path).rglob("data_*.txt"))

    lf_all = []
    for p in paths:
        # scan_csv with eol_char=';' because lines end with ';' in raw files
        lf = pl.scan_csv(str(p), has_header=False, new_columns=file_header, eol_char='\n')
        # derive sensor/device/subject from filename; we'll keep subject_id column from file
        _, sensor, device = str(p).removesuffix('.txt').split('_')[-3:]
        lf = (
            lf
            .with_columns(
                pl.lit(sensor).alias('sensor'),
                pl.lit(device).alias('device'),
                pl.col('subject_id').str.replace_all(r'^\n', '').alias('subject_id')
            )
        )
        lf_all.append(lf)

    lf_all = pl.concat(lf_all).drop_nulls()

    return lf_all

def parse_and_insert(engine, prefix_path: str):
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

    # Ensure expected columns exist
    expected_cols = ['subject_id', 'label', 'timestamp', 'x', 'y', 'z', 'sensor', 'device']
    missing = [c for c in expected_cols if c not in df_pl.columns]
    if missing:
        logging.warning(f"Missing expected columns after collection: {missing}. Attempting to continue with available columns.")

    # Coerce numeric columns (non-strict)
    df_pl = df_pl.with_columns([
        pl.col('timestamp').cast(pl.Int64, strict=False),
        pl.col('x').cast(pl.Float64, strict=False),
        pl.col('y').cast(pl.Float64, strict=False),
        pl.col('z').cast(pl.Float64, strict=False),
    ])

    # keep only 1000 first
    # FIXME: use full dataset later
    df_pl = df_pl.head(1000000)

    # Convert to pandas for the existing SQL insertion path
    logging.info("Converting Polars DataFrame to pandas for DB insertion...")
    df = df_pl.to_pandas()

    # coerce numeric columns again in pandas to be safe
    for c in ['timestamp', 'x', 'y', 'z']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # keep only expected columns if present, fill missing with nulls
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    df = df[expected_cols]

    # Try a fast bulk insert using LOAD DATA LOCAL INFILE if possible; fall back to pandas.to_sql
    try:
        inserted = False
        try:
            inserted = insert_using_load_data(engine, df, table_name='wisdm_raw')
        except Exception as e:
            logging.debug(f"Fast bulk load failed or unavailable: {e}; falling back to pandas.to_sql")

        if not inserted:
            logging.info(f"Appending {len(df)} rows to wisdm_raw using pandas.to_sql")
            # chunked multi-row insert to avoid huge single statements
            df.to_sql('wisdm_raw', con=engine, if_exists='append', index=False, method='multi', chunksize=10000)
            logging.info(f"Appended {len(df)} rows to wisdm_raw using pandas.to_sql.")
    except Exception as e:
        logging.exception(f"Failed to append aggregated data into DB: {e}")
        raise


def insert_using_load_data(engine, df: 'pd.DataFrame', table_name: str = 'wisdm_raw') -> bool:
    """Attempt to bulk-load the dataframe into MySQL using LOAD DATA LOCAL INFILE.

    Returns True on success. Raises on fatal errors. If the server doesn't allow LOCAL INFILE
    this function will raise and the caller will fall back to another method.
    """
    import tempfile
    import os

    if df.empty:
        logging.info("DataFrame is empty, nothing to insert")
        return True

    # Write to a temporary CSV file. Use \N for NULLs which MySQL treats as NULL by default.
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
    tmp_name = tmp.name
    try:
        # Use comma separator and quote minimal fields to be safe
        df.to_csv(tmp_name, index=False, sep=',', na_rep='\\N', quoting=1)
        tmp.close()

        # Ensure path quoting for LOAD DATA
        path_for_sql = tmp_name.replace('\\', '\\\\')

        # Build column list matching our table schema; ensure order matches CSV header
        cols = list(df.columns)
        cols_sql = ','.join(f'`{c}`' for c in cols)

        conn = None
        cursor = None
        try:
            conn = engine.raw_connection()
            cursor = conn.cursor()
            # Use LOCAL because the file is on the client side
            # Build the LOAD DATA LOCAL INFILE statement using format(); ENCLOSED BY '"' is escaped
            sql = (
                "LOAD DATA LOCAL INFILE '{}' INTO TABLE {} "
                "FIELDS TERMINATED BY ',' ENCLOSED BY '\"' "
                "LINES TERMINATED BY '\\n' IGNORE 1 LINES "
                "({});"
            ).format(path_for_sql, table_name, cols_sql)
            logging.info(f"Running bulk LOAD DATA LOCAL INFILE for {len(df)} rows")
            cursor.execute(sql)
            conn.commit()
            return True
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
    finally:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
    return False

# module-level cached status and running flag
_is_populated = False
_is_running = False
_status_lock = threading.Lock()

def is_populated():
    with _status_lock:
        return _is_populated

def is_running():
    with _status_lock:
        return _is_running

def _set_populated(val: bool):
    global _is_populated
    with _status_lock:
        _is_populated = bool(val)

def _set_running(val: bool):
    global _is_running
    with _status_lock:
        _is_running = bool(val)

def populate(prefix_path: str = None):
    if prefix_path is None:
        prefix_path = DataConfig.data_path_prefix

    _set_populated(False)
    engine = get_engine()

    if not wait_for_db(engine, timeout_s=120):
        raise RuntimeError("Database did not become available in time")

    try:
        create_status_table(engine)
    except Exception:
        logging.debug('Could not create init_status table before reading status')

    # Acquire a named lock and hold it for the whole populate operation to prevent
    # concurrent population across multiple containers.
    lock_name = 'wisdm_populate_lock'
    lock_conn = None
    got_lock = False
    try:
        lock_conn = engine.connect()
        try:
            got_lock = bool(lock_conn.execute(text("SELECT GET_LOCK(:name, :timeout)"), {"name": lock_name, "timeout": 10}).scalar())
        except Exception:
            got_lock = False

        if not got_lock:
            logging.info("Another instance is populating (couldn't acquire lock); exiting")
            return

        # If the DB already has is_populated=1, honor it and skip re-population.
        try:
            persisted = get_status_from_db(engine, 'is_populated')
            if persisted in ('1', 'true', 'True'):
                logging.info('Database already populated according to init_status; skipping populate')
                _set_populated(True)
                return
        except Exception:
            logging.debug('Could not read persisted is_populated flag before populate')

        # Run the populate while we hold the lock
        create_table(engine)
        parse_and_insert(engine, prefix_path)
        _set_populated(True)
        try:
            logging.info("Putting is_populated flag to DB")
            set_status_in_db(engine, 'is_populated', '1')
        except Exception:
            logging.exception('Failed to persist is_populated flag to DB')

    except Exception:
        logging.exception('Error during populate with locking')
        return
    finally:
        # release lock if we acquired it
        try:
            if lock_conn is not None:
                lock_conn.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": lock_name})
                lock_conn.close()
        except Exception:
            logging.debug('Failed to release populate lock')

    logging.info("Finished populating database")

logger.info("Starting flask app.")
app = Flask(__name__)

# On module import try to read persisted populated state from DB (best-effort, short timeout).
try:
    _init_engine = get_engine()
    if wait_for_db(_init_engine, timeout_s=5):
        val = get_status_from_db(_init_engine, 'is_populated')
        if val is not None:
            _set_populated(val in ('1', 'true', 'True'))
except Exception:
    # If DB isn't available at import time that's fine; status stays False until updated.
    logging.debug('Could not determine persisted populate status at import time')

def _background_populate(prefix_path: str = None):
    try:
        _set_running(True)
        populate(prefix_path)
    except Exception as exc:
        logging.exception(f"Background populate failed: {exc}")
    finally:
        _set_running(False)

@app.route('/status', methods=['GET'])
def status_endpoint():
    return jsonify({'is_populated': is_populated(), 'is_running': is_running()}), HTTPStatus.OK

@app.route('/populate', methods=['GET'])
def populate_endpoint():
    if is_running():
        return jsonify({'error': 'Populate already running'}), HTTPStatus.CONFLICT

    # Prefer persisted status in DB if available
    try:
        engine = get_engine()
        if wait_for_db(engine, timeout_s=5):
            persisted = get_status_from_db(engine, 'is_populated')
            if persisted in ('1', 'true', 'True'):
                _set_populated(True)
                return jsonify({'started': False, 'reason': 'already populated (persisted)'}), HTTPStatus.CONFLICT
    except Exception:
        logging.debug('Could not check persisted populate status; falling back to in-memory flag')

    # fall back to in-memory cached flag
    if is_populated():
        return jsonify({'started': False, 'reason': 'already populated (in-memory)'}), HTTPStatus.CONFLICT

    # start background job using configured DataConfig path
    t = threading.Thread(target=_background_populate, daemon=True)
    t.start()
    return jsonify({'started': True}), HTTPStatus.ACCEPTED


