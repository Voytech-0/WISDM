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

def set_status_in_db(engine=None, key: str = None, value: str = None):
    """
    Upsert a key/value into init_status. If no engine is provided, create one and try to write.
    """
    if key is None:
        raise ValueError("key is required")

    created_engine = False
    if engine is None:
        engine = get_engine()
        created_engine = True
        if not wait_for_db(engine, timeout_s=10):
            raise RuntimeError("Database not available to set status")

    query = """
    INSERT INTO init_status (`key`, `value`) VALUES (:k, :v)
    ON DUPLICATE KEY UPDATE `value` = :v, updated_at = CURRENT_TIMESTAMP;
    """
    try:
        with engine.connect() as conn:
            conn.execute(text(query), {"k": key, "v": value})
    finally:
        # If we created the engine locally, dispose it to free resources
        if created_engine:
            try:
                engine.dispose()
            except Exception:
                pass


def get_status_from_db(engine=None, key: str = None) -> str | None:
    """
    Read a value from init_status. If no engine is provided, create one and try briefly.
    Returns the stored string value or None if not present / not reachable.
    """
    if key is None:
        raise ValueError("key is required")

    created_engine = False
    if engine is None:
        engine = get_engine()
        created_engine = True
        if not wait_for_db(engine, timeout_s=5):
            logging.debug("DB not available when reading status")
            return None

    q = "SELECT `value` FROM init_status WHERE `key` = :k LIMIT 1"
    try:
        with engine.connect() as conn:
            r = conn.execute(text(q), {"k": key}).fetchone()
            if r:
                return r[0]
    except Exception:
        logging.debug("Could not read init_status from DB (it may not be ready)")
    finally:
        if created_engine:
            try:
                engine.dispose()
            except Exception:
                pass
    return None


def load_data_raw_with_row_per_sensor_device_activity_subject(prefix_path: str) -> pl.LazyFrame:
    file_header = ['subject_id', 'label', 'timestamp', 'x', 'y', 'z']

    paths = list(Path(prefix_path).rglob("data_*.txt"))

    lf_all = []
    for p in paths:
        # scan_csv with eol_char=';' because lines end with ';' in raw files
        lf = pl.scan_csv(str(p), has_header=False, new_columns=file_header, eol_char=';\n')
        # derive sensor/device/subject from filename; we'll keep subject_id column from file
        _, sensor, device = str(p).removesuffix('.txt').split('_')[-3:]
        lf = (
            lf
            .with_columns(
                pl.lit(sensor).alias('sensor'),
                pl.lit(device).alias('device'),
                # pl.col('subject_id').cast(pl.Utf8, strict=False).str.replace_all(r'^\n', '').alias('subject_id')
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
        logging.warning(f"Missing expected columns: {missing}. Filling with nulls.")
        # Add missing columns with null values
        df_pl = df_pl.with_columns([pl.lit(None).alias(c) for c in missing])

    # Coerce numeric columns (non-strict)
    df_pl = df_pl.with_columns([
        pl.col('timestamp').cast(pl.Int64, strict=False),
        pl.col('x').cast(pl.Float64, strict=False),
        pl.col('y').cast(pl.Float64, strict=False),
        pl.col('z').cast(pl.Float64, strict=False),
    ])

    df_pl = df_pl.select(expected_cols)

    try:
        insert_using_load_data(engine, df_pl, table_name='wisdm_raw')
    except Exception as e:
        logging.exception(f"Failed to append aggregated data into DB: {e}")
        raise


def insert_using_load_data(engine, df: pl.DataFrame, table_name: str = 'wisdm_raw') -> bool:
    """
    Bulk-load Polars DataFrame into MySQL using LOAD DATA LOCAL INFILE.
    """
    import tempfile
    import os

    if df.height == 0:
        logging.info("DataFrame is empty, nothing to insert")
        return True

    # Create temp file
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    tmp_name = tmp.name
    tmp.close()  # Close file handle so Polars can open it

    try:
        logging.info("Writing Polars DataFrame to CSV...")
        df.write_csv(
            tmp_name,
            include_header=True,
            separator=',',
            quote_style='always',  # Equivalent to quoting=1 (QUOTE_ALL)
            null_value='\\N'       # MySQL standard for NULL
        )

        # Prepare SQL paths and columns
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
                local_infile=True, # Enables client capability
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
    """Read persisted populated state from the DB."""
    try:
        engine = get_engine()
        if not wait_for_db(engine, timeout_s=5):
            return False
        val = get_status_from_db(engine, 'is_populated')
        return val in ('1', 'true', 'True')
    except Exception:
        logging.debug('Could not determine persisted populate status')
        return False


def populate(prefix_path: str = None) -> tuple[bool, str]:
    """
    Populate the database from raw files. Returns (success: bool, reason: str).

    success == True indicates the database is populated (either already or after this call).
    success == False indicates the operation did not populate the DB (e.g., lock not acquired or error).
    """
    if prefix_path is None:
        prefix_path = DataConfig.data_path_prefix

    engine = get_engine()

    if not wait_for_db(engine, timeout_s=120):
        return (False, "db_unavailable")

    try:
        create_status_table(engine)
    except Exception:
        logging.debug('Could not create init_status table before reading status')

    # Run the populate operation (no locking, single-instance assumption)
    try:
        # If the DB already has is_populated=1, honor it and skip re-population.
        try:
            persisted = get_status_from_db(engine, 'is_populated')
            if persisted in ('1', 'true', 'True'):
                logging.info('Database already populated according to init_status; skipping populate')
                return (True, 'already_populated')
        except Exception:
            logging.debug('Could not read persisted is_populated flag before populate')

        create_table(engine)
        parse_and_insert(engine, prefix_path)
        try:
            logging.info("Putting is_populated flag to DB")
            set_status_in_db(engine, 'is_populated', '1')
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
    # Prefer persisted status in DB if available
    try:
        engine = get_engine()
        if wait_for_db(engine, timeout_s=5):
            persisted = get_status_from_db(engine, 'is_populated')
            if persisted in ('1', 'true', 'True'):
                return jsonify({'started': False, 'populated': True, 'reason': 'already_populated'}), HTTPStatus.OK
    except Exception:
        logging.debug('Could not check persisted populate status; falling back to direct populate')

    success, reason = populate()
    if success:
        return jsonify({'started': True, 'populated': True, 'reason': reason}), HTTPStatus.OK

    if reason == 'db_unavailable':
        return jsonify({'error': 'Database unavailable during populate attempt'}), HTTPStatus.SERVICE_UNAVAILABLE

    logging.error('Populate failed: %s', reason)
    return jsonify({'error': f'Populate failed: {reason}'}), HTTPStatus.INTERNAL_SERVER_ERROR

