import os
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock, patch
import polars as pl
import db_init.populate_db
from db_init.populate_db import (
    wait_for_db,
    create_table,
    create_status_table,
    set_status_in_db,
    get_status_from_db,
    load_data_raw_with_row_per_sensor_device_activity_subject,
    insert_using_load_data,
    populate,
    app
)
from configs.configs import MySQLConfig



@pytest.fixture
def mock_engine():
    mock_eng = MagicMock()
    mock_conn = MagicMock()
    mock_eng.connect.return_value.__enter__.return_value = mock_conn
    return mock_eng, mock_conn


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def test_wait_for_db_success(mock_engine):
    engine, conn = mock_engine
    conn.execute.return_value = None

    assert wait_for_db(engine, timeout_s=1) is True
    args, _ = conn.execute.call_args
    assert "SELECT 1" in str(args[0])


def test_wait_for_db_failure(mock_engine):
    engine, conn = mock_engine
    conn.execute.side_effect = Exception("DB Down")

    with patch("time.sleep", return_value=None):
        assert wait_for_db(engine, timeout_s=0.1) is False


def test_create_table(mock_engine):
    engine, conn = mock_engine
    create_table(engine)
    args, _ = conn.execute.call_args
    assert "CREATE TABLE IF NOT EXISTS wisdm_raw" in str(args[0])


def test_create_status_table(mock_engine):
    engine, conn = mock_engine
    create_status_table(engine)
    args, _ = conn.execute.call_args
    assert "CREATE TABLE IF NOT EXISTS init_status" in str(args[0])


def test_set_status_in_db(mock_engine, monkeypatch):
    engine, conn = mock_engine
    monkeypatch.setattr(db_init.populate_db, "get_connected_engine", lambda timeout_s=10: engine)

    set_status_in_db("test_key", "test_val")

    args, _ = conn.execute.call_args
    assert "INSERT INTO init_status" in str(args[0])
    assert args[1] == {"k": "test_key", "v": "test_val"}


def test_get_status_from_db(mock_engine, monkeypatch):
    engine, conn = mock_engine
    monkeypatch.setattr(db_init.populate_db, "get_connected_engine", lambda timeout_s=5: engine)

    conn.execute.return_value.fetchone.return_value = ("found_value",)
    assert get_status_from_db("some_key") == "found_value"

    conn.execute.return_value.fetchone.return_value = None
    assert get_status_from_db("missing_key") is None


def test_load_data_raw(tmp_dir):
    d = os.path.join(tmp_dir, "subdir")
    os.makedirs(d)

    f_path = os.path.join(d, "data_1_101_accel_phone.txt")

    with open(f_path, "w") as f:
        f.write("101,Walk,1000,0.1,0.2,0.3;\n")

    lf = load_data_raw_with_row_per_sensor_device_activity_subject(tmp_dir)
    df = lf.collect()

    assert df.height == 1
    assert df["subject_id"][0] == '101'
    assert df["sensor"][0] == "accel"
    assert df["device"][0] == "phone"


def test_insert_using_load_data(monkeypatch):
    mock_pymysql = MagicMock()
    monkeypatch.setattr(db_init.populate_db, "pymysql", mock_pymysql)

    monkeypatch.setattr(MySQLConfig, "host", "localhost")
    monkeypatch.setattr(MySQLConfig, "user", "user")
    monkeypatch.setattr(MySQLConfig, "password", "pass")
    monkeypatch.setattr(MySQLConfig, "db", "db")
    monkeypatch.setattr(MySQLConfig, "port", 3306)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_pymysql.connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

    result = insert_using_load_data(df, "test_table")

    assert result is True
    args, _ = mock_cursor.execute.call_args_list[-1]
    sql = args[0]
    assert "LOAD DATA LOCAL INFILE" in sql
    assert "INTO TABLE test_table" in sql

def test_populate_success(monkeypatch, tmp_dir):
    monkeypatch.setattr(db_init.populate_db, "get_status_from_db", lambda k: None)
    monkeypatch.setattr(db_init.populate_db, "get_connected_engine", lambda timeout_s=120: MagicMock())
    monkeypatch.setattr(db_init.populate_db, "create_status_table", lambda e: None)
    monkeypatch.setattr(db_init.populate_db, "create_table", lambda e: None)

    mock_parse = MagicMock()
    monkeypatch.setattr(db_init.populate_db, "parse_and_insert", mock_parse)

    mock_set_status = MagicMock()
    monkeypatch.setattr(db_init.populate_db, "set_status_in_db", mock_set_status)

    success, reason = populate(tmp_dir)

    assert success is True
    assert reason == "populated"
    mock_parse.assert_called_once()
    mock_set_status.assert_called_with('is_populated', '1')


def test_populate_db_unavailable(monkeypatch):

    def raise_err(*args, **kwargs):
        raise RuntimeError("DB Down")

    monkeypatch.setattr(db_init.populate_db, "get_connected_engine", raise_err)

    success, reason = populate("dummy")
    assert success is False
    assert reason == "db_unavailable"


def test_flask_status_endpoint(monkeypatch):
    client = app.test_client()
    monkeypatch.setattr(db_init.populate_db, "is_populated", lambda: True)

    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json == {"is_populated": True}


def test_flask_populate_endpoint_already_done(monkeypatch):
    client = app.test_client()
    monkeypatch.setattr(db_init.populate_db, "get_status_from_db", lambda k: '1')

    resp = client.get("/populate")
    assert resp.status_code == 200
    assert resp.json == {'started': False, 'populated': True, 'reason': 'already_populated'}


def test_flask_populate_endpoint_success(monkeypatch):
    client = app.test_client()
    monkeypatch.setattr(db_init.populate_db, "get_status_from_db", lambda k: None)
    monkeypatch.setattr(db_init.populate_db, "populate", lambda: (True, "populated"))

    resp = client.get("/populate")
    assert resp.status_code == 200
    assert resp.json == {"started": True, "populated": True, "reason": "populated"}


def test_flask_populate_endpoint_fail(monkeypatch):
    client = app.test_client()
    monkeypatch.setattr(db_init.populate_db, "get_status_from_db", lambda k: None)
    monkeypatch.setattr(db_init.populate_db, "populate", lambda: (False, "error:oops"))

    resp = client.get("/populate")
    assert resp.status_code == 500
    assert "Populate failed" in resp.json["error"]
