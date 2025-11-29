#!/usr/bin/env bash
set -euo pipefail

MYSQL_HOST="${MYSQL_HOST:-mysql}"
MYSQL_USER="${MYSQL_USER:-user}"
MYSQL_PASSWORD="${MYSQL_PASSWORD:-password}"

echo "Starting Flask app..."
exec python -m flask run --app db_init.db_init_api --host 0.0.0.0 --port 5001

echo "Waiting for MySQL at ${MYSQL_HOST}..."
until mysqladmin ping -h "${MYSQL_HOST}" -u "${MYSQL_USER}" -p"${MYSQL_PASSWORD}" --silent; do
  sleep 1
done

echo "Running DB population..."
python db_init/populate_db.py

