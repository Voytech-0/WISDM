import json
import logging
import random
import sys
import time
import requests
import src.preprocessing.preprocess as preprocess


def main():
    host = "127.0.0.1"
    port = 8000
    interval_min = 5.0
    interval_max = 10.0
    batch_size = 1

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    url = f"http://{host}:{port}/predict"

    logging.info("Loading local data...")
    lf = preprocess.load_raw_data_from_local()
    if lf is None:
        logging.error("No local data found under DataConfig.data_path_prefix")
        sys.exit(1)

    try:
        lf = preprocess.raw_data_into_xyz_series(lf)
        lf = preprocess.convert_data_to_fixed_length_intervals(lf)
        local_rows = lf.collect().to_dicts()
    except Exception as exc:
        logging.exception("Failed to collect/preprocess local rows: %s", exc)
        sys.exit(1)

    if not local_rows:
        logging.error("No rows produced by preprocessing of local data")
        sys.exit(1)

    logging.info("Loaded %d rows. Starting loop...", len(local_rows))
    logging.info("Posting to %s (Interval: %.1f-%.1fs)", url, interval_min, interval_max)

    try:
        while True:
            batch_rows = random.choices(local_rows, k=batch_size)

            items = [
                {
                    "device": r.get("device"),
                    "sensor": r.get("sensor"),
                    "x": r.get("x"),
                    "y": r.get("y"),
                    "z": r.get("z")
                }
                for r in batch_rows
            ]
            payload = items[0] if batch_size == 1 else items

            try:
                resp = requests.post(url, json=payload, timeout=6.0)
                if resp.ok:
                    logging.info("Received %s — code=%d", json.dumps(resp.json()), resp.status_code)
                else:
                    logging.warning("Non-OK response %d: %s", resp.status_code, resp.text[:200])
            except Exception as exc:
                logging.error("Failed to send payload: %s", exc)

            sleep_t = random.uniform(interval_min, interval_max)
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        logging.info("Interrupted by user — exiting")


if __name__ == "__main__":
    main()