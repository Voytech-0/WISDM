#!/usr/bin/env python3
"""
Simple fake client that POSTs synthetic x/y/z time-series plus `device` and `sensor`
fields to the /predict endpoint at a configurable interval between --interval-min and
--interval-max seconds (defaults to 5-10s).

Usage examples:
  python scripts/fake_client.py --host localhost --port 8000 --count 10
  python scripts/fake_client.py --batch-size 3 --series-len 200 --device phone,watch

The script uses only the `requests` package (added to requirements.txt).
"""

import argparse
import json
import logging
import math
import random
import sys
import time
import requests


def generate_series(length: int, base_freq: float = 1.0, noise_std: float = 0.05):
    """Generate a single axis time-series using a sum of sines plus Gaussian noise.

    Returns a list[float] of length `length`.
    """
    series = []
    for i in range(length):
        t = i / float(length)  # normalized time 0..1
        # base sine + second harmonic + small random jitter
        val = (
            math.sin(2 * math.pi * base_freq * t)
            + 0.5 * math.sin(2 * math.pi * (base_freq * 2) * t + 0.3)
            + (random.gauss(0, noise_std))
        )
        series.append(val)
    return series


def make_item(device: str, sensor: str, series_len: int) -> dict:
    """Create one payload item with x,y,z lists plus device and sensor metadata."""
    # vary base_freq slightly per axis so signals differ
    fx = 1.0 + random.uniform(-0.3, 0.3)
    fy = 1.0 + random.uniform(-0.3, 0.3)
    fz = 1.0 + random.uniform(-0.3, 0.3)

    x = generate_series(series_len, base_freq=fx, noise_std=0.08)
    y = generate_series(series_len, base_freq=fy, noise_std=0.08)
    z = generate_series(series_len, base_freq=fz, noise_std=0.1)

    return {
        "device": device,
        "sensor": sensor,
        "x": x,
        "y": y,
        "z": z,
    }


def send_payload(url: str, payload, max_retries: int = 3, timeout: float = 6.0):
    """POST payload (object or list) to the url with simple retry/backoff logic.

    Returns the requests.Response or raises the last exception.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            # treat 5xx as retriable, other statuses returned to caller
            if 500 <= resp.status_code < 600 and attempt < max_retries:
                sleep_for = 0.5 * (2 ** (attempt - 1))
                logging.warning("Server error %s; retrying after %.1fs (attempt %d)", resp.status_code, sleep_for, attempt)
                time.sleep(sleep_for)
                continue
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                sleep_for = 0.5 * (2 ** (attempt - 1))
                logging.warning("Request failed (%s); retrying after %.1fs (attempt %d)", exc, sleep_for, attempt)
                time.sleep(sleep_for)
            else:
                logging.error("Request failed after %d attempts: %s", attempt, exc)
    if last_exc:
        raise last_exc


def parse_args():
    p = argparse.ArgumentParser(description="Fake client for /predict endpoint (sends x/y/z series)")
    p.add_argument("--host", default="127.0.0.1", help="Server host")
    p.add_argument("--port", default=8000, type=int, help="Server port")
    p.add_argument("--interval-min", default=5.0, type=float, help="Minimum interval between requests (seconds)")
    p.add_argument("--interval-max", default=10.0, type=float, help="Maximum interval between requests (seconds)")
    p.add_argument("--batch-size", default=1, type=int, help="Number of items to send per POST (1 = single object)")
    p.add_argument("--series-len", default=200, type=int, help="Length of x/y/z time-series")
    p.add_argument("--device", default="phone,watch", help="Comma-separated device names to sample from")
    p.add_argument("--sensor", default="accel,gyro", help="Comma-separated sensor names to sample from")
    p.add_argument("--count", default=0, type=int, help="How many POSTs to send (0 = forever)")
    p.add_argument("--seed", default=None, type=int, help="Optional random seed for reproducibility")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.seed is not None:
        random.seed(args.seed)

    devices = [d.strip() for d in args.device.split(",") if d.strip()]
    sensors = [s.strip() for s in args.sensor.split(",") if s.strip()]
    if not devices or not sensors:
        logging.error("At least one device and sensor must be provided")
        sys.exit(2)

    url = f"http://{args.host}:{args.port}/predict"
    logging.info("Posting to %s — batch_size=%d series_len=%d interval=[%.1f,%.1f]", url, args.batch_size, args.series_len, args.interval_min, args.interval_max)

    sent = 0
    try:
        while True:
            # build payload
            items = [make_item(random.choice(devices), random.choice(sensors), args.series_len) for _ in range(args.batch_size)]
            payload = items[0] if args.batch_size == 1 else items

            logging.debug("Payload keys: %s; first x[0]=%.3f", list(items[0].keys()), items[0]["x"][0])

            try:
                resp = send_payload(url, payload)
            except Exception as exc:
                logging.exception("Failed to send payload: %s", exc)
            else:
                try:
                    text = resp.text
                    if resp.ok:
                        data = resp.json()
                        logging.info("Received %s — code=%d", json.dumps(data), resp.status_code)
                    else:
                        logging.warning("Non-OK response %d: %s", resp.status_code, text[:200])
                except ValueError:
                    logging.warning("Response not JSON (status=%d): %s", resp.status_code, resp.text[:200])

            sent += 1
            if args.count > 0 and sent >= args.count:
                logging.info("Reached requested count=%d — exiting", args.count)
                break

            sleep_t = random.uniform(args.interval_min, args.interval_max)
            logging.debug("Sleeping %.2fs before next request", sleep_t)
            time.sleep(sleep_t)
    except KeyboardInterrupt:
        logging.info("Interrupted by user — exiting")


if __name__ == "__main__":
    main()
