from flask import Flask, jsonify
import os
import urllib.request
import urllib.error

app = Flask(__name__)

DB_INIT_HEALTH_URL = os.environ.get("DB_INIT_HEALTH_URL", "http://db_init:5000/health")

@app.route("/health")
def health():
    """Proxy health: return 200 only when db_init reports healthy.

    The app will query the `db_init` service at `DB_INIT_HEALTH_URL` and return 200
    when that endpoint responds with HTTP 200. Otherwise return 503.
    """
    try:
        resp = urllib.request.urlopen(DB_INIT_HEALTH_URL, timeout=2)
        status_code = resp.getcode()
    except Exception:
        status_code = None

    if status_code == 200:
        return jsonify(status="ok"), 200
    return jsonify(status="db_init_pending"), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
