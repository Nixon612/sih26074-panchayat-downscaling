"""04 SERVE — Flask dashboard."""
import os
import json
import time
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory

from ingest import build_dataset, save_dataset
from downscale import run as run_downscale
from advisor import advise_all, to_json

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

DEFAULT_BLOCKS = ["BLK001", "BLK002", "BLK003"]
CACHE_TTL = 60 * 30
_cache = {}


def cget(k):
    e = _cache.get(k)
    return e["v"] if e and time.time() - e["t"] < CACHE_TTL else None


def cset(k, v):
    _cache[k] = {"v": v, "t": time.time()}


def ensure_model():
    if not (os.path.exists("models/rf_downscaler.joblib")
            and os.path.exists("models/feature_columns.json")):
        from train_model import train, save_artifacts
        df = pd.read_csv("data/processed/block_panchayat_features.csv")
        m, f, met = train(df)
        save_artifacts(m, f, met)


def get_downscaled_data():
    """Expensive stage: ingest + train + downscale. Cached across crop changes."""
    cached = cget("downscaled_records")
    if cached is not None:
        return pd.DataFrame(cached)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    save_dataset(build_dataset())
    ensure_model()
    ds = run_downscale("data/processed/block_panchayat_features.csv")
    cset("downscaled_records", ds.to_dict("records"))
    return ds


def run_pipeline(blocks, crop=None):
    """Cheap stage: run advisor on cached downscaled data."""
    ds = get_downscaled_data()
    return json.loads(to_json(advise_all(ds, crop=crop)))


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_blocks=DEFAULT_BLOCKS,
        now=datetime.utcnow().strftime("%d %b %Y, %H:%M UTC"),
    )


@app.route("/api/reports")
def api_reports():
    crop = request.args.get("crop") or None
    key = f"r:default:{crop or 'g'}"

    cached = cget(key)
    if cached:
        return jsonify({"source": "cache", "count": len(cached), "reports": cached})

    reports = run_pipeline(DEFAULT_BLOCKS, crop=crop)
    cset(key, reports)
    return jsonify({"source": "fresh", "count": len(reports), "reports": reports})


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    body = request.get_json(silent=True) or {}
    crop = body.get("crop")
    key = f"r:default:{crop or 'g'}"

    # Cache hit — return immediately
    reports = cget(key)
    if reports:
        return jsonify({"status": "ok", "source": "cache", "count": len(reports)})

    # Cache miss — run cheap advisor pass on cached downscaled data
    t0 = time.time()
    reports = run_pipeline(DEFAULT_BLOCKS, crop=crop)
    cset(key, reports)
    return jsonify({
        "status": "ok",
        "source": "fresh",
        "elapsed_sec": round(time.time() - t0, 2),
        "count": len(reports),
    })


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "model_present": os.path.exists("models/rf_downscaler.joblib"),
        "downscaled_cached": cget("downscaled_records") is not None,
        "time_utc": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/plots/<path:f>")
def serve_plot(f):
    return send_from_directory("static/plots", f)


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ---------- Background warmup on boot ----------
import threading

_warmup_started = False


def _warmup_pipeline():
    global _warmup_started
    if _warmup_started:
        return
    _warmup_started = True
    try:
        print("[warmup] Running full pipeline in background...")
        ds = get_downscaled_data()
        for crop in [None, "rice", "wheat", "cotton", "sugarcane"]:
            reports = json.loads(to_json(advise_all(ds, crop=crop)))
            cset(f"r:default:{crop or 'g'}", reports)
        print("[warmup] Done — cached reports for 5 crop variants")
    except Exception as e:
        print(f"[warmup] Failed: {e}")


threading.Thread(target=_warmup_pipeline, daemon=True).start()


if __name__ == "__main__":
    os.makedirs("static/plots", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False)