"""04 SERVE — Flask dashboard."""
import os, json, time
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
    if not (os.path.exists("models/rf_downscaler.joblib") and os.path.exists("models/feature_columns.json")):
        from train_model import train, save_artifacts
        df = pd.read_csv("data/processed/block_panchayat_features.csv")
        m, f, met = train(df); save_artifacts(m, f, met)

def run_pipeline(blocks, crop=None):
    save_dataset(build_dataset())
    ensure_model()
    ds = run_downscale("data/processed/block_panchayat_features.csv")
    return json.loads(to_json(advise_all(ds, crop=crop)))
@app.route("/")
def index():
    return render_template("index.html", default_blocks=DEFAULT_BLOCKS,
                           now=datetime.utcnow().strftime("%d %b %Y, %H:%M UTC"))

@app.route("/api/reports")
def api_reports():
    blocks = request.args.get("blocks", ",".join(DEFAULT_BLOCKS)).split(",")
    crop = request.args.get("crop") or None
    key = f"r:{','.join(blocks)}:{crop or 'g'}"
    if (c := cget(key)):
        return jsonify({"source": "cache", "count": len(c), "reports": c})
    reports = run_pipeline(blocks, crop=crop)
    cset(key, reports)
    return jsonify({"source": "fresh", "count": len(reports), "reports": reports})

@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    body = request.get_json(silent=True) or {}
    blocks = body.get("blocks") or DEFAULT_BLOCKS
    crop = body.get("crop")
    _cache.clear()
    t0 = time.time()
    reports = run_pipeline(blocks, crop=crop)
    cset(f"r:{','.join(blocks)}:{crop or 'g'}", reports)
    return jsonify({"status": "ok", "elapsed_sec": round(time.time()-t0, 2), "count": len(reports)})

@app.route("/plots/<path:f>")
def plots(f):
    return send_from_directory("static/plots", f)

import threading

_warmup_started = False

def _warmup_pipeline():
    """Run the pipeline once in the background when the server boots."""
    global _warmup_started
    if _warmup_started:
        return
    _warmup_started = True
    try:
        print("[warmup] Running pipeline in background...")
        reports = run_pipeline(DEFAULT_BLOCKS, crop=None)
        cset(f"r:{','.join(DEFAULT_BLOCKS)}:g", reports)
        print(f"[warmup] Done — cached {len(reports)} reports")
    except Exception as e:
        print(f"[warmup] Failed: {e}")

threading.Thread(target=_warmup_pipeline, daemon=True).start()
if __name__ == "__main__":
    os.makedirs("static/plots", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)