"""01 INGEST — real centroids + elevation (OpenTopoData) + weather (OpenWeatherMap)."""
import os
import time
from datetime import date, datetime
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

PANCHAYAT_STATIC_PATH = "data/raw/panchayat_static.csv"
OUT_PATH = "data/processed/block_panchayat_features.csv"
ELEVATION_API = "https://api.opentopodata.org/v1/srtm30m"
OWM_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
REQUEST_TIMEOUT = 30


def load_panchayats():
    df = pd.read_csv(PANCHAYAT_STATIC_PATH)
    df["panchayat_id"] = df["panchayat_id"].astype(str)
    df["block_id"] = df["block_id"].astype(str)
    df["district_id"] = df["district_id"].astype(str)
    df["panchayat_name"] = df["panchayat_name"].astype(str)
    df["block_name"] = df["block_name"].astype(str)
    print(f"[ingest] Loaded {len(df)} panchayats, {df['block_id'].nunique()} blocks")
    return df


def fetch_elevation(lats, lons, batch_size=100):
    values = []
    n = len(lats)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        locations = "|".join(f"{lats[i]},{lons[i]}" for i in range(start, end))
        params = {"locations": locations}
        for attempt in range(3):
            try:
                r = requests.get(ELEVATION_API, params=params, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                for res in r.json().get("results", []):
                    e = res.get("elevation")
                    values.append(float(e) if e is not None else 200.0)
                print(f"[ingest] Elevation batch {start}-{end}: OK")
                break
            except Exception as e:
                print(f"[ingest] batch {start}-{end} attempt {attempt+1}: {e}")
                if attempt == 2:
                    values.extend([200.0] * (end - start))
                else:
                    time.sleep(2 ** attempt)
        time.sleep(1.0)
    return np.array(values, dtype=float)


def derive_ndvi(elev, lat):
    e_norm = (elev - elev.min()) / (elev.max() - elev.min() + 1e-9)
    ndvi = 0.65 - 0.35 * e_norm + 0.05 * np.sin(np.deg2rad(lat))
    return np.clip(ndvi, 0.05, 0.85)


def fetch_block_weather(block_ids, lats, lons):
    """
    Fetch 5-day / 3-hour forecast from OpenWeatherMap for each block centroid,
    aggregate to daily values, return one weather dict per block.
    """
    print("[ingest] Fetching weather from OpenWeatherMap for each block...")
    if not OWM_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY missing in .env")

    df_blocks = pd.DataFrame({"block_id": block_ids, "lat": lats, "lon": lons})
    block_centroids = df_blocks.groupby("block_id").mean()

    weather = {}
    for block_id, row in block_centroids.iterrows():
        params = {
            "lat": row["lat"], "lon": row["lon"],
            "appid": OWM_API_KEY, "units": "metric",
        }
        try:
            r = requests.get(OWM_FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()

            # Aggregate 3-hour entries by day
            daily = {}
            for entry in data["list"]:
                day = datetime.fromtimestamp(entry["dt"]).date()
                if day not in daily:
                    daily[day] = {"tmax": [], "tmin": [], "rain": 0.0, "wind": []}
                daily[day]["tmax"].append(entry["main"]["temp_max"])
                daily[day]["tmin"].append(entry["main"]["temp_min"])
                daily[day]["wind"].append(entry["wind"]["speed"])
                if "rain" in entry:
                    daily[day]["rain"] += entry["rain"].get("3h", 0.0)

            # Use the first (earliest) forecast day as "today"
            first_day = sorted(daily.keys())[0]
            d = daily[first_day]

            weather[block_id] = {
                "rainfall": round(d["rain"], 1),
                "tmax": round(float(np.max(d["tmax"])), 1),
                "tmin": round(float(np.min(d["tmin"])), 1),
                "rh": 75.0,                      # not in free tier
                "wind_speed": round(float(np.mean(d["wind"])) * 3.6, 1),
                "solar_radiation": 18.0,          # not in free tier
            }
            print(f"[ingest]   {block_id}: rain={weather[block_id]['rainfall']}mm "
                  f"tmax={weather[block_id]['tmax']}C "
                  f"tmin={weather[block_id]['tmin']}C")
        except Exception as e:
            print(f"[ingest]   {block_id}: FAILED ({e}) — using mock")
            weather[block_id] = {
                "rainfall": 0.0, "tmax": 30.0, "tmin": 20.0,
                "rh": 70.0, "wind_speed": 8.0, "solar_radiation": 18.0,
            }
        time.sleep(0.3)
    return weather


def build_dataset():
    df = load_panchayats()

    print(f"[ingest] Querying OpenTopoData for {len(df)} panchayats...")
    df["elevation"] = fetch_elevation(df["lat"].values, df["lon"].values)

    df["ndvi"] = derive_ndvi(df["elevation"].values, df["lat"].values)
    df["slope"] = np.clip((df["elevation"].diff().abs().fillna(0) / 100.0), 0, 20).round(2)
    df["dist_water"] = np.round(np.random.default_rng(7).uniform(0.5, 8.0, len(df)), 2)
    df["soil_type"] = "loam"

    bw = fetch_block_weather(df["block_id"].tolist(), df["lat"].values, df["lon"].values)
    for k in ["rainfall", "tmax", "tmin", "rh", "wind_speed", "solar_radiation"]:
        df[k] = df["block_id"].map(lambda bid: bw[bid][k])

    df["date"] = str(date.today())
    return df


def save_dataset(df, path=OUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[ingest] Saved {len(df)} rows -> {path}")


if __name__ == "__main__":
    df = build_dataset()
    save_dataset(df)
    cols = ["panchayat_id", "block_id", "elevation", "rainfall", "tmax", "tmin", "wind_speed"]
    print("\nSample (block weather broadcast to panchayats):")
    print(df[cols].head(10).to_string(index=False))