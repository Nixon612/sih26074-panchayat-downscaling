"""02 PREDICT — Random Forest spatial downscaler."""
import os, json
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

IN_PATH = "data/processed/block_panchayat_features.csv"
MODEL_PATH = "models/rf_downscaler.joblib"
FEATURE_PATH = "models/feature_columns.json"
METRICS_PATH = "models/rf_metrics.json"

BLOCK_F = ["rainfall", "tmax", "tmin", "rh", "wind_speed", "solar_radiation"]
STATIC_F = ["elevation", "slope", "ndvi", "dist_water"]
TARGETS = ["rainfall", "tmax", "tmin", "rh", "wind_speed", "solar_radiation"]
LAPSE = 6.5 / 1000.0

def generate_truth(df):
    elev = df["elevation"].values
    slope = df["slope"].values
    ndvi = df["ndvi"].values
    dw = df["dist_water"].values
    eref = df.groupby("block_id")["elevation"].transform("mean").values
    de = elev - eref
    out = pd.DataFrame(index=df.index)
    out["tmax"] = df["tmax"].values - LAPSE * de - 0.05 * slope
    out["tmin"] = df["tmin"].values - LAPSE * 1.1 * de - 0.03 * slope
    oro = 1.0 + 0.0006 * de + 0.01 * slope
    water = np.exp(-0.15 * dw)
    out["rainfall"] = df["rainfall"].values * oro * (1.0 + 0.15 * water)
    out["rh"] = (df["rh"].values + 0.02 * de + 8.0 * water - 0.2 * slope).clip(10, 100)
    out["wind_speed"] = (df["wind_speed"].values * (1 + 0.0008 * de) * (1 + 0.02 * slope) * (1 - 0.4 * ndvi)).clip(0)
    out["solar_radiation"] = df["solar_radiation"].values * (1 + 0.00005 * de) * np.cos(np.deg2rad(np.clip(slope, 0, 60)))
    rng = np.random.default_rng(42)
    for t in TARGETS:
        out[t] = out[t] * (1 + rng.normal(0, 0.03, size=len(out)))
    return out

def engineer_features(df):
    X = df[BLOCK_F + STATIC_F].copy()
    X["elev_x_slope"] = X["elevation"] * X["slope"]
    X["water_prox"] = np.exp(-0.15 * X["dist_water"])
    X["veg_shelter"] = X["ndvi"] * X["wind_speed"]
    return pd.concat([X, pd.get_dummies(df["soil_type"], prefix="soil", dtype=int)], axis=1)

def train(df):
    truth = generate_truth(df)
    X = engineer_features(df)
    y = truth[TARGETS]
    cols = list(X.columns)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)
    m.fit(Xtr, ytr)
    yp = m.predict(Xte)
    metrics = {}
    print("\n=== Per-target metrics ===")
    for i, t in enumerate(TARGETS):
        rmse = float(np.sqrt(mean_squared_error(yte.iloc[:, i], yp[:, i])))
        mae = float(mean_absolute_error(yte.iloc[:, i], yp[:, i]))
        r2 = float(r2_score(yte.iloc[:, i], yp[:, i]))
        metrics[t] = {"rmse": rmse, "mae": mae, "r2": r2}
        print(f"  {t:16s} RMSE={rmse:7.3f}  R2={r2:6.3f}")
    return m, cols, metrics

def save_artifacts(m, cols, metrics):
    os.makedirs("models", exist_ok=True)
    joblib.dump(m, MODEL_PATH)
    json.dump(cols, open(FEATURE_PATH, "w"), indent=2)
    json.dump(metrics, open(METRICS_PATH, "w"), indent=2)
    print(f"\n[train] Saved model -> {MODEL_PATH}")

if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    print(f"[train] Loaded {len(df)} rows")
    m, c, met = train(df)
    save_artifacts(m, c, met)