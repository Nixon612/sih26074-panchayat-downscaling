"""02b DOWNSCALE — inference + IDW contour plots."""
import os, json
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

IN_PATH = "data/processed/block_panchayat_features.csv"
MODEL_PATH = "models/rf_downscaler.joblib"
FEATURE_PATH = "models/feature_columns.json"
OUT_CSV = "data/processed/panchayat_downscaled.csv"
PLOT_DIR = "static/plots"

# Change this if you switch districts
DISTRICT_LABEL = "Kamrup District, Assam"

TARGETS = ["rainfall", "tmax", "tmin", "rh", "wind_speed", "solar_radiation"]
META = {
    "rainfall": ("YlGnBu", "Rainfall (mm)"),
    "tmax": ("YlOrRd", "Tmax (C)"),
    "tmin": ("coolwarm", "Tmin (C)"),
    "rh": ("BrBG", "RH (%)"),
    "wind_speed": ("viridis", "Wind (km/h)"),
    "solar_radiation": ("inferno", "Solar (MJ/m2)"),
}


def engineer_features(df, cols):
    bf = ["rainfall", "tmax", "tmin", "rh", "wind_speed", "solar_radiation"]
    sn = ["elevation", "slope", "ndvi", "dist_water"]
    X = df[bf + sn].copy()
    X["elev_x_slope"] = X["elevation"] * X["slope"]
    X["water_prox"] = np.exp(-0.15 * X["dist_water"])
    X["veg_shelter"] = X["ndvi"] * X["wind_speed"]
    X = pd.concat([X, pd.get_dummies(df["soil_type"], prefix="soil", dtype=int)], axis=1)
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    return X[cols]


def idw(lons, lats, vals, glon, glat, power=2.0):
    gl, gt = np.meshgrid(glon, glat)
    fl, ft = gl.ravel(), gt.ravel()
    d = np.sqrt((fl[:, None] - lons[None, :]) ** 2 + (ft[:, None] - lats[None, :]) ** 2)
    d = np.maximum(d, 1e-6)
    w = 1.0 / d ** power
    z = (w * vals[None, :]).sum(axis=1) / w.sum(axis=1)
    return z.reshape(gt.shape)


def plot_contour(df, glon, glat, gz, var, out):
    cmap_name, label = META[var]
    cmap = plt.get_cmap(cmap_name)
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=130)
    cf = ax.contourf(glon, glat, gz, levels=14, cmap=cmap, alpha=0.95)
    ax.contour(glon, glat, gz, levels=14, colors="k", linewidths=0.35, alpha=0.5)

    ax.scatter(df["lon"], df["lat"], s=10, c="white", edgecolor="black",
               linewidth=0.4, zorder=3)

    if "panchayat_name" in df.columns:
        top = df.nlargest(5, "elevation")
        for _, row in top.iterrows():
            name = str(row["panchayat_name"])[:18]
            ax.annotate(name, (row["lon"], row["lat"]),
                        fontsize=6, color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.65),
                        ha="center", va="bottom", zorder=5)

    bc = df.groupby("block_id")[["lon", "lat"]].mean()
    ax.scatter(bc["lon"], bc["lat"], s=90, marker="*", c="red",
               edgecolor="black", zorder=4)

    fig.colorbar(ScalarMappable(norm=cf.norm, cmap=cmap), ax=ax).set_label(label)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Panchayat {var.upper()} — Downscaled\n{DISTRICT_LABEL}", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[downscale] Plot -> {out}")


def run(in_path=IN_PATH, resolution=120):
    m = joblib.load(MODEL_PATH)
    cols = json.load(open(FEATURE_PATH))
    df = pd.read_csv(in_path)
    print(f"[downscale] Loaded {len(df)} rows")

    X = engineer_features(df, cols)
    preds = m.predict(X)

    keep = ["panchayat_id", "panchayat_name", "block_id", "block_name",
            "lat", "lon", "elevation", "slope", "ndvi", "dist_water", "soil_type"]
    available = [c for c in keep if c in df.columns]
    out = pd.concat(
        [df[available].reset_index(drop=True),
         pd.DataFrame(preds, columns=TARGETS)],
        axis=1,
    )

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[downscale] Predictions -> {OUT_CSV}")

    lons, lats = out["lon"].values, out["lat"].values
    pad = 0.02
    glon = np.linspace(lons.min() - pad, lons.max() + pad, resolution)
    glat = np.linspace(lats.min() - pad, lats.max() + pad, resolution)
    for v in TARGETS:
        gz = idw(lons, lats, out[v].values, glon, glat)
        plot_contour(out, glon, glat, gz, v, os.path.join(PLOT_DIR, f"contour_{v}.png"))
    return out


if __name__ == "__main__":
    run()