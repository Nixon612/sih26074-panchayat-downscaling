import geopandas as gpd
import os

STATE_NAME = "ASSAM"
DISTRICT_NAME = "Kamrup"

gdf = gpd.read_parquet("data/raw/LGD_Panchayats.parquet")
print(f"Total loaded: {len(gdf)}")

gdf = gdf[(gdf["stname"] == STATE_NAME) & (gdf["dtname"] == DISTRICT_NAME)].copy()
print(f"After filter: {len(gdf)} rows in {DISTRICT_NAME}, {STATE_NAME}")

# Drop rows missing the panchayat code
gdf = gdf[gdf["gp_code"].notna() & (gdf["gp_code"].astype(str).str.strip() != "")].copy()
print(f"After dropping empty gp_code: {len(gdf)} rows")

if gdf.empty:
    raise SystemExit("No rows matched.")

# Convert gp_code to int (it's stored as float/object)
gdf["gp_code"] = gdf["gp_code"].astype(int)

# Reproject to UTM 46N for accurate centroids (Assam is in zone 46)
gdf_utm = gdf.to_crs(32646)
centroids = gdf_utm.geometry.centroid.to_crs(4326)
gdf["lon"] = centroids.x
gdf["lat"] = centroids.y

# Rename to pipeline schema
gdf = gdf.rename(columns={
    "gp_code":    "panchayat_id",
    "gp_name":    "panchayat_name",
    "blkname":    "block_name",
    "blklgdcode": "block_id",
    "dt_lgd":     "district_id",
})

required = ["panchayat_id", "panchayat_name", "block_id", "block_name", "district_id", "lat", "lon"]
missing = [c for c in required if c not in gdf.columns]
if missing:
    print(f"Missing: {missing}")
    print(f"Available: {list(gdf.columns)}")
    raise SystemExit("Fix rename mapping.")

out = gdf[required].drop_duplicates("panchayat_id").reset_index(drop=True)
os.makedirs("data/raw", exist_ok=True)
out.to_csv("data/raw/panchayat_static.csv", index=False)
print(f"Saved {len(out)} centroids -> data/raw/panchayat_static.csv")
print(f"Blocks: {out['block_id'].nunique()}")
print(out.head())