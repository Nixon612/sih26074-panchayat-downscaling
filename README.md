# Panchayat-Level Agro-Meteorological Advisory System

**SIH 2026 · Problem SIH26074 · Ministry of Earth Sciences**

Statistical downscaling of block-level weather forecasts (~12 km) to panchayat-level (~1 km) for hyper-local agro-meteorological advisory services.

---

## The Problem

Indian farmers receive weather advisories at **block level (~12 km)** via IMD's Gramin Krishi Mausam Sewa (GKMS). But farming decisions happen at **panchayat level (~1 km)** where micro-topography — elevation, slope, vegetation, water proximity — creates temperature gradients of 2–4 °C and rainfall variations of 30–60%. This gap materially affects irrigation scheduling, fertilizer timing, spray windows, pest risk, and heat/cold stress management.

## Our Approach

**Hybrid statistical downscaling** — no GPU-bound WRF or 3D-CNN. A lightweight Random Forest spatially regresses panchayat weather on coarse block forecasts plus fine-grained static topography.

**Coarse inputs (block forecast):** Rainfall, Tmax, Tmin, Relative Humidity, Wind speed, Solar radiation

**Fine-grained static inputs (panchayat):** Elevation (SRTM 30 m), slope, NDVI, distance-to-water, soil type

**Proof district:** 142 real panchayats across 14 blocks in Kamrup District, Assam (LGD codes).

CPU inference: **~15 seconds for the full pipeline** across 142 panchayats. Contour plots generated via Inverse Distance Weighting (IDW).

## Architecture — 4-Step Pipeline
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐ ┌──────────────┐
│ 01 INGEST │──▶│ 02 DOWNSCALE │──▶│ 03 ADVISE │──▶│ 04 SERVE │
│ │ │ │ │ │ │ │
│ LGD boundaries │ │ Random Forest │ │ ICAR / GKMS │ │ Flask + │
│ OpenTopoData │ │ IDW contours │ │ rule engine │ │ Leaflet │
│ OpenWeatherMap │ │ 6 weather vars │ │ 7 hazard rules │ │ Dashboard │
└─────────────────┘ └──────────────────┘ └─────────────────┘ └──────────────┘

| Step | File | Role |
|------|------|------|
| 01 Ingest | `ingest.py` | Loads LGD centroids, fetches SRTM elevation, pulls block weather |
| 02 Downscale | `train_model.py` + `downscale.py` | Trains RF, predicts panchayat weather, renders contour PNGs |
| 03 Advise | `advisor.py` | Maps weather to advisory messages using ICAR/GKMS thresholds |
| 04 Serve | `app.py` + templates | Flask dashboard with map, contours, advisory cards |

## Quickstart

**Requirements:** Python 3.11+, `uv` (recommended) or `pip`

```bash
# 1. Clone
git clone https://github.com/Nixon612/sih26074-panchayat-downscaling.git
cd sih26074-panchayat-downscaling

# 2. Create virtualenv
uv venv --python 3.12
source .venv/Scripts/activate     # Windows Git Bash
# source .venv/bin/activate        # Linux/Mac

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and add your OpenWeatherMap API key

# 5. Download LGD panchayat boundaries (one-time, ~351 MB)
# Get LGD_Panchayats.parquet from:
# https://github.com/yashveeeeeeer/india-geodata/releases/tag/admin/panchayats
# Place in data/raw/

# 6. Extract centroids for your chosen district
python extract_centroids.py       # → data/raw/panchayat_static.csv

# 7. Run the pipeline
python ingest.py                  # elevation + weather
python train_model.py             # train downscaler
python downscale.py               # predict + render contours
python app.py                     # launch dashboard
Open http://localhost:5000 and click ↻ Refresh.
Project Structure
sih26074/
├── ingest.py              # 01: LGD centroids + OpenTopoData + OpenWeatherMap
├── train_model.py         # 02a: Random Forest spatial regression
├── downscale.py           # 02b: inference + IDW contour plots
├── advisor.py             # 03: ICAR/GKMS advisory rule engine
├── app.py                 # 04: Flask dashboard
├── extract_centroids.py   # utility: LGD parquet → panchayat centroids
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── plots/             # generated contour PNGs
├── data/
│   ├── raw/               # LGD parquet, panchayat_static.csv
│   └── processed/         # feature table, predictions
├── models/                # trained RF + feature schema
├── requirements.txt
└── README.md
Advisory Rules
The advisor engine implements ICAR-AICRPAM and IMD GKMS calibrated rules across 7 categories:

Rule	Trigger	Advisory
IRR-RAIN-HEAVY	Rain ≥ 10 mm	Do not irrigate; check drainage
IRR-DRY-HEAT	Rain < 1 mm + Tmax ≥ 35 °C or solar ≥ 22 MJ/m²	Irrigate early morning/evening
FERT-RAIN-DELAY	Rain ≥ 5 mm	Delay urea; will leach
SPRAY-WIND	Wind ≥ 10 km/h	Defer spraying; drift risk
SPRAY-RAIN	Rain ≥ 1 mm	Avoid spraying; wash-off risk
PEST-HUMID-HIGH	RH ≥ 90 % + 20 ≤ Tmin ≤ 28 °C	Blast/blight risk; preventive fungicide
PEST-HUMID	RH ≥ 80 %	Fungal/bacterial disease risk; scout crops
STRESS-HEAT-SEV	Tmax ≥ 40 °C	Evening irrigation; mulch; avoid midday work
STRESS-FROST	Tmin ≤ 5 °C	Protect seedlings; light pre-sunset irrigation
WIND-LODGING	Wind ≥ 30 km/h	Stake tall crops; avoid spraying
Every advisory carries a rule_id for full auditability.

Crop-specific rules (Rice, Wheat, Cotton, Sugarcane) fire only on genuine ICAR action thresholds — the engine is scientifically conservative rather than advisory-spammy.
Data Sources
Source	Data	License
Local Government Directory	Panchayat boundaries + LGD codes	Government of India
OpenTopoData	SRTM 30 m elevation	Public API
OpenWeatherMap	Block-level 5-day forecast	Free tier
IMD GKMS	Advisory thresholds	Reference
Production path: The ingest layer is pluggable. IMD block-forecast API and Bhuvan WMS for NDVI/soil can be swapped in once credentials are approved. The current prototype uses OpenWeatherMap to demonstrate the full pipeline without waiting for IP whitelisting.
Validation Strategy
Currently validating against synthetic physical truth (standard lapse rate + orographic rainfall models + vegetation sheltering). Once IMD district AWS observations are available:

Split historical observations 70/15/15 (train/val/test)

Compare RF downscaling against (a) raw block forecast, (b) bilinear interpolation baseline

Report RMSE, MAE, R² per variable per season

Expected realistic performance based on published statistical downscaling literature: R² 0.6–0.85 for temperature, 0.4–0.7 for rainfall.
Roadmap
☑ LGD panchayat boundaries for all India
☑ Real SRTM elevation via OpenTopoData
☑ Real block weather via OpenWeatherMap
☑ 4-step pipeline with Flask dashboard
☑ ICAR/GKMS advisory engine with audit trail
☑ Live search, severity sort, crop-specific rules
□ IMD block-forecast API integration (IP whitelist pending)
□ Real NDVI via MODIS / Bhuvan WMS
□ Historical validation against IMD AWS stations
□ Assamese / Hindi advisory translation
□ SMS / IVR delivery channel for low-connectivity farmers
□ Scale test: 250,000+ panchayats nationally
TEAM -
Simanta Kalita
Bishal Borah
Nishant Chetry
Jangsar Muchahari
Shyamanta Kachari
Daizee Brahma

Acknowledgements
Ministry of Earth Sciences (MoES) · India Meteorological Department (IMD) · ISRO Bhuvan · ICAR-AICRPAM · Local Government Directory