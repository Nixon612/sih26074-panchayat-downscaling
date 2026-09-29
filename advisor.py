"""03 ADVISE — ICAR/GKMS rule engine. Fires only on genuine thresholds."""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict
import pandas as pd


@dataclass
class Advisory:
    panchayat_id: str
    category: str
    severity: str
    message: str
    variables: List[str]
    rule_id: str


@dataclass
class PanchayatReport:
    panchayat_id: str
    block_id: str
    lat: float
    lon: float
    panchayat_name: str = ""
    block_name: str = ""
    weather: Dict[str, float] = field(default_factory=dict)
    advisories: List[Advisory] = field(default_factory=list)
    risk_flags: Dict[str, str] = field(default_factory=dict)


# Thresholds calibrated to ICAR-AICRPAM / IMD GKMS bulletins for kharif/rabi crops.
T = {
    "rain_heavy": 10.0, "rain_moderate": 5.0, "rain_light": 1.0,
    "tmax_heat_stress": 35.0, "tmax_severe_heat": 40.0,
    "tmin_cold_stress": 10.0, "tmin_frost_risk": 5.0,
    "rh_disease_high": 80.0, "rh_disease_very_high": 90.0,
    "rh_spray_max": 60.0,
    "wind_spray_limit": 10.0, "wind_spray_ideal_max": 8.0,
    "wind_lodging_risk": 30.0, "wind_severe": 40.0,
    "solar_high_et": 22.0,
}

# Crop rules fire only on genuine thresholds — no catchall / always-true conditions.
CROP_RULES = {
    "rice": [
        (lambda w: w["rainfall"] < 1 and w["tmax"] > 35,
         "Rice: Heat + dry spell during panicle stage. Maintain 5 cm standing water; "
         "irrigate early morning."),
        (lambda w: w["rh"] > 85 and 22 <= w["tmin"] <= 28,
         "Rice: Blast risk high under humid nights. Apply Tricyclazole 75 WP "
         "@ 0.6 g/l during clear window."),
        (lambda w: w["rainfall"] >= 10,
         "Rice: Heavy rain expected. Check field bunds and drainage to avoid "
         "submergence beyond 10 cm."),
    ],
    "wheat": [
        (lambda w: w["tmax"] > 30 and w["rainfall"] < 1,
         "Wheat: Terminal heat stress risk. Irrigate at crown-root initiation; "
         "skip urea top-dress during heat wave."),
        (lambda w: w["tmin"] < 10,
         "Wheat: Cold stress during tillering. Light afternoon irrigation; "
         "avoid excess nitrogen."),
        (lambda w: w["rh"] > 80 and 15 <= w["tmin"] <= 25,
         "Wheat: Yellow rust risk. Scout lower leaves; apply Propiconazole "
         "25 EC @ 0.5 ml/l if symptoms appear."),
    ],
    "cotton": [
        (lambda w: w["rh"] > 80 and w["rainfall"] > 5,
         "Cotton: Sucking pest / leaf curl risk. Scout; spray Imidacloprid "
         "@ 0.5 ml/l only in clear weather."),
        (lambda w: w["tmax"] > 38,
         "Cotton: Heat stress during boll formation. Irrigate at 10-day "
         "intervals; avoid mid-day field work."),
    ],
    "sugarcane": [
        (lambda w: w["wind_speed"] > 30,
         "Sugarcane: High wind. Tie 3-4 canes together to prevent lodging; "
         "check drainage."),
        (lambda w: w["rainfall"] < 1 and w["tmax"] > 35,
         "Sugarcane: High water demand at grand growth stage. Irrigate "
         "at 7-10 day intervals."),
    ],
}


def _mk(p, cat, sev, msg, vars_, rid):
    return Advisory(p.panchayat_id, cat, sev, msg, vars_, rid)


def rule_irrigation(p):
    r, tm, s = p.weather["rainfall"], p.weather["tmax"], p.weather["solar_radiation"]
    if r >= T["rain_heavy"]:
        return _mk(p, "irrigation", "warning",
                   f"Heavy rain ({r:.1f} mm). Do NOT irrigate. Ensure drainage.",
                   ["rainfall"], "IRR-RAIN-HEAVY")
    if r >= T["rain_moderate"]:
        return _mk(p, "irrigation", "info",
                   f"Moderate rain ({r:.1f} mm). Postpone irrigation 1-2 days.",
                   ["rainfall"], "IRR-RAIN-MOD")
    if r < T["rain_light"] and (tm >= T["tmax_heat_stress"] or s >= T["solar_high_et"]):
        return _mk(p, "irrigation", "warning",
                   f"Dry + high ET (Tmax {tm:.1f}C, solar {s:.1f}). "
                   "Irrigate early morning / evening. Mulch to conserve moisture.",
                   ["rainfall", "tmax", "solar_radiation"], "IRR-DRY-HEAT")
    return None


def rule_fertilizer(p):
    r, w = p.weather["rainfall"], p.weather["wind_speed"]
    if r >= T["rain_moderate"]:
        return _mk(p, "fertilizer", "warning",
                   f"Rain ({r:.1f} mm). Delay urea — will leach or wash away.",
                   ["rainfall"], "FERT-RAIN-DELAY")
    if r >= T["rain_light"]:
        return _mk(p, "fertilizer", "info",
                   f"Light rain ({r:.1f} mm). Urea OK if soil is moist but not saturated.",
                   ["rainfall"], "FERT-LIGHT-RAIN")
    if w >= T["wind_spray_limit"]:
        return _mk(p, "fertilizer", "info",
                   f"Windy ({w:.1f} km/h). Avoid foliar fertilizer — drift losses.",
                   ["wind_speed"], "FERT-WIND")
    return None


def rule_spray(p):
    r, w, rh = p.weather["rainfall"], p.weather["wind_speed"], p.weather["rh"]
    if r >= T["rain_light"]:
        return _mk(p, "spray", "warning",
                   f"Rain ({r:.1f} mm). Avoid spray — will wash off. "
                   "Wait for 6-8 hr clear window.",
                   ["rainfall"], "SPRAY-RAIN")
    if w >= T["wind_spray_limit"]:
        return _mk(p, "spray", "warning",
                   f"Wind {w:.1f} km/h exceeds safe spray limit. Defer — drift risk.",
                   ["wind_speed"], "SPRAY-WIND")
    # Only flag if spray is genuinely blocked; otherwise stay silent
    if rh >= T["rh_disease_high"] and w < T["wind_spray_ideal_max"]:
        return _mk(p, "spray", "info",
                   f"High RH ({rh:.0f}%) with calm wind ({w:.1f} km/h). "
                   "Fungal disease pressure elevated — scout before next spray.",
                   ["rh", "wind_speed"], "SPRAY-HUMID")
    return None


def rule_pest(p):
    """Fires on RH alone. Previous version wrongly required rainfall —
    that missed humid dry days when disease pressure is highest."""
    rh, tm, tn = p.weather["rh"], p.weather["tmax"], p.weather["tmin"]
    if rh >= T["rh_disease_very_high"] and 20 <= tn <= 28:
        return _mk(p, "pest", "critical",
                   f"Very high RH ({rh:.0f}%) + moderate temp. "
                   "High blast/blight risk. Preventive fungicide during clear window.",
                   ["rh", "tmin"], "PEST-HUMID-HIGH")
    if rh >= T["rh_disease_high"]:
        return _mk(p, "pest", "warning",
                   f"High humidity ({rh:.0f}%). Fungal/bacterial disease risk. "
                   "Scout leaves for early symptoms; ensure field drainage.",
                   ["rh"], "PEST-HUMID")
    if tm >= T["tmax_heat_stress"] and rh < 50:
        return _mk(p, "pest", "info",
                   f"Hot, dry (Tmax {tm:.1f}C, RH {rh:.0f}%). Sucking pest "
                   "activity may rise. Monitor with yellow sticky traps.",
                   ["tmax", "rh"], "PEST-DRY-HEAT")
    return None


def rule_heat(p):
    tm = p.weather["tmax"]
    if tm >= T["tmax_severe_heat"]:
        return _mk(p, "stress", "critical",
                   f"Severe heat (Tmax {tm:.1f}C). Evening irrigation to cool canopy; "
                   "mulch; avoid midday field work.",
                   ["tmax"], "STRESS-HEAT-SEV")
    if tm >= T["tmax_heat_stress"]:
        return _mk(p, "stress", "warning",
                   f"Heat stress (Tmax {tm:.1f}C). Ensure soil moisture; "
                   "watch for flower drop and poor fruit set.",
                   ["tmax"], "STRESS-HEAT")
    return None


def rule_cold(p):
    tn = p.weather["tmin"]
    if tn <= T["tmin_frost_risk"]:
        return _mk(p, "stress", "critical",
                   f"Frost risk (Tmin {tn:.1f}C). Cover seedlings; light irrigation "
                   "before sunset; avoid late-evening irrigation.",
                   ["tmin"], "STRESS-FROST")
    if tn <= T["tmin_cold_stress"]:
        return _mk(p, "stress", "warning",
                   f"Cold stress (Tmin {tn:.1f}C). Protect nursery beds; "
                   "delay morning irrigation until temperatures rise.",
                   ["tmin"], "STRESS-COLD")
    return None


def rule_wind(p):
    w = p.weather["wind_speed"]
    if w >= T["wind_severe"]:
        return _mk(p, "field_op", "critical",
                   f"Very high wind ({w:.1f} km/h). Defer all field ops; "
                   "stake tall crops; harvest mature fruits.",
                   ["wind_speed"], "WIND-SEVERE")
    if w >= T["wind_lodging_risk"]:
        return _mk(p, "field_op", "warning",
                   f"Strong wind ({w:.1f} km/h) — lodging risk. Stake vulnerable "
                   "plants; avoid spraying and overhead irrigation.",
                   ["wind_speed"], "WIND-LODGING")
    return None


RULES = [rule_irrigation, rule_fertilizer, rule_spray,
         rule_pest, rule_heat, rule_cold, rule_wind]


def risk_flags(p):
    w = p.weather
    return {
        "heat": "critical" if w["tmax"] >= T["tmax_severe_heat"]
                else "warning" if w["tmax"] >= T["tmax_heat_stress"] else "low",
        "cold": "critical" if w["tmin"] <= T["tmin_frost_risk"]
                else "warning" if w["tmin"] <= T["tmin_cold_stress"] else "low",
        "disease": "critical" if w["rh"] >= T["rh_disease_very_high"]
                   else "warning" if w["rh"] >= T["rh_disease_high"] else "low",
        "spray": "blocked" if (w["wind_speed"] >= T["wind_spray_limit"]
                               or w["rainfall"] >= T["rain_light"]) else "ok",
        "irrigation": "not_needed" if w["rainfall"] >= T["rain_moderate"]
                      else "needed" if (w["rainfall"] < T["rain_light"]
                                        and (w["tmax"] >= T["tmax_heat_stress"]
                                             or w["solar_radiation"] >= T["solar_high_et"]))
                      else "monitor",
        "wind": "critical" if w["wind_speed"] >= T["wind_severe"]
                else "warning" if w["wind_speed"] >= T["wind_lodging_risk"] else "low",
    }


def advise_panchayat(row, crop=None):
    pname = str(row["panchayat_name"]) if "panchayat_name" in row and pd.notna(row["panchayat_name"]) else ""
    bname = str(row["block_name"]) if "block_name" in row and pd.notna(row["block_name"]) else ""

    p = PanchayatReport(
        panchayat_id=str(row["panchayat_id"]),
        block_id=str(row["block_id"]),
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        panchayat_name=pname,
        block_name=bname,
        weather={k: float(row[k]) for k in
                 ["rainfall", "tmax", "tmin", "rh", "wind_speed", "solar_radiation"]},
    )

    for r in RULES:
        a = r(p)
        if a:
            p.advisories.append(a)

    if crop and crop.lower() in CROP_RULES:
        for cond, msg in CROP_RULES[crop.lower()]:
            try:
                if cond(p.weather):
                    p.advisories.append(Advisory(
                        p.panchayat_id, "crop_specific", "info", msg,
                        list(p.weather.keys()), f"CROP-{crop.upper()}"
                    ))
            except Exception:
                pass

    p.risk_flags = risk_flags(p)
    order = {"critical": 0, "warning": 1, "info": 2}
    p.advisories.sort(key=lambda a: order.get(a.severity, 3))
    return p


def _worst_severity(r):
    order = {"critical": 0, "warning": 1, "info": 2}
    if not r.advisories:
        return 3
    return min(order.get(a.severity, 3) for a in r.advisories)


def advise_all(df, crop=None):
    reports = [advise_panchayat(r, crop=crop) for _, r in df.iterrows()]
    reports.sort(key=lambda r: (_worst_severity(r), r.panchayat_name or r.panchayat_id))
    return reports


def to_json(reports):
    return json.dumps([
        {**{k: v for k, v in asdict(r).items() if k != "advisories"},
         "advisories": [asdict(a) for a in r.advisories]}
        for r in reports
    ], indent=2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/panchayat_downscaled.csv")
    ap.add_argument("--panchayat")
    ap.add_argument("--crop")
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    if a.panchayat:
        df = df[df["panchayat_id"].astype(str) == a.panchayat]
    for r in advise_all(df, crop=a.crop)[:5]:
        label = r.panchayat_name or r.panchayat_id
        print(f"\n=== {label} | Block {r.block_name or r.block_id} ===")
        for adv in r.advisories:
            print(f"  [{adv.severity.upper():8s}] {adv.message}")