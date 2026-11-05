"""
================================================================================
GEE WORKFLOW 03: MULTI-TEMPORAL SATELLITE PHENOLOGY TIME SERIES EXTRACTION
Project: Satellite-Based Rice Methane Assessment & Carbon MRV Feasibility
Location: Durg District Pilot (LGD 378), Chhattisgarh, India
Period: 2024 Kharif Season (June 1 - November 30, 2024)
Project ID: ee-akshitvaishnavonline
================================================================================

Server-side parallel reduction for fast multi-sensor extraction:
  1. Sentinel-1 C-Band SAR GRD (IW Mode): VH, VV, VH/VV Ratio
  2. Sentinel-2 Multi-Spectral Surface Reflectance: NDVI, LSWI (SCL cloud-masked)
  3. CHIRPS Daily Rainfall: Precipitation (mm/day)
Exports structured CSV to outputs/tables/durg_kharif_phenology_timeseries.csv
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import ee

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [TIMESERIES_ENGINE] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("extract_timeseries")


def initialize_ee(project_id: str = "ee-akshitvaishnavonline"):
    try:
        ee.Initialize(project=project_id)
        logger.info(f"Initialized Earth Engine with project: '{project_id}'")
    except Exception as e:
        logger.warning(f"Default initialization fallback: {e}")
        ee.Initialize()


def extract_timeseries(output_csv: Path) -> pd.DataFrame:
    initialize_ee()

    # Define representative central Durg rice cultivating zone
    rice_roi = ee.Geometry.Polygon([
        [81.18, 21.08],
        [81.38, 21.08],
        [81.38, 21.28],
        [81.18, 21.28],
        [81.18, 21.08]
    ])

    logger.info("Extracting Sentinel-1 C-Band SAR time series (Server-Side Parallel Reduce)...")
    def reduce_s1(img):
        stats = img.select(["VH", "VV"]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=rice_roi,
            scale=100,
            maxPixels=1e8
        )
        date_str = img.date().format("YYYY-MM-dd")
        return ee.Feature(None, stats).set("date", date_str).set("sensor", "Sentinel-1 SAR")

    s1_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(rice_roi)
        .filterDate("2024-06-01", "2024-11-30")
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .sort("system:time_start")
    )

    s1_features = s1_col.map(reduce_s1).getInfo().get("features", [])
    s1_rows = []
    for f in s1_features:
        props = f.get("properties", {})
        vh = props.get("VH")
        vv = props.get("VV")
        if vh is not None and vv is not None:
            s1_rows.append({
                "date": props.get("date"),
                "vh_mean_db": round(float(vh), 2),
                "vv_mean_db": round(float(vv), 2),
                "vh_vv_ratio_db": round(float(vh) - float(vv), 2)
            })
    df_s1 = pd.DataFrame(s1_rows)
    logger.info(f"Retrieved {len(df_s1)} Sentinel-1 observation dates.")

    logger.info("Extracting Sentinel-2 Multi-Spectral time series (Server-Side Parallel Reduce)...")
    def mask_and_reduce_s2(img):
        scl = img.select("SCL")
        mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        nir = img.select("B8").divide(10000.0)
        red = img.select("B4").divide(10000.0)
        swir1 = img.select("B11").divide(10000.0)

        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        lswi = nir.subtract(swir1).divide(nir.add(swir1)).rename("LSWI")
        fused = ndvi.addBands(lswi).updateMask(mask)

        stats = fused.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=rice_roi,
            scale=100,
            maxPixels=1e8
        )
        date_str = img.date().format("YYYY-MM-dd")
        return ee.Feature(None, stats).set("date", date_str).set("sensor", "Sentinel-2 MSI")

    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(rice_roi)
        .filterDate("2024-06-01", "2024-11-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        .sort("system:time_start")
    )

    s2_features = s2_col.map(mask_and_reduce_s2).getInfo().get("features", [])
    s2_rows = []
    for f in s2_features:
        props = f.get("properties", {})
        ndvi = props.get("NDVI")
        lswi = props.get("LSWI")
        if ndvi is not None and lswi is not None:
            s2_rows.append({
                "date": props.get("date"),
                "ndvi_mean": round(float(ndvi), 3),
                "lswi_mean": round(float(lswi), 3)
            })
    df_s2 = pd.DataFrame(s2_rows)
    logger.info(f"Retrieved {len(df_s2)} Sentinel-2 cloud-masked observation dates.")

    logger.info("Extracting CHIRPS rainfall time series...")
    def reduce_chirps(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=rice_roi,
            scale=5000,
            maxPixels=1e6
        )
        date_str = img.date().format("YYYY-MM-dd")
        return ee.Feature(None, stats).set("date", date_str)

    chirps_col = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(rice_roi)
        .filterDate("2024-06-01", "2024-11-30")
        .sort("system:time_start")
    )
    chirps_features = chirps_col.map(reduce_chirps).getInfo().get("features", [])
    chirps_rows = []
    for f in chirps_features:
        props = f.get("properties", {})
        p = props.get("precipitation")
        chirps_rows.append({
            "date": props.get("date"),
            "rainfall_mm": round(float(p) if p is not None else 0.0, 1)
        })
    df_chirps = pd.DataFrame(chirps_rows)
    logger.info(f"Retrieved {len(df_chirps)} daily CHIRPS rainfall records.")

    # Merge across full Kharif calendar
    all_dates = pd.date_range("2024-06-01", "2024-11-30", freq="D").strftime("%Y-%m-%d")
    df_master = pd.DataFrame({"date": all_dates})

    if not df_s1.empty:
        df_s1_agg = df_s1.groupby("date")[["vh_mean_db", "vv_mean_db", "vh_vv_ratio_db"]].mean().reset_index()
        df_master = df_master.merge(df_s1_agg, on="date", how="left")
    else:
        df_master["vh_mean_db"] = np.nan
        df_master["vv_mean_db"] = np.nan
        df_master["vh_vv_ratio_db"] = np.nan

    if not df_s2.empty:
        df_s2_agg = df_s2.groupby("date")[["ndvi_mean", "lswi_mean"]].mean().reset_index()
        df_master = df_master.merge(df_s2_agg, on="date", how="left")
    else:
        df_master["ndvi_mean"] = np.nan
        df_master["lswi_mean"] = np.nan

    if not df_chirps.empty:
        df_master = df_master.merge(df_chirps, on="date", how="left")
    else:
        df_master["rainfall_mm"] = 0.0

    # Add agronomic phenological phase
    def assign_phase(row_date):
        dt = pd.to_datetime(row_date)
        if dt < pd.Timestamp("2024-06-25"):
            return "Pre-Monsoon Land Preparation"
        elif dt < pd.Timestamp("2024-07-28"):
            return "Puddling & Transplanting (Specular Dip)"
        elif dt < pd.Timestamp("2024-09-12"):
            return "Vegetative Tillering & Culm Elongation"
        elif dt < pd.Timestamp("2024-10-18"):
            return "Panicle Initiation & Heading / Flowering"
        elif dt < pd.Timestamp("2024-11-12"):
            return "Ripening & Grain Filling"
        else:
            return "Physiological Maturity & Harvest"

    df_master["phenology_phase"] = df_master["date"].apply(assign_phase)

    # Inundation status
    df_master["inundation_status"] = np.where(
        (df_master["vh_mean_db"] < -17.5) | (df_master["lswi_mean"] > 0.0),
        "Inundated", "Drainable/Moist"
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(output_csv, index=False)
    logger.info("Successfully exported real satellite phenology time series to: %s (%d records)", output_csv, len(df_master))
    return df_master


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    out_path = project_root / "outputs" / "tables" / "durg_kharif_phenology_timeseries.csv"
    extract_timeseries(out_path)
