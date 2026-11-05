"""
05_extract_statewide_air_quality.py
-----------------------------------
Ingests Sentinel-5P TROPOMI Level-3 observations via Google Earth Engine
for the entire State of Chhattisgarh across all 33 districts during Kharif 2024.

Indicators extracted:
1. Tropospheric Methane (CH4, ppb) - Column volume mixing ratio
2. Carbon Monoxide (CO, mol/m2) - Marker for biomass/crop residue combustion
3. Tropospheric Nitrogen Dioxide (NO2, mol/m2 and umol/m2) - Industrial/thermal power marker
4. UV Absorbing Aerosol Index (AER_AI) - Plume & particulate marker

Outputs:
- outputs/tables/chhattisgarh_33_districts_air_quality_s5p.csv
- data/processed/rasters/chhattisgarh_s5p_air_quality_composite.tif
"""

import sys
import logging
from pathlib import Path
import urllib.request
import geopandas as gpd
import pandas as pd
import ee

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GEE_AirQuality")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPKG_PATH = PROJECT_ROOT / "data" / "processed" / "admin_boundaries" / "chhattisgarh_district_boundaries.gpkg"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
RASTER_DIR = PROJECT_ROOT / "data" / "processed" / "rasters"

GEE_PROJECT = "ee-akshitvaishnavonline"
START_DATE = "2024-06-01"
END_DATE = "2024-11-30"


def main():
    logger.info("Initializing Google Earth Engine with project: %s", GEE_PROJECT)
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine: {e}")
        sys.exit(1)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RASTER_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Reading 33 district boundaries from %s", GPKG_PATH)
    gdf = gpd.read_file(GPKG_PATH, layer="chhattisgarh_districts_33")
    gdf_4326 = gdf.to_crs(4326)

    # Chhattisgarh statewide bounding rectangle
    minx, miny, maxx, maxy = gdf_4326.total_bounds
    bbox = ee.Geometry.Rectangle([float(minx - 0.2), float(miny - 0.2), float(maxx + 0.2), float(maxy + 0.2)])

    logger.info("Building Sentinel-5P TROPOMI L3 composites (%s to %s)...", START_DATE, END_DATE)
    ch4 = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4")
        .filterBounds(bbox)
        .filterDate(START_DATE, END_DATE)
        .select("CH4_column_volume_mixing_ratio_dry_air")
        .mean()
        .rename("ch4_ppb")
    )

    co = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CO")
        .filterBounds(bbox)
        .filterDate(START_DATE, END_DATE)
        .select("CO_column_number_density")
        .mean()
        .rename("co_mol_m2")
    )

    no2 = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
        .filterBounds(bbox)
        .filterDate(START_DATE, END_DATE)
        .select("tropospheric_NO2_column_number_density")
        .mean()
        .rename("no2_mol_m2")
    )

    aer = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_AER_AI")
        .filterBounds(bbox)
        .filterDate(START_DATE, END_DATE)
        .select("absorbing_aerosol_index")
        .mean()
        .rename("aer_ai")
    )

    s5p_composite = ee.Image.cat([ch4, co, no2, aer]).clip(bbox)

    # --- 1. District Zonal Reduction ---
    logger.info("Building FeatureCollection for 33 districts...")
    features = []
    for _, row in gdf_4326.iterrows():
        simplified_geom = row.geometry.simplify(0.005)
        f = ee.Feature(
            ee.Geometry(simplified_geom.__geo_interface__),
            {
                "lgd_code": int(row["lgd_code"]),
                "district_name": str(row["district_name"]),
                "division": str(row.get("division", "NA")),
                "district_area_ha": float(row.get("district_area_ha", 0.0)),
            },
        )
        features.append(f)

    fc = ee.FeatureCollection(features)

    logger.info("Executing server-side reduceRegions at 3500m resolution...")
    reduced = s5p_composite.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=3500,
    ).getInfo()

    records = []
    for f in reduced["features"]:
        props = f["properties"]
        ch4_val = props.get("ch4_ppb")
        co_val = props.get("co_mol_m2")
        no2_val = props.get("no2_mol_m2")
        aer_val = props.get("aer_ai")

        no2_umol = (no2_val * 1e6) if (no2_val is not None) else None

        records.append({
            "lgd_code": props["lgd_code"],
            "district_name": props["district_name"],
            "division": props["division"],
            "district_area_ha": props["district_area_ha"],
            "s5p_ch4_ppb": ch4_val,
            "s5p_co_mol_m2": co_val,
            "s5p_no2_mol_m2": no2_val,
            "s5p_no2_umol_m2": no2_umol,
            "s5p_aer_ai": aer_val,
            "observation_sensor": "Sentinel-5P TROPOMI Level-3",
            "observation_period": f"{START_DATE} to {END_DATE}",
        })

    df_air = pd.DataFrame(records)
    df_air = df_air.sort_values("lgd_code").reset_index(drop=True)
    out_csv = TABLE_DIR / "chhattisgarh_33_districts_air_quality_s5p.csv"
    df_air.to_csv(out_csv, index=False)
    logger.info(f"Saved {len(df_air)} district records to {out_csv}")

    # --- 2. Download Statewide Composite GeoTIFF (EPSG:32644, 4000m resolution) ---
    logger.info("Requesting statewide S5P composite GeoTIFF download URL...")
    out_tif = RASTER_DIR / "chhattisgarh_s5p_air_quality_composite.tif"
    try:
        url = s5p_composite.getDownloadURL({
            "scale": 4000,
            "crs": "EPSG:32644",
            "region": bbox,
            "format": "GEO_TIFF",
        })
        logger.info("Downloading statewide raster from Earth Engine to %s...", out_tif)
        urllib.request.urlretrieve(url, out_tif)
        logger.info("Successfully downloaded statewide GeoTIFF: %s (%.2f MB)", out_tif, out_tif.stat().st_size / (1024 * 1024))
    except Exception as e:
        logger.warning(f"Could not download full GeoTIFF (fallback to district data): {e}")

    logger.info("Statewide Sentinel-5P air quality processing complete.")


if __name__ == "__main__":
    main()
