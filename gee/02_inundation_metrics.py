"""
================================================================================
GEE WORKFLOW 02: INUNDATION METRICS & HYDRO-PERIOD DYNAMICS
Pilot: Durg District, Chhattisgarh (LGD Code: 378)
Project ID: ee-akshitvaishnavonline
CRS: EPSG:32644 (UTM Zone 44N)
================================================================================

This module quantifies seasonal water inundation dynamics across rice paddies:
1. Dynamic Inundation Tracking:
   - Uses Sentinel-1 SAR dual-pol (VV, VH) to detect standing water underneath
     monsoon cloud cover. Standing water thresholding: VH < -20.0 dB or VV < -14.0 dB.
   - Sentinel-2 Modified Normalized Difference Water Index (MNDWI) and LSWI for
     cloud-free observations: (LSWI > 0.1 or MNDWI > 0).
2. Per-Pixel Inundation Metrics:
   - wet_observation_count: Total passes where pixel was detected with standing water.
   - total_valid_observations: Total valid satellite passes (clear optical + SAR).
   - inundation_fraction: wet_observation_count / total_valid_observations.
   - estimated_inundation_duration_days: inundation_fraction * cultivation_duration_days.
   - inundation_confidence: Reliability metric based on observation density:
     confidence = min(1.0, total_valid_observations / expected_passes).
3. Non-blocking Export:
   - Dispatches non-blocking COG export tasks to Google Drive or Cloud Storage.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
import yaml
import ee

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inundation_metrics")


def initialize_earth_engine(project_id: str) -> None:
    """Initializes Earth Engine with specific Cloud Project."""
    try:
        ee.Initialize(project=project_id)
        logger.info(f"Earth Engine successfully initialized with project: '{project_id}'")
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine with project '{project_id}': {e}")
        ee.Initialize()


def load_durg_geometry(geojson_path: Path) -> ee.Geometry:
    """Loads Durg pilot boundary from local GeoJSON or fallback bbox."""
    if geojson_path.exists():
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        features = data.get("features", [])
        if features:
            coords = features[0]["geometry"]["coordinates"]
            geom_type = features[0]["geometry"]["type"]
            if geom_type == "Polygon":
                return ee.Geometry.Polygon(coords)
            elif geom_type == "MultiPolygon":
                return ee.Geometry.MultiPolygon(coords)

    return ee.Geometry.Polygon([
        [81.08, 20.80],
        [81.66, 20.80],
        [81.66, 21.58],
        [81.08, 21.58],
        [81.08, 20.80]
    ])


def detect_sar_water(image: ee.Image) -> ee.Image:
    """
    Identifies standing surface water using Sentinel-1 dual-polarization SAR backscatter.
    Calm open water or flooded paddy soil exhibits low backscatter due to specular reflection.
    """
    # VH threshold for water/puddled soil
    vh = image.select("VH")
    vv = image.select("VV")

    # Water condition: specular dip in both VH and VV
    is_water = vh.lt(-19.5).Or(vv.lt(-14.0)).rename("is_wet")
    valid_mask = image.select("VV").mask().rename("is_valid")

    return is_water.addBands(valid_mask).set("system:time_start", image.get("system:time_start"))


def detect_s2_water(image: ee.Image) -> ee.Image:
    """
    Identifies water/flooded surface using Sentinel-2 LSWI and MNDWI on clear pixels.
    """
    # Cloud mask check
    qa = image.select("QA60")
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))

    green = image.select("B3").divide(10000.0)
    swir1 = image.select("B11").divide(10000.0)
    nir = image.select("B8").divide(10000.0)

    # MNDWI = (Green - SWIR1) / (Green + SWIR1)
    mndwi = green.subtract(swir1).divide(green.add(swir1))
    lswi = nir.subtract(swir1).divide(nir.add(swir1))

    is_wet = (mndwi.gt(0.0).Or(lswi.gt(0.1))).And(cloud_mask).rename("is_wet")
    is_valid = cloud_mask.rename("is_valid")

    return is_wet.addBands(is_valid).set("system:time_start", image.get("system:time_start"))


def build_inundation_metrics(
    geometry: ee.Geometry,
    start_date: str = "2024-06-01",
    end_date: str = "2024-11-30",
    cultivation_duration_days: int = 105,
    expected_satellite_passes: int = 30
) -> ee.Image:
    """
    Builds the per-pixel inundation metrics collection and composite.
    """
    logger.info(f"Computing dynamic inundation metrics over period: {start_date} to {end_date}...")

    # 1. Sentinel-1 SAR observations
    s1_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .map(detect_sar_water)
    )

    # 2. Sentinel-2 Optical observations
    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        .map(detect_s2_water)
    )

    # 3. Combine valid observations and wet observations across both sensors
    combined_col = s1_col.merge(s2_col)

    wet_obs_count = combined_col.select("is_wet").sum().rename("wet_observation_count")
    total_valid_obs = combined_col.select("is_valid").sum().rename("total_valid_observations")

    # Avoid division by zero: where valid == 0, fraction = 0
    inundation_fraction = (
        wet_obs_count.divide(total_valid_obs.max(1))
        .rename("inundation_fraction")
    )

    # Estimated Inundation Duration (days)
    estimated_inundation_days = (
        inundation_fraction.multiply(cultivation_duration_days)
        .rename("estimated_inundation_duration_days")
    )

    # Observation Confidence Score [0.0 to 1.0]
    # Penalizes pixels with inadequate observations due to persistent cloud cover / sensor gaps
    inundation_confidence = (
        total_valid_obs.divide(float(expected_satellite_passes))
        .min(1.0)
        .rename("inundation_confidence")
    )

    # Water regime categorization band
    # 0: Low/Intermittent (< 30% inundation)
    # 1: Moderate/Intermittent (30-65% inundation)
    # 2: Continuously Flooded (> 65% inundation)
    regime_class = (
        ee.Image(0)
        .where(inundation_fraction.gte(0.30).And(inundation_fraction.lt(0.65)), 1)
        .where(inundation_fraction.gte(0.65), 2)
        .rename("water_regime_classification")
    )

    metrics_composite = (
        wet_obs_count
        .addBands(total_valid_obs)
        .addBands(inundation_fraction)
        .addBands(estimated_inundation_days)
        .addBands(inundation_confidence)
        .addBands(regime_class)
        .clip(geometry)
    )

    return metrics_composite


def export_inundation_cog(
    metrics_image: ee.Image,
    geometry: ee.Geometry,
    output_target: str,
    export_folder: str,
    bucket_name: str,
    scale: int = 10,
    crs: str = "EPSG:32644"
) -> ee.batch.Task:
    """Submits non-blocking COG export for inundation metrics."""
    task_name = "durg_inundation_metrics_2024"
    logger.info(f"Submitting non-blocking export task: '{task_name}'...")

    if output_target == "toCloudStorage":
        task = ee.batch.Export.image.toCloudStorage(
            image=metrics_image,
            description=task_name,
            bucket=bucket_name,
            fileNamePrefix=f"{export_folder}/{task_name}",
            region=geometry,
            scale=scale,
            crs=crs,
            maxPixels=1e9,
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True}
        )
    else:
        task = ee.batch.Export.image.toDrive(
            image=metrics_image,
            description=task_name,
            folder=export_folder,
            fileNamePrefix=task_name,
            region=geometry,
            scale=scale,
            crs=crs,
            maxPixels=1e9,
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True}
        )

    task.start()
    logger.info(f"Inundation export submitted! Task ID: {task.id}, Status: {task.status().get('state')}")
    return task


def main():
    parser = argparse.ArgumentParser(
        description="WP-GEE-02: Durg District Dynamic Inundation Metrics Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/parameters.yaml",
        help="Path to pipeline configuration YAML.",
    )
    parser.add_argument(
        "--geojson",
        type=str,
        default="data/processed/admin_boundaries/durg_pilot_boundary.geojson",
        help="Path to Durg pilot boundary GeoJSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Builds GEE processing graph and computes spatial summary without launching export.",
    )

    args = parser.parse_args()

    # Paths resolution
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name in ["gee", "scripts"] else script_dir

    config_path = project_root / args.config
    geojson_path = project_root / args.geojson

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_id = cfg["gee"]["project_id"]
    initialize_earth_engine(project_id)

    durg_geom = load_durg_geometry(geojson_path)

    metrics_image = build_inundation_metrics(
        geometry=durg_geom,
        start_date=cfg["project"]["kharif_season"]["start_date"],
        end_date=cfg["project"]["kharif_season"]["end_date"],
        cultivation_duration_days=cfg["ipcc_tier2"]["cultivation_duration"]["default_days"],
        expected_satellite_passes=25
    )

    # Compute district mean metrics
    stats = metrics_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=durg_geom,
        scale=50,
        maxPixels=1e8
    )

    logger.info("Computing mean inundation statistics for Durg District...")
    try:
        results = stats.getInfo()
        for k, v in results.items():
            if v is not None:
                logger.info(f"  {k}: {v:.3f}")
            else:
                logger.info(f"  {k}: None")
    except Exception as e:
        logger.warning(f"Could not retrieve synchronous inundation reduction: {e}")

    if not args.dry_run:
        export_inundation_cog(
            metrics_image=metrics_image,
            geometry=durg_geom,
            output_target=cfg["gee"].get("export_target", "toDrive"),
            export_folder=cfg["gee"].get("export_folder", "GEE_Rice_Methane_MRV"),
            bucket_name=cfg["gee"].get("cloud_storage_bucket", "ee-akshitvaishnavonline-mrv"),
            scale=cfg["gee"].get("resolution_meters", 10),
            crs=cfg["crs"]["projected_crs"]
        )
    else:
        logger.info("Dry-run complete. Cloud export skipped.")


if __name__ == "__main__":
    main()
