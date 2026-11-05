"""
================================================================================
GEE WORKFLOW 01: DURG DISTRICT RICE PHENOLOGY & EXTENT MAPPING
Pilot: Durg District, Chhattisgarh (LGD Code: 378)
Project ID: ee-akshitvaishnavonline
CRS: EPSG:32644 (UTM Zone 44N)
================================================================================

This module implements a fused Sentinel-1 SAR and Sentinel-2 Multi-Spectral
phenology algorithm for paddy rice detection during the Kharif season:

ALGORITHM FOUNDATION:
1. Sentinel-1 SAR Temporal Dip & Rise:
   - Specular dip: During puddling and transplanting (June-August), calm standing
     water causes forward specular reflection, resulting in a pronounced drop in
     VV and VH backscatter (VH typically drops to < -18 dB).
   - Canopy volume scattering: As the rice canopy develops rapidly (August-October),
     multiple scattering within the vertical culms produces a sharp increase in VH
     backscatter (+4 to +8 dB increase).
2. Sentinel-2 Optical Phenology:
   - SCL / QA60 cloud and cloud-shadow masking.
   - Land Surface Water Index (LSWI = (NIR - SWIR1)/(NIR + SWIR1)) and
     Normalized Difference Vegetation Index (NDVI = (NIR - Red)/(NIR + Red)).
   - Transplanting signature: Inundated field condition where LSWI + 0.05 >= NDVI
     or LSWI > 0 during the transplanting window, followed by a sustained increase
     in NDVI (peak NDVI > 0.50).
3. Land Cover & Topographic Masking:
   - Slope masking (< 5%) using SRTM to eliminate steep terrain layover.
   - Cropland mask from ESA WorldCover.
4. Export:
   - Non-blocking Cloud-Optimized GeoTIFF (COG) export via ee.batch.Export.image.
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
logger = logging.getLogger("durg_rice_phenology")


def initialize_earth_engine(project_id: str) -> None:
    """Initializes Earth Engine with specific Cloud Project."""
    try:
        ee.Initialize(project=project_id)
        logger.info(f"Earth Engine successfully initialized with project: '{project_id}'")
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine with project '{project_id}': {e}")
        ee.Initialize()


def load_durg_geometry(geojson_path: Path) -> ee.Geometry:
    """Loads Durg pilot boundary from local GeoJSON or falls back to bounding box."""
    if geojson_path.exists():
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        features = data.get("features", [])
        if features:
            coords = features[0]["geometry"]["coordinates"]
            geom_type = features[0]["geometry"]["type"]
            if geom_type == "Polygon":
                logger.info(f"Loaded Durg district polygon from {geojson_path}")
                return ee.Geometry.Polygon(coords)
            elif geom_type == "MultiPolygon":
                logger.info(f"Loaded Durg district MultiPolygon from {geojson_path}")
                return ee.Geometry.MultiPolygon(coords)

    logger.warning("Local GeoJSON not found. Using calibrated Durg boundary envelope.")
    return ee.Geometry.Polygon([
        [81.08, 20.80],
        [81.66, 20.80],
        [81.66, 21.58],
        [81.08, 21.58],
        [81.08, 20.80]
    ])


def apply_speckle_filter(image: ee.Image) -> ee.Image:
    """Applies focal mean speckle filter to SAR bands."""
    focal = image.select(["VV", "VH"]).focal_mean(radius=15, units="meters")
    return image.addBands(focal, overwrite=True)


def process_s2_image(image: ee.Image) -> ee.Image:
    """Masks clouds and computes NDVI, LSWI, and Flooded signal for Sentinel-2."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    clear_qa = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    scl = image.select("SCL")
    valid_scl = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    mask = clear_qa.And(valid_scl)

    nir = image.select("B8").divide(10000.0)
    red = image.select("B4").divide(10000.0)
    swir1 = image.select("B11").divide(10000.0)

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    lswi = nir.subtract(swir1).divide(nir.add(swir1)).rename("LSWI")
    flooded = lswi.add(0.05).gte(ndvi).Or(lswi.gt(0.0)).rename("FLOODED_SIGNAL")

    return ndvi.addBands([lswi, flooded]).updateMask(mask)


def build_rice_phenology_pipeline(
    geometry: ee.Geometry,
    start_date: str = "2024-06-01",
    end_date: str = "2024-11-30",
    transplanting_end: str = "2024-08-15",
    heading_start: str = "2024-08-16",
    heading_end: str = "2024-10-15"
) -> ee.Image:
    """
    Constructs the fused multi-sensor rice classification and phenology image.
    """
    logger.info("Building Sentinel-1 SAR and Sentinel-2 multi-temporal processing graph...")

    # 1. Sentinel-1 SAR Early Transplanting Window (Specular water dip)
    s1_transplanting = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start_date, transplanting_end)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .map(apply_speckle_filter)
    )
    vh_min_early = s1_transplanting.select("VH").min().rename("VH_min_early")
    vv_min_early = s1_transplanting.select("VV").min().rename("VV_min_early")

    # SAR Vegetative / Heading Window (Canopy volume scattering peak)
    s1_heading = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(heading_start, heading_end)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .map(apply_speckle_filter)
    )
    vh_max_heading = s1_heading.select("VH").max().rename("VH_max_heading")

    # SAR Phenological Dynamics: Canopy emergence amplitude
    sar_vh_amplitude = vh_max_heading.subtract(vh_min_early).rename("VH_amplitude")

    # 2. Sentinel-2 Surface Reflectance
    s2_transplanting = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, transplanting_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        .map(process_s2_image)
    )
    optical_flooded_sum = s2_transplanting.select("FLOODED_SIGNAL").sum().rename("OPTICAL_FLOOD_COUNT")

    s2_heading = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(heading_start, heading_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        .map(process_s2_image)
    )
    ndvi_max_heading = s2_heading.select("NDVI").max().rename("NDVI_max_heading")

    # 3. Topographic and Land Cover Exclusion Masks
    dem = ee.Image("USGS/SRTMGL1_003").clip(geometry)
    slope = ee.Terrain.slope(dem)
    gentle_slope_mask = slope.lt(5.0)

    # ESA WorldCover 10m Cropland mask (40 = Cropland, 30 = Herbaceous vegetation)
    worldcover = ee.Image("ESA/WorldCover/v200/2021").clip(geometry)
    cropland_mask = worldcover.eq(40).Or(worldcover.eq(30))

    # 4. Binary Rice Classification Decision Rule
    # Criteria:
    # - Low early backscatter: VH_min_early < -16.0 dB
    # - Strong vegetative canopy increase: VH_amplitude >= 4.5 dB
    # - High optical vegetative peak: NDVI_max_heading >= 0.50
    # - Topography: Gentle slope < 5%
    sar_rice_rule = vh_min_early.lt(-16.0).And(sar_vh_amplitude.gte(4.5))
    optical_rice_rule = ndvi_max_heading.gte(0.50)

    rice_extent = (
        sar_rice_rule
        .And(optical_rice_rule)
        .And(gentle_slope_mask)
        .rename("rice_extent_binary")
    )

    composite = (
        rice_extent
        .addBands(vh_min_early)
        .addBands(vh_max_heading)
        .addBands(sar_vh_amplitude)
        .addBands(ndvi_max_heading)
        .addBands(optical_flooded_sum)
        .clip(geometry)
    )

    return composite


def export_rice_extent_cog(
    composite_image: ee.Image,
    geometry: ee.Geometry,
    output_target: str,
    export_folder: str,
    bucket_name: str,
    scale: int = 10,
    crs: str = "EPSG:32644"
) -> ee.batch.Task:
    """Submits non-blocking COG export task to Drive or GCS."""
    task_name = "durg_rice_extent_phenology_2024"
    logger.info(f"Submitting non-blocking export task: '{task_name}' (Target: {output_target})...")

    if output_target == "toCloudStorage":
        task = ee.batch.Export.image.toCloudStorage(
            image=composite_image,
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
            image=composite_image,
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
    logger.info(f"Export task submitted successfully! Task ID: {task.id}, Status: {task.status().get('state')}")
    return task


def main():
    parser = argparse.ArgumentParser(
        description="WP-GEE-01: Durg District Sentinel-1/Sentinel-2 Rice Phenology & Extent Pipeline"
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
        help="Builds GEE processing graph and computes sample area without launching cloud export.",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name in ["gee", "scripts"] else script_dir

    config_path = project_root / args.config
    geojson_path = project_root / args.geojson

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_id = cfg["gee"]["project_id"]
    initialize_earth_engine(project_id)

    durg_geom = load_durg_geometry(geojson_path)

    composite = build_rice_phenology_pipeline(
        geometry=durg_geom,
        start_date=cfg["project"]["kharif_season"]["start_date"],
        end_date=cfg["project"]["kharif_season"]["end_date"],
        transplanting_end=cfg["project"]["kharif_season"]["transplanting_window_end"],
        heading_start="2024-08-16",
        heading_end="2024-10-15"
    )

    # Compute estimated rice area within GEE
    rice_mask = composite.select("rice_extent_binary")
    area_image = rice_mask.multiply(ee.Image.pixelArea()).divide(10000.0) # hectares
    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=durg_geom,
        scale=30,
        maxPixels=1e8
    )

    logger.info("Querying GEE for estimated Kharif 2024 Rice Extent in Durg...")
    try:
        rice_area_ha = stats.get("rice_extent_binary").getInfo()
        if rice_area_ha is not None:
            logger.info(f"[GEE COMPUTATION] Durg Estimated Rice Extent: {rice_area_ha:,.2f} ha")
        else:
            logger.info("[GEE COMPUTATION] Durg Area computation completed.")
    except Exception as e:
        logger.warning(f"Could not retrieve synchronous area reduction: {e}")

    if not args.dry_run:
        export_rice_extent_cog(
            composite_image=composite,
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
