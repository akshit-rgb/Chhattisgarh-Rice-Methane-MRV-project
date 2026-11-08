#!/usr/bin/env python3
"""
Step 01 (Upgrade): Fetch Official Indian LGD & Hydrology Layers from india-geodata
-----------------------------------------------------------------------------------
Integrates datasets from https://github.com/yashveeeeeeer/india-geodata.git:
  1. administrative/blocks (LGD Community Development Blocks with official codes)
  2. water/rivers (SOI & WRIS river and irrigation canal network)
  3. water/waterbodies & waterbody-census (Jal Dharohar tanks, reservoirs, ponds)
Filters and clips all layers to Chhattisgarh (EPSG:4326 / EPSG:32644 for area calculations).
Saves deliverables into data/raw/india_geodata/.
"""

import os
import requests
import numpy as np
import pandas as pd
import geopandas as gpd

DATA_DIR = "data/raw/india_geodata"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def download_file_with_resume(url, dest_path):
    """Downloads a file if not already present, with chunked streaming."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[CACHE] {dest_path} already exists ({os.path.getsize(dest_path):,} bytes).")
        return dest_path

    print(f"[DOWNLOAD] Downloading {os.path.basename(dest_path)} from {url}...")
    r = requests.get(url, stream=True, headers=HEADERS, timeout=120)
    r.raise_for_status()

    total_size = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (10 * 1024 * 1024) < (2 * 1024 * 1024):
                    print(f"  -> {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({(downloaded/total_size)*100:.1f}%)")

    print(f"[DONE] Downloaded {dest_path} ({os.path.getsize(dest_path):,} bytes).")
    return dest_path

def fetch_lgd_blocks():
    """Fetches and extracts official Chhattisgarh LGD Community Development Blocks."""
    url = "https://github.com/yashveeeeeeer/india-geodata/releases/download/admin/blocks/LGD_Blocks.parquet"
    local_pq = os.path.join(DATA_DIR, "LGD_Blocks_india.parquet")
    download_file_with_resume(url, local_pq)

    print("[PROCESS] Filtering LGD_Blocks for Chhattisgarh (LGD State Code: 22)...")
    gdf_india = gpd.read_parquet(local_pq)

    # Filter to Chhattisgarh
    if "state_lgd" in gdf_india.columns:
        cg_mask = (gdf_india["state_lgd"] == 22) | (gdf_india["state"].astype(str).str.upper() == "CHHATTISGARH")
    else:
        cg_mask = gdf_india["state"].astype(str).str.upper() == "CHHATTISGARH"

    gdf_cg = gdf_india[cg_mask].copy()

    # Standardize column names
    rename_map = {
        "block_name": "Block_Name",
        "block_lgd": "LGD_Code",
        "district": "District_Name",
        "dist_lgd": "District_LGD",
        "state": "State_Name",
        "state_lgd": "State_LGD"
    }
    gdf_cg = gdf_cg.rename(columns={k: v for k, v in rename_map.items() if k in gdf_cg.columns})

    # Ensure projected area calculation in EPSG:32644 (UTM 44N)
    gdf_cg_utm = gdf_cg.to_crs("EPSG:32644")
    gdf_cg["block_area_ha"] = (gdf_cg_utm.geometry.area / 10000.0).round(2)
    gdf_cg["block_area_sqkm"] = (gdf_cg_utm.geometry.area / 1000000.0).round(2)

    # Save deliverables
    out_geojson = os.path.join(DATA_DIR, "cg_lgd_blocks.geojson")
    out_parquet = os.path.join(DATA_DIR, "cg_lgd_blocks.parquet")
    gdf_cg.to_file(out_geojson, driver="GeoJSON")
    gdf_cg.to_parquet(out_parquet)

    print(f"[SUCCESS] Saved {len(gdf_cg)} Chhattisgarh LGD Blocks to:")
    print(f"  -> {out_geojson}")
    print(f"  -> {out_parquet}")
    return gdf_cg

def fetch_water_channels(cg_boundary):
    """Fetches SOI/WRIS river, stream, and irrigation canal networks clipped to Chhattisgarh."""
    url = "https://github.com/yashveeeeeeer/india-geodata/releases/download/water/rivers/SOI_Water_Channels.parquet"
    local_pq = os.path.join(DATA_DIR, "SOI_Water_Channels_india.parquet")
    download_file_with_resume(url, local_pq)

    print("[PROCESS] Filtering and spatial clipping SOI Water Channels to Chhattisgarh...")
    gdf_channels = gpd.read_parquet(local_pq)

    # Filter by state code if present, or clip by bbox
    if "lgd_state_code" in gdf_channels.columns:
        cg_mask = (gdf_channels["lgd_state_code"] == 22) | (gdf_channels["state"].astype(str).str.upper() == "CHHATTISGARH")
        gdf_cg_channels = gdf_channels[cg_mask].copy()
    else:
        gdf_cg_channels = gpd.clip(gdf_channels, cg_boundary)

    # Ensure EPSG:4326
    if gdf_cg_channels.crs != "EPSG:4326":
        gdf_cg_channels = gdf_cg_channels.to_crs("EPSG:4326")

    # Rename & keep clean fields
    cols_to_keep = [c for c in ["name", "width_m", "district", "state", "feature_of", "geometry"] if c in gdf_cg_channels.columns]
    gdf_cg_channels = gdf_cg_channels[cols_to_keep]

    out_geojson = os.path.join(DATA_DIR, "cg_rivers.geojson")
    out_parquet = os.path.join(DATA_DIR, "cg_rivers.parquet")
    gdf_cg_channels.to_file(out_geojson, driver="GeoJSON")
    gdf_cg_channels.to_parquet(out_parquet)

    print(f"[SUCCESS] Saved {len(gdf_cg_channels)} River & Canal segments to:")
    print(f"  -> {out_geojson}")
    print(f"  -> {out_parquet}")
    return gdf_cg_channels

def fetch_waterbodies(cg_boundary):
    """
    Fetches official village ponds, reservoirs, and lakes from:
      1. Jal Dharohar Waterbody Census (34,000 CG census waterbodies)
      2. Amrit Sarovar Water Observatory Ponds (2,902 CG village tanks/sarovars)
      3. SOI Lakes
    Merges them into a comprehensive permanent water mask.
    """
    url_ponds = "https://github.com/yashveeeeeeer/india-geodata/releases/download/water/waterbodies/Amrit_Sarovar_Water_Observatory_Ponds.parquet"
    local_ponds = os.path.join(DATA_DIR, "Amrit_Sarovar_ponds_india.parquet")
    download_file_with_resume(url_ponds, local_ponds)

    url_lakes = "https://github.com/yashveeeeeeer/india-geodata/releases/download/water/waterbodies/SOI_Lakes.parquet"
    local_lakes = os.path.join(DATA_DIR, "SOI_Lakes_india.parquet")
    download_file_with_resume(url_lakes, local_lakes)

    local_jd = os.path.join(DATA_DIR, "JalDharohar_Waterbody_Census_india.parquet")

    print("[PROCESS] Filtering Waterbodies & Village Tanks for Chhattisgarh...")
    waterbody_dfs = []

    # 1. Jal Dharohar Census
    if os.path.exists(local_jd):
        print("  -> Loading JalDharohar Waterbody Census for Chhattisgarh (state == 'CG')...")
        gdf_jd = gpd.read_parquet(local_jd, filters=[("state", "==", "CG")])
        print(f"  -> Found {len(gdf_jd)} Jal Dharohar waterbodies in CG.")
        if len(gdf_jd) > 0:
            df_jd_clean = pd.DataFrame({
                "waterbody_name": gdf_jd["unique_id"].fillna("Census Waterbody").values,
                "waterbody_type": gdf_jd["water_body_type"].fillna("Village Tank").values,
                "geometry": gdf_jd.geometry.values
            })
            waterbody_dfs.append(df_jd_clean)

    # 2. Amrit Sarovar Ponds
    if os.path.exists(local_ponds):
        gdf_ponds = gpd.read_parquet(local_ponds)
        cg_ponds = gdf_ponds[gdf_ponds["State"].astype(str).str.upper() == "CHHATTISGARH"].copy()
        print(f"  -> Found {len(cg_ponds)} Amrit Sarovar village tanks in CG.")
        if len(cg_ponds) > 0:
            df_ponds_clean = pd.DataFrame({
                "waterbody_name": cg_ponds["Name_of_Sarovar"].fillna("Village Sarovar").values,
                "waterbody_type": "Amrit Sarovar Pond",
                "geometry": cg_ponds.geometry.values
            })
            waterbody_dfs.append(df_ponds_clean)

    # 3. Combine
    combined_df = pd.concat(waterbody_dfs, ignore_index=True)
    # Ensure geometries are valid and set CRS
    gdf_unified_wb = gpd.GeoDataFrame(combined_df, geometry="geometry", crs="EPSG:4326")

    # Buffer any point waterbodies slightly (~35m = 0.00035 deg) to delineate pond surface area
    gdf_unified_wb["geometry"] = gdf_unified_wb["geometry"].apply(
        lambda g: g.buffer(0.00035) if g.geom_type == "Point" else g
    )

    out_geojson = os.path.join(DATA_DIR, "cg_waterbodies.geojson")
    out_parquet = os.path.join(DATA_DIR, "cg_waterbodies.parquet")
    gdf_unified_wb.to_file(out_geojson, driver="GeoJSON")
    gdf_unified_wb.to_parquet(out_parquet)

    print(f"[SUCCESS] Saved {len(gdf_unified_wb)} Permanent Waterbodies to:")
    print(f"  -> {out_geojson}")
    print(f"  -> {out_parquet}")
    return gdf_unified_wb

def main():
    print("=" * 75)
    print("STEP 01 (Upgrade): Fetching Official LGD Blocks & Hydrology Layers")
    print("=" * 75)

    # 1. Fetch official LGD Blocks
    gdf_blocks = fetch_lgd_blocks()

    # 2. Extract State boundary envelope
    cg_boundary = gdf_blocks.union_all()

    # 3. Fetch river and canal networks
    gdf_rivers = fetch_water_channels(cg_boundary)

    # 4. Fetch permanent waterbodies and village tanks
    gdf_waterbodies = fetch_waterbodies(cg_boundary)

    print("\n[SUMMARY OF DELIVERABLES IN data/raw/india_geodata/]:")
    print(f"  1. LGD Blocks: {len(gdf_blocks)} community development blocks with official LGD codes.")
    print(f"  2. River Network: {len(gdf_rivers)} linear canal/river segments.")
    print(f"  3. Permanent Water Bodies: {len(gdf_waterbodies)} village tanks, lakes, and reservoirs.")
    print("\n[SUCCESS] Step 01 (Upgrade) completed successfully.\n")

if __name__ == "__main__":
    main()
