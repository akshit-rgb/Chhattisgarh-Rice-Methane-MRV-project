#!/usr/bin/env python3
"""
Step 06 (Upgrade): Web GIS Dashboard Asset Builder & FTW Live Maps Generator
--------------------------------------------------------------------------------------
Builds self-contained GeoJSON assets and a responsive, publication-grade
interactive Web GIS Dashboard (index.html) deployable directly via GitHub Pages:
  - data/processed/cg_blocks_mrv_web.geojson (146 blocks with FTW 10m parcel census, XGBoost predictions & GW)
  - data/processed/durg_parcels_web.geojson (2,000 real 10m field parcels for cadastral overlay)
  - index.html with 3 Live Panels:
      1. 🛰️ Live Interactive GIS Map (Leaflet.js + Dual-Season Simulation + FTW 10m parcels)
      2. 📊 FTW Cartography Atlas (7 High-Res Map Plates @ 300 DPI with dynamic reload)
      3. 🔬 146-Block Cadastral Maps Explorer (Searchable gallery for all 146 block field maps)
"""

import glob
import json
import os
import time

import geopandas as gpd
import pandas as pd
from shapely import from_wkb
from shapely.geometry import Point

GWP_CH4 = 27.9

BLOCKS_SHP = "data/processed/shapefiles/chhattisgarh_blocks_mrv.shp"
FTW_CSV = "outputs/reports/chhattisgarh_block_methane_analysis_agriculture_field.csv"
PANEL_CSV = "data/processed/chhattisgarh_statewide_kharif_rabi_groundwater_panel_2019_2026.csv"
FUTURE_CSV = "data/processed/chhattisgarh_block_future_forecasts_2027_2035.csv"
PATAN_PARQUET = "data/processed/block_fields/parquet/block_3639_patan.parquet"
BLOCK_MAPS_DIR = "outputs/figures/block_field_maps"
WEB_BLOCKS_GEOJSON = "data/processed/cg_blocks_mrv_web.geojson"
WEB_PARCELS_GEOJSON = "data/processed/durg_parcels_web.geojson"


def build_parcels_geojson():
    """Generates 2,000 real 10m field parcels GeoJSON for Patan/Durg zoom overlay."""
    if os.path.exists(WEB_PARCELS_GEOJSON) and os.path.getsize(WEB_PARCELS_GEOJSON) > 10000:
        print(f"[INFO] Using existing parcels GeoJSON: {WEB_PARCELS_GEOJSON}")
        return

    if not os.path.exists(PATAN_PARQUET):
        print(f"[WARN] Patan parquet not found at {PATAN_PARQUET}, skipping parcels GeoJSON.")
        return

    print("[INFO] Generating sample 10m field parcels GeoJSON from Patan...")
    df = pd.read_parquet(PATAN_PARQUET)
    geoms = from_wkb(df["geometry"].values)
    gdf = gpd.GeoDataFrame(
        df[["parcel_id", "area_ha", "confidence_mean"]],
        geometry=geoms,
        crs="EPSG:4326"
    )

    minx, miny, maxx, maxy = gdf.total_bounds
    center_pt = Point((minx + maxx) / 2.0, (miny + maxy) / 2.0)
    gdf["dist"] = gdf.geometry.centroid.distance(center_pt)
    sample = gdf.nsmallest(2000, "dist").copy()
    sample["area_ha"] = sample["area_ha"].round(3)
    sample["conf"] = sample["confidence_mean"].round(3)
    sample = sample[["parcel_id", "area_ha", "conf", "geometry"]]
    sample.to_file(WEB_PARCELS_GEOJSON, driver="GeoJSON")
    print(f"[SUCCESS] Exported {len(sample):,} parcels to {WEB_PARCELS_GEOJSON} ({os.path.getsize(WEB_PARCELS_GEOJSON)/1024:.1f} KB)")


def build_blocks_geojson() -> gpd.GeoDataFrame:
    """Builds and enriches 146 blocks with FTW ground truth parcel census & predictions."""
    print("[INFO] Building enriched 146-block GeoJSON with FTW census and methane analysis...")
    if not os.path.exists(BLOCKS_SHP):
        raise FileNotFoundError(f"Blocks shapefile missing: {BLOCKS_SHP}")

    gdf_blocks = gpd.read_file(BLOCKS_SHP)
    if gdf_blocks.crs != "EPSG:4326":
        gdf_blocks = gdf_blocks.to_crs("EPSG:4326")

    # Load FTW empirical methane analysis
    if not os.path.exists(FTW_CSV):
        raise FileNotFoundError(f"FTW methane analysis CSV missing: {FTW_CSV}")
    df_ftw = pd.read_csv(FTW_CSV)

    # Load dual-season groundwater panel (2024)
    df_panel_2024 = None
    if os.path.exists(PANEL_CSV):
        df_p = pd.read_csv(PANEL_CSV)
        df_panel_2024 = df_p[df_p["year"] == 2024].copy()

    # Load future forecasts (2035)
    f_2035_bau = None
    f_2035_mit = None
    if os.path.exists(FUTURE_CSV):
        df_f = pd.read_csv(FUTURE_CSV)
        f_2035_bau = df_f[(df_f["forecast_year"] == 2035) & (df_f["scenario"] == "Scenario 1 (BAU)")][
            ["lgd_code", "total_ch4_tonnes", "gw_stage_pct"]
        ].rename(columns={"total_ch4_tonnes": "ch4_2035_bau", "gw_stage_pct": "gw_2035_bau"})
        f_2035_mit = df_f[(df_f["forecast_year"] == 2035) & (df_f["scenario"] == "Scenario 3 (Rabi Groundwater Conservation + DSR)")][
            ["lgd_code", "avoided_co2e_tonnes", "annual_gw_savings_mcm"]
        ].rename(columns={"avoided_co2e_tonnes": "avoided_co2e_2035", "annual_gw_savings_mcm": "gw_saved_mcm"})

    # Prepare FTW attributes
    ftw_keep = [
        "LGD_Code", "Field_Count", "Rice_ha", "Mean_Field_ha", "Marginal_pct",
        "Annual_CH4_Tonnes", "Seasonal_CH4_Tonnes_Kharif", "Seasonal_CH4_Tonnes_Rabi",
        "Annual_CO2e_Tonnes", "AWD_Total_CH4_Saved_Tonnes", "AWD_Total_Carbon_Offsets_tCO2e",
        "Annual_Carbon_Revenue_INR_Crore", "Annual_Carbon_Revenue_USD_15", "Annual_Emission_Intensity_kg_ha"
    ]
    df_ftw_sub = df_ftw[ftw_keep].rename(columns={
        "Field_Count": "ftw_parcels",
        "Rice_ha": "ftw_rice_ha",
        "Mean_Field_ha": "ftw_mean_field",
        "Marginal_pct": "ftw_marg_pct",
        "Annual_CH4_Tonnes": "ftw_ch4_t",
        "Seasonal_CH4_Tonnes_Kharif": "ftw_k_ch4_t",
        "Seasonal_CH4_Tonnes_Rabi": "ftw_r_ch4_t",
        "Annual_CO2e_Tonnes": "ftw_co2e_t",
        "AWD_Total_CH4_Saved_Tonnes": "ftw_awd_ch4",
        "AWD_Total_Carbon_Offsets_tCO2e": "ftw_carbon_offsets",
        "Annual_Carbon_Revenue_INR_Crore": "ftw_revenue_cr",
        "Annual_Carbon_Revenue_USD_15": "ftw_revenue_usd",
        "Annual_Emission_Intensity_kg_ha": "ftw_intensity"
    })

    merged = gdf_blocks.merge(df_ftw_sub, on="LGD_Code", how="left")

    if df_panel_2024 is not None:
        p_cols = ["lgd_code", "cgwb_gw_stage_pct", "cgwb_aquifer_category", "gldas_summer_drawdown_mm",
                  "methane_gw_cobenefit_index", "shc_soc_pct", "shc_ph", "shc_nitrogen_kg_ha",
                  "aez_region", "soil_indigenous_type"]
        p_sub = df_panel_2024[[c for c in p_cols if c in df_panel_2024.columns]].rename(columns={
            "cgwb_gw_stage_pct": "gw_stage",
            "cgwb_aquifer_category": "gw_cat",
            "gldas_summer_drawdown_mm": "drawdown_mm",
            "methane_gw_cobenefit_index": "cobenefit_idx",
            "shc_soc_pct": "soc_pct",
            "shc_ph": "ph",
            "shc_nitrogen_kg_ha": "nitrogen",
            "aez_region": "aez",
            "soil_indigenous_type": "soil"
        })
        merged = merged.merge(p_sub, left_on="LGD_Code", right_on="lgd_code", how="left")

    if f_2035_bau is not None:
        merged = merged.merge(f_2035_bau, left_on="LGD_Code", right_on="lgd_code", how="left")
    if f_2035_mit is not None:
        merged = merged.merge(f_2035_mit, left_on="LGD_Code", right_on="lgd_code", how="left")

    # Simplify geometries to ~0.002 degrees for lightning-fast web rendering (< 800 KB)
    merged_sim = merged.copy()
    merged_sim["geometry"] = merged.geometry.simplify(0.002, preserve_topology=True)

    # Standardize column names
    col_rename = {
        "Block": "block_name",
        "District": "district_name",
        "LGD_Code": "lgd_code"
    }
    merged_sim = merged_sim.rename(columns={k: v for k, v in col_rename.items() if k in merged_sim.columns})

    # Fill defaults for any missing attributes
    num_cols = ["ftw_parcels", "ftw_rice_ha", "ftw_mean_field", "ftw_marg_pct",
                "ftw_ch4_t", "ftw_k_ch4_t", "ftw_r_ch4_t", "ftw_co2e_t",
                "ftw_awd_ch4", "ftw_carbon_offsets", "ftw_revenue_cr", "ftw_revenue_usd", "ftw_intensity",
                "gw_stage", "drawdown_mm", "cobenefit_idx", "soc_pct", "ph", "nitrogen",
                "ch4_2035_bau", "avoided_co2e_2035", "gw_saved_mcm"]
    for c in num_cols:
        if c in merged_sim.columns:
            merged_sim[c] = merged_sim[c].fillna(0).round(2)

    str_cols = ["gw_cat", "aez", "soil"]
    for c in str_cols:
        if c in merged_sim.columns:
            merged_sim[c] = merged_sim[c].fillna("Standard")

    merged_sim.to_file(WEB_BLOCKS_GEOJSON, driver="GeoJSON")
    print(f"[SUCCESS] Exported web blocks GeoJSON: {WEB_BLOCKS_GEOJSON} ({os.path.getsize(WEB_BLOCKS_GEOJSON)/1024:.1f} KB)")
    return merged_sim


def get_block_maps_catalog(df_blocks):
    """Builds an index of all 146 individual block field maps."""
    png_files = glob.glob(os.path.join(BLOCK_MAPS_DIR, "block_*_field_map.png"))
    map_dict = {}
    for p in png_files:
        base = os.path.basename(p)
        parts = base.split("_")
        try:
            lgd = int(parts[1])
            map_dict[lgd] = base
        except Exception:
            continue

    catalog = []
    for _, r in df_blocks.sort_values(["district_name", "block_name"]).iterrows():
        lgd = int(r["lgd_code"])
        fname = map_dict.get(lgd, f"block_{lgd}_{r['block_name'].lower()}_field_map.png")
        catalog.append({
            "lgd": lgd,
            "block": str(r["block_name"]).upper(),
            "district": str(r["district_name"]).title(),
            "parcels": int(r.get("ftw_parcels", 0)),
            "rice_ha": float(r.get("ftw_rice_ha", 0)),
            "ch4_t": float(r.get("ftw_ch4_t", 0)),
            "awd_ch4": float(r.get("ftw_awd_ch4", 0)),
            "rev_cr": float(r.get("ftw_revenue_cr", 0)),
            "file": f"outputs/figures/block_field_maps/{fname}"
        })
    return catalog


def generate_index_html(blocks_gdf, catalog):
    """Generates the upgraded index.html with 3 live panels and dynamic map updates."""
    print("[INFO] Assembling full index.html with FTW live maps and cartography atlas...")
    with open(WEB_BLOCKS_GEOJSON, "r", encoding="utf-8") as f:
        blocks_geojson_str = f.read()

    parcels_geojson_str = '{"type":"FeatureCollection","features":[]}'
    if os.path.exists(WEB_PARCELS_GEOJSON):
        with open(WEB_PARCELS_GEOJSON, "r", encoding="utf-8") as f:
            parcels_geojson_str = f.read()

    catalog_json_str = json.dumps(catalog)
    timestamp_ver = int(time.time())

    # Calculate statewide totals for header cards
    tot_parcels = int(blocks_gdf["ftw_parcels"].sum())
    tot_rice_ha = float(blocks_gdf["ftw_rice_ha"].sum())
    tot_ch4 = float(blocks_gdf["ftw_ch4_t"].sum())
    tot_rabi_ch4 = float(blocks_gdf["ftw_r_ch4_t"].sum())
    tot_offsets = float(blocks_gdf["ftw_carbon_offsets"].sum())
    tot_revenue_cr = float(blocks_gdf["ftw_revenue_cr"].sum())

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Chhattisgarh 146-Block Rice Methane & Groundwater MRV (14.4M FTW Parcels)</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }}
    body {{ display: flex; flex-direction: column; height: 100vh; background: #0b1120; color: #f8fafc; overflow: hidden; }}
    
    /* Top Navbar */
    #navbar {{
      height: 52px; background: #0f172a; border-bottom: 1px solid #1e293b;
      display: flex; align-items: center; justify-content: space-between; padding: 0 18px; z-index: 2000;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }}
    .nav-brand {{ display: flex; align-items: center; gap: 10px; }}
    .nav-brand h1 {{ font-size: 15px; font-weight: 700; color: #38bdf8; letter-spacing: -0.2px; }}
    .nav-brand .badge-tag {{
      background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 2px 7px; border-radius: 4px; font-size: 10px; font-weight: 600; font-family: 'JetBrains Mono', monospace;
    }}
    .nav-tabs {{ display: flex; gap: 6px; }}
    .nav-tab-btn {{
      background: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 7px 14px;
      border-radius: 6px; font-size: 11.5px; font-weight: 600; cursor: pointer; transition: all 0.2s;
      display: flex; align-items: center; gap: 6px;
    }}
    .nav-tab-btn:hover {{ background: #334155; color: #f8fafc; }}
    .nav-tab-btn.active {{ background: #0284c7; color: white; border-color: #38bdf8; box-shadow: 0 0 12px rgba(2, 132, 199, 0.4); }}

    /* Main Workspace Views */
    #app-workspace {{ flex: 1; display: flex; position: relative; overflow: hidden; }}
    .app-panel {{ width: 100%; height: 100%; display: none; }}
    .app-panel.active {{ display: flex; }}

    /* Panel 1: Interactive GIS Map */
    #sidebar {{
      width: 440px; background: #111827; border-right: 1px solid #1f2937;
      display: flex; flex-direction: column; z-index: 1000; box-shadow: 4px 0 18px rgba(0,0,0,0.4);
    }}
    #map-container {{ flex: 1; position: relative; height: 100%; }}
    #map {{ width: 100%; height: 100%; }}
    .scrollable {{ flex: 1; overflow-y: auto; padding: 14px 18px; }}
    .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }}
    .metric-card {{ background: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 9px 12px; }}
    .metric-card .title {{ font-size: 9.5px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-card .value {{ font-size: 16px; font-weight: 700; color: #f9fafb; margin-top: 3px; }}
    .metric-card.accent .value {{ color: #4ade80; }}
    .metric-card.warning .value {{ color: #fbbf24; }}
    .metric-card.danger .value {{ color: #f87171; }}
    .metric-card.cyan .value {{ color: #38bdf8; }}
    .section-title {{ font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: #cbd5e1; margin: 12px 0 6px 0; letter-spacing: 0.5px; }}
    .control-group {{ background: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 11px; margin-bottom: 12px; }}
    .slider-row {{ margin-bottom: 9px; }}
    .slider-label {{ display: flex; justify-content: space-between; font-size: 11px; color: #d1d5db; margin-bottom: 3px; }}
    .slider-val {{ font-weight: 700; color: #38bdf8; }}
    input[type=range] {{ -webkit-appearance: none; width: 100%; background: #374151; height: 5px; border-radius: 3px; outline: none; }}
    input[type=range]::-webkit-slider-thumb {{ -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #38bdf8; cursor: pointer; }}
    select, button {{ background: #111827; color: #f9fafb; border: 1px solid #4b5563; padding: 6px 9px; border-radius: 6px; font-size: 11px; outline: none; cursor: pointer; width: 100%; }}
    select:focus {{ border-color: #38bdf8; }}
    .btn-action {{ background: #0284c7; color: white; border: none; font-weight: 600; padding: 8px; border-radius: 6px; margin-top: 6px; display: flex; align-items: center; justify-content: center; gap: 6px; text-decoration: none; font-size: 11.5px; transition: background 0.2s; }}
    .btn-action:hover {{ background: #0369a1; }}
    
    #inspector {{ background: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 11px; margin-top: 6px; min-height: 140px; }}
    #inspector h3 {{ font-size: 13px; color: #38bdf8; font-weight: 700; display: flex; align-items: center; justify-content: space-between; }}
    #inspector .sub {{ font-size: 10px; color: #9ca3af; margin: 2px 0 6px 0; }}
    #inspector .stat-row {{ display: flex; justify-content: space-between; font-size: 11px; padding: 3px 0; border-bottom: 1px solid #374151; }}
    #inspector .stat-row:last-child {{ border-bottom: none; }}
    .badge {{ padding: 2px 5px; border-radius: 4px; font-size: 9px; font-weight: 600; text-transform: uppercase; }}
    .badge.safe {{ background: rgba(74, 222, 128, 0.2); color: #4ade80; }}
    .badge.semi {{ background: rgba(251, 191, 36, 0.2); color: #fbbf24; }}
    .badge.critical {{ background: rgba(248, 113, 113, 0.2); color: #f87171; }}
    .badge.hotspot {{ background: rgba(244, 63, 94, 0.2); color: #fb7185; }}
    .legend-box {{ position: absolute; bottom: 20px; right: 20px; background: rgba(17, 24, 39, 0.94); border: 1px solid #374151; padding: 9px 13px; border-radius: 8px; z-index: 1000; font-size: 10.5px; backdrop-filter: blur(4px); }}
    .legend-gradient {{ height: 9px; width: 170px; border-radius: 4px; margin: 4px 0; }}
    .toggle-row {{ display: flex; align-items: center; gap: 7px; font-size: 11px; color: #d1d5db; margin-top: 7px; cursor: pointer; }}

    /* Panel 2: FTW Cartography Atlas */
    #panel-atlas {{ flex-direction: column; background: #0b1120; overflow-y: auto; padding: 20px 24px; }}
    .atlas-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid #1f2937; padding-bottom: 12px; }}
    .atlas-header h2 {{ font-size: 18px; font-weight: 700; color: #38bdf8; }}
    .atlas-header p {{ font-size: 12px; color: #94a3b8; margin-top: 3px; }}
    .plate-btn-grid {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
    .plate-btn {{
      background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 8px 14px;
      border-radius: 6px; font-size: 11.5px; font-weight: 600; cursor: pointer; transition: all 0.2s;
    }}
    .plate-btn:hover {{ background: #334155; color: white; }}
    .plate-btn.active {{ background: #0284c7; color: white; border-color: #38bdf8; }}
    .plate-display-card {{
      background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 18px;
      display: flex; flex-direction: column; gap: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    .plate-info-bar {{ display: flex; justify-content: space-between; align-items: center; }}
    .plate-title-group h3 {{ font-size: 15px; font-weight: 700; color: #f8fafc; }}
    .plate-title-group p {{ font-size: 11.5px; color: #94a3b8; margin-top: 2px; }}
    .plate-meta-badges {{ display: flex; gap: 6px; }}
    .plate-img-wrapper {{
      width: 100%; min-height: 580px; max-height: 780px; background: #030712; border: 1px solid #1f2937;
      border-radius: 8px; display: flex; align-items: center; justify-content: center; overflow: hidden;
      cursor: zoom-in; position: relative;
    }}
    .plate-img-wrapper img {{ max-width: 100%; max-height: 100%; object-fit: contain; transition: transform 0.2s; }}

    /* Panel 3: 146-Block Cadastral Maps Gallery */
    #panel-blocks {{ display: flex; background: #0b1120; height: 100%; }}
    #blocks-catalog-sidebar {{
      width: 380px; background: #111827; border-right: 1px solid #1f2937;
      display: flex; flex-direction: column; padding: 14px 16px; overflow-y: auto;
    }}
    #blocks-catalog-sidebar h2 {{ font-size: 15px; font-weight: 700; color: #38bdf8; margin-bottom: 4px; }}
    #blocks-catalog-sidebar p {{ font-size: 11px; color: #9ca3af; margin-bottom: 12px; }}
    .block-search-box {{ margin-bottom: 10px; }}
    .block-search-box input {{
      width: 100%; background: #1f2937; border: 1px solid #374151; color: white;
      padding: 7px 11px; border-radius: 6px; font-size: 11.5px; outline: none;
    }}
    .block-search-box input:focus {{ border-color: #38bdf8; }}
    .block-meta-card {{
      background: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 12px;
      margin-top: 12px; display: flex; flex-direction: column; gap: 5px;
    }}
    .block-meta-card h4 {{ font-size: 13px; color: #38bdf8; font-weight: 700; }}
    .block-meta-card .stat-line {{ display: flex; justify-content: space-between; font-size: 11px; padding: 2px 0; border-bottom: 1px solid #374151; }}
    .block-meta-card .stat-line:last-child {{ border-bottom: none; }}
    #block-viewer-main {{ flex: 1; padding: 18px 22px; display: flex; flex-direction: column; overflow-y: auto; }}
    .block-viewer-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
    .block-viewer-header h3 {{ font-size: 16px; font-weight: 700; color: #f8fafc; }}
    .block-img-container {{
      flex: 1; min-height: 600px; background: #030712; border: 1px solid #1f2937;
      border-radius: 8px; display: flex; align-items: center; justify-content: center; overflow: hidden;
      cursor: zoom-in; position: relative;
    }}
    .block-img-container img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}

    /* Lightbox Modal */
    #lightbox {{
      position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(3, 7, 18, 0.95);
      z-index: 5000; display: none; align-items: center; justify-content: center; backdrop-filter: blur(8px);
    }}
    #lightbox.active {{ display: flex; }}
    #lightbox img {{ max-width: 95vw; max-height: 92vh; object-fit: contain; border-radius: 6px; box-shadow: 0 0 30px rgba(0,0,0,0.8); }}
    #lightbox-close {{
      position: absolute; top: 16px; right: 24px; color: #94a3b8; font-size: 28px; font-weight: 700;
      cursor: pointer; transition: color 0.2s;
    }}
    #lightbox-close:hover {{ color: white; }}
  </style>
</head>
<body>

  <!-- Top Navigation Bar -->
  <div id="navbar">
    <div class="nav-brand">
      <h1>🌾 Chhattisgarh Rice Methane & Groundwater MRV</h1>
      <span class="badge-tag">14.4M FTW PARCELS</span>
      <span class="badge-tag" style="color:#4ade80; border-color:rgba(74,222,128,0.3); background:rgba(74,222,128,0.15)">146 BLOCKS</span>
    </div>
    <div class="nav-tabs">
      <button class="nav-tab-btn active" onclick="switchTab('map')">
        <span>🛰️</span> Live Interactive GIS Map
      </button>
      <button class="nav-tab-btn" onclick="switchTab('atlas')">
        <span>📊</span> FTW Cartography Atlas (7 Plates)
      </button>
      <button class="nav-tab-btn" onclick="switchTab('blocks')">
        <span>🔬</span> 146-Block Field Maps Gallery
      </button>
      <a class="nav-tab-btn" href="https://code.earthengine.google.com/?scriptPath=users/akshitvaishnavonline:chhattisgarh_statewide_kharif_rabi_future_dashboard" target="_blank" style="text-decoration:none;">
        <span>☁️</span> GEE Dashboard
      </a>
    </div>
  </div>

  <!-- Application Workspace Container -->
  <div id="app-workspace">

    <!-- PANEL 1: Interactive GIS Map & Dual-Season Simulation -->
    <div id="panel-map" class="app-panel active">
      <div id="sidebar">
        <div class="scrollable">
          
          <!-- FTW Ground-Truth Metrics Grid -->
          <div class="metric-grid">
            <div class="metric-card cyan">
              <div class="title">Total Annual CH4 (2024)</div>
              <div class="value" id="card-tot-ch4">{tot_ch4:,.0f} t</div>
            </div>
            <div class="metric-card warning">
              <div class="title">Rabi Summer CH4</div>
              <div class="value" id="card-rabi-ch4">{tot_rabi_ch4:,.0f} t</div>
            </div>
            <div class="metric-card accent">
              <div class="title">AWD Carbon Offsets</div>
              <div class="value" id="card-co2e">{tot_offsets/1e6:.2f}M t</div>
            </div>
            <div class="metric-card danger">
              <div class="title">Carbon Value (₹ Cr)</div>
              <div class="value" id="card-revenue">₹{tot_revenue_cr:,.1f} Cr</div>
            </div>
          </div>

          <!-- Scenario Simulation Engine -->
          <div class="section-title">Scenario Simulation Engine (2024–2035)</div>
          <div class="control-group">
            <div class="slider-row">
              <div class="slider-label">
                <span>Simulation Horizon Year:</span>
                <span class="slider-val" id="lbl-year">2024</span>
              </div>
              <input type="range" id="sim-year" min="2024" max="2035" value="2024" step="1" oninput="updateSimulation()">
            </div>
            <div class="slider-row">
              <div class="slider-label">
                <span>AWD Pani-Pipe Adoption:</span>
                <span class="slider-val" id="lbl-awd">40%</span>
              </div>
              <input type="range" id="sim-awd" min="0" max="100" value="40" step="5" oninput="updateSimulation()">
            </div>
            <div class="slider-row" style="margin-bottom: 2px;">
              <div class="slider-label">
                <span>Monsoon Rainfall Anomaly:</span>
                <span class="slider-val" id="lbl-rain">0%</span>
              </div>
              <input type="range" id="sim-rain" min="-30" max="30" value="0" step="5" oninput="updateSimulation()">
            </div>
          </div>

          <!-- Statewide Layer & District Controls -->
          <div class="section-title">Statewide Layer & District Controls</div>
          <div class="control-group">
            <div style="margin-bottom: 5px; font-size: 11px; color: #cbd5e1; font-weight: 600;">Active Metric Visualized:</div>
            <select id="metric-select" onchange="updateChoropleth()">
              <option value="ftw_ch4_t">🌾 FTW Annual Methane (Tonnes CH4 | XGBoost)</option>
              <option value="ftw_k_ch4_t">🌾 FTW Kharif Monsoon Methane (Tonnes CH4)</option>
              <option value="ftw_r_ch4_t">🌾 FTW Rabi Summer Methane (Garma Dhan)</option>
              <option value="ftw_awd_ch4">🌿 FTW Verra VM0042 AWD Abatement (t CH4)</option>
              <option value="ftw_carbon_offsets">💰 FTW Annual Carbon Credits (tCO2e)</option>
              <option value="ftw_revenue_cr">💵 FTW Carbon Revenue Potential (₹ Crore/yr)</option>
              <option value="ftw_rice_ha">🚜 FTW Measured Rice Extent (Hectares)</option>
              <option value="ftw_parcels">📐 FTW Physical Field Parcels Count (Medh)</option>
              <option value="ftw_intensity">⚡ FTW Emission Intensity (kg CH4/ha)</option>
              <option value="gw_stage">💧 CGWB Groundwater Extraction Stage (%)</option>
              <option value="cobenefit_idx">🔄 Methane–GW Co-Benefit Index (0-100)</option>
              <option value="soc_pct">🌱 Soil Organic Carbon (% SHC)</option>
              <option value="avoided_co2e_2035">🎯 2035 Policy Avoided CO2e (Tonnes)</option>
              <option value="gw_saved_mcm">💧 2035 Annual GW Savings (Million m³)</option>
            </select>

            <div style="margin: 8px 0 4px 0; font-size: 11px; color: #cbd5e1; font-weight: 600;">District / Territory Focus:</div>
            <select id="district-filter" onchange="zoomToDistrict()">
              <option value="ALL">ALL CHHATTISGARH (146 Blocks)</option>
              <option value="Durg">Durg (Command & Pumping Basin)</option>
              <option value="Raipur">Raipur (Mahanadi Plains)</option>
              <option value="Balod">Balod (Over-Exploited Aquifers)</option>
              <option value="Bemetara">Bemetara (High Extraction)</option>
              <option value="Bilaspur">Bilaspur (Hasdeo Basin)</option>
              <option value="Janjgir-Champa">Janjgir-Champa (Highest Paddy Area)</option>
              <option value="Rajnandgaon">Rajnandgaon (Western Plains)</option>
              <option value="Bastar">Bastar (Bastar Plateau Uplands)</option>
              <option value="Surguja">Surguja (Northern Undulating Hills)</option>
            </select>

            <label class="toggle-row">
              <input type="checkbox" id="toggle-parcels" checked onchange="toggleParcelsLayer()">
              <span>Show 10m Field-Parcel Overlay (Zoom &ge; 11 | Patan)</span>
            </label>
          </div>

          <!-- Block / Parcel Inspector -->
          <div class="section-title">Block / Parcel Inspector</div>
          <div id="inspector">
            <div style="color: #94a3b8; font-size: 11px; text-align: center; margin-top: 35px;">
              Click any block on the map to inspect dual-season methane, FTW 10m parcel census, and carbon mitigation revenue.
            </div>
          </div>

          <!-- Top Emitting Blocks Chart -->
          <div class="section-title">Top 5 Methane Hotspot Blocks</div>
          <div style="height: 150px; background: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 8px;">
            <canvas id="rankingChart"></canvas>
          </div>

        </div>
      </div>

      <!-- Map Element & Floating Legend -->
      <div id="map-container">
        <div id="map"></div>
        <div class="legend-box" id="legend">
          <div id="legend-title" style="font-weight: 600; margin-bottom: 2px;">FTW Annual Methane</div>
          <div class="legend-gradient" id="legend-bar" style="background: linear-gradient(to right, #ffffcc, #fecc5c, #fd8d3c, #f03b20, #bd0026);"></div>
          <div style="display: flex; justify-content: space-between; font-size: 9.5px; color: #94a3b8;">
            <span id="legend-min">0 t</span>
            <span id="legend-max">25,000 t</span>
          </div>
        </div>
      </div>
    </div>

    <!-- PANEL 2: FTW Cartography Atlas (7 High-Res Plates) -->
    <div id="panel-atlas" class="app-panel">
      <div class="atlas-header">
        <div>
          <h2>🗺️ Statewide FTW Cartography Atlas & Methane MRV Plates</h2>
          <p>Publication-Grade 300 DPI Maps Derived from 14,405,396 Physical Field Boundaries (Zero Distortion | UTM Zone 44N)</p>
        </div>
        <button class="nav-tab-btn" onclick="refreshAtlasImages()" title="Reload newest rendered map files">
          🔄 Refresh Updated Maps
        </button>
      </div>

      <!-- Plate Switcher Buttons -->
      <div class="plate-btn-grid">
        <button class="plate-btn active" onclick="loadPlate(1)">Plate 01: Statewide 146-Block Methane Atlas</button>
        <button class="plate-btn" onclick="loadPlate(2)">Plate 02: Dual-Season Kharif vs. Rabi</button>
        <button class="plate-btn" onclick="loadPlate(3)">Plate 03: AWD Abatement & Carbon Credits</button>
        <button class="plate-btn" onclick="loadPlate(4)">Plate 04: Holding Size Distribution</button>
        <button class="plate-btn" onclick="loadPlate(5)">Plate 05: 146-Block Field Footprint Atlas</button>
        <button class="plate-btn" onclick="loadPlate(10)">Plate 10: Statewide Block Field Density</button>
        <button class="plate-btn" onclick="loadPlate(11)">Plate 11: 4-Panel Key Blocks Cadastral</button>
      </div>

      <!-- Plate Display Card -->
      <div class="plate-display-card">
        <div class="plate-info-bar">
          <div class="plate-title-group">
            <h3 id="plate-title">Plate 01: Statewide 146-Block Agricultural Methane Atlas</h3>
            <p id="plate-desc">Comprehensive 146-Block Choropleth with 3 Cadastral Insets (Patan, Jagdalpur, Masturi) @ 300 DPI</p>
          </div>
          <div class="plate-meta-badges">
            <span class="badge-tag" id="plate-badge-dpi">300 DPI</span>
            <span class="badge-tag" style="color:#4ade80; border-color:rgba(74,222,128,0.3); background:rgba(74,222,128,0.15)">EPSG:32644</span>
            <button class="nav-tab-btn" style="padding:4px 10px; font-size:11px;" onclick="openCurrentPlateLightbox()">🔍 Enlarge</button>
          </div>
        </div>
        <div class="plate-img-wrapper" onclick="openCurrentPlateLightbox()">
          <img id="plate-img" src="outputs/figures/01_agriculture_field_statewide_methane_analysis_300dpi.png?v={timestamp_ver}" alt="Cartography Plate">
        </div>
      </div>
    </div>

    <!-- PANEL 3: 146-Block Cadastral Maps Gallery -->
    <div id="panel-blocks" class="app-panel">
      <div id="blocks-catalog-sidebar">
        <h2>🔬 146-Block Field Maps Gallery</h2>
        <p>Browse dedicated 10m cadastral parcel maps for all 146 Community Development Blocks across Chhattisgarh.</p>

        <div class="block-search-box">
          <input type="text" id="block-search" placeholder="🔍 Search Block or District..." oninput="filterBlockList()">
        </div>

        <div style="font-size:11px; color:#cbd5e1; font-weight:600; margin-bottom:5px;">Select Block to Inspect:</div>
        <select id="block-catalog-select" size="12" style="height:260px; overflow-y:auto;" onchange="selectBlockFromCatalog()">
          <!-- Populated by JS -->
        </select>

        <div class="block-meta-card" id="catalog-meta-card">
          <h4 id="meta-block-name">PATAN BLOCK</h4>
          <div class="stat-line"><span>District:</span><strong id="meta-district">Durg</strong></div>
          <div class="stat-line"><span>LGD Code:</span><strong id="meta-lgd">3639</strong></div>
          <div class="stat-line"><span>Physical Field Parcels:</span><strong id="meta-parcels" style="color:#38bdf8">104,478</strong></div>
          <div class="stat-line"><span>Measured Rice Extent:</span><strong id="meta-rice">18,342.1 ha</strong></div>
          <div class="stat-line"><span>Annual Methane Footprint:</span><strong id="meta-ch4">3,892 t CH4</strong></div>
          <div class="stat-line"><span>AWD Abatement Potential:</span><strong id="meta-awd" style="color:#4ade80">1,712 t CH4</strong></div>
          <div class="stat-line"><span>Annual Carbon Revenue:</span><strong id="meta-revenue" style="color:#4ade80">₹5.98 Crore</strong></div>
          <button class="btn-action" style="margin-top:8px;" onclick="zoomToBlockFromCatalog()">
            📍 Locate on Interactive Leaflet Map
          </button>
        </div>
      </div>

      <div id="block-viewer-main">
        <div class="block-viewer-header">
          <div>
            <h3 id="block-viewer-title">Patan Block (LGD: 3639) - 10m Agricultural Field Footprint Map</h3>
            <p style="font-size:11.5px; color:#94a3b8;">High-Resolution Cadastral Parcel Map Grounded Directly in agriculture_field.shp</p>
          </div>
          <button class="nav-tab-btn" onclick="openCurrentBlockLightbox()">🔍 View Full Resolution</button>
        </div>
        <div class="block-img-container" onclick="openCurrentBlockLightbox()">
          <img id="block-main-img" src="outputs/figures/block_field_maps/block_3639_patan_field_map.png?v={timestamp_ver}" alt="Block Field Map">
        </div>
      </div>
    </div>

  </div>

  <!-- Fullscreen Lightbox Modal -->
  <div id="lightbox" onclick="closeLightbox(event)">
    <span id="lightbox-close" onclick="closeLightbox(event)">&times;</span>
    <img id="lightbox-img" src="" alt="Fullscreen Map View">
  </div>

  <script>
    const blocksData = {blocks_geojson_str};
    const parcelsData = {parcels_geojson_str};
    const blockCatalog = {catalog_json_str};
    let currentPlateIdx = 1;
    let currentSelectedLgd = 3639;
    let blocksLayer, parcelsLayer;
    let currentMetric = 'ftw_ch4_t';

    const platesCatalog = {{
      1: {{
        title: "Plate 01: Statewide 146-Block Agricultural Methane Atlas",
        desc: "Statewide Choropleth with 3 Cadastral Insets (Patan, Jagdalpur, Masturi) @ 300 DPI | EPSG:32644",
        dpi: "300 DPI",
        file: "outputs/figures/01_agriculture_field_statewide_methane_analysis_300dpi.png"
      }},
      2: {{
        title: "Plate 02: Dual-Season Kharif Monsoon vs. Rabi Summer Methane Emissions",
        desc: "Panel A (Rainfed Floodwater Kharif: 72.8%) vs. Panel B (Groundwater Garma Dhan Rabi: 27.2%) | 250 DPI",
        dpi: "250 DPI",
        file: "outputs/figures/02_statewide_kharif_vs_rabi_methane_blockwise.png"
      }},
      3: {{
        title: "Plate 03: Verra VM0042 AWD Mitigation & Carbon Offset Credit Atlas",
        desc: "Annual CH4 Abatement Hotspots (Panel A) and Carbon Revenue Potential (Panel B: ₹398 Crore/yr) | 250 DPI",
        dpi: "250 DPI",
        file: "outputs/figures/03_statewide_awd_methane_mitigation_and_carbon_credits.png"
      }},
      4: {{
        title: "Plate 04: Landholding Size Distribution of Emissions & Smallholder Co-Benefits",
        desc: "Statistical 4-Panel Diagnostics Proving 94.2% of Emissions Originate from Marginal Farmers (<1 ha) | 300 DPI",
        dpi: "300 DPI",
        file: "outputs/figures/04_methane_emissions_by_field_size_distribution.png"
      }},
      5: {{
        title: "Plate 05: Statewide 146-Block Agriculture Field Footprint Atlas",
        desc: "Derived Directly from 14,405,396 Physical Parcels (1.45M ha) with 3 Cadastral Insets | 300 DPI",
        dpi: "300 DPI",
        file: "outputs/figures/01_agriculture_field_statewide_cartography_300dpi.png"
      }},
      10: {{
        title: "Plate 10: Statewide Block Field Density & Cadastral Parcel Insets",
        desc: "Statewide 146-Block Density Atlas with Patan & Jagdalpur 10m Medh Boundaries | 300 DPI",
        dpi: "300 DPI",
        file: "outputs/figures/10_statewide_block_field_density_and_parcels.png"
      }},
      11: {{
        title: "Plate 11: 4-Panel Key Blocks Cadastral Plate (10m Parcels)",
        desc: "Patan (Durg), Abhanpur (Raipur), Masturi (Bilaspur), and Ambikapur (Surguja) 10m Medh Boundaries | 250 DPI",
        dpi: "250 DPI",
        file: "outputs/figures/11_key_blocks_cadastral_10m_field_maps.png"
      }}
    }};

    // Switch between the 3 main application panels
    function switchTab(tabKey) {{
      document.querySelectorAll('.nav-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.app-panel').forEach(p => p.classList.remove('active'));

      if (tabKey === 'map') {{
        event.currentTarget.classList.add('active');
        document.getElementById('panel-map').classList.add('active');
        if (map) setTimeout(() => map.invalidateSize(), 150);
      }} else if (tabKey === 'atlas') {{
        event.currentTarget.classList.add('active');
        document.getElementById('panel-atlas').classList.add('active');
      }} else if (tabKey === 'blocks') {{
        event.currentTarget.classList.add('active');
        document.getElementById('panel-blocks').classList.add('active');
      }}
    }}

    // Plate loading
    function loadPlate(idx) {{
      currentPlateIdx = idx;
      const info = platesCatalog[idx];
      if (!info) return;

      document.querySelectorAll('.plate-btn').forEach((b, i) => {{
        b.classList.toggle('active', [1, 2, 3, 4, 5, 10, 11][i] === idx);
      }});

      document.getElementById('plate-title').innerText = info.title;
      document.getElementById('plate-desc').innerText = info.desc;
      document.getElementById('plate-badge-dpi').innerText = info.dpi;
      const ts = new Date().getTime();
      document.getElementById('plate-img').src = info.file + '?v=' + ts;
    }}

    function refreshAtlasImages() {{
      const ts = new Date().getTime();
      const info = platesCatalog[currentPlateIdx];
      document.getElementById('plate-img').src = info.file + '?v=' + ts;
      alert('Map plates refreshed successfully!');
    }}

    function openCurrentPlateLightbox() {{
      const img = document.getElementById('plate-img');
      document.getElementById('lightbox-img').src = img.src;
      document.getElementById('lightbox').classList.add('active');
    }}

    function openCurrentBlockLightbox() {{
      const img = document.getElementById('block-main-img');
      document.getElementById('lightbox-img').src = img.src;
      document.getElementById('lightbox').classList.add('active');
    }}

    function closeLightbox(e) {{
      if (e.target.id === 'lightbox' || e.target.id === 'lightbox-close') {{
        document.getElementById('lightbox').classList.remove('active');
      }}
    }}

    // Metric color gradients and scale config
    const metricConfigs = {{
      'ftw_ch4_t': {{ label: 'FTW Annual Methane (Tonnes CH4)', min: 0, max: 8000, grad: ['#ffffcc', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'] }},
      'ftw_k_ch4_t': {{ label: 'FTW Kharif Methane (Tonnes CH4)', min: 0, max: 6000, grad: ['#ffffd4', '#fed98e', '#fe9929', '#d95f0e', '#993404'] }},
      'ftw_r_ch4_t': {{ label: 'FTW Rabi Summer Methane (Tonnes CH4)', min: 0, max: 2500, grad: ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'] }},
      'ftw_awd_ch4': {{ label: 'FTW AWD Abatement (Tonnes CH4)', min: 0, max: 3500, grad: ['#f7fcf5', '#c7e9c0', '#74c476', '#238b45', '#00441b'] }},
      'ftw_carbon_offsets': {{ label: 'FTW Annual Carbon Credits (tCO2e)', min: 0, max: 95000, grad: ['#e5f5e0', '#a1d99b', '#41ab5d', '#238b45', '#00441b'] }},
      'ftw_revenue_cr': {{ label: 'FTW Carbon Revenue (₹ Crore/year)', min: 0, max: 12.0, grad: ['#e5f5f9', '#99d8c9', '#41ae76', '#238b45', '#005824'] }},
      'ftw_rice_ha': {{ label: 'FTW Measured Rice Extent (ha)', min: 0, max: 30000, grad: ['#ffffe5', '#d9f0a3', '#78c679', '#238443', '#004529'] }},
      'ftw_parcels': {{ label: 'FTW Physical Parcels Count', min: 0, max: 220000, grad: ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c'] }},
      'ftw_intensity': {{ label: 'FTW Emission Intensity (kg CH4/ha)', min: 100, max: 260, grad: ['#fff5eb', '#fdd0a2', '#fdae6b', '#fd8d3c', '#d94701'] }},
      'gw_stage': {{ label: 'GW Extraction Stage (%)', min: 20, max: 105, grad: ['#e0f2fe', '#38bdf8', '#fbbf24', '#f87171', '#b91c1c'] }},
      'cobenefit_idx': {{ label: 'Methane–GW Co-Benefit Index', min: 10, max: 95, grad: ['#f3e8ff', '#c084fc', '#9333ea', '#6b21a8', '#3b0764'] }},
      'soc_pct': {{ label: 'Soil Organic Carbon (%)', min: 0.45, max: 1.25, grad: ['#fef08a', '#84cc16', '#4d7c0f', '#14532d'] }},
      'avoided_co2e_2035': {{ label: '2035 Policy Avoided CO2e (t)', min: 0, max: 160000, grad: ['#dcfce7', '#4ade80', '#16a34a', '#14532d'] }},
      'gw_saved_mcm': {{ label: '2035 GW Saved (MCM/yr)', min: 0, max: 85, grad: ['#e0e7ff', '#818cf8', '#4338ca', '#312e81'] }}
    }};

    function getColor(val, cfg) {{
      const norm = Math.max(0, Math.min(1, (val - cfg.min) / (cfg.max - cfg.min)));
      const colors = cfg.grad;
      const idx = Math.min(colors.length - 1, Math.floor(norm * (colors.length - 1)));
      return colors[idx];
    }}

    function styleBlock(feature) {{
      const val = feature.properties[currentMetric] || 0;
      const cfg = metricConfigs[currentMetric] || metricConfigs['ftw_ch4_t'];
      return {{
        fillColor: getColor(val, cfg),
        weight: 1,
        opacity: 0.85,
        color: '#475569',
        fillOpacity: 0.85
      }};
    }}

    function onEachBlock(feature, layer) {{
      layer.on({{
        mouseover: (e) => {{
          const l = e.target;
          l.setStyle({{ weight: 2.5, color: '#38bdf8', fillOpacity: 0.95 }});
          l.bringToFront();
        }},
        mouseout: (e) => {{
          blocksLayer.resetStyle(e.target);
        }},
        click: (e) => {{
          showBlockDetails(feature.properties);
        }}
      }});
    }}

    function showBlockDetails(p) {{
      const insp = document.getElementById('inspector');
      const gwBadgeClass = p.gw_cat === 'Critical' ? 'critical' : (p.gw_cat === 'Semi-Critical' ? 'semi' : 'safe');
      currentSelectedLgd = p.lgd_code;
      insp.innerHTML = `
        <h3>
          ${{p.block_name}}
          <span class="badge ${{gwBadgeClass}}">${{p.gw_cat}}</span>
        </h3>
        <div class="sub">${{p.district_name}} District | LGD: ${{p.lgd_code}} | ${{p.aez || 'Central Plains'}}</div>
        <div class="stat-row"><span>FTW Physical Parcels:</span><strong style="color:#38bdf8">${{Number(p.ftw_parcels).toLocaleString()}}</strong></div>
        <div class="stat-row"><span>Measured Rice Area:</span><strong>${{Number(p.ftw_rice_ha).toLocaleString()}} ha</strong></div>
        <div class="stat-row"><span>Mean Parcel / Marginal %:</span><strong>${{p.ftw_mean_field}} ha (${{p.ftw_marg_pct}}%)</strong></div>
        <div class="stat-row"><span>Annual Methane Footprint:</span><strong style="color:#f43f5e">${{Number(p.ftw_ch4_t).toLocaleString()}} t CH4</strong></div>
        <div class="stat-row"><span>Kharif vs. Rabi CH4:</span><strong>${{Number(p.ftw_k_ch4_t).toLocaleString()}} / ${{Number(p.ftw_r_ch4_t).toLocaleString()}} t</strong></div>
        <div class="stat-row"><span>AWD Abatement Potential:</span><strong style="color:#4ade80">${{Number(p.ftw_awd_ch4).toLocaleString()}} t CH4</strong></div>
        <div class="stat-row"><span>Annual Carbon Offsets:</span><strong style="color:#4ade80">${{Number(p.ftw_carbon_offsets).toLocaleString()}} tCO2e</strong></div>
        <div class="stat-row"><span>Carbon Revenue Potential:</span><strong style="color:#4ade80">₹${{p.ftw_revenue_cr}} Crore / yr</strong></div>
        <div class="stat-row"><span>Emission Intensity:</span><strong>${{p.ftw_intensity}} kg CH4/ha</strong></div>
        <div class="stat-row"><span>GW Extraction Stage:</span><strong>${{p.gw_stage}}%</strong></div>
        <button class="btn-action" style="margin-top:8px; padding:6px;" onclick="viewBlockInGallery(${{p.lgd_code}})">
          🔬 Open Dedicated 10m Field Map
        </button>
      `;
    }}

    function viewBlockInGallery(lgd) {{
      const bItem = blockCatalog.find(b => b.lgd === lgd);
      if (!bItem) return;
      document.querySelectorAll('.nav-tab-btn')[2].click();
      document.getElementById('block-catalog-select').value = lgd;
      selectBlockFromCatalog();
    }}

    function updateSimulation() {{
      const yr = parseInt(document.getElementById('sim-year').value);
      const awd = parseInt(document.getElementById('sim-awd').value);
      const rain = parseInt(document.getElementById('sim-rain').value);

      document.getElementById('lbl-year').innerText = yr;
      document.getElementById('lbl-awd').innerText = awd + '%';
      document.getElementById('lbl-rain').innerText = (rain > 0 ? '+' : '') + rain + '%';

      const baseMethane = 258457.1 * (1.0 + 0.008 * (yr - 2024)) * (1.0 + 0.010 * rain / 100.0);
      const abatedMethane = baseMethane * (awd / 100.0) * 0.44;
      const netMethane = baseMethane - abatedMethane;
      const avoidedCO2e = abatedMethane * 27.9;
      const rabiMethane = 70300.3 * (1.0 - (awd / 100.0) * 0.38);
      const revenueCr = (avoidedCO2e * 15.0 * 83.5) / 1e7;

      document.getElementById('card-tot-ch4').innerText = Math.round(netMethane).toLocaleString() + ' t';
      document.getElementById('card-rabi-ch4').innerText = Math.round(rabiMethane).toLocaleString() + ' t';
      document.getElementById('card-co2e').innerText = (avoidedCO2e / 1e6).toFixed(2) + 'M t';
      document.getElementById('card-revenue').innerText = '₹' + revenueCr.toFixed(1) + ' Cr';
    }}

    function zoomToDistrict() {{
      const dist = document.getElementById('district-filter').value;
      if (dist === 'ALL') {{
        map.setView([21.2787, 81.8661], 7);
        return;
      }}
      const matched = blocksData.features.filter(f => f.properties.district_name.toLowerCase() === dist.toLowerCase());
      if (matched.length > 0) {{
        const bounds = L.geoJSON({{ type: 'FeatureCollection', features: matched }}).getBounds();
        map.fitBounds(bounds, {{ padding: [20, 20] }});
      }}
    }}

    function toggleParcelsLayer() {{
      const isChecked = document.getElementById('toggle-parcels').checked;
      if (!parcelsLayer) return;
      if (isChecked) {{
        if (!map.hasLayer(parcelsLayer)) map.addLayer(parcelsLayer);
      }} else {{
        if (map.hasLayer(parcelsLayer)) map.removeLayer(parcelsLayer);
      }}
    }}

    function updateChoropleth() {{
      currentMetric = document.getElementById('metric-select').value;
      const cfg = metricConfigs[currentMetric] || metricConfigs['ftw_ch4_t'];
      blocksLayer.setStyle(styleBlock);

      document.getElementById('legend-title').innerText = cfg.label;
      document.getElementById('legend-min').innerText = cfg.min.toLocaleString();
      document.getElementById('legend-max').innerText = cfg.max.toLocaleString();
      document.getElementById('legend-bar').style.background = `linear-gradient(to right, ${{cfg.grad.join(', ')}})`;
    }}

    // Map initialization
    const map = L.map('map', {{
      center: [21.2787, 81.8661],
      zoom: 7,
      zoomControl: true
    }});

    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; CartoDB & OpenStreetMap | Chhattisgarh MRV',
      subdomains: 'abcd',
      maxZoom: 19
    }}).addTo(map);

    blocksLayer = L.geoJSON(blocksData, {{
      style: styleBlock,
      onEachFeature: onEachBlock
    }}).addTo(map);

    if (parcelsData && parcelsData.features.length > 0) {{
      parcelsLayer = L.geoJSON(parcelsData, {{
        style: {{ color: '#22c55e', weight: 1.0, fillColor: '#4ade80', fillOpacity: 0.65 }}
      }});
      map.on('zoomend', () => {{
        const isChecked = document.getElementById('toggle-parcels').checked;
        if (map.getZoom() >= 11 && isChecked) {{
          if (!map.hasLayer(parcelsLayer)) map.addLayer(parcelsLayer);
        }} else {{
          if (map.hasLayer(parcelsLayer)) map.removeLayer(parcelsLayer);
        }}
      }});
    }}

    // Populate Block Gallery Catalog List
    function populateBlockCatalog() {{
      const sel = document.getElementById('block-catalog-select');
      sel.innerHTML = '';
      blockCatalog.forEach(b => {{
        const opt = document.createElement('option');
        opt.value = b.lgd;
        opt.innerText = `${{b.block}} (${{b.district}}) - ${{b.parcels.toLocaleString()}} parcels`;
        if (b.lgd === 3639) opt.selected = true;
        sel.appendChild(opt);
      }});
    }}

    function filterBlockList() {{
      const query = document.getElementById('block-search').value.toLowerCase();
      const sel = document.getElementById('block-catalog-select');
      sel.innerHTML = '';
      blockCatalog.filter(b => b.block.toLowerCase().includes(query) || b.district.toLowerCase().includes(query))
        .forEach(b => {{
          const opt = document.createElement('option');
          opt.value = b.lgd;
          opt.innerText = `${{b.block}} (${{b.district}}) - ${{b.parcels.toLocaleString()}} parcels`;
          sel.appendChild(opt);
        }});
      if (sel.options.length > 0) {{
        sel.selectedIndex = 0;
        selectBlockFromCatalog();
      }}
    }}

    function selectBlockFromCatalog() {{
      const sel = document.getElementById('block-catalog-select');
      const lgd = parseInt(sel.value);
      const bItem = blockCatalog.find(b => b.lgd === lgd);
      if (!bItem) return;

      currentSelectedLgd = lgd;
      document.getElementById('meta-block-name').innerText = bItem.block + ' BLOCK';
      document.getElementById('meta-district').innerText = bItem.district;
      document.getElementById('meta-lgd').innerText = bItem.lgd;
      document.getElementById('meta-parcels').innerText = bItem.parcels.toLocaleString();
      document.getElementById('meta-rice').innerText = bItem.rice_ha.toLocaleString() + ' ha';
      document.getElementById('meta-ch4').innerText = Math.round(bItem.ch4_t).toLocaleString() + ' t CH4';
      document.getElementById('meta-awd').innerText = Math.round(bItem.awd_ch4).toLocaleString() + ' t CH4';
      document.getElementById('meta-revenue').innerText = '₹' + bItem.rev_cr.toFixed(2) + ' Crore';

      document.getElementById('block-viewer-title').innerText = `${{bItem.block}} Block (${{bItem.district}} District) - 10m Agricultural Field Footprint`;
      const ts = new Date().getTime();
      document.getElementById('block-main-img').src = bItem.file + '?v=' + ts;
    }}

    function zoomToBlockFromCatalog() {{
      document.querySelectorAll('.nav-tab-btn')[0].click();
      const matched = blocksData.features.find(f => f.properties.lgd_code === currentSelectedLgd);
      if (matched) {{
        const bLayer = L.geoJSON(matched);
        map.fitBounds(bLayer.getBounds(), {{ padding: [40, 40] }});
        showBlockDetails(matched.properties);
      }}
    }}

    // Ranking Chart: Top 5 Emitters
    const ctx = document.getElementById('rankingChart').getContext('2d');
    const sortedHotspots = [...blocksData.features].sort((a,b) => b.properties.ftw_ch4_t - a.properties.ftw_ch4_t).slice(0, 5);
    new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels: sortedHotspots.map(f => f.properties.block_name),
        datasets: [{{
          label: 'Annual CH4 (t)',
          data: sortedHotspots.map(f => f.properties.ftw_ch4_t),
          backgroundColor: '#f43f5e',
          borderRadius: 4
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8', font: {{ size: 8.5 }} }}, grid: {{ display: false }} }},
          y: {{ ticks: {{ color: '#94a3b8', font: {{ size: 8.5 }} }}, grid: {{ color: '#1e293b' }} }}
        }}
      }}
    }});

    // Initialize block gallery catalog
    populateBlockCatalog();
    updateSimulation();
  </script>
</body>
</html>
"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[SUCCESS] Exported upgraded interactive dashboard to index.html ({len(html_content):,} chars)")


def main():
    print("=" * 80)
    print("🌾 CHHATTISGARH RICE MRV: WEB GIS DASHBOARD ASSET BUILDER & FTW LIVE MAPS")
    print("   Protocol: Ground-Truth 14.4M FTW Parcels, 146 Blocks, Zero QGIS")
    print("=" * 80)
    t0 = time.time()
    build_parcels_geojson()
    blocks_gdf = build_blocks_geojson()
    catalog = get_block_maps_catalog(blocks_gdf)
    generate_index_html(blocks_gdf, catalog)
    print(f"🎉 Web dashboard assets & index.html successfully generated in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
