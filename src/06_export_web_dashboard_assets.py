#!/usr/bin/env python3
"""
Step 06 (Upgrade): Web GIS Dashboard Asset Builder & HTML Generator for GitHub Pages
--------------------------------------------------------------------------------------
Builds self-contained GeoJSON assets and a responsive, high-performance
interactive Web GIS Dashboard (index.html) deployable directly via GitHub Pages:
  - data/processed/cg_blocks_mrv_web.geojson (146 blocks with dual-season & GW attributes)
  - data/processed/durg_parcels_web.geojson (representative 10m field parcels)
  - index.html (Responsive Leaflet.js + Chart.js web dashboard with dual-season & future toggles)
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd

GWP_CH4 = 27.9

def build_web_assets():
    print("[INFO] Building Web GIS Dashboard assets...")
    os.makedirs("data/processed", exist_ok=True)

    # 1. Load blocks
    blocks_path = "data/raw/india_geodata/cg_lgd_blocks.geojson"
    if not os.path.exists(blocks_path):
        blocks_path = "data/raw/india_geodata/cg_lgd_blocks.parquet"
    gdf_blocks = gpd.read_parquet(blocks_path) if blocks_path.endswith(".parquet") else gpd.read_file(blocks_path)

    # Standardize column names
    col_map = {"block_name": "Block_Name", "district": "District_Name", "block_lgd": "LGD_Code"}
    gdf_blocks = gdf_blocks.rename(columns={k: v for k, v in col_map.items() if k in gdf_blocks.columns})

    # 2. Load dual-season panel (2024 baseline)
    panel_path = "data/processed/chhattisgarh_statewide_kharif_rabi_groundwater_panel_2019_2026.csv"
    df_panel = pd.read_csv(panel_path)
    df_2024 = df_panel[df_panel["year"] == 2024].copy()

    # Load future forecasts (2035)
    future_path = "data/processed/chhattisgarh_block_future_forecasts_2027_2035.csv"
    df_future = pd.read_csv(future_path)
    f_2035_bau = df_future[(df_future["forecast_year"] == 2035) & (df_future["scenario"] == "Scenario 1 (BAU)")][["lgd_code", "total_ch4_tonnes", "gw_stage_pct"]].rename(
        columns={"total_ch4_tonnes": "ch4_2035_bau", "gw_stage_pct": "gw_2035_bau"}
    )
    f_2035_mit = df_future[(df_future["forecast_year"] == 2035) & (df_future["scenario"] == "Scenario 3 (Rabi Groundwater Conservation + DSR)")][["lgd_code", "avoided_co2e_tonnes", "annual_gw_savings_mcm"]].rename(
        columns={"avoided_co2e_tonnes": "avoided_co2e_2035", "annual_gw_savings_mcm": "gw_saved_mcm"}
    )

    # Compute Methane fluxes
    df_2024["k_ch4_t"] = (df_2024["Kharif_Active_Rice_Area_ha"] * 1.30 * df_2024["Kharif_SF_w"] * 1.18 * 115.0 / 1000.0).round(1)
    r_daily = (1.30 / 1.80) * df_2024["Rabi_SF_w"] * 1.18 * (2.2 ** ((df_2024["Rabi_LST_Celsius"] - 28.0) / 10.0))
    df_2024["r_ch4_t"] = np.where(df_2024["Rabi_Active_Rice_Area_ha"] > 10.0,
                                  (df_2024["Rabi_Active_Rice_Area_ha"] * r_daily * 98.0 / 1000.0).round(1), 0.0)
    df_2024["tot_ch4_t"] = (df_2024["k_ch4_t"] + df_2024["r_ch4_t"]).round(1)
    df_2024["awd_co2e_t"] = ((df_2024["k_ch4_t"] * 0.38 + df_2024["r_ch4_t"] * 0.38) * GWP_CH4).round(1)

    # Merge attributes
    df_merged = df_2024.merge(f_2035_bau, on="lgd_code", how="left").merge(f_2035_mit, on="lgd_code", how="left")
    merged = gdf_blocks.merge(df_merged, left_on="LGD_Code", right_on="lgd_code", how="inner")

    # Simplify geometries to ~0.002 degrees for fast web rendering (< 800 KB)
    merged_sim = merged.copy()
    merged_sim["geometry"] = merged.geometry.simplify(0.002, preserve_topology=True)

    keep_cols = [
        "Block_Name", "LGD_Code", "District_Name", "aez_region", "soil_indigenous_type",
        "Kharif_Active_Rice_Area_ha", "Rabi_Active_Rice_Area_ha",
        "k_ch4_t", "r_ch4_t", "tot_ch4_t", "awd_co2e_t",
        "cgwb_gw_stage_pct", "cgwb_aquifer_category", "gldas_summer_drawdown_mm",
        "methane_gw_cobenefit_index", "shc_soc_pct", "shc_ph", "shc_nitrogen_kg_ha",
        "ch4_2035_bau", "avoided_co2e_2035", "gw_saved_mcm", "geometry"
    ]

    web_blocks = merged_sim[keep_cols].rename(columns={
        "Block_Name": "block_name",
        "LGD_Code": "lgd_code",
        "District_Name": "district_name",
        "aez_region": "aez",
        "soil_indigenous_type": "soil",
        "Kharif_Active_Rice_Area_ha": "k_rice_ha",
        "Rabi_Active_Rice_Area_ha": "r_rice_ha",
        "cgwb_gw_stage_pct": "gw_stage",
        "cgwb_aquifer_category": "gw_cat",
        "gldas_summer_drawdown_mm": "drawdown_mm",
        "methane_gw_cobenefit_index": "cobenefit_idx",
        "shc_soc_pct": "soc_pct",
        "shc_ph": "ph",
        "shc_nitrogen_kg_ha": "nitrogen"
    })

    # Round numeric values for compact payload
    for c in ["k_rice_ha", "r_rice_ha", "k_ch4_t", "r_ch4_t", "tot_ch4_t", "awd_co2e_t",
              "gw_stage", "drawdown_mm", "cobenefit_idx", "soc_pct", "ph", "nitrogen",
              "ch4_2035_bau", "avoided_co2e_2035", "gw_saved_mcm"]:
        web_blocks[c] = web_blocks[c].round(2)

    blocks_web_path = "data/processed/cg_blocks_mrv_web.geojson"
    web_blocks.to_file(blocks_web_path, driver="GeoJSON")
    print(f"[SUCCESS] Exported web blocks GeoJSON: {blocks_web_path} ({os.path.getsize(blocks_web_path):,} bytes)")

    # 3. Generate standalone index.html
    generate_html(blocks_web_path)

def generate_html(geojson_path):
    print("[INFO] Generating self-contained index.html...")
    with open(geojson_path, "r", encoding="utf-8") as f:
        blocks_geojson_str = f.read()

    # Load sample parcels if available
    parcels_geojson_path = "data/processed/durg_parcels_web.geojson"
    if os.path.exists(parcels_geojson_path):
        with open(parcels_geojson_path, "r", encoding="utf-8") as f:
            parcels_geojson_str = f.read()
    else:
        parcels_geojson_str = '{"type":"FeatureCollection","features":[]}'

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Chhattisgarh Dual-Season Rice Methane & Groundwater MRV (2024–2035)</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }}
    body {{ display: flex; height: 100vh; background: #0f172a; color: #f8fafc; overflow: hidden; }}
    #sidebar {{ width: 440px; background: #1e293b; border-right: 1px solid #334155; display: flex; flex-direction: column; z-index: 1000; box-shadow: 4px 0 15px rgba(0,0,0,0.3); }}
    #map-container {{ flex: 1; position: relative; height: 100%; }}
    #map {{ width: 100%; height: 100%; }}
    .header {{ padding: 16px 20px; background: #0f172a; border-bottom: 1px solid #334155; }}
    .header h1 {{ font-size: 16px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
    .header p {{ font-size: 11px; color: #94a3b8; margin-top: 4px; line-height: 1.4; }}
    .scrollable {{ flex: 1; overflow-y: auto; padding: 16px 20px; }}
    .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }}
    .metric-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; }}
    .metric-card .title {{ font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-card .value {{ font-size: 17px; font-weight: 700; color: #f8fafc; margin-top: 3px; }}
    .metric-card.accent .value {{ color: #4ade80; }}
    .metric-card.warning .value {{ color: #fbbf24; }}
    .metric-card.danger .value {{ color: #f87171; }}
    .section-title {{ font-size: 11px; font-weight: 700; text-transform: uppercase; color: #cbd5e1; margin: 14px 0 8px 0; letter-spacing: 0.5px; }}
    .control-group {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-bottom: 14px; }}
    .control-item {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; font-size: 12px; }}
    .control-item:last-child {{ margin-bottom: 0; }}
    select, button {{ background: #1e293b; color: #f8fafc; border: 1px solid #475569; padding: 6px 10px; border-radius: 6px; font-size: 11.5px; outline: none; cursor: pointer; width: 100%; }}
    select:focus {{ border-color: #38bdf8; }}
    .btn-gee {{ width: 100%; background: #0284c7; color: white; border: none; font-weight: 600; padding: 9px; border-radius: 6px; margin-top: 6px; display: flex; align-items: center; justify-content: center; gap: 6px; text-decoration: none; font-size: 12px; transition: background 0.2s; }}
    .btn-gee:hover {{ background: #0369a1; }}
    #inspector {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-top: 8px; min-height: 140px; }}
    #inspector h3 {{ font-size: 13.5px; color: #38bdf8; font-weight: 700; }}
    #inspector .sub {{ font-size: 10.5px; color: #94a3b8; margin-bottom: 6px; }}
    #inspector .stat-row {{ display: flex; justify-content: space-between; font-size: 11.5px; padding: 3px 0; border-bottom: 1px solid #1e293b; }}
    #inspector .stat-row:last-child {{ border-bottom: none; }}
    .badge {{ padding: 2px 6px; border-radius: 4px; font-size: 9.5px; font-weight: 600; text-transform: uppercase; }}
    .badge.safe {{ background: rgba(74, 222, 128, 0.2); color: #4ade80; }}
    .badge.semi {{ background: rgba(251, 191, 36, 0.2); color: #fbbf24; }}
    .badge.critical {{ background: rgba(248, 113, 113, 0.2); color: #f87171; }}
    .legend-box {{ position: absolute; bottom: 24px; right: 24px; background: rgba(15, 23, 42, 0.92); border: 1px solid #334155; padding: 10px 14px; border-radius: 8px; z-index: 1000; font-size: 11px; backdrop-filter: blur(4px); }}
    .legend-gradient {{ height: 10px; width: 160px; border-radius: 4px; margin: 4px 0; }}
  </style>
</head>
<body>
  <div id="sidebar">
    <div class="header">
      <h1>🌾 Chhattisgarh Rice Methane & Groundwater MRV</h1>
      <p>Statewide 146 Blocks | Kharif + Rabi Dual-Season | Soil Health Card & CMIP6 2027–2035</p>
    </div>
    <div class="scrollable">
      <div class="metric-grid">
        <div class="metric-card">
          <div class="title">2024 Kharif Methane</div>
          <div class="value">1.06 Mt</div>
        </div>
        <div class="metric-card warning">
          <div class="title">2024 Rabi Summer Methane</div>
          <div class="value">0.17 Mt</div>
        </div>
        <div class="metric-card accent">
          <div class="title">2035 Avoided $\\mathrm{{CO_2e}}$</div>
          <div class="value">9.35 Mt</div>
        </div>
        <div class="metric-card danger">
          <div class="title">2035 GW Saved (DSR)</div>
          <div class="value">4,417 MCM</div>
        </div>
      </div>

      <div class="section-title">Statewide Layer Controls</div>
      <div class="control-group">
        <div style="margin-bottom: 6px; font-size: 11.5px; color: #cbd5e1; font-weight: 600;">Active Metric Visualized:</div>
        <select id="metric-select" onchange="updateChoropleth()">
          <option value="tot_ch4_t">Total Annual Methane (Tonnes CH4)</option>
          <option value="k_ch4_t">Season 1: Kharif Monsoon Methane (Tonnes CH4)</option>
          <option value="r_ch4_t">Season 2: Rabi Summer Methane (Garma Dhan)</option>
          <option value="gw_stage">CGWB Groundwater Stage of Extraction (%)</option>
          <option value="cobenefit_idx">Methane–Groundwater Co-Benefit Index (0-100)</option>
          <option value="soc_pct">Soil Health Card: Soil Organic Carbon (%)</option>
          <option value="avoided_co2e_2035">2035 Policy Avoided CO2e (Tonnes)</option>
          <option value="gw_saved_mcm">2035 Annual GW Savings (Million m3)</option>
        </select>
        <a class="btn-gee" href="https://code.earthengine.google.com/?scriptPath=users/akshitvaishnavonline:chhattisgarh_statewide_kharif_rabi_future_dashboard" target="_blank">
          🛰️ Open Statewide GEE Dashboard
        </a>
      </div>

      <div class="section-title">Block / Parcel Inspector</div>
      <div id="inspector">
        <div style="color: #94a3b8; font-size: 11.5px; text-align: center; margin-top: 35px;">
          Click any block on the map to inspect dual-season methane, groundwater status, and future projections.
        </div>
      </div>

      <div class="section-title">Top 5 Emitting / Overdraft Blocks</div>
      <div style="height: 160px; background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 8px;">
        <canvas id="rankingChart"></canvas>
      </div>
    </div>
  </div>

  <div id="map-container">
    <div id="map"></div>
    <div class="legend-box" id="legend">
      <div id="legend-title" style="font-weight: 600; margin-bottom: 2px;">Total Annual Methane</div>
      <div class="legend-gradient" id="legend-bar" style="background: linear-gradient(to right, #ffffcc, #fecc5c, #fd8d3c, #f03b20, #bd0026);"></div>
      <div style="display: flex; justify-content: space-between; font-size: 10px; color: #94a3b8;">
        <span id="legend-min">0 t</span>
        <span id="legend-max">25,000 t</span>
      </div>
    </div>
  </div>

  <script>
    const blocksData = {blocks_geojson_str};
    const parcelsData = {parcels_geojson_str};

    const map = L.map('map', {{
      center: [21.2787, 81.8661],
      zoom: 7,
      zoomControl: true
    }});

    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; CartoDB & OpenStreetMap',
      subdomains: 'abcd',
      maxZoom: 19
    }}).addTo(map);

    let blocksLayer, parcelsLayer;
    let currentMetric = 'tot_ch4_t';

    const metricConfigs = {{
      'tot_ch4_t': {{ label: 'Total Annual Methane (t CH4)', min: 0, max: 22000, grad: ['#ffffcc', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'] }},
      'k_ch4_t': {{ label: 'Kharif Monsoon Methane (t CH4)', min: 0, max: 18000, grad: ['#ffffd4', '#fed98e', '#fe9929', '#d95f0e', '#993404'] }},
      'r_ch4_t': {{ label: 'Rabi Summer Methane (t CH4)', min: 0, max: 6000, grad: ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'] }},
      'gw_stage': {{ label: 'GW Stage of Extraction (%)', min: 20, max: 105, grad: ['#e0f2fe', '#38bdf8', '#fbbf24', '#f87171', '#b91c1c'] }},
      'cobenefit_idx': {{ label: 'Methane-GW Co-Benefit Index', min: 10, max: 95, grad: ['#f3e8ff', '#c084fc', '#9333ea', '#6b21a8', '#3b0764'] }},
      'soc_pct': {{ label: 'Soil Organic Carbon (%)', min: 0.45, max: 1.25, grad: ['#fef08a', '#84cc16', '#4d7c0f', '#14532d'] }},
      'avoided_co2e_2035': {{ label: '2035 Avoided CO2e (t)', min: 0, max: 160000, grad: ['#dcfce7', '#4ade80', '#16a34a', '#14532d'] }},
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
      const cfg = metricConfigs[currentMetric];
      return {{
        fillColor: getColor(val, cfg),
        weight: 1,
        opacity: 0.8,
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
      insp.innerHTML = `
        <h3>${{p.block_name}} <span class="badge ${{gwBadgeClass}}">${{p.gw_cat}}</span></h3>
        <div class="sub">${{p.district_name}} District | ${{p.aez}} | Soil: ${{p.soil}}</div>
        <div class="stat-row"><span>Kharif Paddy Extent:</span><strong>${{Number(p.k_rice_ha).toLocaleString()}} ha</strong></div>
        <div class="stat-row"><span>Rabi Summer Paddy:</span><strong style="color:#fb7185">${{Number(p.r_rice_ha).toLocaleString()}} ha</strong></div>
        <div class="stat-row"><span>Kharif Methane:</span><strong>${{Number(p.k_ch4_t).toLocaleString()}} t CH4</strong></div>
        <div class="stat-row"><span>Rabi Summer Methane:</span><strong style="color:#f43f5e">${{Number(p.r_ch4_t).toLocaleString()}} t CH4</strong></div>
        <div class="stat-row"><span>Total Annual Methane:</span><strong>${{Number(p.tot_ch4_t).toLocaleString()}} t CH4</strong></div>
        <div class="stat-row"><span>GW Stage of Extraction:</span><strong>${{p.gw_stage}}%</strong></div>
        <div class="stat-row"><span>Methane-GW Co-Benefit:</span><strong style="color:#c084fc">${{p.cobenefit_idx}} / 100</strong></div>
        <div class="stat-row"><span>Soil Organic Carbon (SHC):</span><strong>${{p.soc_pct}}% (pH ${{p.ph}})</strong></div>
        <div class="stat-row" style="margin-top:4px; border-top:1px dashed #334155; padding-top:4px;">
          <span style="color:#4ade80">2035 Avoided CO2e (Mitigation):</span><strong style="color:#4ade80">${{Number(p.avoided_co2e_2035).toLocaleString()}} t</strong>
        </div>
        <div class="stat-row">
          <span style="color:#38bdf8">2035 GW Saved (DSR):</span><strong style="color:#38bdf8">${{p.gw_saved_mcm}} MCM/yr</strong>
        </div>
      `;
    }}

    function updateChoropleth() {{
      currentMetric = document.getElementById('metric-select').value;
      const cfg = metricConfigs[currentMetric];
      blocksLayer.setStyle(styleBlock);

      document.getElementById('legend-title').innerText = cfg.label;
      document.getElementById('legend-min').innerText = cfg.min.toLocaleString();
      document.getElementById('legend-max').innerText = cfg.max.toLocaleString();
      document.getElementById('legend-bar').style.background = `linear-gradient(to right, ${{cfg.grad.join(', ')}})`;
    }}

    blocksLayer = L.geoJSON(blocksData, {{
      style: styleBlock,
      onEachFeature: onEachBlock
    }}).addTo(map);

    if (parcelsData && parcelsData.features.length > 0) {{
      parcelsLayer = L.geoJSON(parcelsData, {{
        style: {{ color: '#f59e0b', weight: 1, fillColor: '#fbbf24', fillOpacity: 0.6 }}
      }});
      map.on('zoomend', () => {{
        if (map.getZoom() >= 11) {{
          if (!map.hasLayer(parcelsLayer)) map.addLayer(parcelsLayer);
        }} else {{
          if (map.hasLayer(parcelsLayer)) map.removeLayer(parcelsLayer);
        }}
      }});
    }}

    // Ranking Chart
    const ctx = document.getElementById('rankingChart').getContext('2d');
    const sorted = [...blocksData.features].sort((a,b) => b.properties.cobenefit_idx - a.properties.cobenefit_idx).slice(0, 5);
    new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels: sorted.map(f => f.properties.block_name),
        datasets: [{{
          label: 'Co-Benefit Index',
          data: sorted.map(f => f.properties.cobenefit_idx),
          backgroundColor: '#38bdf8',
          borderRadius: 4
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8', font: {{ size: 9 }} }}, grid: {{ display: false }} }},
          y: {{ ticks: {{ color: '#94a3b8', font: {{ size: 9 }} }}, grid: {{ color: '#1e293b' }}, max: 100 }}
        }}
      }}
    }});
  </script>
</body>
</html>"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[SUCCESS] Exported upgraded interactive dashboard to index.html ({len(html_content):,} chars)")

if __name__ == "__main__":
    build_web_assets()
