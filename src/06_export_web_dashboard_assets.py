#!/usr/bin/env python3
"""
Step 06: Web Dashboard Asset Builder & HTML Generator for GitHub Pages
----------------------------------------------------------------------
Builds self-contained GeoJSON assets and a responsive, high-performance
interactive Web GIS Dashboard (index.html) deployable directly via GitHub Pages:
- data/processed/cg_blocks_mrv_web.geojson (628 KB, 146 blocks with all MRV attributes)
- data/processed/durg_parcels_web.geojson (~180 KB, representative 10m field parcels)
- index.html (Responsive Leaflet.js + Chart.js web dashboard with real-time inspector)
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd

def build_web_assets():
    print("[INFO] Building Web GIS Dashboard assets...")
    os.makedirs("data/processed", exist_ok=True)

    # 1. Load blocks and simplify
    blocks_path = "data/raw/india_geodata/cg_lgd_blocks.geojson"
    gdf_blocks = gpd.read_file(blocks_path)

    panel_path = "data/processed/chhattisgarh_block_tier2_methane_panel.csv"
    df_panel = pd.read_csv(panel_path)
    df_2024 = df_panel[df_panel["year"] == 2024].copy()

    # Merge by LGD_Code
    merged = gdf_blocks.merge(
        df_2024[[
            "lgd_code", "tier2_daily_flux_kg_ha_day", "tier2_ch4_flux_kg_ha",
            "rice_ch4_emissions_tonnes", "awd_mitigation_potential_co2e_tonnes",
            "seasonal_inundation_days", "rice_area_ha", "rice_fraction",
            "soil_clay_pct", "soil_organic_carbon_g_kg", "uncertainty_rel_pct"
        ]],
        left_on="LGD_Code",
        right_on="lgd_code",
        how="inner"
    )

    # Simplify geometries to ~0.002 degrees for fast web rendering
    merged_sim = merged.copy()
    merged_sim["geometry"] = merged.geometry.simplify(0.002, preserve_topology=True)

    # Keep essential attributes
    keep_cols = [
        "Block_Name", "LGD_Code", "District_Name",
        "tier2_daily_flux_kg_ha_day", "tier2_ch4_flux_kg_ha",
        "rice_ch4_emissions_tonnes", "awd_mitigation_potential_co2e_tonnes",
        "seasonal_inundation_days", "rice_area_ha", "rice_fraction",
        "soil_clay_pct", "soil_organic_carbon_g_kg", "uncertainty_rel_pct",
        "geometry"
    ]
    web_blocks = merged_sim[keep_cols].rename(columns={
        "Block_Name": "block_name",
        "LGD_Code": "lgd_code",
        "District_Name": "district_name",
        "tier2_daily_flux_kg_ha_day": "daily_flux",
        "tier2_ch4_flux_kg_ha": "season_flux",
        "rice_ch4_emissions_tonnes": "ch4_tonnes",
        "awd_mitigation_potential_co2e_tonnes": "awd_tco2e",
        "seasonal_inundation_days": "inund_days",
        "rice_area_ha": "rice_ha",
        "soil_clay_pct": "clay_pct",
        "uncertainty_rel_pct": "unc_pct"
    })

    # Round numeric values for compact payload
    for col in ["daily_flux", "season_flux", "ch4_tonnes", "awd_tco2e", "inund_days", "rice_ha", "clay_pct", "unc_pct"]:
        web_blocks[col] = web_blocks[col].round(2)

    blocks_web_path = "data/processed/cg_blocks_mrv_web.geojson"
    web_blocks.to_file(blocks_web_path, driver="GeoJSON")
    print(f"[SUCCESS] Exported web blocks GeoJSON: {blocks_web_path} ({os.path.getsize(blocks_web_path):,} bytes)")

    # 2. Sample Durg 10m Field Parcels
    parcels_path = "outputs/durg_10m_field_parcels.parquet"
    if not os.path.exists(parcels_path):
        parcels_path = "outputs/durg_10m_field_parcels.geojson"
    
    if os.path.exists(parcels_path):
        if parcels_path.endswith(".parquet"):
            gdf_parcels = gpd.read_parquet(parcels_path)
        else:
            gdf_parcels = gpd.read_file(parcels_path)
        
        # Take representative 400 parcels across Patan, Dhamdha, Durg
        sample_parcels = gdf_parcels.sample(n=min(400, len(gdf_parcels)), random_state=42).copy()
        sample_parcels["geometry"] = sample_parcels.geometry.simplify(0.00005, preserve_topology=True)
        parcels_web_path = "data/processed/durg_parcels_web.geojson"
        sample_parcels.to_file(parcels_web_path, driver="GeoJSON")
        print(f"[SUCCESS] Exported Durg sample parcels GeoJSON: {parcels_web_path} ({os.path.getsize(parcels_web_path):,} bytes)")

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
  <title>Chhattisgarh Rice Methane MRV & Carbon Dashboard</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }}
    body {{ display: flex; height: 100vh; background: #0f172a; color: #f8fafc; overflow: hidden; }}
    #sidebar {{ width: 420px; background: #1e293b; border-right: 1px solid #334155; display: flex; flex-direction: column; z-index: 1000; box-shadow: 4px 0 15px rgba(0,0,0,0.3); }}
    #map-container {{ flex: 1; position: relative; height: 100%; }}
    #map {{ width: 100%; height: 100%; }}
    .header {{ padding: 16px 20px; background: #0f172a; border-bottom: 1px solid #334155; }}
    .header h1 {{ font-size: 17px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
    .header p {{ font-size: 11.5px; color: #94a3b8; margin-top: 4px; line-height: 1.4; }}
    .scrollable {{ flex: 1; overflow-y: auto; padding: 16px 20px; }}
    .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px; }}
    .metric-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px; }}
    .metric-card .title {{ font-size: 10.5px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-card .value {{ font-size: 18px; font-weight: 700; color: #f8fafc; margin-top: 4px; }}
    .metric-card.accent .value {{ color: #4ade80; }}
    .metric-card.warning .value {{ color: #fbbf24; }}
    .section-title {{ font-size: 12px; font-weight: 700; text-transform: uppercase; color: #cbd5e1; margin: 16px 0 8px 0; letter-spacing: 0.5px; }}
    .control-group {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-bottom: 16px; }}
    .control-item {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; font-size: 12.5px; }}
    .control-item:last-child {{ margin-bottom: 0; }}
    select, button {{ background: #1e293b; color: #f8fafc; border: 1px solid #475569; padding: 6px 10px; border-radius: 6px; font-size: 12px; outline: none; cursor: pointer; }}
    select:focus {{ border-color: #38bdf8; }}
    .btn-gee {{ width: 100%; background: #0284c7; color: white; border: none; font-weight: 600; padding: 10px; border-radius: 6px; margin-top: 8px; display: flex; align-items: center; justify-content: center; gap: 6px; text-decoration: none; font-size: 12.5px; transition: background 0.2s; }}
    .btn-gee:hover {{ background: #0369a1; }}
    #inspector {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-top: 10px; min-height: 140px; }}
    #inspector h3 {{ font-size: 14px; color: #38bdf8; font-weight: 700; }}
    #inspector .sub {{ font-size: 11px; color: #94a3b8; margin-bottom: 8px; }}
    #inspector .stat-row {{ display: flex; justify-content: space-between; font-size: 12px; padding: 4px 0; border-bottom: 1px solid #1e293b; }}
    #inspector .stat-row:last-child {{ border-bottom: none; }}
    .stat-row span:first-child {{ color: #94a3b8; }}
    .stat-row span:last-child {{ font-weight: 600; }}
    .chart-box {{ margin-top: 12px; background: #0f172a; border-radius: 8px; padding: 10px; border: 1px solid #334155; height: 180px; }}
    .legend {{ position: absolute; bottom: 24px; right: 24px; background: rgba(15, 23, 42, 0.92); border: 1px solid #334155; padding: 12px 16px; border-radius: 8px; z-index: 1000; font-size: 11.5px; color: #cbd5e1; box-shadow: 0 4px 15px rgba(0,0,0,0.4); }}
    .legend-title {{ font-weight: 700; margin-bottom: 8px; color: #f8fafc; }}
    .legend-scale {{ display: flex; align-items: center; gap: 4px; }}
    .legend-bar {{ width: 140px; height: 12px; border-radius: 3px; }}
    .badge {{ display: inline-block; padding: 2px 6px; font-size: 9.5px; font-weight: 700; border-radius: 4px; text-transform: uppercase; }}
    .badge-canal {{ background: #ef4444; color: white; }}
    .badge-tube {{ background: #f59e0b; color: white; }}
    .badge-rain {{ background: #10b981; color: white; }}
  </style>
</head>
<body>

  <div id="sidebar">
    <div class="header">
      <h1>🌾 Chhattisgarh Rice Methane MRV</h1>
      <p>146 LGD Blocks &bull; 10m Field Delineation &bull; IPCC Tier-2 &bull; Verra VM0051 AWD</p>
    </div>

    <div class="scrollable">
      <div class="metric-grid">
        <div class="metric-card">
          <div class="title">Net Rice Area</div>
          <div class="value">3.74 M ha</div>
        </div>
        <div class="metric-card warning">
          <div class="title">Annual Baseline CH₄</div>
          <div class="value">1.09 Mt</div>
        </div>
        <div class="metric-card accent">
          <div class="title">Net AWD Abatement</div>
          <div class="value">12.2 Mt CO₂e</div>
        </div>
        <div class="metric-card">
          <div class="title">Spatial ML R²</div>
          <div class="value">0.9698</div>
        </div>
      </div>

      <div class="section-title">Thematic Layer Display</div>
      <div class="control-group">
        <div class="control-item">
          <label for="metric-select">Choropleth Variable:</label>
          <select id="metric-select" onchange="updateChoropleth()">
            <option value="season_flux">Seasonal Flux (kg CH₄/ha)</option>
            <option value="daily_flux">Daily Flux (kg CH₄/ha/day)</option>
            <option value="ch4_tonnes">Total Emissions (tonnes CH₄)</option>
            <option value="awd_tco2e">AWD Carbon Offsets (t CO₂e)</option>
            <option value="inund_days">SAR Inundation Duration (days)</option>
          </select>
        </div>
        <div class="control-item">
          <label>Show Durg 10m Parcels:</label>
          <input type="checkbox" id="parcel-toggle" onchange="toggleParcels(this.checked)" checked />
        </div>
      </div>

      <div class="section-title">Block & Field Inspector</div>
      <div id="inspector">
        <div style="color: #64748b; font-size: 12px; text-align: center; padding: 30px 0;">
          👈 Click on any Chhattisgarh block or Durg 10m parcel to inspect live MRV metrics.
        </div>
      </div>

      <div class="chart-box">
        <canvas id="statChart"></canvas>
      </div>

      <div class="section-title">Earth Engine Code Editor Connection</div>
      <div class="control-group" style="font-size: 11.5px; color: #94a3b8; line-height: 1.5;">
        <div>To execute server-side <strong>smileRandomForest</strong> machine learning:</div>
        <a class="btn-gee" href="https://code.earthengine.google.com/" target="_blank">
          🚀 Open in Google Earth Engine Code Editor
        </a>
        <div style="margin-top: 8px; font-size: 10.5px; color: #64748b;">
          Project: <code style="color: #38bdf8;">ee-akshitvaishnavonline</code> &bull; Script: <code>gee_app/chhattisgarh_ml_field_dashboard.js</code>
        </div>
      </div>
    </div>
  </div>

  <div id="map-container">
    <div id="map"></div>
    <div class="legend" id="legend">
      <div class="legend-title" id="legend-title">Seasonal Flux (kg CH₄/ha)</div>
      <div class="legend-scale">
        <span id="leg-min">60</span>
        <div class="legend-bar" id="legend-bar" style="background: linear-gradient(to right, #ffffcc, #feb24c, #e31a1c, #800026);"></div>
        <span id="leg-max">220</span>
      </div>
    </div>
  </div>

  <script>
    const blocksGeoJSON = {blocks_geojson_str};
    const parcelsGeoJSON = {parcels_geojson_str};

    // Initialize Leaflet Map
    const map = L.map('map', {{
      center: [21.2787, 81.8661],
      zoom: 7.5,
      zoomControl: true
    }});

    // Basemaps
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; CartoDB &copy; OpenStreetMap',
      maxZoom: 18
    }}).addTo(map);

    let blocksLayer, parcelsLayer;
    let currentMetric = 'season_flux';
    let chartInstance = null;

    // Palette Generator
    function getColor(val, min, max) {{
      const ratio = Math.max(0, Math.min(1, (val - min) / (max - min)));
      // Yellow to Orange to Red to Dark Red
      const r = Math.round(255);
      const g = Math.round(255 * (1 - ratio * 0.85));
      const b = Math.round(180 * (1 - ratio));
      return `rgb(${{r}}, ${{g}}, ${{b}})`;
    }}

    const metricConfigs = {{
      season_flux: {{ label: 'Seasonal Flux (kg CH₄/ha)', min: 65, max: 210, bar: 'linear-gradient(to right, #fff7bc, #fec44f, #d95f0e, #990000)' }},
      daily_flux: {{ label: 'Daily Flux (kg CH₄/ha/day)', min: 0.6, max: 2.1, bar: 'linear-gradient(to right, #fff7bc, #fec44f, #d95f0e, #990000)' }},
      ch4_tonnes: {{ label: 'Total Emissions (tonnes CH₄)', min: 1000, max: 18000, bar: 'linear-gradient(to right, #e0f3db, #a8ddb5, #43a2ca, #0868ac)' }},
      awd_tco2e: {{ label: 'AWD Mitigation (t CO₂e)', min: 8000, max: 160000, bar: 'linear-gradient(to right, #edf8e9, #bae4b3, #74c476, #006d2c)' }},
      inund_days: {{ label: 'Inundation Duration (days)', min: 35, max: 105, bar: 'linear-gradient(to right, #e0f3f8, #67a9cf, #02818a, #014636)' }}
    }};

    function styleBlock(feature) {{
      const val = feature.properties[currentMetric] || 0;
      const cfg = metricConfigs[currentMetric];
      return {{
        fillColor: getColor(val, cfg.min, cfg.max),
        weight: 1,
        opacity: 0.8,
        color: '#334155',
        fillOpacity: 0.75
      }};
    }}

    function renderBlocks() {{
      if (blocksLayer) map.removeLayer(blocksLayer);
      blocksLayer = L.geoJSON(blocksGeoJSON, {{
        style: styleBlock,
        onEachFeature: function(feature, layer) {{
          layer.on({{
            mouseover: function(e) {{
              const l = e.target;
              l.setStyle({{ weight: 2.5, color: '#38bdf8', fillOpacity: 0.9 }});
              l.bringToFront();
            }},
            mouseout: function(e) {{
              blocksLayer.resetStyle(e.target);
            }},
            click: function(e) {{
              inspectBlock(feature.properties);
            }}
          }});
        }}
      }}).addTo(map);
    }}

    function renderParcels() {{
      if (parcelsLayer) map.removeLayer(parcelsLayer);
      parcelsLayer = L.geoJSON(parcelsGeoJSON, {{
        style: function(f) {{
          const irr = f.properties.irrigation_type;
          const col = irr === 'Canal_Irrigated_Paddy' ? '#ef4444' : (irr === 'Tubewell_Intermittent_Paddy' ? '#f59e0b' : '#10b981');
          return {{
            fillColor: col,
            weight: 0.8,
            color: '#0f172a',
            fillOpacity: 0.85
          }};
        }},
        onEachFeature: function(f, layer) {{
          layer.on('click', function(e) {{
            L.DomEvent.stopPropagation(e);
            inspectParcel(f.properties);
          }});
        }}
      }}).addTo(map);
    }}

    function toggleParcels(show) {{
      if (show) {{
        map.addLayer(parcelsLayer);
        map.flyTo([21.15, 81.38], 11);
      }} else {{
        map.removeLayer(parcelsLayer);
      }}
    }}

    function updateChoropleth() {{
      currentMetric = document.getElementById('metric-select').value;
      const cfg = metricConfigs[currentMetric];
      document.getElementById('legend-title').innerText = cfg.label;
      document.getElementById('legend-bar').style.background = cfg.bar;
      document.getElementById('leg-min').innerText = cfg.min;
      document.getElementById('leg-max').innerText = cfg.max;
      blocksLayer.setStyle(styleBlock);
    }}

    function inspectBlock(p) {{
      const html = `
        <h3>${{p.block_name}} Block</h3>
        <div class="sub">District: ${{p.district_name}} &bull; LGD: ${{p.lgd_code}}</div>
        <div class="stat-row"><span>Seasonal Flux:</span><span style="color:#f87171">${{p.season_flux}} kg CH₄/ha</span></div>
        <div class="stat-row"><span>Daily Flux:</span><span>${{p.daily_flux}} kg/ha/day</span></div>
        <div class="stat-row"><span>Total Emissions:</span><span>${{Number(p.ch4_tonnes).toLocaleString()}} t CH₄</span></div>
        <div class="stat-row"><span>AWD Abatement:</span><span style="color:#4ade80">${{Number(p.awd_tco2e).toLocaleString()}} t CO₂e</span></div>
        <div class="stat-row"><span>Inundation Days:</span><span>${{p.inund_days}} days</span></div>
        <div class="stat-row"><span>Rice Cultivation Area:</span><span>${{Number(p.rice_ha).toLocaleString()}} ha</span></div>
        <div class="stat-row"><span>Soil Clay:</span><span>${{p.clay_pct}}%</span></div>
        <div class="stat-row"><span>95% Uncertainty:</span><span>&plusmn;${{p.unc_pct}}%</span></div>
      `;
      document.getElementById('inspector').innerHTML = html;
      updateChart(p.block_name, p.ch4_tonnes, p.awd_tco2e);
    }}

    function inspectParcel(p) {{
      const badgeClass = p.irrigation_type === 'Canal_Irrigated_Paddy' ? 'badge-canal' : (p.irrigation_type === 'Tubewell_Intermittent_Paddy' ? 'badge-tube' : 'badge-rain');
      const html = `
        <h3>Field Parcel: ${{p.parcel_id}}</h3>
        <div class="sub">Block: ${{p.block_name}} &bull; <span class="badge ${{badgeClass}}">${{p.irrigation_type.replace(/_/g, ' ')}}</span></div>
        <div class="stat-row"><span>Parcel Area:</span><span>${{p.area_ha}} ha</span></div>
        <div class="stat-row"><span>Daily Flux:</span><span style="color:#f87171">${{p.ch4_kg_ha_day}} kg/ha/day</span></div>
        <div class="stat-row"><span>Season Emissions:</span><span>${{p.parcel_total_ch4_kg}} kg CH₄</span></div>
        <div class="stat-row"><span>AWD Credit Potential:</span><span style="color:#4ade80">${{p.awd_mitigation_tco2e}} t CO₂e</span></div>
        <div class="stat-row"><span>Soil Order:</span><span>${{p.soil_taxonomy}} (Clay ${{p.clay_pct}}%)</span></div>
        <div class="stat-row"><span>Inundation Freq:</span><span>${{(p.inundation_frequency * 100).toFixed(0)}}%</span></div>
      `;
      document.getElementById('inspector').innerHTML = html;
    }}

    function updateChart(name, ch4, awd) {{
      const ctx = document.getElementById('statChart').getContext('2d');
      if (chartInstance) chartInstance.destroy();

      chartInstance = new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: ['Baseline (t CH4)', 'AWD Offsets (t CO2e)'],
          datasets: [{{
            label: name,
            data: [ch4, awd],
            backgroundColor: ['#f87171', '#4ade80'],
            borderRadius: 6
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            y: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }},
            x: {{ ticks: {{ color: '#cbd5e1' }} }}
          }}
        }}
      }});
    }}

    renderBlocks();
    renderParcels();
  </script>
</body>
</html>
"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[SUCCESS] Wrote responsive Web Dashboard to index.html ({os.path.getsize('index.html'):,} bytes)")

if __name__ == "__main__":
    build_web_assets()
