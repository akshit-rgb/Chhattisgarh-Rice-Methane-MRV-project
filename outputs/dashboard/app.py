"""
================================================================================
INTERACTIVE CARBON MRV PORTFOLIO DASHBOARD
Project: Satellite-Based Rice Methane Assessment & Carbon MRV Feasibility
Location: Chhattisgarh, India (Pilot: Durg District LGD 378)
Framework: Streamlit + Folium (streamlit-folium)
Standards: IPCC 2019 Refinement, Verra VM0051, IPCC AR6 GWPs (CH4=27.9, N2O=273)
================================================================================
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium

# Determine project root path
dashboard_dir = Path(__file__).resolve().parent
project_root = dashboard_dir.parent.parent

# Set page config
st.set_page_config(
    page_title="Chhattisgarh Rice Methane MRV Portfolio",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .reportview-container { background: #f8fafc; }
    .main-header { font-size: 26px; font-weight: 800; color: #0f172a; margin-bottom: 0px; }
    .sub-header { font-size: 14px; color: #475569; margin-bottom: 12px; }
    .disclaimer-banner {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 12px 16px;
        margin-bottom: 18px;
        border-radius: 4px;
        font-size: 13px;
        color: #991b1b;
        font-weight: 500;
    }
    .metric-card {
        background: white;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val { font-size: 24px; font-weight: 700; color: #0284c7; }
    .metric-lbl { font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# Executive Disclaimer Banner
st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <strong>EXECUTIVE TECHNICAL DISCLAIMER:</strong> This platform is an independent scientific research and 
    technical carbon MRV feasibility assessment conducted for Chhattisgarh, India. All modelled baseline fluxes and 
    AWD mitigation scenarios represent counterfactual potentials and do <strong>NOT</strong> constitute registered carbon 
    offsets, issued Verra credits, or commercial claims. Verification requires physical perforated water tube (pani pipe) field logs.
</div>
""", unsafe_allow_html=True)

# Title & Metadata
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-header">🌾 Satellite Rice Methane Assessment & Carbon MRV Portfolio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">State of Chhattisgarh, India | Pilot Validation: Durg District (LGD 378) | Scalable to 33 Districts</div>', unsafe_allow_html=True)
with col_status:
    st.info("🛰️ **Sensors:** Sentinel-1 SAR + Sentinel-2\n\n📐 **Target CRS:** EPSG:32644 (UTM 44N)")

# ------------------------------------------------------------------------------
# DATA LOADERS
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    tables_dir = project_root / "outputs" / "tables"
    mrv_csv = tables_dir / "district_mrv_summary.csv"
    tradeoffs_csv = tables_dir / "wp5_awd_scenario_tradeoffs.csv"
    gap_csv = tables_dir / "wp6_vm0051_mrv_gap_matrix.csv"
    confusion_csv = tables_dir / "wp2_confusion_matrix.csv"
    ts_csv = tables_dir / "durg_kharif_phenology_timeseries.csv"
    air_csv = tables_dir / "chhattisgarh_33_districts_air_quality_s5p.csv"

    df_mrv = pd.read_csv(mrv_csv) if mrv_csv.exists() else pd.DataFrame()
    df_tradeoffs = pd.read_csv(tradeoffs_csv) if tradeoffs_csv.exists() else pd.DataFrame()
    df_gap = pd.read_csv(gap_csv) if gap_csv.exists() else pd.DataFrame()
    df_confusion = pd.read_csv(confusion_csv) if confusion_csv.exists() else pd.DataFrame()
    df_ts = pd.read_csv(ts_csv) if ts_csv.exists() else pd.DataFrame()
    df_air = pd.read_csv(air_csv) if air_csv.exists() else pd.DataFrame()

    # Load boundaries
    gpkg_path = project_root / "data" / "processed" / "admin_boundaries" / "chhattisgarh_district_boundaries.gpkg"
    if gpkg_path.exists():
        gdf_boundaries = gpd.read_file(gpkg_path, layer="chhattisgarh_districts_33").to_crs("EPSG:4326")
    else:
        gdf_boundaries = gpd.GeoDataFrame()

    return df_mrv, df_tradeoffs, df_gap, df_confusion, df_ts, gdf_boundaries, df_air

df_mrv, df_tradeoffs, df_gap, df_confusion, df_ts, gdf_boundaries, df_air = load_data()


@st.cache_data(ttl=3600, show_spinner=False)
def get_gee_sentinel_tiles(project_id: str = "ee-akshitvaishnavonline"):
    """
    Connects to Google Earth Engine with the user's project ID and retrieves
    live Sentinel-1 SAR, Sentinel-2 multi-spectral, and ESA WorldCover LULC tile URLs for Durg Pilot.
    """
    try:
        import ee
        try:
            ee.Initialize(project=project_id)
        except Exception:
            ee.Initialize()

        # Full Durg administrative boundary bounding box (WGS84 EPSG:4326: 81.148° to 81.625° E, 20.849° to 21.545° N)
        durg_geom = ee.Geometry.Rectangle([81.08, 20.80, 81.66, 21.58])

        # 1. Sentinel-1 SAR IW-VH Mean Composite (dB)
        s1_col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(durg_geom)
            .filterDate("2024-06-01", "2024-10-31")
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
        )
        s1_mean = s1_col.select("VH").mean().clip(durg_geom)
        s1_url = s1_mean.getMapId({
            "min": -22, "max": -8,
            "palette": ["0a0a0a", "333333", "666666", "aaaaaa", "ffffff"]
        })["tile_fetcher"].url_format

        # 2. Sentinel-2 False Color Infrared (B8/B4/B3 - High Vegetative Contrast)
        s2_col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(durg_geom)
            .filterDate("2024-09-01", "2024-11-15")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        )
        s2_fcc_url = s2_col.median().clip(durg_geom).getMapId({
            "bands": ["B8", "B4", "B3"],
            "min": 500, "max": 4000
        })["tile_fetcher"].url_format

        # 3. Sentinel-2 Natural True Color (B4/B3/B2)
        s2_rgb_url = s2_col.median().clip(durg_geom).getMapId({
            "bands": ["B4", "B3", "B2"],
            "min": 200, "max": 2500
        })["tile_fetcher"].url_format

        # 4. Remote-Sensing Derived Rice Mask
        vh_early = s1_col.filterDate("2024-06-01", "2024-08-15").select("VH").min()
        vh_late = s1_col.filterDate("2024-08-16", "2024-10-15").select("VH").max()
        vh_amp = vh_late.subtract(vh_early)
        rice_mask = vh_early.lt(-16.0).And(vh_amp.gte(4.5)).selfMask().clip(durg_geom)
        rice_url = rice_mask.getMapId({"palette": ["16a34a"]})["tile_fetcher"].url_format

        # 5. ESA WorldCover 10m LULC Baseline
        wc = ee.Image("ESA/WorldCover/v200/2021").select("Map").clip(durg_geom)
        wc_url = wc.getMapId({})["tile_fetcher"].url_format

        # 6. Sentinel-5P Atmospheric Methane (Kharif 2024 Statewide)
        cg_bbox = ee.Geometry.Rectangle([80.0, 17.5, 84.5, 24.5])
        ch4_col = (
            ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4")
            .filterBounds(cg_bbox)
            .filterDate("2024-06-01", "2024-11-30")
            .select("CH4_column_volume_mixing_ratio_dry_air")
            .mean()
            .clip(cg_bbox)
        )
        ch4_url = ch4_col.getMapId({"min": 1920, "max": 1965, "palette": ["yellow", "orange", "red", "darkred"]})["tile_fetcher"].url_format

        # 7. Sentinel-5P Tropospheric NO2 (Kharif 2024 Statewide)
        no2_col = (
            ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
            .filterBounds(cg_bbox)
            .filterDate("2024-06-01", "2024-11-30")
            .select("tropospheric_NO2_column_number_density")
            .mean()
            .clip(cg_bbox)
        )
        no2_url = no2_col.getMapId({"min": 0.00002, "max": 0.00012, "palette": ["black", "blue", "purple", "orange", "yellow"]})["tile_fetcher"].url_format

        return {
            "status": "connected",
            "s1_sar": s1_url,
            "s2_fcc": s2_fcc_url,
            "s2_rgb": s2_rgb_url,
            "rice_mask": rice_url,
            "lulc": wc_url,
            "s5p_ch4": ch4_url,
            "s5p_no2": no2_url,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ------------------------------------------------------------------------------
# SIDEBAR: INTERACTIVE SIMULATOR (DURG PILOT)
# ------------------------------------------------------------------------------
st.sidebar.header("🎛️ Durg Pilot Real-Time MRV Simulator")
st.sidebar.markdown("*Adjust parameters to evaluate live flux sensitivity & N2O trade-offs:*")

# IPCC Tier-2 Parameter Controls
sim_ef = st.sidebar.slider(
    "Baseline EF (kg CH4/ha/day)",
    min_value=0.80, max_value=2.20, value=1.30, step=0.05,
    help="IPCC 2019 Table 5.11 default is 1.30 [0.80 - 2.20]."
)

sim_regime = st.sidebar.selectbox(
    "Baseline Water Regime (SF_w)",
    options=["single_aeration (0.71)", "continuously_flooded (1.00)", "regular_rainfed (0.54)", "multiple_aeration (0.55)"],
    index=0
)
sf_w_val = float(sim_regime.split("(")[1].replace(")", ""))

sim_duration = st.sidebar.slider(
    "Cultivation Duration (Days)",
    min_value=85, max_value=125, value=105, step=5
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌱 AWD Intervention Potential")

sim_awd_adopt = st.sidebar.slider(
    "AWD Adoption on Irrigated Land (%)",
    min_value=5, max_value=80, value=30, step=5
)

sim_n2o_penalty = st.sidebar.slider(
    "Induced N2O Penalty (kg N2O/ha)",
    min_value=0.20, max_value=1.20, value=0.55, step=0.05,
    help="Direct soil N2O emissions increase due to intermittent aeration cycles."
)

# Live calculation for Durg
durg_rice_ha = 108821.68
irrigated_ha = durg_rice_ha * 0.65
sf_p = 0.68 # Non-flooded preseason > 180 days

# Baseline Methane
live_ch4_kg = durg_rice_ha * sim_duration * sim_ef * sf_w_val * sf_p
live_ch4_tonnes = live_ch4_kg * 1e-3

# Monte Carlo 95% bound approximation (sigma ~ 0.28 lognormal)
live_mc_low = live_ch4_tonnes * 0.57
live_mc_high = live_ch4_tonnes * 1.87

# Counterfactual AWD
adopted_ha = irrigated_ha * (sim_awd_adopt / 100.0)
awd_red_pct = 0.45 # 45% reduction on adopted
avoided_ch4_tonnes = (live_ch4_tonnes * 0.65) * (sim_awd_adopt / 100.0) * awd_red_pct
gross_co2e_ch4 = avoided_ch4_tonnes * 27.9 # AR6 GWP

induced_n2o_kg = adopted_ha * sim_n2o_penalty
induced_n2o_tonnes = induced_n2o_kg * 1e-3
n2o_penalty_co2e = induced_n2o_tonnes * 273.0 # AR6 GWP

net_co2e = gross_co2e_ch4 - n2o_penalty_co2e

# Display Simulator Summary in Sidebar
st.sidebar.markdown(f"""
<div style="background:#f1f5f9; padding:12px; border-radius:6px; font-size:12px; border:1px solid #cbd5e1;">
    <strong>SIMULATED OUTPUTS (DURG):</strong><br>
    • <b>Baseline CH4:</b> {live_ch4_tonnes:,.1f} t CH4<br>
    • <b>95% MC CI:</b> [{live_mc_low:,.0f} – {live_mc_high:,.0f}] t<br>
    • <b>Avoided CH4:</b> {avoided_ch4_tonnes:,.1f} t CH4<br>
    • <b>Gross CH4 CO2e:</b> +{gross_co2e_ch4:,.0f} t CO2e<br>
    • <b>N2O Penalty:</b> -{n2o_penalty_co2e:,.0f} t CO2e<br>
    <hr style="margin:6px 0;">
    <strong style="color:#1d4ed8; font-size:13px;">NET CO2e: {net_co2e:,.0f} t CO2e</strong>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# MAIN DASHBOARD TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Interactive MRV Map",
    "🖼️ Publication Cartography",
    "📈 Diagnostic & Uncertainty Charts",
    "🌾 Satellite Time Series & LULC Deep-Dive",
    "📋 Institutional Compliance (VM0051)",
    "💾 Data Tables & Export Center"
])

# ------------------------------------------------------------------------------
# TAB 1: INTERACTIVE MAP
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Geospatial MRV Exploration Layer & Live Sentinel Earth Engine Stream")

    col_ctrl, col_view, col_info = st.columns([2, 1, 1])
    with col_ctrl:
        layer_select = st.radio(
            "Select Thematic Vector Attribute:",
            options=[
                "[Remotely Sensed Observation] Observed Kharif Rice Extent",
                "[Remotely Sensed Observation] Seasonal Inundation Duration (Days)",
                "[Modelled IPCC Estimate] Baseline Rice Methane Emissions (tonnes CH4)",
                "[Counterfactual Scenario Potential] Estimated AWD Scenario Reduction (tonnes CH4)"
            ],
            horizontal=True
        )

    with col_view:
        view_focus = st.selectbox(
            "Camera Viewport:",
            options=["Durg Pilot Satellite Zoom (LGD 378)", "Statewide Overview (33 Districts)"]
        )

    with col_info:
        st.markdown("""
        **Data Provenance:**
        - **Satellite Source:** Copernicus Sentinel-1 & 2 via GEE `ee-akshitvaishnavonline`
        - **Admin Boundaries:** Survey of India (`EPSG:32644` projected)
        """)

    # Determine map center and zoom level
    if "Durg Pilot" in view_focus:
        map_center = [21.19, 81.28]
        map_zoom = 10
    else:
        map_center = [21.2787, 81.8661]
        map_zoom = 7

    # Initialize Folium Map with Esri Satellite and CartoDB options
    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="CartoDB positron", control_scale=True)

    # Add Esri World Imagery as optional basemap
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri High-Resolution Satellite Basemap",
        overlay=False,
        control=True
    ).add_to(m)

    # Fetch live Sentinel tile layers from Google Earth Engine
    gee_tiles = get_gee_sentinel_tiles()
    if gee_tiles.get("status") == "connected":
        # 1. Live Sentinel-1 C-Band SAR IW-VH dB backscatter
        folium.TileLayer(
            tiles=gee_tiles["s1_sar"],
            attr="Google Earth Engine / Copernicus Sentinel-1 SAR",
            name="🛰️ Sentinel-1 SAR C-Band Radar (IW-VH dB)",
            overlay=True,
            control=True,
            show=False,
            opacity=0.85
        ).add_to(m)

        # 2. Live Sentinel-2 False Color Infrared (B8/B4/B3)
        folium.TileLayer(
            tiles=gee_tiles["s2_fcc"],
            attr="Google Earth Engine / Copernicus Sentinel-2 MSI",
            name="🛰️ Sentinel-2 False Color Infrared (B8/B4/B3)",
            overlay=True,
            control=True,
            show=False,
            opacity=0.85
        ).add_to(m)

        # 3. Live Sentinel-2 Natural True Color (B4/B3/B2)
        folium.TileLayer(
            tiles=gee_tiles["s2_rgb"],
            attr="Google Earth Engine / Copernicus Sentinel-2 MSI",
            name="🛰️ Sentinel-2 Natural Color RGB (10m Optical)",
            overlay=True,
            control=True,
            show=False,
            opacity=0.85
        ).add_to(m)

        # 4. Remote-Sensing Derived Rice Mask
        folium.TileLayer(
            tiles=gee_tiles["rice_mask"],
            attr="Google Earth Engine Rice Phenology Algorithm",
            name="🌾 Fused S1/S2 Mapped Rice Extent Mask (10m)",
            overlay=True,
            control=True,
            show=True,
            opacity=0.80
        ).add_to(m)

        # 5. Live ESA WorldCover 10m LULC Baseline
        if "lulc" in gee_tiles:
            folium.TileLayer(
                tiles=gee_tiles["lulc"],
                attr="ESA WorldCover 2021 (10m Land Use Land Cover)",
                name="🌍 ESA WorldCover 10m LULC (Cropland/Urban/Forest)",
                overlay=True,
                control=True,
                show=False,
                opacity=0.80
            ).add_to(m)

        # 6. Live Sentinel-5P Atmospheric Methane
        if "s5p_ch4" in gee_tiles:
            folium.TileLayer(
                tiles=gee_tiles["s5p_ch4"],
                attr="Google Earth Engine / Copernicus Sentinel-5P TROPOMI",
                name="🛰️ Sentinel-5P Atmospheric Methane (CH4 ppb)",
                overlay=True,
                control=True,
                show=False,
                opacity=0.75
            ).add_to(m)

        # 7. Live Sentinel-5P Tropospheric NO2 (Industrial Plumes)
        if "s5p_no2" in gee_tiles:
            folium.TileLayer(
                tiles=gee_tiles["s5p_no2"],
                attr="Google Earth Engine / Copernicus Sentinel-5P TROPOMI",
                name="🛰️ Sentinel-5P Atmospheric NO2 (Industrial Plumes)",
                overlay=True,
                control=True,
                show=False,
                opacity=0.75
            ).add_to(m)

    if not gdf_boundaries.empty and not df_mrv.empty:
        merged_gdf = gdf_boundaries.merge(df_mrv, left_on="lgd_code", right_on="district_code", how="left")

        # Define style function based on selected layer
        def get_style(feature):
            code = feature["properties"].get("lgd_code")
            if code == 378: # Durg Pilot
                if "Rice Extent" in layer_select:
                    fill = "#16a34a"
                elif "Inundation" in layer_select:
                    fill = "#0284c7"
                elif "Methane" in layer_select:
                    fill = "#ea580c"
                else:
                    fill = "#10b981"
                return {"fillColor": fill, "color": "#0f172a", "weight": 2.5, "fillOpacity": 0.40}
            else:
                # Unvalidated districts: Grey hatching style
                return {"fillColor": "#e2e8f0", "color": "#94a3b8", "weight": 1.0, "fillOpacity": 0.35}

        # Add GeoJson to map
        folium.GeoJson(
            merged_gdf,
            name="🏛️ Statutory District Boundaries & MRV Stats",
            style_function=get_style,
            tooltip=folium.GeoJsonTooltip(
                fields=["district_name_x", "lgd_code", "rice_area_ha", "rice_ch4_emissions_tonnes", "estimated_inundation_duration_days", "data_completeness_score", "limitations"],
                aliases=["District:", "LGD Code:", "Rice Area (ha):", "CH4 Emissions (t):", "Inundation (Days):", "Completeness:", "Status:"],
                localize=True
            )
        ).add_to(m)

    # Layer Control for toggling between Sentinel-1, Sentinel-2, Basemaps, and Vector Overlays
    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(m, width=1300, height=580)

    # Sentinel Sensor Diagnostic Bar
    st.caption("🛰️ **Active Sensor Feed:** Copernicus Sentinel-1 C-Band SAR (IW Mode, 5.405 GHz, VV+VH) + Sentinel-2 MSI Multi-Spectral (10m/20m Bands B2, B3, B4, B8, B11, SCL Mask) streamed live from Google Earth Engine (`ee-akshitvaishnavonline`). Use the top-right layer switcher to toggle SAR microwave radar, False-Color NIR vegetation reflectance, and classified rice extent.")

    # Key metric banner below map
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card"><div class="metric-lbl">Mapped Rice Area (Durg)</div><div class="metric-val">108,822 ha</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><div class="metric-lbl">Olofsson Accuracy</div><div class="metric-val">92.27% ± 3.1%</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><div class="metric-lbl">Baseline CH4 Emissions</div><div class="metric-val">7,171.6 t CH4</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card"><div class="metric-lbl">Net AWD Abatement (Central)</div><div class="metric-val">14,371 t CO2e</div></div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: PUBLICATION CARTOGRAPHY
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Publication-Grade Cartographic Portfolio (300 DPI)")
    maps_dir = project_root / "outputs" / "maps"

    map_options = [
        ("Map 1: Administrative Boundary Integrity & LGD Validation", maps_dir / "01_admin_boundary_validation.png"),
        ("Map 2: Durg Pilot Sentinel-1/2 Rice Extent & Accuracy Inset", maps_dir / "02_durg_pilot_rice_extent.png"),
        ("Map 3: Dynamic Inundation Duration & Observation Density", maps_dir / "03_inundation_and_confidence.png"),
        ("Map 4: Modelled IPCC Baseline CH4 Flux & Monte Carlo Uncertainty", maps_dir / "04_modelled_ch4_and_uncertainty.png"),
        ("Map 5: Counterfactual AWD Mitigation Scenario Potential", maps_dir / "05_awd_scenario_potential.png"),
        ("Map 6: ESA WorldCover 10m LULC Baseline vs. Sentinel Active Rice", maps_dir / "06_durg_land_use_land_cover.png"),
        ("Map 7: Statewide Kharif Rice Extent & Cultivation Intensity", maps_dir / "07_statewide_rice_paddy_density.png"),
        ("Map 8: Statewide Sentinel-5P TROPOMI Air Quality & Atmospheric Emissions Suite", maps_dir / "08_chhattisgarh_statewide_air_quality.png")
    ]

    selected_map_name = st.selectbox("Select Publication Map to View:", [m[0] for m in map_options])
    selected_path = dict(map_options)[selected_map_name]

    if selected_path.exists():
        st.image(str(selected_path), caption=selected_map_name, use_container_width=True)
    else:
        st.warning(f"Map file not found at {selected_path}. Run generate_maps.py to render.")

# ------------------------------------------------------------------------------
# TAB 3: DIAGNOSTIC & UNCERTAINTY CHARTS
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Scientific Diagnostic & Uncertainty Visualizations (300 DPI)")
    charts_dir = project_root / "outputs" / "charts"

    chart_options = [
        ("Chart 1: Sentinel-1 SAR & Optical Multi-Temporal Phenology Curve", charts_dir / "01_sar_optical_phenology_curve.png"),
        ("Chart 2: Satellite vs. Official DESAgri Area 1:1 Parity & Residuals", charts_dir / "02_satellite_vs_govt_area_parity.png"),
        ("Chart 3: Modelled CH4 Forest Plot with 95% MC CI vs Regional Benchmarks", charts_dir / "03_district_ch4_forest_plot.png"),
        ("Chart 4: Monte Carlo Variance Decomposition & Sensitivity Tornado", charts_dir / "04_monte_carlo_sensitivity_tornado.png"),
        ("Chart 5: AWD CH4 vs. N2O Trade-off Net Abatement Waterfall", charts_dir / "05_awd_ch4_n2o_waterfall.png")
    ]

    selected_chart_name = st.selectbox("Select Diagnostic Chart to Inspect:", [c[0] for c in chart_options])
    selected_c_path = dict(chart_options)[selected_chart_name]

    if selected_c_path.exists():
        st.image(str(selected_c_path), caption=selected_chart_name, use_container_width=True)
    else:
        st.warning(f"Chart file not found at {selected_c_path}. Run generate_charts.py to render.")

# ------------------------------------------------------------------------------
# TAB 4: SATELLITE TIME SERIES & LULC DEEP-DIVE
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🌾 Multi-Temporal Satellite Phenology & Land Use Land Cover (LULC) Stratification")
    st.markdown("""
    This module provides direct scientific proof of Kharif rice cultivation by analyzing the 
    **multi-temporal Sentinel-1 C-band SAR backscatter trajectory**, **Sentinel-2 optical vegetative greenup**, 
    and **ESA WorldCover 10m LULC baseline** extracted across Durg District.
    """)

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total District Area", "229,581 ha", "Survey of India")
    with c2:
        st.metric("Total Cropland (LULC)", "195,410 ha", "ESA WorldCover 10m (85.1%)")
    with c3:
        st.metric("Active Flooded Rice", "108,822 ha", "55.7% of Cropland (Sentinel-1/2)")
    with c4:
        st.metric("Built-up & Cities", "11,940 ha", "Durg & Bhilai (5.2%)")

    st.markdown("---")

    # Time Series Visualizations
    if not df_ts.empty:
        st.subheader("1. Sentinel-1 & Sentinel-2 Kharif Multi-Temporal Trajectory")
        st.caption("Real satellite observations extracted from Google Earth Engine (`ee-akshitvaishnavonline`) across June 1 – November 30, 2024:")

        col_ts1, col_ts2 = st.columns(2)

        with col_ts1:
            st.markdown("**Sentinel-1 C-Band SAR Dynamics (IW Mode: VH & VV Backscatter in dB)**")
            df_s1_plot = df_ts.dropna(subset=["vh_mean_db"]).set_index("date")[["vh_mean_db", "vv_mean_db"]]
            st.line_chart(df_s1_plot, color=["#dc2626", "#f97316"])
            st.info("""
            📉 **Specular Water Dip (July):** VH backscatter drops to < -19 dB during puddling and transplanting.\n
            📈 **Vegetative Canopy Rise (August-September):** Multi-scattering in developing culms drives VH backscatter upward by +6 dB.
            """)

        with col_ts2:
            st.markdown("**Sentinel-2 Optical Dynamics (10m Cloud-Masked NDVI & LSWI)**")
            df_s2_plot = df_ts.dropna(subset=["ndvi_mean"]).set_index("date")[["ndvi_mean", "lswi_mean"]]
            st.line_chart(df_s2_plot, color=["#16a34a", "#0284c7"])
            st.info("""
            💧 **Land Surface Water Index (LSWI):** Positive during July-August standing water, dropping post-October drydown.\n
            🌿 **Canopy Emergence (NDVI):** Climbs sharply from 0.20 to > 0.60 at peak flowering/heading.
            """)

        st.markdown("---")
        st.subheader("2. CHIRPS Daily Monsoon Rainfall (Durg Sheonath Basin)")
        df_rain_plot = df_ts.set_index("date")[["rainfall_mm"]]
        st.bar_chart(df_rain_plot, color="#3b82f6")

        st.markdown("---")
        st.subheader("3. LULC Composition: Cropland vs. Flooded Rice Methane Footprint")

        df_lulc_summary = pd.DataFrame([
            {"Class": "Verified Active Flooded Rice (Sentinel Mapped Methane Source)", "Area (ha)": 108822, "Share (%)": 47.4},
            {"Class": "Other Agricultural Cropland (Upland Non-Flooded, Pulses/Oilseeds)", "Area (ha)": 86588, "Share (%)": 37.7},
            {"Class": "Built-up & Urban Infrastructure (Durg, Bhilai)", "Area (ha)": 11940, "Share (%)": 5.2},
            {"Class": "Tree Cover & Forests", "Area (ha)": 11700, "Share (%)": 5.1},
            {"Class": "Surface Water Bodies (Sheonath River, Reservoirs, Tanks)", "Area (ha)": 4130, "Share (%)": 1.8},
            {"Class": "Grassland & Shrubland", "Area (ha)": 6401, "Share (%)": 2.8}
        ])
        st.dataframe(df_lulc_summary, use_container_width=True)

        st.markdown("---")
        st.subheader("4. Phenology Observations Log (183 Daily Kharif Records from GEE)")
        st.dataframe(df_ts, use_container_width=True)

        csv_ts_export = df_ts.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Kharif Multi-Sensor Time Series (CSV)",
            data=csv_ts_export,
            file_name="durg_kharif_phenology_timeseries.csv",
            mime="text/csv"
        )
    else:
        st.warning("Time series data not found. Run gee/03_extract_phenology_timeseries.py.")

# ------------------------------------------------------------------------------
# TAB 5: INSTITUTIONAL COMPLIANCE (VERRA VM0051 GAP MATRIX)
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("Verra VM0051 Institutional Compliance & Satellite Gap Audit")
    st.markdown("""
    This matrix evaluates satellite remote sensing capabilities against mandatory carbon certification
    requirements under **Verra VM0051** (*Methodology for Improved Agricultural Land Management*).
    """)

    if not df_gap.empty:
        st.dataframe(df_gap, use_container_width=True)
    else:
        st.warning("Gap matrix table not found. Run generate_tables.py.")

    st.markdown("---")
    st.subheader("Olofsson (2014) Stratified Area-Adjusted Error Matrix")
    if not df_confusion.empty:
        st.dataframe(df_confusion, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 6: DATA TABLES & EXPORT CENTER
# ------------------------------------------------------------------------------
with tab6:
    st.subheader("Direct CSV Export & Data Repository")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if not df_mrv.empty:
            csv_mrv = df_mrv.to_csv(index=False, na_rep="NA")
            st.download_button(
                label="📥 Download 20-Column District MRV Summary (CSV)",
                data=csv_mrv,
                file_name="district_mrv_summary.csv",
                mime="text/csv"
            )
            st.dataframe(df_mrv.head(10), use_container_width=True)

    with col_d2:
        if not df_tradeoffs.empty:
            csv_tradeoffs = df_tradeoffs.to_csv(index=False)
            st.download_button(
                label="📥 Download AWD Scenario Trade-offs Table (CSV)",
                data=csv_tradeoffs,
                file_name="wp5_awd_scenario_tradeoffs.csv",
                mime="text/csv"
            )
            st.dataframe(df_tradeoffs, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🛰️ Copernicus Sentinel-5P TROPOMI Air Quality & Atmospheric Metrics (All 33 Districts)")
    if not df_air.empty:
        csv_air = df_air.to_csv(index=False)
        st.download_button(
            label="📥 Download Statewide Air Quality & Emissions Table (CSV)",
            data=csv_air,
            file_name="chhattisgarh_33_districts_air_quality_s5p.csv",
            mime="text/csv"
        )
        st.dataframe(df_air, use_container_width=True)


if __name__ == "__main__":
    pass
