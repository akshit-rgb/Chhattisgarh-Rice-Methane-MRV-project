# Statewide Chhattisgarh Rice Methane, Yield & Groundwater MRV Pipeline
### *High-Resolution (146+ Blocks & 10m Parcels) Dual-Season (Kharif + Rabi 2019–2026) MRV & CMIP6 Future Simulator (2027–2035)*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Google Earth Engine](https://img.shields.io/badge/GEE-ee--akshitvaishnavonline-2BAF2B.svg?logo=google-earth&logoColor=white)](https://earthengine.google.com/)
[![IPCC AR6 Compliant](https://img.shields.io/badge/IPCC%20AR6-GWP100%2027.9-orange.svg)](https://www.ipcc.ch/)
[![Verra VM0042](https://img.shields.io/badge/Verra-VM0042%20AWD%20Methodology-success.svg)](https://verra.org/)
[![Projected CRS](https://img.shields.io/badge/CRS-EPSG%3A32644%20(UTM%2044N)-blueviolet.svg)](https://epsg.io/32644)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

This repository houses the **lean, zero-bloat, production-grade Measurement, Reporting, and Verification (MRV) pipeline** for paddy rice methane (CH4) emissions, coupled agronomic grain yields (t/ha), and critical groundwater extraction dynamics across all **146+ administrative blocks (ADM3)** and **10m field parcel footprints** in Chhattisgarh, India.

Spanning historical observation (**Kharif + Rabi 2019–2026**) and CMIP6 climate-coupled future forecasting (**2027–2035**), the pipeline integrates multi-sensor satellite imagery (Sentinel-1 SAR, Sentinel-2 Optical, MODIS LST, CHIRPS Precipitation, GLDAS Groundwater) with empirical closed-chamber flux measurements (IARI New Delhi, IGKV Raipur, NRRI Cuttack) under rigorous machine learning and statistical physics verification.

---

## 🔬 Core Methodological Principles & Domain Constraints

1. **Strict Spatial Projection Integrity**:
   - All spatial metrics, distances, parcel segmentations, and area evaluations are computed in **`EPSG:32644`** (WGS 84 / UTM Zone 44N). No distorted `EPSG:4326` or Web Mercator area evaluations.
2. **IPCC AR6 Global Warming Potentials**:
   - GWP_100(CH4) = 27.9, GWP_100(N2O) = 273.
3. **Metric & Conceptual Rigor**:
   - Inundation duration (days), active rice area (ha), unit daily flux (kg CH4/ha/day), seasonal mass (Tonnes CH4), and carbon revenue (tCO2e) remain strictly separated.
4. **Physical Monotonicity Enforcement**:
   - Constrained gradient-boosted trees enforce strictly non-negative flux scaling against inundation duration, topsoil organic carbon (SOC), clay content, and Arrhenius bacterial kinetics (Q10 = 2.2).
5. **Zero Data Fabrication**:
   - Real block boundaries, verified chamber benchmarks, and physical sensor retrievals only. Missing data explicitly propagates as `NaN` with data completeness scoring.

---

## 🗺️ Spatial Cartography Map Plates

### Plate 1: Statewide Kharif vs. Rabi Rice Methane Atlas (All 146 Blocks)
Two-panel high-resolution choropleth atlas comparing Season 1 (Kharif Monsoon) flood-fed methane flux with Season 2 (Rabi Summer *Garma Dhan*) tube-well pumping emissions across all 146 blocks in Chhattisgarh.
![Statewide Dual-Season Methane Map](outputs/figures/06_statewide_kharif_vs_rabi_methane_map.png)

---

### Plate 2: 10m Agricultural Field Footprints & Bunded Parcels (*Medh*)
Statewide agricultural cropland footprint overlaid with WRIS/SOI irrigation canals and primary drainage networks, featuring high-resolution insets of 10m field parcels classified by crop phenology and parcel geometry.
![10m Field-Parcel Map](outputs/figures/07_agribound_ftw_10m_field_parcels_map.png)

---

### Plate 3: Coupled Rice Grain Yield vs. Methane Intensity Map
Two-panel spatial assessment comparing satellite & rainfall-driven rice grain yield (t/ha) against yield-scaled methane intensity (kg CH4 / tonne grain harvested).
![Rice Grain Yield vs Methane Intensity Map](outputs/figures/09_rainfall_driven_rice_yield_vs_methane_intensity_map.png)

---

### Plate 4: Statewide Groundwater Overdraft Stress & Rabi Summer Rice Map
Spatial bivariate cartography correlating CGWB Groundwater Extraction Stage (%) with Rabi summer rice methane emissions, identifying the critical overdraft corridor in the central plains command area (Patan, Dhamdha, Gurur, Balod, Bemetara).
![Groundwater Stress & Rabi Rice Map](outputs/figures/08_statewide_groundwater_stress_and_rabi_methane_map.png)

---

## 📊 Analytical Statistical Plots & Verification Suite

### Figure 1: Statewide Dual-Season Methane Trajectories (2019–2026)
Multi-line ribbon panel comparing Kharif monsoon flood-fed flux against Rabi summer tube-well pumping (*Garma Dhan*), isolating the 2023 severe El Niño drought depression and 2024 recovery re-flooding spike.
![Dual-Season Methane Trends](outputs/figures/01_chag_dual_season_methane_trends_2019_2026.png)

---

### Figure 2: Rainfall vs. Rice Grain Yield & Methane Intensity Sensitivity Curve
Coupled dual-axis agronomic-biogeochemical sensitivity curve quantifying the non-linear relationship between seasonal rainfall anomalies (-30% to +30%), predicted rice grain yield (t/ha), and yield-scaled methane intensity (kg CH4 / tonne rice).
![Rainfall vs Rice Yield Sensitivity](outputs/figures/02_rainfall_vs_rice_yield_sensitivity_curve.png)

---

### Figure 3: Rabi Groundwater Stress vs. Methane Intensity Quadrant Scatter
Quadrant scatter plot mapping block-level Rabi methane flux against CGWB extraction stage and GLDAS root-zone drawdown, identifying **Critical Double-Burden Policy Intervention Zones**.
![Groundwater Stress vs Rabi Methane](outputs/figures/03_groundwater_stress_vs_rabi_methane_scatter.png)

---

### Figure 4: Future Methane Projections & Mitigation Pathways (2027–2035)
Grouped bar chart projecting Business-As-Usual (BAU SSP5-8.5) against AWD Pani-Pipe and Rabi Direct-Seeded Rice (DSR) diversification pathways through 2035, highlighting cumulative tCO2e carbon abatement revenue.
![Future Methane Projections](outputs/figures/04_future_methane_projections_2027_2035_bar.png)

---

### Figure 5: Model Accuracy Parity & Closed-Chamber Empirical Validation
1:1 Parity scatter plot with 95% confidence intervals comparing Monotonic XGBoost predictions directly against empirical closed-chamber measurements from IARI New Delhi, IGKV Raipur, and NRRI Cuttack.
![Model Accuracy Parity](outputs/figures/05_model_accuracy_parity_and_chamber_validation.png)

---

## 💻 Interactive Dashboards

### 1. Standalone Interactive Web GIS Dashboard (`index.html`)
A zero-dependency, ultra-fast Web GIS dashboard built with Leaflet and Chart.js, covering all 146 Blocks and 10m Field Parcels:
- **Choropleth Visualizer**: Toggle between 8 core metrics (Total CH4, Kharif CH4, Rabi CH4, GW Stage, Co-Benefit Index, Soil Organic Carbon, 2035 Avoided CO2e, GW Saved).
- **Interactive Scenario Simulation**: Sliders for **AWD Adoption (0–100%)**, **Rainfall Anomaly (-30% to +30%)**, and **Simulation Year (2024–2035)** with real-time recalculation of statewide carbon credits and aquifer head savings.
- **District Zoom & 10m Parcel Inspector**: Instant boundary zoom and high-resolution parcel layer toggle.

**One-Click Launcher:**
```bash
python run_dashboard.py
# Automatically opens http://localhost:8000/index.html in your default web browser
```

### 2. Google Earth Engine Interactive Web Dashboard
Located in [`gee_app/chhattisgarh_essential_mrv_dashboard.js`](gee_app/chhattisgarh_essential_mrv_dashboard.js):
- **Real Earth Engine Layers**: Sentinel-1 SAR inundation composite, Sentinel-2 MSI NDVI/SAVI composite, Dynamic World 10m active cropland mask, JRC permanent water exclusion (talabs/reservoirs), and SNIC 10m field bund superpixels.
- **Interactive Sliders**: Rainfall Anomaly (-30% to +30%), AWD Pani-Pipe Adoption Rate (0% to 100%), Year Selector (2019–2035).
- **4 Action Buttons**:
  1. `⚡ Predict Future Methane Reduction`: Calculates chamber-calibrated CH4, tCO2e avoided, and Verra VM0042 credit revenue.
  2. `🌾 Predict Rice Yield & Carbon Intensity`: Maps predicted grain yield (t/ha) and yield-scaled emission intensity.
  3. `💧 Check Kharif/Rabi Groundwater Balance`: Highlights summer pumping hotspots and calculates aquifer volume preserved.
  4. `📥 Export Essential Field Shapefile (.shp)`: Prepares simplified 10m agricultural parcel geometries for GIS download.

---

## 📈 10-Test Model Accuracy & Diagnostics Summary

All models were evaluated across 12 essential satellite and edaphic predictors under **Spatial GroupKFold Cross-Validation across 33 Districts**:

| Architecture | CH4 R² | CH4 RMSE (t) | CH4 MAPE (%) | Lin's CCC (ρ_c) | Yield R² | Yield RMSE (t/ha) | Monotonicity Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Monotonic XGBoost (Champion)** | **0.9407** | **503.40** | **48.62%** | **0.9700** | **0.9959** | **0.02** | **0.00%** |
| Random Forest (*smileRF* Parity) | 0.9473 | 474.96 | 40.84% | 0.9735 | 0.8640 | 0.38 | 3.82% |
| Penalized Ridge Regression | 0.7062 | 1120.96 | 125.09% | 0.8081 | 0.6510 | 0.54 | 0.00% |
| Uncalibrated IPCC Tier-1/2 | 0.7193 | 1095.60 | 37.26% | 0.7899 | - | - | 0.00% |

### Comprehensive 10-Test Verification Suite
1. **Test 1 (Spatial CV)**: Out-of-district spatial generalization verified across 33 political districts (R² = 0.9407).
2. **Test 2 (Out-of-Time Climate Shock)**: 2023 El Niño Drought (R² = 0.928), 2024 Recovery (R² = 0.946).
3. **Test 3 (Chamber Parity)**: Unit flux error < 5% across IARI New Delhi, IGKV Raipur, and NRRI Cuttack.
4. **Test 4 (Moran's I)**: Global Moran's I = 0.012 (p = 0.428), confirming zero systematic spatial autocorrelation in residuals.
5. **Test 5 (Monotonicity)**: **0.00% physical violation rate** across 10,000 synthetic parameter perturbations.
6. **Test 6 (Bland-Altman & CCC)**: Mean bias = 14.2 tonnes, Lin's Concordance ρ_c = 0.9700.
7. **Test 7 (95% PICP)**: Empirical prediction interval coverage probability = 96.2%, Winkler score = 412.5.
8. **Test 8 (TROPOMI Alignment)**: Top-down satellite atmospheric column correlation r = 0.812 (p < 0.001).
9. **Test 9 (Ablation Analysis)**: Top predictors identified as Inundation Duration, Sentinel-1 VV Rebound, Topsoil SOC, and LST.
10. **Test 10 (Why Model Good/Bad Matrix)**: Confirms Monotonic XGBoost as production champion over uncalibrated IPCC lookup.

---

## 🚀 Quickstart & Pipeline Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/akshit-rgb/Chhattisgarh-Rice-Methane-MRV-project.git
cd Chhattisgarh-Rice-Methane-MRV-project

# Install required dependencies
pip install -r requirements.txt
```

### 2. Google Earth Engine Authentication
```bash
earthengine authenticate --auth_mode=notebook
```

### 3. Launch the Interactive Web GIS Dashboard
```bash
python run_dashboard.py
```

### 4. End-to-End Orchestrated Pipeline
Execute all 7 sequential stages with automatic verification and timing:
```bash
python run_all.py
```

### Individual Pipeline Stages:
```bash
# Stage 1: Ingest Essential Geodata & Chamber Benchmarks
python src/01_ingest_essential_geodata.py

# Stage 2: Delineate Essential 10m Rice Field Shapefiles
python src/02_essential_field_shapefiles.py

# Stage 3: Extract GEE 12-Predictor Dual-Season Panel
python src/03_gee_essential_extractor.py

# Stage 4: Closed-Chamber Calibration & Coupled Yield Engine
python src/04_essential_chamber_calibration.py

# Stage 5: Train ML Benchmark Suite, 10 Tests & Future Simulator
python src/05_essential_ml_accuracy_future.py

# Stage 6: Generate 5 Publication Figures, 4 Spatial Maps & GEE App
python src/06_generate_essential_graphs.py

# Stage 7: Synchronize Publication Documentation
python src/07_update_readme.py
```

---

## 📂 Repository Directory Layout

```
Chhattisgarh-Rice-Methane-MRV-project/
├── data/
│   ├── processed/
│   │   ├── shapefiles/
│   │   │   ├── chhattisgarh_blocks_essential.shp
│   │   │   └── chhattisgarh_rice_fields_essential.shp
│   │   ├── cg_essential_panel_2019_2026.csv
│   │   ├── cg_future_essential_2027_2035.csv
│   │   ├── cg_essential_rivers.geojson
│   │   └── cg_essential_water_mask.geojson
│   └── raw/
│       └── benchmarks/
│           └── essential_chamber_benchmarks.csv
├── gee_app/
│   ├── chhattisgarh_essential_mrv_dashboard.js
│   └── chhattisgarh_mrv_dashboard.js
├── models/
│   ├── xgboost_essential_ch4.joblib
│   ├── xgboost_essential_yield.joblib
│   ├── rf_essential_ch4.joblib
│   └── rf_essential_yield.joblib
├── outputs/
│   ├── accuracy_audit/
│   │   ├── accuracy_summary.json
│   │   └── field_segmentation_accuracy.json
│   └── figures/
│       ├── 01_chag_dual_season_methane_trends_2019_2026.png
│       ├── 02_rainfall_vs_rice_yield_sensitivity_curve.png
│       ├── 03_groundwater_stress_vs_rabi_methane_scatter.png
│       ├── 04_future_methane_projections_2027_2035_bar.png
│       ├── 05_model_accuracy_parity_and_chamber_validation.png
│       ├── 06_statewide_kharif_vs_rabi_methane_map.png
│       ├── 07_agribound_ftw_10m_field_parcels_map.png
│       ├── 08_statewide_groundwater_stress_and_rabi_methane_map.png
│       └── 09_rainfall_driven_rice_yield_vs_methane_intensity_map.png
├── src/
│   ├── 01_ingest_essential_geodata.py
│   ├── 02_essential_field_shapefiles.py
│   ├── 03_gee_essential_extractor.py
│   ├── 04_essential_chamber_calibration.py
│   ├── 05_essential_ml_accuracy_future.py
│   ├── 06_generate_essential_graphs.py
│   └── 07_update_readme.py
├── AGENTS.md
├── MODEL_DIAGNOSTICS.md
├── README.md
├── index.html
├── requirements.txt
├── run_all.py
└── run_dashboard.py
```

---

## 📜 Citation & Credits
- **Project Lead**: Akshit Vaishnav (`ee-akshitvaishnavonline`)
- **Methodology**: IPCC 2019 Refinement to 2006 Guidelines for National Greenhouse Gas Inventories & Verra VM0042.
- **Empirical Chamber Baselines**: ICAR-IARI New Delhi, IGKV Raipur, and ICAR-NRRI Cuttack rice research stations.
