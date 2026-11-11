#!/usr/bin/env python3
"""
Master Orchestration Script: Statewide Chhattisgarh Rice Methane, Yield & Groundwater MRV Pipeline
--------------------------------------------------------------------------------------------------
Executes all 7 essential, lean pipeline stages end-to-end:
  1. Ingest Essential Geodata & Closed-Chamber Benchmarks
  2. Delineate Essential 10m Rice Field Shapefiles (AgriBound + FTW + CropHarvest)
  3. Extract GEE Dual-Season Satellite & Groundwater Panel (12 Essential Predictors)
  4. Closed-Chamber Calibration, Coupled Rice Yield & Monte Carlo Uncertainty Engine
  5. Dual-Target ML Benchmark Suite, 10 Accuracy Tests & CMIP6 Future Forecasts (2027-2035)
  6. Automated Publication Graph Generation Suite (5 Plates @ 300 DPI) & Lean GEE App
  7. Professional Publication Documentation (README.md & MODEL_DIAGNOSTICS.md)
"""

import sys
import subprocess
import time

PIPELINE_SCRIPTS = [
    ("Stage 01: Essential Geodata & Chamber Benchmarks", "src/01_ingest_essential_geodata.py"),
    ("Stage 02: 146-Block Shapefile Incorporation & Census", "src/01_incorporate_agriculture_field_blockwise.py"),
    ("Stage 03: Exact Local Shapefile Dual-Season Extraction", "src/01_extract_local_shapefile_kharif_rabi.py"),
    ("Stage 04: Chamber Biogeochemical Engine & Monte Carlo", "src/02_biogeochemical_methane_engine.py"),
    ("Stage 05: Monotonic XGBoost v2 & 10 Accuracy Tests", "src/03_train_and_predict_ml_methane.py"),
    ("Stage 06: Essential 10m Field Parcels (All 33 Districts)", "src/02_essential_field_shapefiles.py"),
    ("Stage 07: GEE 12-Predictor Dual-Season Extractor", "src/03_gee_essential_extractor.py"),
    ("Stage 08: Chamber Calibration & Coupled Yield Engine", "src/04_essential_chamber_calibration.py"),
    ("Stage 09: Essential ML Suite, 10 Tests & Future Simulator", "src/05_essential_ml_accuracy_future.py"),
    ("Stage 10: Automated Graph Suite & GEE Dashboard", "src/06_generate_essential_graphs.py"),
    ("Stage 11: 146 Block-Wise Field Parcel Maps Suite", "src/13_generate_block_field_maps.py"),
    ("Stage 12: Shapefile Block-Level Statewide Cartography", "src/12_create_maps_from_user_shapefile.py"),
    ("Stage 13: Comprehensive Soil Science & Typology Suite", "src/10_generate_soil_maps_and_typology.py"),
    ("Stage 14: Professional Publication Documentation", "src/07_update_readme.py"),
    ("Stage 15: Automated Git & GitHub Remote Synchronization", "src/04_push_to_github.py")
]

def main():
    print("=" * 85)
    print("🌾 STATEWIDE CHHATTISGARH RICE METHANE, YIELD & GROUNDWATER MRV PIPELINE (2019-2035)")
    print("=" * 85)
    start_time = time.time()

    for idx, (stage_name, script_path) in enumerate(PIPELINE_SCRIPTS, 1):
        print(f"\n[RUNNING STEP {idx}/{len(PIPELINE_SCRIPTS)}] {stage_name} ({script_path})...")
        t0 = time.time()
        ret = subprocess.run([sys.executable, script_path], check=False)
        elapsed = time.time() - t0

        if ret.returncode != 0:
            print(f"\n❌ [FAILED] Stage {idx} ({script_path}) exited with return code {ret.returncode}.")
            sys.exit(ret.returncode)
        else:
            print(f"✅ [SUCCESS] {stage_name} completed in {elapsed:.1f}s.")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 85)
    print(f"🎉 ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_elapsed:.1f}s!")
    print("=" * 85)

if __name__ == "__main__":
    main()
