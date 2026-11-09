#!/usr/bin/env python3
"""
Master Orchestration Script: Statewide Chhattisgarh Rice Methane & Groundwater MRV Pipeline
--------------------------------------------------------------------------------------------
Executes all 6 pipeline stages end-to-end:
  1. Ingest All Open Geodata & Closed-Chamber Benchmarks
  2. Extract Agricultural Field Footprints & 10m Field-Boundary Delineation
  3. Statewide Dual-Season (Kharif + Rabi) & Groundwater GEE Feature Extractor (2019-2026)
  4. Biogeochemical Chamber Calibration & Monte Carlo Engine (10,000 draws)
  5. Train ML Benchmark Suite, Execute 10 Accuracy Tests & CMIP6 Future Forecasts (2027-2035)
  6. Generate Statewide Atlas Maps, Policy Tables & GEE Interactive Dashboard
"""

import sys
import subprocess
import time

PIPELINE_SCRIPTS = [
    ("Stage 01: Open Geodata & Chamber Benchmarks", "src/01_ingest_all_open_geodata_and_chamber_benchmarks.py"),
    ("Stage 02: 10m Agricultural Field Footprints", "src/02_extract_agricultural_field_footprints_and_boundaries.py"),
    ("Stage 03: Statewide Dual-Season GEE Panel Extractor", "src/03_gee_statewide_kharif_rabi_groundwater_extractor.py"),
    ("Stage 04: Biogeochemical Calibration & Monte Carlo", "src/04_biogeochemical_chamber_calibration_monte_carlo.py"),
    ("Stage 05: 4-Model Benchmark, 10 Tests & CMIP6 Forecast", "src/05_train_ml_10_accuracy_tests_and_future_forecast.py"),
    ("Stage 06: Atlas Cartography, Policy & GEE Dashboard", "src/06_generate_statewide_atlas_maps_and_diagnostics.py")
]

def main():
    print("=" * 85)
    print("🚀 STATEWIDE CHHATTISGARH RICE METHANE & GROUNDWATER MRV PIPELINE (2019-2035)")
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
