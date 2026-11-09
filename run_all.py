#!/usr/bin/env python3
"""
Master Orchestration Script: Statewide Chhattisgarh Rice Methane, Yield & Groundwater MRV Pipeline
--------------------------------------------------------------------------------------------------
Executes all 6 pipeline stages end-to-end:
  1. Ingest All Open Geodata & Closed-Chamber Ground-Truth Benchmarks
  2. AgriBound + FTW + CropHarvest 10m Agricultural Field Footprints & Shapefile Export
  3. Statewide Dual-Season (Kharif + Rabi) GEE Groundwater & Coupled Yield Panel (2019-2026)
  4. Closed-Chamber Calibrated Methane, Yield & Monte Carlo Engine (10,000 draws)
  5. Dual-Target ML Benchmark Suite, 10 Accuracy Tests & CMIP6 Future Simulator (2027-2035)
  6. Generate Statewide Atlas Cartography, Policy Tables & GEE Interactive Dashboard
"""

import sys
import subprocess
import time

PIPELINE_SCRIPTS = [
    ("Stage 01: Open Geodata & Chamber Benchmarks", "src/01_ingest_geodata_and_chamber_benchmarks.py"),
    ("Stage 02: AgriBound & FTW 10m Field Shapefiles", "src/02_agribound_ftw_cropharvest_field_shapefiles.py"),
    ("Stage 03: GEE Dual-Season Groundwater & Yield Panel", "src/03_gee_statewide_kharif_rabi_groundwater_yield_panel.py"),
    ("Stage 04: Chamber Calibration & Monte Carlo Engine", "src/04_chamber_calibrated_methane_and_yield_monte_carlo.py"),
    ("Stage 05: Dual-Target ML, 10 Tests & Future Simulator", "src/05_train_ml_10_accuracy_tests_and_future_simulator.py"),
    ("Stage 06: Statewide Atlas, Field Maps & GEE App", "src/06_generate_statewide_atlas_and_field_maps.py")
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
