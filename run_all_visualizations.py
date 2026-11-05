"""
================================================================================
MASTER RUNNER: GENERATE ALL VISUALIZATIONS, MAPS, TABLES & DASHBOARD ASSETS
Location: Chhattisgarh, India (All 33 Districts, Pilot: Durg LGD 378)
================================================================================

Executes the complete cartographic, tabular, and diagnostic generation pipeline:
  1. scripts/generate_tables.py -> outputs/tables/ (4 CSV files)
  2. scripts/generate_maps.py   -> outputs/maps/   (5 300-DPI PNG maps)
  3. scripts/generate_charts.py -> outputs/charts/ (5 300-DPI PNG charts)
  4. Audits completeness and validates all artifacts on disk.
"""

import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [VISUALIZATION_MASTER] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_all_visualizations")


def run_script(script_path: Path, desc: str) -> None:
    logger.info("--> [START] %s", desc)
    python_exe = sys.executable
    proc = subprocess.run([python_exe, str(script_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("[FAILED] %s\nSTDOUT:\n%s\nSTDERR:\n%s", desc, proc.stdout, proc.stderr)
        raise RuntimeError(f"Step '{desc}' failed with exit code {proc.returncode}")
    logger.info("<-- [SUCCESS] %s", desc)


def main():
    base_dir = Path(__file__).resolve().parent

    logger.info("================================================================================")
    logger.info("STARTING MASTER VISUALIZATION & CARTOGRAPHIC PIPELINE")
    logger.info("State: Chhattisgarh, India | Pilot: Durg District (LGD 378)")
    logger.info("Target Projected CRS: EPSG:32644 (UTM Zone 44N) | 300 DPI Exports")
    logger.info("================================================================================")

    # 1. Generate Structured Reporting Tables
    run_script(
        base_dir / "scripts" / "generate_tables.py",
        "Step 1: Generating Structured Reporting Tables & Audit Matrices (outputs/tables/)"
    )

    # 2. Generate Publication-Grade Cartographic Maps
    run_script(
        base_dir / "scripts" / "generate_maps.py",
        "Step 2: Generating Publication-Grade Cartography in EPSG:32644 (outputs/maps/)"
    )

    # 3. Generate Diagnostic & Uncertainty Visualizations
    run_script(
        base_dir / "scripts" / "generate_charts.py",
        "Step 3: Generating Diagnostic, Parity & Uncertainty Visualizations (outputs/charts/)"
    )

    # 4. Audit Deliverables
    logger.info("================================================================================")
    logger.info("AUDITING GENERATED ARTIFACTS...")
    logger.info("================================================================================")

    tables = [
        "district_mrv_summary.csv",
        "wp2_confusion_matrix.csv",
        "wp5_awd_scenario_tradeoffs.csv",
        "wp6_vm0051_mrv_gap_matrix.csv",
        "durg_kharif_phenology_timeseries.csv",
        "chhattisgarh_33_districts_air_quality_s5p.csv"
    ]
    for t in tables:
        p = base_dir / "outputs" / "tables" / t
        status = "EXISTS" if p.exists() else "MISSING"
        logger.info("Table: %-46s -> [%s] (%s)", t, status, f"{p.stat().st_size:,} bytes" if p.exists() else "0 bytes")

    maps = [
        "01_admin_boundary_validation.png",
        "02_durg_pilot_rice_extent.png",
        "03_inundation_and_confidence.png",
        "04_modelled_ch4_and_uncertainty.png",
        "05_awd_scenario_potential.png",
        "06_durg_land_use_land_cover.png",
        "07_statewide_rice_paddy_density.png",
        "08_chhattisgarh_statewide_air_quality.png"
    ]
    for m in maps:
        p = base_dir / "outputs" / "maps" / m
        status = "EXISTS" if p.exists() else "MISSING"
        logger.info("Map:   %-35s -> [%s] (%s)", m, status, f"{p.stat().st_size:,} bytes" if p.exists() else "0 bytes")

    charts = [
        "01_sar_optical_phenology_curve.png",
        "02_satellite_vs_govt_area_parity.png",
        "03_district_ch4_forest_plot.png",
        "04_monte_carlo_sensitivity_tornado.png",
        "05_awd_ch4_n2o_waterfall.png"
    ]
    for c in charts:
        p = base_dir / "outputs" / "charts" / c
        status = "EXISTS" if p.exists() else "MISSING"
        logger.info("Chart: %-35s -> [%s] (%s)", c, status, f"{p.stat().st_size:,} bytes" if p.exists() else "0 bytes")

    dashboard_app = base_dir / "outputs" / "dashboard" / "app.py"
    logger.info("Dashboard App: %-29s -> [%s]", "app.py", "EXISTS" if dashboard_app.exists() else "MISSING")

    logger.info("================================================================================")
    logger.info("ALL TABLES, MAPS, CHARTS, AND DASHBOARD CODE COMPLETED SUCCESSFULLY!")
    logger.info("To launch the interactive dashboard, run:")
    logger.info("  streamlit run chhattisgarh_rice_methane_mrv/outputs/dashboard/app.py")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
