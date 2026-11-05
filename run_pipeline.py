"""
================================================================================
MASTER ORCHESTRATION PIPELINE: CHHATTISGARH RICE METHANE MRV
Pilot District: Durg (LGD Code: 378) | Scalable to all 33 Districts
================================================================================

Executes the complete, modular satellite MRV pipeline end-to-end:
  - WP1: Boundary Validation, Topology Audit, and UTM Zone 44N Area Calculation
  - WP2: Olofsson Area-Adjusted Accuracy Assessment & Error Matrix
  - WP4: Vectorized IPCC 2019 Tier-2 CH4 Flux & 10,000-Draw Monte Carlo Engine
  - WP5: Counterfactual AWD Scenario Mitigation with N2O Trade-off Penalty
  - WP6: Schema Validation & 20-Column Table Aggregation (CSV & GPKG)
"""

import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PIPELINE_ORCHESTRATOR] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mrv_orchestrator")


def run_command_logged(cmd: list, desc: str) -> None:
    logger.info(f"--> [START] {desc}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(f"[FAILED] {desc}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        raise RuntimeError(f"Step '{desc}' failed with exit code {proc.returncode}")
    logger.info(f"<-- [SUCCESS] {desc}")


def main():
    base_dir = Path(__file__).resolve().parent
    python_exe = sys.executable

    logger.info("================================================================================")
    logger.info("STARTING SATELLITE-BASED RICE METHANE ASSESSMENT & MRV PIPELINE")
    logger.info("State: Chhattisgarh, India (33 Districts, Pilot: Durg LGD 378)")
    logger.info("================================================================================")

    # 1. WP1: Boundary Ingestion & Validation
    run_command_logged(
        [python_exe, str(base_dir / "scripts" / "wp1_boundary_validator.py")],
        "Work Package 1: Administrative Boundary Validator & UTM Zone 44N Reprojection"
    )

    # 2. WP2: Accuracy Assessment
    run_command_logged(
        [python_exe, str(base_dir / "scripts" / "wp2_accuracy_assessment.py")],
        "Work Package 2: Olofsson (2014) Area-Adjusted Accuracy Engine"
    )

    # 3. WP4: IPCC Tier-2 Vectorized CH4 Engine
    run_command_logged(
        [python_exe, str(base_dir / "scripts" / "wp4_ipcc_tier2_engine.py")],
        "Work Package 4: IPCC Tier-2 Flux Model & 10,000-Draw Monte Carlo Uncertainty"
    )

    # 4. WP5: AWD Counterfactual Scenarios
    run_command_logged(
        [python_exe, str(base_dir / "scripts" / "wp5_awd_scenarios.py")],
        "Work Package 5: AWD Scenarios & N2O Trade-off Deduction"
    )

    # 5. WP6: District Aggregator & Schema Verification
    run_command_logged(
        [python_exe, str(base_dir / "scripts" / "wp6_district_aggregator.py")],
        "Work Package 6: 20-Column Schema Verification & District Aggregator"
    )

    logger.info("================================================================================")
    logger.info("ALL PIPELINE WORK PACKAGES EXECUTED AND VALIDATED SUCCESSFULLY!")
    logger.info(f"Final 20-Column CSV: {base_dir / 'data' / 'outputs' / 'chhattisgarh_33_districts_rice_methane_mrv.csv'}")
    logger.info(f"Final Spatial GPKG: {base_dir / 'data' / 'outputs' / 'chhattisgarh_33_districts_rice_methane_mrv.gpkg'}")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
