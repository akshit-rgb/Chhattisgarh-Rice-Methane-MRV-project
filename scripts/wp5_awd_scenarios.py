"""
================================================================================
WORK PACKAGE 5: COUNTERFACTUAL AWD SCENARIO CALCULATOR & N2O TRADE-OFF DEDUCTION
Standards: IPCC 2019 Refinement & IPCC AR6 WG1 100-Year GWPs (CH4=27.9, N2O=273)
Target: Alternate Wetting and Drying (AWD) Carbon MRV Feasibility
================================================================================

This module quantifies the counterfactual mitigation potential of transitioning
from continuous flooding / conventional irrigation to Alternate Wetting and Drying (AWD).

MANDATORY MRV & ARCHITECTURAL PRINCIPLES:
1. Conceptual Separation:
   - Baseline CH4 emissions (kg and tonnes CH4)
   - Avoided CH4 flux (kg and tonnes CH4)
   - Direct N2O trade-off penalty from intermittent aeration (kg and tonnes N2O)
   - Gross CO2-equivalent conversions using IPCC AR6 100-year GWP:
     * GWP_CH4 = 27.9
     * GWP_N2O = 273.0
   - Net CO2-equivalent abatement = (Avoided CH4 * 27.9) - (Induced N2O * 273.0)
2. Scenario Hierarchy:
   - Conservative (25% adoption, 28% CH4 reduction, 0.35 kg N2O/ha trade-off)
   - Central (50% adoption, 45% CH4 reduction, 0.60 kg N2O/ha trade-off)
   - Optimistic (80% adoption, 55% CH4 reduction, 0.85 kg N2O/ha trade-off)
3. No Fabricated Data: Districts without validated rice area output np.nan.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("wp5_awd_scenarios")

# IPCC AR6 100-year Global Warming Potentials (WG1 Table 7.15)
GWP_CH4_AR6 = 27.9
GWP_N2O_AR6 = 273.0


class AWDScenarioCalculator:
    """
    Computes AWD counterfactual methane mitigation and net carbon credits after N2O trade-off.
    """

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.cfg = self._load_config()
        self.scenarios_cfg = self.cfg["awd_scenarios"]["scenarios"]

    def _load_config(self) -> Dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def calculate_scenario(
        self,
        rice_area_ha: float,
        baseline_ch4_tonnes: float,
        scenario_key: str = "central"
    ) -> Dict:
        """
        Calculates CH4 reduction, N2O penalty, and net CO2e abatement for a specific scenario.
        """
        if pd.isna(rice_area_ha) or pd.isna(baseline_ch4_tonnes) or rice_area_ha <= 0:
            return {
                "scenario": scenario_key,
                "awd_adopted_area_ha": np.nan,
                "ch4_reduction_kg": np.nan,
                "ch4_reduction_tonnes": np.nan,
                "gross_ch4_abatement_co2e_tonnes": np.nan,
                "n2o_increase_kg": np.nan,
                "n2o_increase_tonnes": np.nan,
                "n2o_tradeoff_penalty_co2e_tonnes": np.nan,
                "net_ghg_abatement_co2e_tonnes": np.nan,
            }

        sc = self.scenarios_cfg[scenario_key]
        adoption_rate = float(sc["adoption_rate"])
        ch4_red_pct = float(sc["ch4_reduction_factor"])
        n2o_penalty_kg_ha = float(sc["n2o_tradeoff_kg_per_ha"])

        # 1. Adopted area under AWD
        adopted_area_ha = rice_area_ha * adoption_rate

        # 2. Avoided Methane (kg and tonnes CH4)
        # Baseline emissions on adopted area * reduction percentage
        ch4_avoided_tonnes = baseline_ch4_tonnes * adoption_rate * ch4_red_pct
        ch4_avoided_kg = ch4_avoided_tonnes * 1e3

        # 3. Gross CO2e Abatement from CH4 Avoidance (GWP = 27.9)
        gross_co2e_ch4 = ch4_avoided_tonnes * GWP_CH4_AR6

        # 4. Increased N2O Emissions due to intermittent aeration (kg and tonnes N2O)
        n2o_increase_kg = adopted_area_ha * n2o_penalty_kg_ha
        n2o_increase_tonnes = n2o_increase_kg * 1e-3

        # 5. N2O Trade-off Penalty in CO2e (GWP = 273.0)
        n2o_penalty_co2e = n2o_increase_tonnes * GWP_N2O_AR6

        # 6. Net GHG Mitigation (tonnes CO2e)
        net_co2e = gross_co2e_ch4 - n2o_penalty_co2e

        return {
            "scenario": scenario_key,
            "awd_adopted_area_ha": float(adopted_area_ha),
            "ch4_reduction_kg": float(ch4_avoided_kg),
            "ch4_reduction_tonnes": float(ch4_avoided_tonnes),
            "gross_ch4_abatement_co2e_tonnes": float(gross_co2e_ch4),
            "n2o_increase_kg": float(n2o_increase_kg),
            "n2o_increase_tonnes": float(n2o_increase_tonnes),
            "n2o_tradeoff_penalty_co2e_tonnes": float(n2o_penalty_co2e),
            "net_ghg_abatement_co2e_tonnes": float(net_co2e),
        }

    def evaluate_districts_awd(
        self,
        tier2_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluates Conservative, Central, and Optimistic AWD scenarios across districts.
        Returns:
            1. Detailed multi-scenario DataFrame
            2. District summary DataFrame containing estimated_awd_scenario_reduction
        """
        logger.info(f"Evaluating AWD counterfactual scenarios across {len(tier2_df)} districts...")
        detailed_records = []
        central_reductions = []

        for _, row in tier2_df.iterrows():
            d_name = row["district_name"]
            d_code = row["district_code"]
            area_ha = row.get("rice_area_ha", np.nan)
            base_ch4_tonnes = row.get("rice_ch4_emissions_tonnes", np.nan)

            # Central scenario for primary table column
            res_central = self.calculate_scenario(area_ha, base_ch4_tonnes, "central")
            central_reductions.append(res_central["ch4_reduction_tonnes"])

            for scen in ["conservative", "central", "optimistic"]:
                res = self.calculate_scenario(area_ha, base_ch4_tonnes, scen)
                detailed_records.append({
                    "district_name": d_name,
                    "district_code": d_code,
                    "rice_area_ha": area_ha,
                    "baseline_ch4_tonnes": base_ch4_tonnes,
                    "scenario": scen,
                    "awd_adopted_area_ha": res["awd_adopted_area_ha"],
                    "ch4_reduction_kg": res["ch4_reduction_kg"],
                    "ch4_reduction_tonnes": res["ch4_reduction_tonnes"],
                    "gross_co2e_ch4_tonnes": res["gross_ch4_abatement_co2e_tonnes"],
                    "n2o_increase_kg": res["n2o_increase_kg"],
                    "n2o_increase_tonnes": res["n2o_increase_tonnes"],
                    "n2o_penalty_co2e_tonnes": res["n2o_tradeoff_penalty_co2e_tonnes"],
                    "net_co2e_abatement_tonnes": res["net_ghg_abatement_co2e_tonnes"],
                    "gwp_ch4_reference": GWP_CH4_AR6,
                    "gwp_n2o_reference": GWP_N2O_AR6,
                })

        detailed_df = pd.DataFrame(detailed_records)
        summary_df = tier2_df.copy()
        # Strictly populate estimated_awd_scenario_reduction (Central scenario avoided CH4 tonnes)
        summary_df["estimated_awd_scenario_reduction"] = central_reductions

        return detailed_df, summary_df


def main():
    parser = argparse.ArgumentParser(
        description="WP5: Counterfactual AWD Scenario & N2O Trade-off Calculator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/parameters.yaml",
        help="Path to parameters.yaml.",
    )
    parser.add_argument(
        "--input-tier2",
        type=str,
        default="data/outputs/wp4_tier2_methane_results.csv",
        help="Path to input WP4 Tier-2 results CSV.",
    )
    parser.add_argument(
        "--output-detailed",
        type=str,
        default="data/outputs/wp5_awd_scenarios_detailed.csv",
        help="Path to output multi-scenario detailed CSV.",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="data/outputs/wp5_awd_summary.csv",
        help="Path to output summary CSV.",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name in ["gee", "scripts"] else script_dir

    config_path = project_root / args.config
    input_path = project_root / args.input_tier2
    out_detailed_path = project_root / args.output_detailed
    out_summary_path = project_root / args.output_summary

    calculator = AWDScenarioCalculator(config_path=config_path)
    tier2_df = pd.read_csv(input_path)

    detailed_df, summary_df = calculator.evaluate_districts_awd(tier2_df)

    detailed_df.to_csv(out_detailed_path, index=False)
    summary_df.to_csv(out_summary_path, index=False)

    logger.info(f"Saved detailed multi-scenario analysis to: {out_detailed_path}")
    logger.info(f"Saved summary to: {out_summary_path}")

    # Display Durg Pilot Scenario Analysis
    durg_scenarios = detailed_df[detailed_df["district_code"] == 378]
    logger.info("=== DURG PILOT AWD SCENARIO BREAKDOWN ===")
    for _, sc_row in durg_scenarios.iterrows():
        sc_name = sc_row["scenario"].upper()
        logger.info(
            f"[{sc_name}] Adopted: {sc_row['awd_adopted_area_ha']:,.0f} ha | "
            f"CH4 Avoided: {sc_row['ch4_reduction_tonnes']:,.2f} t CH4 | "
            f"Gross CO2e: {sc_row['gross_co2e_ch4_tonnes']:,.2f} t | "
            f"N2O Penalty: -{sc_row['n2o_penalty_co2e_tonnes']:,.2f} t CO2e | "
            f"NET ABATEMENT: {sc_row['net_co2e_abatement_tonnes']:,.2f} t CO2e"
        )


if __name__ == "__main__":
    main()
