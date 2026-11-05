"""
================================================================================
UNIT & INTEGRATION TEST SUITE: CHHATTISGARH RICE METHANE MRV PIPELINE
Tests compliance with strict architectural and IPCC QA/QC constraints.
Standard Library unittest runner.
================================================================================
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

# Ensure project modules are importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.wp1_boundary_validator import DistrictBoundaryValidator, TARGET_CRS, EXPECTED_DISTRICT_COUNT
from scripts.wp2_accuracy_assessment import OlofssonAccuracyAssessment
from scripts.wp4_ipcc_tier2_engine import IPCCTier2MethaneEngine
from scripts.wp5_awd_scenarios import AWDScenarioCalculator, GWP_CH4_AR6, GWP_N2O_AR6
from scripts.wp6_district_aggregator import MANDATED_SCHEMA


class TestMRVPipeline(unittest.TestCase):

    def test_district_crosswalk_integrity(self):
        """Validates that crosswalk contains exactly 33 unique LGD codes and Durg (378)."""
        crosswalk_path = project_root / "metadata" / "district_name_crosswalk.csv"
        self.assertTrue(crosswalk_path.exists(), "Crosswalk file missing!")
        df = pd.read_csv(crosswalk_path)
        self.assertEqual(len(df), EXPECTED_DISTRICT_COUNT, f"Expected 33 districts, found {len(df)}")
        self.assertEqual(df["lgd_code"].nunique(), EXPECTED_DISTRICT_COUNT, "LGD codes must be unique")
        self.assertIn(378, df["lgd_code"].values, "Durg Pilot (LGD 378) must be present")

    def test_crs_projected_area_calculation(self):
        """Strict test: confirms area calculations are executed in EPSG:32644 (metres)."""
        crosswalk_path = project_root / "metadata" / "district_name_crosswalk.csv"
        validator = DistrictBoundaryValidator(crosswalk_path=crosswalk_path)

        # Create dummy 1 deg x 1 deg box in geographic CRS
        poly = Polygon([(81.0, 21.0), (81.0, 22.0), (82.0, 22.0), (82.0, 21.0), (81.0, 21.0)])
        gdf_geo = gpd.GeoDataFrame(
            [{"lgd_code": 378, "district_name": "Durg", "geometry": poly}],
            crs="EPSG:4326"
        )

        gdf_projected = validator.reproject_and_calculate_area(gdf_geo)
        self.assertEqual(str(gdf_projected.crs), TARGET_CRS, f"Expected {TARGET_CRS}, got {gdf_projected.crs}")
        self.assertIn("district_area_ha", gdf_projected.columns)
        self.assertGreater(gdf_projected["district_area_ha"].iloc[0], 10000.0)

    def test_olofsson_accuracy_engine(self):
        """Tests Olofsson (2014) area-adjusted accuracy calculations."""
        strata_areas = {"rice": 100000.0, "non_rice": 100000.0}
        engine = OlofssonAccuracyAssessment(strata_areas_ha=strata_areas)

        counts = np.array([
            [90, 10],
            [10, 90]
        ])
        results = engine.compute_from_confusion_matrix(counts)

        self.assertAlmostEqual(results["overall_accuracy"], 0.90, places=2)
        self.assertAlmostEqual(results["users_accuracy"]["rice"], 0.90, places=2)
        self.assertAlmostEqual(results["producers_accuracy"]["rice"], 0.90, places=2)
        self.assertGreater(results["overall_accuracy_se"], 0.0)

    def test_ipcc_tier2_flux_and_no_fabricated_data(self):
        """Validates Tier-2 formula and strict NaN preservation for missing districts."""
        config_path = project_root / "config" / "parameters.yaml"
        engine = IPCCTier2MethaneEngine(config_path=config_path)

        test_df = pd.DataFrame([
            {
                "district_name": "Durg",
                "district_code": 378,
                "boundary_vintage": 2000,
                "rice_area_ha": 10000.0,
                "water_regime_assumption": "single_aeration",
                "cultivation_duration_days": 105,
                "wet_observation_count": 12.0,
                "rice_map_accuracy": 0.92
            },
            {
                "district_name": "Unmapped_District",
                "district_code": 999,
                "boundary_vintage": 2012,
                "rice_area_ha": np.nan,
                "water_regime_assumption": "single_aeration",
                "cultivation_duration_days": 105
            }
        ])

        results = engine.compute_district_emissions_vectorized(test_df)

        # Durg: Area 10,000 ha * 105 days * 1.30 * 0.71 * 0.68 * 1e-3 = ~658.7 tonnes CH4
        durg_tonnes = results.loc[results["district_code"] == 378, "rice_ch4_emissions_tonnes"].iloc[0]
        self.assertAlmostEqual(durg_tonnes, 658.74, delta=2.0)
        self.assertEqual(results.loc[results["district_code"] == 378, "data_completeness_score"].iloc[0], 1.0)

        # Unmapped district: MUST be NaN and have penalized score
        unmapped_tonnes = results.loc[results["district_code"] == 999, "rice_ch4_emissions_tonnes"].iloc[0]
        self.assertTrue(pd.isna(unmapped_tonnes), "Missing district must have NaN emissions")
        self.assertLess(results.loc[results["district_code"] == 999, "data_completeness_score"].iloc[0], 0.50)
        self.assertIn("Missing satellite-derived rice extent", results.loc[results["district_code"] == 999, "limitations"].iloc[0])

    def test_awd_n2o_tradeoff_and_gwp_ar6(self):
        """Validates AWD counterfactual scenarios and AR6 GWP constants."""
        config_path = project_root / "config" / "parameters.yaml"
        calculator = AWDScenarioCalculator(config_path=config_path)

        self.assertEqual(GWP_CH4_AR6, 27.9, "IPCC AR6 CH4 GWP must be 27.9")
        self.assertEqual(GWP_N2O_AR6, 273.0, "IPCC AR6 N2O GWP must be 273.0")

        res = calculator.calculate_scenario(rice_area_ha=1000.0, baseline_ch4_tonnes=100.0, scenario_key="central")
        # Central: 50% adoption, 45% CH4 reduction -> 100 * 0.50 * 0.45 = 22.5 t CH4 avoided
        self.assertAlmostEqual(res["ch4_reduction_tonnes"], 22.5, places=2)
        gross_co2e = 22.5 * 27.9 # 627.75 t CO2e
        self.assertAlmostEqual(res["gross_ch4_abatement_co2e_tonnes"], gross_co2e, places=2)

        # N2O penalty: 500 ha * 0.60 kg/ha = 300 kg = 0.30 t N2O -> 0.30 * 273 = 81.9 t CO2e
        self.assertAlmostEqual(res["n2o_tradeoff_penalty_co2e_tonnes"], 81.9, places=2)
        # Net abatement: 627.75 - 81.9 = 545.85 t CO2e
        self.assertAlmostEqual(res["net_ghg_abatement_co2e_tonnes"], 545.85, places=2)

    def test_final_20_column_schema_compliance(self):
        """Validates exact 20-column schema ordering, count, and typing."""
        out_csv = project_root / "data" / "outputs" / "chhattisgarh_33_districts_rice_methane_mrv.csv"
        self.assertTrue(out_csv.exists(), "Final 20-column CSV missing!")
        df = pd.read_csv(out_csv)
        self.assertEqual(len(df.columns), 20, f"Expected 20 columns, got {len(df.columns)}")
        self.assertEqual(list(df.columns), MANDATED_SCHEMA, "Column names or order do not match mandated schema")
        self.assertEqual(len(df), 33, f"Expected 33 rows, got {len(df)}")


if __name__ == "__main__":
    unittest.main()
