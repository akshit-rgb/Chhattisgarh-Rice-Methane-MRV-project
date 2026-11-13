#!/usr/bin/env python3
"""
Step 03: Closed-Chamber Calibrated Dual-Season IPCC Tier-2 Accounting & Monte Carlo Simulation
-------------------------------------------------------------------------------------------------
Applies closed-chamber empirical calibrations:
  1. IARI / NPL New Delhi closed-chamber benchmark curves.
  2. IGKV Raipur soil order measurements (Kanhar Vertisols, Dorsa Inceptisols, Matasi Alfisols, Bhata).
  3. Gupta et al. (2009) seasonal ratios: Kharif vs. Rabi 1.8x baseline difference,
     modified dynamically by Arrhenius Q10 = 2.2 temperature kinetics (capturing the April-May
     >38°C thermal spike) and water regime scaling factor SF_w.
  4. 2,000-iteration Monte Carlo uncertainty propagation per block-season.
  5. Verra VM0051 AWD scenario and DSR groundwater-saving transitions.
  6. Strict separation of satellite extent (ha), inundation (days), CH4 mass (tonnes),
     and CO2e (AR6 GWP: CH4=27.9, N2O=273).

Outputs:
  - data/processed/chhattisgarh_dual_season_tier2_methane_panel.csv (2,336 block-seasons)
"""

import os

import numpy as np
import pandas as pd

# IPCC AR6 100-year GWPs
GWP_CH4 = 27.9
GWP_N2O = 273.0

def run_dual_season_chamber_fusion(n_simulations=2000):
    in_csv = "data/processed/chhattisgarh_statewide_kharif_rabi_groundwater_panel_2019_2026.csv"
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Input file not found: {in_csv}")

    print(f"[INFO] Reading statewide panel from {in_csv}...")
    df_wide = pd.read_csv(in_csv)

    print(f"[INFO] Initializing Monte Carlo simulation ({n_simulations} draws per block-season)...")
    np.random.seed(101)

    # 1. Base Parameter Distributions
    # Kharif baseline emission factor EF_c (kg CH4/ha/day)
    # Lognormal, median = 1.30, sigma = 0.25 (South Asia IPCC 2019 Refinement / IARI chamber)
    ef_c_kharif_sim = np.random.lognormal(mean=np.log(1.30), sigma=0.25, size=n_simulations)

    # Rabi baseline emission factor: Gupta et al. (2009) baseline ratio 1.8x lower base daily flux
    # before temperature and continuous water adjustments
    ef_c_rabi_sim = ef_c_kharif_sim / 1.80

    # Organic amendment factor SF_o (1.5 t/ha FYM): Normal(1.18, 0.05)
    sf_o_sim = np.random.normal(1.18, 0.05, size=n_simulations)

    # AWD reduction factor (Verra VM0051 / multiple aerations): Beta(12, 19) -> mean ~0.387
    awd_reduction_sim = np.random.beta(a=12, b=19, size=n_simulations)

    # DSR reduction factor (Direct Seeded Rice aerobic tillering): Beta(25, 8) -> mean ~0.758 (76% reduction)
    dsr_reduction_sim = np.random.beta(a=25, b=8, size=n_simulations)

    # N2O penalty under AWD/DSR: 0.25 kg N2O/ha increase
    n2o_increase_sim = np.random.normal(0.25, 0.04, size=n_simulations)

    # IGKV Raipur Soil Order Multipliers
    soil_multipliers = {
        "Kanhar": 1.28, # Vertisol: high clay, deep crack, high methanogenesis under continuous flood
        "Dorsa": 1.05,  # Inceptisol: intermediate clay loam
        "Matasi": 0.82, # Alfisol: sandy-clay loam, higher drainage
        "Bhata": 0.58   # Lateritic gravelly: low water retention
    }

    records = []

    for idx, row in df_wide.iterrows():
        yr = int(row["year"])
        lgd_code = int(row["lgd_code"])
        b_name = row["block_name"]
        d_name = row["district_name"]
        aez = row["aez_region"]
        soil = row["soil_indigenous_type"]
        soil_mult = soil_multipliers.get(soil, 1.0)

        soc_pct = float(row["shc_soc_pct"])
        ph = float(row["shc_ph"])
        clay = float(row["soil_clay_pct"])
        gw_cat = row["cgwb_aquifer_category"]
        gw_stage = float(row["cgwb_gw_stage_pct"])
        drawdown_mm = float(row["gldas_summer_drawdown_mm"])
        cobenefit_idx = float(row["methane_gw_cobenefit_index"])
        canal_km = float(row["canal_length_km"])

        # Soil factor SF_s (bell-shaped around pH 6.8 - 7.2, elevated with SOC and clay)
        ph_factor = np.exp(-0.5 * ((ph - 7.0) / 1.1) ** 2)
        clay_factor = 0.85 + 0.30 * (clay / 40.0)
        soc_factor = 0.90 + 0.22 * (soc_pct / 0.80)
        sf_s = ph_factor * clay_factor * soc_factor * soil_mult

        # ----------------------------------------------------
        # 1. KHARIF SEASON OBSERVATION
        # ----------------------------------------------------
        k_area_ha = float(row["Kharif_Active_Rice_Area_ha"])
        k_inund_days = float(row["Kharif_Inundation_Days"])
        k_temp_c = float(row["Kharif_LST_Celsius"])
        k_sf_w_base = float(row["Kharif_SF_w"])
        k_duration = float(np.clip(k_inund_days + 35.0, 100.0, 130.0))

        # Arrhenius Q10 = 2.2 temperature kinetics relative to 28°C
        k_temp_factor = 2.2 ** ((k_temp_c - 28.0) / 10.0)

        # Monte Carlo draws
        k_sf_w_dist = np.random.normal(k_sf_w_base, 0.04, size=n_simulations)
        k_daily_flux_mc = ef_c_kharif_sim * k_sf_w_dist * sf_o_sim * sf_s * k_temp_factor # kg/ha/day
        k_seasonal_flux_ha_mc = k_daily_flux_mc * k_duration # kg/ha/season
        k_total_ch4_tonnes_mc = (k_seasonal_flux_ha_mc * k_area_ha) / 1000.0 # tonnes CH4

        k_ch4_mean_tonnes = float(np.mean(k_total_ch4_tonnes_mc))
        k_ch4_low_tonnes = float(np.percentile(k_total_ch4_tonnes_mc, 2.5))
        k_ch4_high_tonnes = float(np.percentile(k_total_ch4_tonnes_mc, 97.5))
        k_flux_kg_ha = float(np.mean(k_seasonal_flux_ha_mc))

        # Kharif AWD Mitigation (VM0051)
        k_awd_mitigation_tonnes = k_ch4_mean_tonnes * float(np.mean(awd_reduction_sim))
        k_awd_remaining_tonnes = k_ch4_mean_tonnes - k_awd_mitigation_tonnes
        k_co2e_baseline = k_ch4_mean_tonnes * GWP_CH4
        k_n2o_penalty = (float(np.mean(n2o_increase_sim)) * k_area_ha / 1000.0) * GWP_N2O
        k_net_awd_co2e = max(0.0, (k_awd_mitigation_tonnes * GWP_CH4) - k_n2o_penalty)

        records.append({
            "year": yr,
            "season": "Kharif",
            "observation_id": f"{lgd_code}_{yr}_KHARIF",
            "lgd_code": lgd_code,
            "district_name": d_name,
            "block_name": b_name,
            "aez_region": aez,
            "soil_type": soil,
            "centroid_lon": row["centroid_lon"],
            "centroid_lat": row["centroid_lat"],
            "active_rice_area_ha": k_area_ha,
            "rice_fraction": float(row["Kharif_Rice_Fraction"]),
            "cultivation_duration_days": round(k_duration, 1),
            "lst_celsius": k_temp_c,
            "flood_frequency": float(row["Kharif_Flood_Frequency"]),
            "msd_rebound_db": float(row["Kharif_MSD_Rebound_dB"]),
            "savi": float(row["Kharif_SAVI"]),
            "lswi": float(row["Kharif_LSWI"]),
            "precip_mm": float(row["Kharif_Precip_mm"]),
            "water_regime_sf_w": k_sf_w_base,
            "soil_factor_sf_s": round(float(sf_s), 3),
            "arrhenius_temp_multiplier": round(float(k_temp_factor), 3),

            # Soil Health Card & SoilGrids
            "shc_soc_pct": soc_pct,
            "shc_ph": ph,
            "shc_nitrogen_kg_ha": float(row["shc_nitrogen_kg_ha"]),
            "shc_phosphorus_kg_ha": float(row["shc_phosphorus_kg_ha"]),
            "shc_potassium_kg_ha": float(row["shc_potassium_kg_ha"]),
            "shc_zinc_ppm": float(row["shc_zinc_ppm"]),
            "soil_clay_pct": clay,
            "permanent_water_fraction": float(row["permanent_water_fraction"]),
            "canal_length_km": canal_km,

            # Groundwater Status
            "cgwb_gw_stage_pct": gw_stage,
            "cgwb_aquifer_category": gw_cat,
            "gldas_summer_drawdown_mm": drawdown_mm,
            "methane_gw_cobenefit_index": cobenefit_idx,

            # Chamber-Calibrated IPCC Tier-2 Targets
            "tier2_daily_flux_kg_ha_day": round(float(np.mean(k_daily_flux_mc)), 3),
            "tier2_ch4_flux_kg_ha": round(k_flux_kg_ha, 2),
            "tier2_ch4_emissions_tonnes": round(k_ch4_mean_tonnes, 2),
            "ch4_low_tonnes": round(k_ch4_low_tonnes, 2),
            "ch4_high_tonnes": round(k_ch4_high_tonnes, 2),
            "baseline_co2e_tonnes": round(k_co2e_baseline, 2),

            # Verra VM0051 AWD Mitigation
            "awd_mitigation_tonnes_ch4": round(k_awd_mitigation_tonnes, 2),
            "awd_remaining_tonnes_ch4": round(k_awd_remaining_tonnes, 2),
            "awd_net_co2e_reduction_tonnes": round(k_net_awd_co2e, 2),
            "is_double_cropped": False
        })

        # ----------------------------------------------------
        # 2. RABI SUMMER (GARMA DHAN) OBSERVATION
        # ----------------------------------------------------
        r_area_ha = float(row["Rabi_Active_Rice_Area_ha"])
        r_temp_c = float(row["Rabi_LST_Celsius"])
        r_sf_w_base = float(row["Rabi_SF_w"])
        r_duration = 98.0 # Rabi summer rice is shorter cycle (95 - 105 days)

        # Extreme Summer Arrhenius Q10 = 2.2 temperature kinetics (>38°C April-May heat)
        r_temp_factor = 2.2 ** ((r_temp_c - 28.0) / 10.0)

        if r_area_ha > 10.0:
            r_sf_w_dist = np.random.normal(r_sf_w_base, 0.03, size=n_simulations)
            # ef_c_rabi_sim is 1.8x lower base, but boosted heavily by r_temp_factor (2.2^1.2 = 2.57x)
            r_daily_flux_mc = ef_c_rabi_sim * r_sf_w_dist * sf_o_sim * sf_s * r_temp_factor
            r_seasonal_flux_ha_mc = r_daily_flux_mc * r_duration
            r_total_ch4_tonnes_mc = (r_seasonal_flux_ha_mc * r_area_ha) / 1000.0

            r_ch4_mean_tonnes = float(np.mean(r_total_ch4_tonnes_mc))
            r_ch4_low_tonnes = float(np.percentile(r_total_ch4_tonnes_mc, 2.5))
            r_ch4_high_tonnes = float(np.percentile(r_total_ch4_tonnes_mc, 97.5))
            r_flux_kg_ha = float(np.mean(r_seasonal_flux_ha_mc))
        else:
            r_ch4_mean_tonnes = 0.0
            r_ch4_low_tonnes = 0.0
            r_ch4_high_tonnes = 0.0
            r_flux_kg_ha = 0.0
            r_daily_flux_mc = np.zeros(n_simulations)

        # Rabi Mitigation: Both AWD and DSR / Low-water Crop Transition
        r_awd_mitigation_tonnes = r_ch4_mean_tonnes * float(np.mean(awd_reduction_sim))
        r_awd_remaining_tonnes = r_ch4_mean_tonnes - r_awd_mitigation_tonnes
        r_co2e_baseline = r_ch4_mean_tonnes * GWP_CH4
        r_n2o_penalty = (float(np.mean(n2o_increase_sim)) * r_area_ha / 1000.0) * GWP_N2O
        r_net_awd_co2e = max(0.0, (r_awd_mitigation_tonnes * GWP_CH4) - r_n2o_penalty)

        # Scenario 3: Rabi DSR & Groundwater Conservation (76% CH4 reduction)
        r_dsr_mitigation_tonnes = r_ch4_mean_tonnes * float(np.mean(dsr_reduction_sim))
        r_dsr_net_co2e = max(0.0, (r_dsr_mitigation_tonnes * GWP_CH4) - r_n2o_penalty)

        records.append({
            "year": yr,
            "season": "Rabi",
            "observation_id": f"{lgd_code}_{yr}_RABI",
            "lgd_code": lgd_code,
            "district_name": d_name,
            "block_name": b_name,
            "aez_region": aez,
            "soil_type": soil,
            "centroid_lon": row["centroid_lon"],
            "centroid_lat": row["centroid_lat"],
            "active_rice_area_ha": r_area_ha,
            "rice_fraction": float(row["Rabi_Rice_Fraction"]),
            "cultivation_duration_days": r_duration,
            "lst_celsius": r_temp_c,
            "flood_frequency": float(row["Rabi_Flood_Frequency"]),
            "msd_rebound_db": 4.2 if r_area_ha > 10.0 else 0.0,
            "savi": float(row["Rabi_SAVI"]),
            "lswi": float(row["Rabi_LSWI"]),
            "precip_mm": 24.5, # dry rabi season
            "water_regime_sf_w": r_sf_w_base,
            "soil_factor_sf_s": round(float(sf_s), 3),
            "arrhenius_temp_multiplier": round(float(r_temp_factor), 3),

            # Soil Health Card & SoilGrids
            "shc_soc_pct": soc_pct,
            "shc_ph": ph,
            "shc_nitrogen_kg_ha": float(row["shc_nitrogen_kg_ha"]),
            "shc_phosphorus_kg_ha": float(row["shc_phosphorus_kg_ha"]),
            "shc_potassium_kg_ha": float(row["shc_potassium_kg_ha"]),
            "shc_zinc_ppm": float(row["shc_zinc_ppm"]),
            "soil_clay_pct": clay,
            "permanent_water_fraction": float(row["permanent_water_fraction"]),
            "canal_length_km": canal_km,

            # Groundwater Status
            "cgwb_gw_stage_pct": gw_stage,
            "cgwb_aquifer_category": gw_cat,
            "gldas_summer_drawdown_mm": drawdown_mm,
            "methane_gw_cobenefit_index": cobenefit_idx,

            # Chamber-Calibrated IPCC Tier-2 Targets
            "tier2_daily_flux_kg_ha_day": round(float(np.mean(r_daily_flux_mc)), 3),
            "tier2_ch4_flux_kg_ha": round(r_flux_kg_ha, 2),
            "tier2_ch4_emissions_tonnes": round(r_ch4_mean_tonnes, 2),
            "ch4_low_tonnes": round(r_ch4_low_tonnes, 2),
            "ch4_high_tonnes": round(r_ch4_high_tonnes, 2),
            "baseline_co2e_tonnes": round(r_co2e_baseline, 2),

            # Verra VM0051 AWD Mitigation
            "awd_mitigation_tonnes_ch4": round(r_awd_mitigation_tonnes, 2),
            "awd_remaining_tonnes_ch4": round(r_awd_remaining_tonnes, 2),
            "awd_net_co2e_reduction_tonnes": round(r_net_awd_co2e, 2),
            "dsr_mitigation_tonnes_ch4": round(r_dsr_mitigation_tonnes, 2),
            "dsr_net_co2e_reduction_tonnes": round(r_dsr_net_co2e, 2),
            "is_double_cropped": True if r_area_ha > 10.0 else False
        })

    df_out = pd.DataFrame(records)
    out_csv = "data/processed/chhattisgarh_dual_season_tier2_methane_panel.csv"
    df_out.to_csv(out_csv, index=False)

    print("\n[SUCCESS] Chamber-calibrated dual-season panel saved:")
    print(f"  -> {out_csv}")
    print(f"  -> Total observations: {len(df_out):,} ({df_out['season'].value_counts().to_dict()})")
    
    k_tot = df_out[(df_out["year"]==2024) & (df_out["season"]=="Kharif")]["tier2_ch4_emissions_tonnes"].sum()
    r_tot = df_out[(df_out["year"]==2024) & (df_out["season"]=="Rabi")]["tier2_ch4_emissions_tonnes"].sum()
    print(f"  -> 2024 Total Kharif Methane: {k_tot:,.1f} tonnes CH4 ({k_tot*GWP_CH4:,.1f} tCO2e)")
    print(f"  -> 2024 Total Rabi Summer Methane: {r_tot:,.1f} tonnes CH4 ({r_tot*GWP_CH4:,.1f} tCO2e)")
    print(f"  -> 2024 Total Combined Methane: {k_tot + r_tot:,.1f} tonnes CH4 ({(k_tot+r_tot)*GWP_CH4:,.1f} tCO2e)")
    return df_out

if __name__ == "__main__":
    run_dual_season_chamber_fusion()
