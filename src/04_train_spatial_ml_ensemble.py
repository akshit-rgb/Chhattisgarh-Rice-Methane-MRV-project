#!/usr/bin/env python3
"""
Step 04: Spatial Cross-Validation and ML Ensemble Training (RF + XGBoost)
-------------------------------------------------------------------------
Trains a spatial machine learning ensemble (Random Forest + XGBoost Regressor)
to predict block-level rice methane emissions from satellite SAR, optical,
climate, soil, and topography features.
Enforces:
  1. Spatial GroupKFold CV grouped by district (tests spatial generalizability).
  2. Out-of-Time Temporal Validation on 2023–2024 test horizon.
Outputs model weights, performance tables, and validation prediction datasets.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

FEATURE_COLS = [
    # Sentinel-1 SAR
    "s1_vv_db", "s1_vh_db", "s1_vh_vv_ratio_db",
    # Sentinel-2 MSI
    "s2_ndvi_mean", "s2_ndvi_max", "s2_evi_mean", "s2_lswi_mean",
    # Inferred Inundation
    "seasonal_inundation_days", "inundation_frequency",
    # Climate
    "chirps_rainfall_mm", "heavy_rain_days", "era5_temp_c",
    # Soil & Topography
    "soil_organic_carbon_g_kg", "soil_clay_pct", "soil_ph",
    "elevation_m", "slope_deg",
    # Cultivation Extent
    "rice_area_ha", "rice_fraction"
]

TARGET_COL = "rice_ch4_emissions_tonnes"

def run_spatial_cross_validation(df_train, n_splits=5):
    """
    Performs Spatial GroupKFold cross-validation grouped strictly by district_name.
    Prevents spatial autocorrelation leakage between train and test folds.
    """
    print(f"\n[INFO] Running {n_splits}-Fold Spatial GroupKFold Cross-Validation (grouped by district)...")
    gkf = GroupKFold(n_splits=n_splits)

    groups = df_train["district_name"].values
    X = df_train[FEATURE_COLS].values
    y = df_train[TARGET_COL].values

    rf_r2_scores, rf_rmse_scores = [], []
    xgb_r2_scores, xgb_rmse_scores = [], []
    ens_r2_scores, ens_rmse_scores = [], []

    fold_predictions = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        val_df = df_train.iloc[val_idx].copy()

        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        pred_rf = rf.predict(X_val)

        # Train XGBoost
        model_xgb = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.06, subsample=0.85, random_state=42, n_jobs=-1)
        model_xgb.fit(X_tr, y_tr)
        pred_xgb = model_xgb.predict(X_val)

        # 50/50 Ensemble
        pred_ens = 0.5 * pred_rf + 0.5 * pred_xgb

        # Evaluate Fold
        r2_rf = r2_score(y_val, pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_val, pred_rf))
        r2_xgb = r2_score(y_val, pred_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_val, pred_xgb))
        r2_ens = r2_score(y_val, pred_ens)
        rmse_ens = np.sqrt(mean_squared_error(y_val, pred_ens))

        rf_r2_scores.append(r2_rf)
        rf_rmse_scores.append(rmse_rf)
        xgb_r2_scores.append(r2_xgb)
        xgb_rmse_scores.append(rmse_xgb)
        ens_r2_scores.append(r2_ens)
        ens_rmse_scores.append(rmse_ens)

        val_df["cv_fold"] = fold
        val_df["pred_rf_tonnes"] = pred_rf
        val_df["pred_xgb_tonnes"] = pred_xgb
        val_df["pred_ensemble_tonnes"] = pred_ens
        fold_predictions.append(val_df)

        print(f"  Fold {fold}: Districts={val_df['district_name'].nunique()} | RF R2={r2_rf:.3f} | XGB R2={r2_xgb:.3f} | Ensemble R2={r2_ens:.3f} (RMSE={rmse_ens:,.1f} t)")

    mean_rf_r2 = np.mean(rf_r2_scores)
    mean_xgb_r2 = np.mean(xgb_r2_scores)
    mean_ens_r2 = np.mean(ens_r2_scores)
    mean_ens_rmse = np.mean(ens_rmse_scores)

    print(f"\n[SPATIAL CV RESULT] Ensemble Mean Spatial R2 = {mean_ens_r2:.4f} | Mean Spatial RMSE = {mean_ens_rmse:,.2f} tonnes CH4")

    df_cv_preds = pd.concat(fold_predictions, axis=0)
    return {
        "mean_rf_r2": mean_rf_r2,
        "mean_xgb_r2": mean_xgb_r2,
        "mean_ens_r2": mean_ens_r2,
        "mean_ens_rmse": mean_ens_rmse,
        "cv_predictions": df_cv_preds
    }

def run_out_of_time_validation(df):
    """
    Evaluates Out-of-Time generalization:
    - Train set: 2019–2022 (584 observations)
    - Test / Validation set: 2023–2024 (292 observations)
    - Forecasting horizon: 2025–2026 (292 observations)
    """
    print("\n[INFO] Running Out-of-Time Temporal Validation (Train: 2019-2022 | Test: 2023-2024)...")
    train_mask = (df["year"] >= 2019) & (df["year"] <= 2022)
    test_mask = (df["year"] >= 2023) & (df["year"] <= 2024)
    forecast_mask = (df["year"] >= 2025) & (df["year"] <= 2026)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    df_forecast = df[forecast_mask].copy()

    X_train, y_train = df_train[FEATURE_COLS], df_train[TARGET_COL]
    X_test, y_test = df_test[FEATURE_COLS], df_test[TARGET_COL]
    X_fore = df_forecast[FEATURE_COLS]

    # Fit final models on 2019-2022
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    model_xgb = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
    model_xgb.fit(X_train, y_train)

    # Test predictions
    pred_test_rf = rf.predict(X_test)
    pred_test_xgb = model_xgb.predict(X_test)
    pred_test_ens = 0.5 * pred_test_rf + 0.5 * pred_test_xgb

    oot_r2_rf = r2_score(y_test, pred_test_rf)
    oot_r2_xgb = r2_score(y_test, pred_test_xgb)
    oot_r2_ens = r2_score(y_test, pred_test_ens)
    oot_rmse_ens = np.sqrt(mean_squared_error(y_test, pred_test_ens))
    oot_mae_ens = mean_absolute_error(y_test, pred_test_ens)
    oot_mape_ens = np.mean(np.abs((y_test - pred_test_ens) / y_test)) * 100.0

    print(f"[OOT TEST 2023-2024] Ensemble Out-of-Time R2   = {oot_r2_ens:.4f}")
    print(f"[OOT TEST 2023-2024] Ensemble Out-of-Time RMSE = {oot_rmse_ens:,.2f} tonnes CH4")
    print(f"[OOT TEST 2023-2024] Ensemble Out-of-Time MAE  = {oot_mae_ens:,.2f} tonnes CH4")
    print(f"[OOT TEST 2023-2024] Ensemble Out-of-Time MAPE = {oot_mape_ens:.2f}%")

    df_test["pred_rf_tonnes"] = pred_test_rf
    df_test["pred_xgb_tonnes"] = pred_test_xgb
    df_test["pred_ensemble_tonnes"] = pred_test_ens

    # Forecast on 2025-2026
    pred_fore_ens = 0.5 * rf.predict(X_fore) + 0.5 * model_xgb.predict(X_fore)
    df_forecast["pred_ensemble_tonnes"] = pred_fore_ens

    # Save trained models
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/random_forest_mrv.joblib")
    model_xgb.save_model("models/xgboost_mrv.json")
    print("[SAVED] models/random_forest_mrv.joblib")
    print("[SAVED] models/xgboost_mrv.json")

    # Feature Importances
    fi_rf = rf.feature_importances_
    fi_xgb = model_xgb.feature_importances_
    fi_ens = 0.5 * fi_rf + 0.5 * fi_xgb

    df_fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance_random_forest": fi_rf,
        "importance_xgboost": fi_xgb,
        "importance_ensemble": fi_ens
    }).sort_values("importance_ensemble", ascending=False).reset_index(drop=True)

    return {
        "oot_r2_ens": oot_r2_ens,
        "oot_rmse_ens": oot_rmse_ens,
        "oot_mae_ens": oot_mae_ens,
        "oot_mape_ens": oot_mape_ens,
        "df_test": df_test,
        "df_forecast": df_forecast,
        "df_fi": df_fi
    }

def main():
    print("=" * 70)
    print("STEP 04: Spatial ML Ensemble Training & Validation")
    print("=" * 70)

    fused_path = "data/processed/chhattisgarh_block_tier2_methane_panel.csv"
    if not os.path.exists(fused_path):
        raise FileNotFoundError(f"Missing fused dataset: {fused_path}. Please run step 03 first.")

    df = pd.read_csv(fused_path)
    print(f"[OK] Loaded {len(df)} block-year records.")

    # 1. Spatial Cross-Validation on Historical (2019-2022)
    df_hist = df[(df["year"] >= 2019) & (df["year"] <= 2022)].copy()
    cv_res = run_spatial_cross_validation(df_hist, n_splits=5)

    # 2. Out-of-Time Validation (2023-2024) and Forecasting (2025-2026)
    oot_res = run_out_of_time_validation(df)

    # 3. Save Validation Predictions Table
    df_all_preds = pd.concat([
        cv_res["cv_predictions"],
        oot_res["df_test"],
        oot_res["df_forecast"]
    ], ignore_index=True)
    pred_path = "data/processed/spatial_ml_validation_predictions.csv"
    df_all_preds.to_csv(pred_path, index=False)
    print(f"[SAVED] {pred_path}")

    # 4. Save Feature Importance Table
    fi_path = "data/processed/feature_importances.csv"
    oot_res["df_fi"].to_csv(fi_path, index=False)
    print(f"[SAVED] {fi_path}")

    # 5. Save Model Performance Metrics Table
    metrics_records = [
        {"validation_scheme": "5-Fold Spatial GroupKFold CV (by District)", "r2_score": round(cv_res["mean_ens_r2"], 4), "rmse_tonnes": round(cv_res["mean_ens_rmse"], 2), "description": "Tests spatial generalizability to unseen geographical districts"},
        {"validation_scheme": "Out-of-Time Horizon (2023-2024 Test Set)", "r2_score": round(oot_res["oot_r2_ens"], 4), "rmse_tonnes": round(oot_res["oot_rmse_ens"], 2), "description": "Trained on 2019-2022, tested strictly on future 2023-2024 observations"},
        {"validation_scheme": "Out-of-Time MAE", "r2_score": None, "rmse_tonnes": round(oot_res["oot_mae_ens"], 2), "description": "Mean Absolute Error in tonnes CH4 on 2023-2024"},
        {"validation_scheme": "Out-of-Time MAPE (%)", "r2_score": None, "rmse_tonnes": round(oot_res["oot_mape_ens"], 2), "description": "Mean Absolute Percentage Error on test horizon"}
    ]
    df_metrics = pd.DataFrame(metrics_records)
    metrics_path = "data/processed/model_performance_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"[SAVED] {metrics_path}")

    print("\n--- Model Performance Summary Table ---")
    print(df_metrics.to_string(index=False))

    print("\n--- Top 7 Biophysical & Remote Sensing Drivers ---")
    print(oot_res["df_fi"].head(7)[["feature", "importance_ensemble"]].to_string(index=False))

    print("\n[SUCCESS] Step 04 complete.\n")

if __name__ == "__main__":
    main()
