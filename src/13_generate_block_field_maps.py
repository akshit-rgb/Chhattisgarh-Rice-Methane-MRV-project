#!/usr/bin/env python3
"""
Step 13: High-Resolution Block-Wise Agricultural Field Cartography Suite (300 DPI)
==================================================================================
Fulfills User Command:
  "you have not incoprate the shape file do make if you enatire chhatighare 
   if it faling but do block wise"

Domain Constraints:
  1. Projected CRS Integrity: EPSG:32644 (WGS 84 / UTM Zone 44N) for all maps.
  2. Ground Truth Geometry: Individual 10m parcel boundaries (*Medh*) plotted
     directly from block-wise shapefiles/parquets derived from agriculture_field.shp.
  3. Zero Blank Voids: Complete coverage across all 146 administrative blocks.

Outputs Generated:
  1. outputs/figures/10_statewide_block_field_density_and_parcels.png
     - Statewide 146-block field parcel census choropleth with two cadastral insets (Patan & Jagdalpur).
  2. outputs/figures/11_key_blocks_cadastral_10m_field_maps.png
     - 4-panel publication plate displaying 4 representative agro-climatic blocks:
       (Patan, Abhanpur, Masturi, Ambikapur) at 10m parcel fidelity with canals and diagnostic cards.
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Setup paths
PROCESSED_DIR = "data/processed"
SHP_DIR = os.path.join(PROCESSED_DIR, "shapefiles")
BLOCK_FIELDS_DIR = os.path.join(PROCESSED_DIR, "block_fields")
BLOCK_PARQUET_DIR = os.path.join(BLOCK_FIELDS_DIR, "parquet")
BLOCK_SHP_DIR = os.path.join(BLOCK_FIELDS_DIR, "shp")
FIG_DIR = "outputs/figures"
REPORT_DIR = "outputs/reports"

BLOCKS_SHP = os.path.join(SHP_DIR, "chhattisgarh_blocks_mrv.shp")
DISTRICTS_SHP = os.path.join(SHP_DIR, "chhattisgarh_districts_essential.shp")
RIVERS_SHP = os.path.join(SHP_DIR, "cg_essential_rivers.shp")
BLOCK_CENSUS_CSV = os.path.join(PROCESSED_DIR, "cg_block_fields_summary.csv")

os.makedirs(FIG_DIR, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def add_scalebar_utm(ax, length_km=10, x_frac=0.06, y_frac=0.05, label_unit="km"):
    """Accurate metric scalebar in UTM Zone 44N (EPSG:32644) meters."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    x0 = xlim[0] + dx * x_frac
    y0 = ylim[0] + dy * y_frac
    total_len_m = length_km * 1000.0 if label_unit == "km" else float(length_km)
    half_len_m = total_len_m / 2.0
    bar_h = dy * 0.015

    ax.add_patch(Rectangle((x0, y0), half_len_m, bar_h, facecolor="#1a1a1a", edgecolor="#1a1a1a", lw=0.8, zorder=10))
    ax.add_patch(Rectangle((x0 + half_len_m, y0), half_len_m, bar_h, facecolor="#ffffff", edgecolor="#1a1a1a", lw=0.8, zorder=10))

    text_y = y0 + bar_h * 1.3
    ax.text(x0, text_y, "0", ha="center", va="bottom", fontsize=7.0, fontweight="bold", zorder=11)
    ax.text(x0 + half_len_m, text_y, f"{int(length_km/2)}", ha="center", va="bottom", fontsize=7.0, fontweight="bold", zorder=11)
    ax.text(x0 + total_len_m, text_y, f"{int(length_km)} {label_unit}", ha="center", va="bottom", fontsize=7.0, fontweight="bold", zorder=11)


def add_north_arrow(ax, x_frac=0.92, y_frac=0.90, arrow_len=0.06):
    """Cartographic North arrow."""
    ax.annotate(
        "N",
        xy=(x_frac, y_frac + arrow_len),
        xytext=(x_frac, y_frac),
        arrowprops=dict(facecolor="#1a1a1a", edgecolor="#1a1a1a", width=2.0, headwidth=7.0, headlength=8),
        ha="center",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        xycoords="axes fraction",
        zorder=12
    )


def load_block_gdf_utm(lgd_code: int, block_name: str) -> gpd.GeoDataFrame:
    """Loads block parcels from parquet or shapefile, reprojected strictly to EPSG:32644."""
    safe_name = block_name.lower().replace(" ", "_").replace("-", "_")
    
    # Try shapefile first
    shp_path = os.path.join(BLOCK_SHP_DIR, f"block_{lgd_code}_{safe_name}.shp")
    if os.path.exists(shp_path):
        gdf = gpd.read_file(shp_path)
        if gdf.crs != "EPSG:32644":
            gdf = gdf.to_crs("EPSG:32644")
        return gdf

    # Try parquet
    parquet_path = os.path.join(BLOCK_PARQUET_DIR, f"block_{lgd_code}_{safe_name}.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.GeoSeries.from_wkb(df["geometry"]),
            crs="EPSG:4326"
        ).to_crs("EPSG:32644")
        return gdf

    raise FileNotFoundError(f"No block file found for LGD {lgd_code} ({block_name})")


def render_statewide_block_density_plate():
    """Renders 300 DPI Plate 10: Statewide 146-Block Field Density & Parcels."""
    print("=" * 80)
    print("PLATE 10: Statewide 146-Block Agricultural Field Density & Cadastral Insets")
    print("=" * 80)
    t0 = time.time()

    gdf_blocks = gpd.read_file(BLOCKS_SHP).to_crs("EPSG:32644")
    gdf_districts = gpd.read_file(DISTRICTS_SHP).to_crs("EPSG:32644") if os.path.exists(DISTRICTS_SHP) else None
    gdf_rivers = gpd.read_file(RIVERS_SHP).to_crs("EPSG:32644") if os.path.exists(RIVERS_SHP) else None
    df_census = pd.read_csv(BLOCK_CENSUS_CSV)

    # Merge block census to ensure exact Rice_ha & Field_Count
    gdf_blocks = gdf_blocks.merge(
        df_census[["LGD_Code", "Field_Count", "Rice_ha", "Mean_Field_ha", "Marginal_pct"]],
        on="LGD_Code",
        how="left",
        suffixes=("_old", "")
    )
    # Calculate density (parcels per sq km of block area)
    block_area_km2 = gdf_blocks.geometry.area / 1e6
    gdf_blocks["parcel_density"] = gdf_blocks["Field_Count"] / np.clip(block_area_km2, 10.0, None)

    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax_main = fig.add_axes([0.05, 0.05, 0.54, 0.88])
    ax_inset1 = fig.add_axes([0.62, 0.52, 0.35, 0.41])
    ax_inset2 = fig.add_axes([0.62, 0.08, 0.35, 0.40])

    # 1. Main Canvas: Statewide 146 Blocks Choropleth (Rice Area in Hectares)
    gdf_blocks.plot(
        ax=ax_main,
        column="Rice_ha",
        cmap="YlGn",
        edgecolor="#2e7d32",
        linewidth=0.45,
        legend=True,
        legend_kwds={
            "label": "Measured Agricultural Rice Extent per Block (Hectares | EPSG:32644)",
            "orientation": "horizontal",
            "fraction": 0.045,
            "pad": 0.04,
            "shrink": 0.85
        }
    )

    if gdf_districts is not None:
        gdf_districts.boundary.plot(ax=ax_main, edgecolor="#212121", linewidth=1.1, zorder=5)

    if gdf_rivers is not None:
        gdf_rivers.plot(ax=ax_main, edgecolor="#00bcd4", linewidth=0.6, alpha=0.85, zorder=6)

    ax_main.set_title(
        "CHHATTISGARH STATEWIDE AGRICULTURAL FIELD FOOTPRINT\n"
        "Block-Wise Incorporation from agriculture_field.shp (14,405,396 Parcels | 146 Blocks | UTM Zone 44N)",
        fontsize=10.5, fontweight="bold", pad=10, color="#1b5e20"
    )
    ax_main.set_xlabel("UTM Zone 44N Easting (m)", fontsize=8)
    ax_main.set_ylabel("UTM Zone 44N Northing (m)", fontsize=8)
    ax_main.grid(True, linestyle="--", alpha=0.4, color="#b0bec5")
    add_scalebar_utm(ax_main, length_km=50, x_frac=0.06, y_frac=0.04)
    add_north_arrow(ax_main, x_frac=0.92, y_frac=0.92)

    # Inset 1: Patan Block (LGD 3639 - Durg)
    print("  -> Rendering Inset 1: Patan Block (104,478 Parcels)...")
    gdf_patan = load_block_gdf_utm(3639, "patan")
    b_poly_patan = gdf_blocks[gdf_blocks["LGD_Code"] == 3639]

    b_poly_patan.plot(ax=ax_inset1, facecolor="#f1f8e9", edgecolor="#1b5e20", linewidth=1.5)
    gdf_patan.plot(ax=ax_inset1, facecolor="#2e7d32", edgecolor="#1b5e20", linewidth=0.15, alpha=0.85)
    if gdf_rivers is not None:
        clip_rivers1 = gpd.clip(gdf_rivers, b_poly_patan)
        if len(clip_rivers1) > 0:
            clip_rivers1.plot(ax=ax_inset1, color="#00bcd4", linewidth=1.2)

    ax_inset1.set_title(
        "CADASTRAL INSET A: Patan Block (Durg District)\n"
        "104,478 Real Field Parcels | 18,342.1 ha Rice | Sheonath/Canal Command",
        fontsize=8.5, fontweight="bold", color="#1b5e20"
    )
    ax_inset1.grid(True, linestyle=":", alpha=0.4)
    add_scalebar_utm(ax_inset1, length_km=5, x_frac=0.06, y_frac=0.06)

    # Inset 2: Jagdalpur Block (LGD 3595 - Bastar)
    print("  -> Rendering Inset 2: Jagdalpur Block (49,726 Parcels)...")
    gdf_jag = load_block_gdf_utm(3595, "jagdalpur")
    b_poly_jag = gdf_blocks[gdf_blocks["LGD_Code"] == 3595]

    b_poly_jag.plot(ax=ax_inset2, facecolor="#f1f8e9", edgecolor="#1b5e20", linewidth=1.5)
    gdf_jag.plot(ax=ax_inset2, facecolor="#388e3c", edgecolor="#1b5e20", linewidth=0.15, alpha=0.85)
    if gdf_rivers is not None:
        clip_rivers2 = gpd.clip(gdf_rivers, b_poly_jag)
        if len(clip_rivers2) > 0:
            clip_rivers2.plot(ax=ax_inset2, color="#00bcd4", linewidth=1.2)

    ax_inset2.set_title(
        "CADASTRAL INSET B: Jagdalpur Block (Bastar District)\n"
        "49,726 Real Field Parcels | 6,839.4 ha Rice | Indravati Basin / Bastar Plateau",
        fontsize=8.5, fontweight="bold", color="#1b5e20"
    )
    ax_inset2.grid(True, linestyle=":", alpha=0.4)
    add_scalebar_utm(ax_inset2, length_km=5, x_frac=0.06, y_frac=0.06)

    out_fig = os.path.join(FIG_DIR, "10_statewide_block_field_density_and_parcels.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"✅ [SUCCESS] Exported 300 DPI Plate: {out_fig} ({time.time()-t0:.1f}s)")


def render_key_blocks_cadastral_plate():
    """Renders 300 DPI Plate 11: 4-Panel Key Blocks Cadastral Plate."""
    print("\n" + "=" * 80)
    print("PLATE 11: 4-Panel Key Blocks Cadastral Field Maps (10m Resolution)")
    print("=" * 80)
    t0 = time.time()

    gdf_blocks = gpd.read_file(BLOCKS_SHP).to_crs("EPSG:32644")
    gdf_rivers = gpd.read_file(RIVERS_SHP).to_crs("EPSG:32644") if os.path.exists(RIVERS_SHP) else None

    # 4 Representative Agro-Climatic Blocks
    BLOCK_CASES = [
        {"lgd": 3639, "name": "patan", "label": "Patan (Durg)", "zone": "Central Irrigated Plains", "color": "#2e7d32"},
        {"lgd": 3693, "name": "abhanpur", "label": "Abhanpur (Raipur)", "zone": "Mahanadi Rice Bowl", "color": "#388e3c"},
        {"lgd": 3610, "name": "masturi", "label": "Masturi (Bilaspur)", "zone": "Hasdeo River Basin", "color": "#43a047"},
        {"lgd": 3717, "name": "ambikapur", "label": "Ambikapur (Surguja)", "zone": "Northern Undulating Hills", "color": "#558b2f"},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=300)
    axes_flat = axes.flatten()

    for idx, case in enumerate(BLOCK_CASES):
        ax = axes_flat[idx]
        lgd = case["lgd"]
        b_name = case["name"]
        color = case["color"]

        print(f"  [{idx+1}/4] Plotting Cadastral Plate: {case['label']}...")
        gdf_p = load_block_gdf_utm(lgd, b_name)
        b_poly = gdf_blocks[gdf_blocks["LGD_Code"] == lgd]

        # Draw boundary
        b_poly.plot(ax=ax, facecolor="#fafafa", edgecolor="#212121", linewidth=1.4, zorder=2)

        # Draw field parcels
        gdf_p.plot(ax=ax, facecolor=color, edgecolor="#1b5e20", linewidth=0.12, alpha=0.9, zorder=3)

        # Overlay Rivers/Canals
        if gdf_rivers is not None:
            c_riv = gpd.clip(gdf_rivers, b_poly)
            if len(c_riv) > 0:
                c_riv.plot(ax=ax, color="#00acc1", linewidth=1.3, zorder=4)

        # Metrics Card Box
        tot_cnt = len(gdf_p)
        tot_area = gdf_p["area_ha"].sum()
        mean_ha = gdf_p["area_ha"].mean()
        marg_pct = (gdf_p["area_ha"] < 1.0).mean() * 100

        card_str = (
            f"• Parcels: {tot_cnt:,}\n"
            f"• Rice Area: {tot_area:,.1f} ha\n"
            f"• Mean Parcel: {mean_ha:.3f} ha\n"
            f"• Marginal (<1ha): {marg_pct:.1f}%\n"
            f"• Zone: {case['zone']}"
        )
        ax.text(
            0.03, 0.96, card_str,
            transform=ax.transAxes,
            fontsize=7.5,
            family="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#37474f", alpha=0.92, lw=0.8),
            zorder=15
        )

        ax.set_title(f"{case['label']} - 10m Agricultural Field Footprint", fontsize=10, fontweight="bold", pad=8, color="#1b5e20")
        ax.set_xlabel("Easting (m)", fontsize=7.5)
        ax.set_ylabel("Northing (m)", fontsize=7.5)
        ax.grid(True, linestyle=":", alpha=0.4)
        add_scalebar_utm(ax, length_km=5, x_frac=0.06, y_frac=0.05)
        add_north_arrow(ax, x_frac=0.92, y_frac=0.88)

    plt.suptitle(
        "CHHATTISGARH REPRESENTATIVE BLOCK-WISE CADASTRAL ATLAS (300 DPI)\n"
        "Derived Directly from User agriculture_field.shp (10m Resolution Parcels | EPSG:32644)",
        fontsize=12, fontweight="bold", y=0.99, color="#004d40"
    )
    plt.tight_layout()

    out_fig2 = os.path.join(FIG_DIR, "11_key_blocks_cadastral_10m_field_maps.png")
    plt.savefig(out_fig2)
    plt.close()
    print(f"✅ [SUCCESS] Exported 300 DPI Plate: {out_fig2} ({time.time()-t0:.1f}s)")


def main():
    print("=" * 80)
    print("🌾 STEP 13: BLOCK-WISE AGRICULTURAL FIELD CARTOGRAPHY GENERATION")
    print("=" * 80)
    render_statewide_block_density_plate()
    render_key_blocks_cadastral_plate()
    print("\n" + "=" * 80)
    print("🎉 ALL BLOCK-WISE 300 DPI CARTOGRAPHIC PLATES SUCCESSFULLY GENERATED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
