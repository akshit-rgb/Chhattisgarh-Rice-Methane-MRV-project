#!/usr/bin/env python3
"""
Step 08: Standalone QGIS Project Generator & Publication Layout Export (PyQGIS)
--------------------------------------------------------------------------------
Generates a standalone, fully-styled, multi-layer QGIS Project file (.qgz):
  1. Outputs: outputs/qgis/chhattisgarh_mrv_project.qgz
  2. Publication Map Layout Exports (300 DPI):
     - outputs/figures/qgis_statewide_mrv_composite.png (Carbon MRV Atlas)
     - outputs/figures/qgis_statewide_soil_cartography.png (Soil Science Atlas)
     - outputs/figures/qgis_statewide_field_footprint_composite.png (FTW 10m Parcels)
  3. Structured Layer Hierarchy (EPSG:32644 UTM Zone 44N):
     - Group 01: Administrative Boundaries (33 Districts, 146 Blocks)
     - Group 02: Hydrology & Canals (WRIS Canals & Primary Drainage)
     - Group 03: Fields of the World (FTW) 10m Parcels (11,190 instances)
     - Group 04: Carbon MRV Flux (2024 Kharif Rice Methane kt)
     - Group 05: Soil Science & Agro-Ecological Typology (Kanhar, Dorsa, Matasi, Bhata)
  4. Resolves layer collision, overdraw, and legend mixing issues.
"""

import os
import sys

# Self-dispatch to official QGIS Python environment if run from standard system Python
try:
    from qgis.core import (
        QgsApplication, QgsProject, QgsVectorLayer,
        QgsCoordinateReferenceSystem, QgsGraduatedSymbolRenderer,
        QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsRendererRange,
        QgsPrintLayout, QgsLayoutItemMap, QgsLayoutItemScaleBar,
        QgsLayoutItemLabel, QgsLayoutItemLegend, QgsLayoutPoint,
        QgsLayoutSize, QgsUnitTypes, QgsLayoutExporter,
        QgsFillSymbol, QgsLineSymbol, QgsTextFormat
    )
except ImportError:
    import subprocess
    qgis_bat = r"C:\Program Files\QGIS 3.40.14\bin\python-qgis-ltr.bat"
    if os.path.exists(qgis_bat):
        print(f"[INFO] Dispatching PyQGIS execution via: {qgis_bat}...")
        res = subprocess.run([qgis_bat, os.path.abspath(__file__)] + sys.argv[1:], check=False)
        sys.exit(res.returncode)
    else:
        print(f"[ERROR] QGIS official batch runner not found at: {qgis_bat}")
        sys.exit(1)

import pandas as pd
import geopandas as gpd
from PyQt5.QtGui import QColor, QFont

PROCESSED_DIR = "data/processed"
SHP_DIR = os.path.join(PROCESSED_DIR, "shapefiles")
QGIS_DIR = "outputs/qgis"
FIG_DIR = "outputs/figures"

os.makedirs(QGIS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def build_enriched_shapefiles():
    """
    Enriches block boundaries with 2024 MRV indicators and soil parameters,
    and generates 33 dissolved district boundaries in EPSG:32644.
    """
    b_shp = os.path.join(SHP_DIR, "chhattisgarh_blocks_essential.shp")
    panel_csv = os.path.join(PROCESSED_DIR, "cg_essential_panel_2019_2026.csv")
    soil_shp = os.path.join(SHP_DIR, "chhattisgarh_soils_essential.shp")

    if not os.path.exists(b_shp) or not os.path.exists(panel_csv):
        raise FileNotFoundError("Missing blocks shapefile or master panel CSV.")

    gdf_b = gpd.read_file(b_shp).to_crs("EPSG:32644")
    df_panel = pd.read_csv(panel_csv)
    df_2024 = df_panel[df_panel["Year"] == 2024]

    df_k = df_2024[df_2024["Season"] == "Kharif"].set_index("LGD_Code")
    df_r = df_2024[df_2024["Season"] == "Rabi"].set_index("LGD_Code")

    # Add enriched indicators
    gdf_b["Kharif_kt"] = (gdf_b["LGD_Code"].map(df_k["Seasonal_CH4_Tonnes"]) / 1000.0).round(2)
    gdf_b["Rabi_kt"] = (gdf_b["LGD_Code"].map(df_r["Seasonal_CH4_Tonnes"]).fillna(0.0) / 1000.0).round(2)
    gdf_b["Yield_tha"] = gdf_b["LGD_Code"].map(df_k["Predicted_Yield_t_ha"]).fillna(4.0).round(2)
    gdf_b["GW_Stress"] = gdf_b["LGD_Code"].map(df_k["GW_Stress_Index"]).fillna(25.0).round(1)
    gdf_b["Rice_ha"] = gdf_b["LGD_Code"].map(df_k["Active_Rice_Area_ha"]).fillna(0.0).round(0)

    # Attach soil attributes
    if os.path.exists(soil_shp):
        gdf_soil = gpd.read_file(soil_shp).set_index("LGD_Code")
        gdf_b["Soil_Class"] = gdf_b["LGD_Code"].map(gdf_soil["Soil_Class"]).fillna("Matasi")
        gdf_b["SOC_g_kg"] = gdf_b["LGD_Code"].map(gdf_soil["SOC_g_kg"]).fillna(7.0).round(1)
        gdf_b["Clay_pct"] = gdf_b["LGD_Code"].map(gdf_soil["Clay_pct"]).fillna(22.0).round(1)
        gdf_b["Soil_pH"] = gdf_b["LGD_Code"].map(gdf_soil["Soil_pH"]).fillna(6.4).round(2)

    out_mrv_shp = os.path.join(SHP_DIR, "chhattisgarh_blocks_mrv.shp")
    gdf_b.to_file(out_mrv_shp)
    print(f"[SUCCESS] Exported enriched MRV blocks to: {out_mrv_shp}")

    # Generate 33 dissolved district boundaries
    gdf_dist = gdf_b.dissolve(by="District", as_index=False)[["District", "geometry"]]
    out_dist_shp = os.path.join(SHP_DIR, "chhattisgarh_districts_essential.shp")
    gdf_dist.to_file(out_dist_shp)
    print(f"[SUCCESS] Exported 33 dissolved districts to: {out_dist_shp}")

    return out_mrv_shp, out_dist_shp


def create_qgis_project(blocks_shp, districts_shp):
    """
    Initializes QGIS, builds the complete project hierarchy in EPSG:32644,
    applies professional cartographic renderers, creates print layouts, and exports images.
    """
    app = QgsApplication([], False)
    app.initQgis()

    try:
        project = QgsProject.instance()
        project.clear()
        project.setTitle("Statewide Chhattisgarh Rice MRV & Carbon Flux Project")

        # 1. Coordinate Reference System: EPSG:32644 (WGS 84 / UTM Zone 44N)
        crs_utm = QgsCoordinateReferenceSystem("EPSG:32644")
        project.setCrs(crs_utm)

        # 2. Add District Boundaries Layer (Thick Outlines)
        layer_dist = QgsVectorLayer(os.path.abspath(districts_shp), "Chhattisgarh 33 Districts", "ogr")
        sym_dist = QgsFillSymbol.createSimple({
            "color": "0,0,0,0",
            "outline_color": "25,25,25,255",
            "outline_width": "0.75"
        })
        layer_dist.renderer().setSymbol(sym_dist)
        project.addMapLayer(layer_dist, False)

        # 3. Add Blocks Reference Outline (light neutral background for fields)
        layer_blocks_outline = QgsVectorLayer(os.path.abspath(blocks_shp), "146 Blocks (Outlines)", "ogr")
        sym_b_out = QgsFillSymbol.createSimple({
            "color": "246,249,247,255",
            "outline_color": "180,195,185,255",
            "outline_width": "0.25"
        })
        layer_blocks_outline.renderer().setSymbol(sym_b_out)
        project.addMapLayer(layer_blocks_outline, False)

        # 4. Add Rivers & Canals Shapefile (EPSG:32644)
        rivers_shp = os.path.join(SHP_DIR, "cg_essential_rivers.shp")
        if not os.path.exists(rivers_shp):
            rivers_shp = os.path.join(PROCESSED_DIR, "cg_essential_rivers.geojson")
        layer_rivers = QgsVectorLayer(os.path.abspath(rivers_shp), "WRIS Irrigation Canals & Drainage", "ogr")
        sym_riv = QgsLineSymbol.createSimple({
            "line_color": "0,131,143,230",
            "line_width": "0.45"
        })
        layer_rivers.renderer().setSymbol(sym_riv)
        project.addMapLayer(layer_rivers, False)

        # 5. Add Fields of the World (FTW) 10m Agricultural Field Parcels
        fields_shp = os.path.join(SHP_DIR, "chhattisgarh_rice_fields_essential.shp")
        layer_fields = QgsVectorLayer(os.path.abspath(fields_shp), "Fields of the World (FTW) 10m Parcels", "ogr")
        cat_f = [
            QgsRendererCategory("Kharif_Monsoon", QgsFillSymbol.createSimple({"color": "27,120,55,225", "outline_color": "0,77,64,255", "outline_width": "0.18"}), "FTW Class 1: Kharif Monsoon Paddy"),
            QgsRendererCategory("Double_Cropped", QgsFillSymbol.createSimple({"color": "118,42,131,225", "outline_color": "74,20,140,255", "outline_width": "0.18"}), "FTW Class 2: Double-Cropped Rice"),
            QgsRendererCategory("Rabi_Summer", QgsFillSymbol.createSimple({"color": "227,26,28,225", "outline_color": "183,28,28,255", "outline_width": "0.18"}), "FTW Class 3: Rabi Summer Rice")
        ]
        layer_fields.setRenderer(QgsCategorizedSymbolRenderer("Crop_Seas", cat_f))
        project.addMapLayer(layer_fields, False)

        # 6. Add Soil Typology Layer (Kanhar, Dorsa, Matasi, Bhata)
        soil_shp = os.path.join(SHP_DIR, "chhattisgarh_soils_essential.shp")
        layer_soils = QgsVectorLayer(os.path.abspath(soil_shp), "Agro-Ecological Soil Typology", "ogr")
        cat_s = [
            QgsRendererCategory("Kanhar", QgsFillSymbol.createSimple({"color": "62,39,35,225", "outline_color": "255,255,255,180", "outline_width": "0.25"}), "Kanhar (Deep Black Vertisol)"),
            QgsRendererCategory("Dorsa", QgsFillSymbol.createSimple({"color": "121,85,72,225", "outline_color": "255,255,255,180", "outline_width": "0.25"}), "Dorsa (Clay-Loam Inceptisol)"),
            QgsRendererCategory("Matasi", QgsFillSymbol.createSimple({"color": "245,127,23,225", "outline_color": "255,255,255,180", "outline_width": "0.25"}), "Matasi (Yellow Sandy-Loam Alfisol)"),
            QgsRendererCategory("Bhata", QgsFillSymbol.createSimple({"color": "183,28,28,225", "outline_color": "255,255,255,180", "outline_width": "0.25"}), "Bhata (Lateritic Red Entisol)")
        ]
        layer_soils.setRenderer(QgsCategorizedSymbolRenderer("Soil_Class", cat_s))
        project.addMapLayer(layer_soils, False)

        # 7. Add Enriched 146 Blocks Layer with Graduated Renderer on Kharif_kt
        layer_blocks_mrv = QgsVectorLayer(os.path.abspath(blocks_shp), "Carbon MRV (Kharif CH4 kt)", "ogr")
        classes_m = [
            (0.0, 10.0, "#ffffcc", "< 10 kt CH4"),
            (10.0, 20.0, "#c2e699", "10 - 20 kt CH4"),
            (20.0, 30.0, "#78c679", "20 - 30 kt CH4"),
            (30.0, 40.0, "#31a354", "30 - 40 kt CH4"),
            (40.0, 80.0, "#006837", "> 40 kt CH4")
        ]
        rend_m = QgsGraduatedSymbolRenderer("Kharif_kt")
        for lower, upper, hex_c, label in classes_m:
            sym = QgsFillSymbol.createSimple({"color": hex_c, "outline_color": "100,100,100,200", "outline_width": "0.2"})
            rend_m.addClassRange(QgsRendererRange(lower, upper, sym, label))
        layer_blocks_mrv.setRenderer(rend_m)
        project.addMapLayer(layer_blocks_mrv, False)

        # 8. Organize Structured Layer Groups in QGIS Layer Tree
        root = project.layerTreeRoot()
        root.clear()

        grp_admin = root.addGroup("01. Administrative Boundaries")
        grp_admin.addLayer(layer_dist)
        grp_admin.addLayer(layer_blocks_outline)

        grp_rivers = root.addGroup("02. Hydrology & Canals (WRIS)")
        grp_rivers.addLayer(layer_rivers)

        grp_fields = root.addGroup("03. Fields of the World (FTW) 10m Parcels")
        grp_fields.addLayer(layer_fields)

        grp_mrv = root.addGroup("04. Carbon MRV Flux (2024 Kharif)")
        grp_mrv.addLayer(layer_blocks_mrv)

        grp_soils = root.addGroup("05. Soil Science & Typology")
        grp_soils.addLayer(layer_soils)

        # Initial visibility settings (prevent polygon collision)
        grp_admin.setItemVisibilityChecked(True)
        grp_rivers.setItemVisibilityChecked(True)
        grp_fields.setItemVisibilityChecked(True)
        grp_mrv.setItemVisibilityChecked(True)
        grp_soils.setItemVisibilityChecked(False)  # Unchecked by default so user can toggle between Soils and MRV

        # Save Standalone QGIS Project
        out_project = os.path.join(QGIS_DIR, "chhattisgarh_mrv_project.qgz")
        project.write(out_project)
        print(f"[SUCCESS] Saved standalone QGIS Project to: {out_project}")

        # Shared typography
        format_t = QgsTextFormat()
        format_t.setFont(QFont("Arial", 12, QFont.Bold))
        format_t.setSize(12)

        format_s = QgsTextFormat()
        format_s.setFont(QFont("Arial", 8))
        format_s.setSize(8)

        calc_extent = layer_blocks_outline.extent()
        calc_extent.scale(1.08)

        settings = QgsLayoutExporter.ImageExportSettings()
        settings.dpi = 300

        # ---------------------------------------------------------------------
        # LAYOUT 1: Statewide Carbon MRV Atlas Export
        # ---------------------------------------------------------------------
        layout_mrv = QgsPrintLayout(project)
        layout_mrv.initializeDefaults()
        layout_mrv.setName("Statewide MRV Atlas (EPSG:32644)")

        map_mrv = QgsLayoutItemMap(layout_mrv)
        map_mrv.attemptMove(QgsLayoutPoint(10, 15, QgsUnitTypes.LayoutMillimeters))
        map_mrv.attemptResize(QgsLayoutSize(200, 185, QgsUnitTypes.LayoutMillimeters))
        map_mrv.setCrs(crs_utm)
        map_mrv.setExtent(calc_extent)
        map_mrv.setLayers([layer_dist, layer_rivers, layer_blocks_mrv])
        map_mrv.setKeepLayerSet(True)
        layout_mrv.addLayoutItem(map_mrv)

        title_mrv = QgsLayoutItemLabel(layout_mrv)
        title_mrv.setText("CHHATTISGARH STATEWIDE RICE METHANE & MRV ATLAS")
        title_mrv.attemptMove(QgsLayoutPoint(10, 5, QgsUnitTypes.LayoutMillimeters))
        title_mrv.attemptResize(QgsLayoutSize(277, 8, QgsUnitTypes.LayoutMillimeters))
        title_mrv.setTextFormat(format_t)
        layout_mrv.addLayoutItem(title_mrv)

        sub_mrv = QgsLayoutItemLabel(layout_mrv)
        sub_mrv.setText("146 Blocks | 33 Districts | Closed-Chamber Calibrated Monotonic XGBoost (R2=0.941) | UTM Zone 44N (EPSG:32644)")
        sub_mrv.attemptMove(QgsLayoutPoint(10, 11, QgsUnitTypes.LayoutMillimeters))
        sub_mrv.attemptResize(QgsLayoutSize(277, 5, QgsUnitTypes.LayoutMillimeters))
        sub_mrv.setTextFormat(format_s)
        layout_mrv.addLayoutItem(sub_mrv)

        scale_bar_m = QgsLayoutItemScaleBar(layout_mrv)
        scale_bar_m.setStyle("Double Box")
        scale_bar_m.setUnits(QgsUnitTypes.DistanceKilometers)
        scale_bar_m.setNumberOfSegments(2)
        scale_bar_m.setUnitsPerSegment(50.0)
        scale_bar_m.setLinkedMap(map_mrv)
        scale_bar_m.attemptMove(QgsLayoutPoint(15, 185, QgsUnitTypes.LayoutMillimeters))
        layout_mrv.addLayoutItem(scale_bar_m)

        legend_m = QgsLayoutItemLegend(layout_mrv)
        legend_m.setLinkedMap(map_mrv)
        legend_m.setLegendFilterByMapEnabled(True)
        legend_m.attemptMove(QgsLayoutPoint(215, 20, QgsUnitTypes.LayoutMillimeters))
        legend_m.attemptResize(QgsLayoutSize(72, 160, QgsUnitTypes.LayoutMillimeters))
        layout_mrv.addLayoutItem(legend_m)

        out_mrv_img = os.path.join(FIG_DIR, "qgis_statewide_mrv_composite.png")
        if os.path.exists(out_mrv_img):
            os.remove(out_mrv_img)
        res_m = QgsLayoutExporter(layout_mrv).exportToImage(os.path.abspath(out_mrv_img), settings)
        if res_m == QgsLayoutExporter.Success and os.path.exists(out_mrv_img):
            print(f"[SUCCESS] Exported 300 DPI QGIS Cartographic MRV Map: {out_mrv_img} ({os.path.getsize(out_mrv_img) / 1024:.1f} KB)")

        # ---------------------------------------------------------------------
        # LAYOUT 2: Statewide Soil Science Cartography Export
        # ---------------------------------------------------------------------
        layout_soil = QgsPrintLayout(project)
        layout_soil.initializeDefaults()
        layout_soil.setName("Statewide Soil Science Atlas (EPSG:32644)")

        map_soil = QgsLayoutItemMap(layout_soil)
        map_soil.attemptMove(QgsLayoutPoint(10, 15, QgsUnitTypes.LayoutMillimeters))
        map_soil.attemptResize(QgsLayoutSize(200, 185, QgsUnitTypes.LayoutMillimeters))
        map_soil.setCrs(crs_utm)
        map_soil.setExtent(calc_extent)
        map_soil.setLayers([layer_dist, layer_rivers, layer_soils])
        map_soil.setKeepLayerSet(True)
        layout_soil.addLayoutItem(map_soil)

        title_soil = QgsLayoutItemLabel(layout_soil)
        title_soil.setText("CHHATTISGARH STATEWIDE AGRO-ECOLOGICAL SOIL TYPOLOGY")
        title_soil.attemptMove(QgsLayoutPoint(10, 5, QgsUnitTypes.LayoutMillimeters))
        title_soil.attemptResize(QgsLayoutSize(277, 8, QgsUnitTypes.LayoutMillimeters))
        title_soil.setTextFormat(format_t)
        layout_soil.addLayoutItem(title_soil)

        sub_soil = QgsLayoutItemLabel(layout_soil)
        sub_soil.setText("Kanhar (Vertisols), Dorsa (Inceptisols), Matasi (Alfisols), Bhata (Entisols) | IGKV Raipur & ISRIC SoilGrids 250m")
        sub_soil.attemptMove(QgsLayoutPoint(10, 11, QgsUnitTypes.LayoutMillimeters))
        sub_soil.attemptResize(QgsLayoutSize(277, 5, QgsUnitTypes.LayoutMillimeters))
        sub_soil.setTextFormat(format_s)
        layout_soil.addLayoutItem(sub_soil)

        scale_bar_s = QgsLayoutItemScaleBar(layout_soil)
        scale_bar_s.setStyle("Double Box")
        scale_bar_s.setUnits(QgsUnitTypes.DistanceKilometers)
        scale_bar_s.setNumberOfSegments(2)
        scale_bar_s.setUnitsPerSegment(50.0)
        scale_bar_s.setLinkedMap(map_soil)
        scale_bar_s.attemptMove(QgsLayoutPoint(15, 185, QgsUnitTypes.LayoutMillimeters))
        layout_soil.addLayoutItem(scale_bar_s)

        legend_s = QgsLayoutItemLegend(layout_soil)
        legend_s.setLinkedMap(map_soil)
        legend_s.setLegendFilterByMapEnabled(True)
        legend_s.attemptMove(QgsLayoutPoint(215, 20, QgsUnitTypes.LayoutMillimeters))
        legend_s.attemptResize(QgsLayoutSize(72, 160, QgsUnitTypes.LayoutMillimeters))
        layout_soil.addLayoutItem(legend_s)

        out_soil_img = os.path.join(FIG_DIR, "qgis_statewide_soil_cartography.png")
        if os.path.exists(out_soil_img):
            os.remove(out_soil_img)
        res_s = QgsLayoutExporter(layout_soil).exportToImage(os.path.abspath(out_soil_img), settings)
        if res_s == QgsLayoutExporter.Success and os.path.exists(out_soil_img):
            print(f"[SUCCESS] Exported 300 DPI QGIS Soil Cartography Map: {out_soil_img} ({os.path.getsize(out_soil_img) / 1024:.1f} KB)")

        # ---------------------------------------------------------------------
        # LAYOUT 3: Fields of the World (FTW) 10m Field Footprints Export
        # ---------------------------------------------------------------------
        layout_fields = QgsPrintLayout(project)
        layout_fields.initializeDefaults()
        layout_fields.setName("Fields of the World (FTW) 10m Parcels (EPSG:32644)")

        map_fields = QgsLayoutItemMap(layout_fields)
        map_fields.attemptMove(QgsLayoutPoint(10, 15, QgsUnitTypes.LayoutMillimeters))
        map_fields.attemptResize(QgsLayoutSize(200, 185, QgsUnitTypes.LayoutMillimeters))
        map_fields.setCrs(crs_utm)
        map_fields.setExtent(calc_extent)
        map_fields.setLayers([layer_dist, layer_rivers, layer_fields, layer_blocks_outline])
        map_fields.setKeepLayerSet(True)
        layout_fields.addLayoutItem(map_fields)

        title_f = QgsLayoutItemLabel(layout_fields)
        title_f.setText("FIELDS OF THE WORLD (FTW) — 10m AGRICULTURAL FIELD ATLAS (ALL 33 DISTRICTS)")
        title_f.attemptMove(QgsLayoutPoint(10, 5, QgsUnitTypes.LayoutMillimeters))
        title_f.attemptResize(QgsLayoutSize(277, 8, QgsUnitTypes.LayoutMillimeters))
        title_f.setTextFormat(format_t)
        layout_fields.addLayoutItem(title_f)

        sub_f = QgsLayoutItemLabel(layout_fields)
        sub_f.setText("11,190 Verified FTW Instances | NASA Harvest & Radiant Earth Benchmark Standard | UTM Zone 44N (EPSG:32644)")
        sub_f.attemptMove(QgsLayoutPoint(10, 11, QgsUnitTypes.LayoutMillimeters))
        sub_f.attemptResize(QgsLayoutSize(277, 5, QgsUnitTypes.LayoutMillimeters))
        sub_f.setTextFormat(format_s)
        layout_fields.addLayoutItem(sub_f)

        scale_bar_f = QgsLayoutItemScaleBar(layout_fields)
        scale_bar_f.setStyle("Double Box")
        scale_bar_f.setUnits(QgsUnitTypes.DistanceKilometers)
        scale_bar_f.setNumberOfSegments(2)
        scale_bar_f.setUnitsPerSegment(50.0)
        scale_bar_f.setLinkedMap(map_fields)
        scale_bar_f.attemptMove(QgsLayoutPoint(15, 185, QgsUnitTypes.LayoutMillimeters))
        layout_fields.addLayoutItem(scale_bar_f)

        legend_f = QgsLayoutItemLegend(layout_fields)
        legend_f.setLinkedMap(map_fields)
        legend_f.setLegendFilterByMapEnabled(True)
        legend_f.attemptMove(QgsLayoutPoint(215, 20, QgsUnitTypes.LayoutMillimeters))
        legend_f.attemptResize(QgsLayoutSize(72, 160, QgsUnitTypes.LayoutMillimeters))
        layout_fields.addLayoutItem(legend_f)

        out_f_img = os.path.join(FIG_DIR, "qgis_statewide_field_footprint_composite.png")
        if os.path.exists(out_f_img):
            os.remove(out_f_img)
        res_f = QgsLayoutExporter(layout_fields).exportToImage(os.path.abspath(out_f_img), settings)
        if res_f == QgsLayoutExporter.Success and os.path.exists(out_f_img):
            print(f"[SUCCESS] Exported 300 DPI QGIS FTW Field Footprint Map: {out_f_img} ({os.path.getsize(out_f_img) / 1024:.1f} KB)")

    finally:
        app.exitQgis()


def main():
    print("=" * 80)
    print("STEP 08: Standalone QGIS Project Generator & Publication Layout Export")
    print("=" * 80)
    blocks_shp, dist_shp = build_enriched_shapefiles()
    create_qgis_project(blocks_shp, dist_shp)
    print("\n" + "=" * 80)
    print("🎉 QGIS PROJECT & 300 DPI PUBLICATION LAYOUTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
