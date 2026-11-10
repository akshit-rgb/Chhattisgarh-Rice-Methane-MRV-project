"""
02_extract_ftw_real_tiles.py
============================
Extracts, reprojects, and partitions real 2024 Fields of the World (FTW) 
Sentinel-2 10m agricultural parcel predictions across all 33 districts of Chhattisgarh.

Strict Domain Adherence:
- Areas calculated strictly in EPSG:32644 (UTM Zone 44N)
- Output partitioned district-wise into data/processed/district_fields/
- Master statewide GeoParquet compiled
- Summary metrics compiled into outputs/reports/chhattisgarh_ftw_district_summary.csv
"""

import os
import glob
import time
import gc
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import pyarrow as pa
import pyarrow.parquet as pq

def sanitize_filename(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")

def main():
    print("=" * 70)
    print("CHHATTISGARH FTW 2024 REAL AGRICULTURAL FIELD EXTRACTION PIPELINE")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. Setup paths
    raw_dir = r"chhattisgarh_rice_methane all in one"
    districts_shp = r"data/processed/shapefiles/chhattisgarh_districts_essential.shp"
    out_district_dir = r"data/processed/district_fields"
    out_report_dir = r"outputs/reports"
    out_master_parquet = r"data/processed/chhattisgarh_fields_ftw_2024.parquet"
    out_sample_shp = r"data/processed/shapefiles/chhattisgarh_ftw_fields_sampled.shp"
    
    os.makedirs(out_district_dir, exist_ok=True)
    os.makedirs(out_report_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_sample_shp), exist_ok=True)
    
    # 2. Load district boundaries in EPSG:4326 for fast spatial filtering
    print(f"\n[1/5] Loading district boundaries from {districts_shp}...")
    districts_32644 = gpd.read_file(districts_shp)
    districts_4326 = districts_32644.to_crs(epsg=4326)
    
    # Pre-compute spatial index on districts
    districts_sindex = districts_4326.sindex
    print(f"Loaded {len(districts_4326)} districts in Chhattisgarh.")
    
    # 3. Identify and sort unique parquet tiles
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    unique_tiles = [f for f in raw_files if " - Copy" not in f and "_copy" not in f.lower()]
    print(f"\n[2/5] Discovered {len(unique_tiles)} unique 1x1 degree GeoParquet tiles.")
    
    # Temporary directory for chunk-level district files
    temp_dir = os.path.join(out_district_dir, "_temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Clean previous temp files if any
    for old_f in glob.glob(os.path.join(temp_dir, "*.parquet")):
        try:
            os.remove(old_f)
        except Exception:
            pass
            
    total_processed_raw = 0
    total_cg_fields = 0
    
    print("\n[3/5] Processing tiles and spatial-joining to Chhattisgarh districts...")
    for idx, tile_path in enumerate(unique_tiles, start=1):
        t0 = time.time()
        tile_name = os.path.basename(tile_path)
        
        # Parse lat/lon from tile name e.g. 2024_N21E081.parquet
        parts = tile_name.replace(".parquet", "").split("_")
        tile_code = parts[1]
        lat = int(tile_code[1:3])
        lon = int(tile_code[4:7])
        tile_geom = box(lon, lat, lon + 1, lat + 1)
        
        # Check which districts intersect this tile box
        possible_district_matches = list(districts_sindex.intersection(tile_geom.bounds))
        if not possible_district_matches:
            print(f"  [{idx:02d}/{len(unique_tiles):02d}] {tile_name}: No district intersection. Skipping.")
            continue
            
        candidate_districts = districts_4326.iloc[possible_district_matches]
        intersecting_districts = candidate_districts[candidate_districts.intersects(tile_geom)].copy()
        
        if len(intersecting_districts) == 0:
            print(f"  [{idx:02d}/{len(unique_tiles):02d}] {tile_name}: 0 districts intersect bounding box. Skipping.")
            continue
            
        # Read parquet tile
        tile_gdf = gpd.read_parquet(tile_path)
        n_raw = len(tile_gdf)
        total_processed_raw += n_raw
        
        # Spatial join via centroids (fast, robust, no edge fragmentation)
        centroids = tile_gdf.geometry.centroid
        pts_gdf = gpd.GeoDataFrame({"orig_idx": tile_gdf.index}, geometry=centroids, crs="EPSG:4326")
        
        # sjoin with intersecting districts
        joined = gpd.sjoin(pts_gdf, intersecting_districts[["District", "geometry"]], how="inner", predicate="within")
        
        n_matched = len(joined)
        total_cg_fields += n_matched
        
        if n_matched > 0:
            # Subset fields
            matched_indices = joined["orig_idx"].values
            matched_districts = joined["District"].values
            
            sub_fields = tile_gdf.iloc[matched_indices].copy()
            sub_fields["district"] = matched_districts
            
            # Compute precise area in EPSG:32644 (UTM 44N)
            sub_fields_proj = sub_fields.to_crs(epsg=32644)
            sub_fields["area_ha"] = (sub_fields_proj.geometry.area / 10000.0).round(4)
            
            # Keep clean schema
            keep_cols = ["district", "area_ha", "confidence_mean", "confidence_median", "confidence_min", "chunk_id", "geometry"]
            sub_fields = sub_fields[[c for c in keep_cols if c in sub_fields.columns]]
            
            # Group by district and save temp chunk
            for d_name, d_group in sub_fields.groupby("district"):
                s_name = sanitize_filename(d_name)
                chunk_file = os.path.join(temp_dir, f"{s_name}_{tile_code}.parquet")
                d_group.to_parquet(chunk_file, index=False)
                
            del sub_fields_proj, sub_fields
            
        del tile_gdf, pts_gdf, joined
        gc.collect()
        
        elapsed = time.time() - t0
        print(f"  [{idx:02d}/{len(unique_tiles):02d}] {tile_name}: {n_raw:,} rows -> {n_matched:,} in CG ({elapsed:.1f}s)")
        
    print(f"\nExtracted {total_cg_fields:,} fields across Chhattisgarh from {total_processed_raw:,} total predictions.")
    
    # 4. Consolidate chunks per district
    print("\n[4/5] Consolidating district-wise GeoParquet files...")
    district_stats = []
    sample_dfs = []
    
    all_district_names = sorted(districts_4326["District"].unique())
    
    for d_name in all_district_names:
        s_name = sanitize_filename(d_name)
        pattern = os.path.join(temp_dir, f"{s_name}_*.parquet")
        chunk_files = glob.glob(pattern)
        
        dist_out_file = os.path.join(out_district_dir, f"district_{s_name}_ftw.parquet")
        
        if chunk_files:
            dfs = [gpd.read_parquet(cf) for cf in chunk_files]
            dist_gdf = pd.concat(dfs, ignore_index=True)
            dist_gdf = gpd.GeoDataFrame(dist_gdf, geometry="geometry", crs="EPSG:4326")
            
            # Save final district parquet
            dist_gdf.to_parquet(dist_out_file, index=False)
            
            count = len(dist_gdf)
            tot_area = dist_gdf["area_ha"].sum()
            mean_area = dist_gdf["area_ha"].mean()
            median_area = dist_gdf["area_ha"].median()
            mean_conf = dist_gdf["confidence_mean"].mean()
            
            district_stats.append({
                "District": d_name,
                "Field_Count": count,
                "Total_Area_ha": round(tot_area, 2),
                "Mean_Field_ha": round(mean_area, 4),
                "Median_Field_ha": round(median_area, 4),
                "Mean_Confidence": round(mean_conf, 4)
            })
            
            # Take representative sample for statewide visualization (~1,500 per district = ~50k fields total)
            sample_size = min(len(dist_gdf), 1500)
            sample_dfs.append(dist_gdf.sample(n=sample_size, random_state=42))
            
            file_mb = os.path.getsize(dist_out_file) / (1024 * 1024)
            print(f"  Saved {dist_out_file} ({count:,} fields, {tot_area:,.1f} ha, {file_mb:.1f} MB)")
        else:
            print(f"  Warning: No fields found for district {d_name}!")
            district_stats.append({
                "District": d_name,
                "Field_Count": 0,
                "Total_Area_ha": 0.0,
                "Mean_Field_ha": 0.0,
                "Median_Field_ha": 0.0,
                "Mean_Confidence": 0.0
            })
            
    # Save district statistics report
    stats_df = pd.DataFrame(district_stats)
    stats_csv = os.path.join(out_report_dir, "chhattisgarh_ftw_district_summary.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"\nDistrict statistics exported to: {stats_csv}")
    
    # 5. Export statewide master parquet and lightweight sampled shapefile
    print("\n[5/5] Generating statewide representative sampled shapefile & master parquet...")
    
    # Compile master statewide GeoParquet streamingly
    print(f"Writing statewide master GeoParquet to: {out_master_parquet} ...")
    master_writer = None
    district_files = sorted(glob.glob(os.path.join(out_district_dir, "district_*_ftw.parquet")))
    for df_path in district_files:
        tbl = pq.read_table(df_path)
        if master_writer is None:
            master_writer = pq.ParquetWriter(out_master_parquet, tbl.schema, compression="snappy")
        master_writer.write_table(tbl)
    if master_writer:
        master_writer.close()
        master_size_mb = os.path.getsize(out_master_parquet) / (1024 * 1024)
        print(f"Master statewide GeoParquet written: {out_master_parquet} ({master_size_mb:.1f} MB)")
    
    if sample_dfs:
        sample_gdf = pd.concat(sample_dfs, ignore_index=True)
        sample_gdf = gpd.GeoDataFrame(sample_gdf, geometry="geometry", crs="EPSG:4326")
        
        # Project sampled shapefile to EPSG:32644 for native QGIS performance
        sample_gdf_32644 = sample_gdf.to_crs(epsg=32644)
        sample_gdf_32644.to_file(out_sample_shp)
        print(f"Exported statewide sampled shapefile ({len(sample_gdf_32644):,} fields) to: {out_sample_shp}")
        
    # Clean up temp chunks
    for cf in glob.glob(os.path.join(temp_dir, "*.parquet")):
        try:
            os.remove(cf)
        except Exception:
            pass
    try:
        os.rmdir(temp_dir)
    except Exception:
        pass
        
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time/60.0:.2f} MINUTES")
    print(f"Extracted {total_cg_fields:,} fields across {len(districts_4326)} districts.")
    print("=" * 70)

if __name__ == "__main__":
    main()
