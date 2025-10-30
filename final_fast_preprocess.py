#!/usr/bin/env python3
"""
Final ultra-fast preprocessing - tested and optimized for maximum speed
Handles exactly 206 CSV files with 666 features each
Output: (26304, 206, 666) numpy array compatible with PM2.5-GNN project
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import glob
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

def process_single_file(csv_file, time_index_map):
    """Process one CSV file with proven fast method"""
    try:
        station_name = os.path.basename(csv_file).replace('.csv', '')

        # Read CSV - let pandas auto-detect dtypes for speed
        df = pd.read_csv(csv_file, engine='c')

        # Convert DateTime column efficiently
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed', errors='coerce')
        df = df.dropna(subset=['DateTime'])

        # Get feature columns (skip first 5: DateTime, City, Station_name, Lon, Lat)
        feature_cols = df.columns[5:].tolist()

        # Ensure we have 666 features
        if len(feature_cols) != 666:
            print(f"Warning: {station_name} has {len(feature_cols)} features (expected 666)")

        # Initialize result array with zeros
        result = np.zeros((len(time_index_map), len(feature_cols)), dtype=np.float32)

        # Fill data efficiently
        valid_rows = 0
        for _, row in df.iterrows():
            dt = row['DateTime']
            if dt in time_index_map:
                idx = time_index_map[dt]
                # Fill with feature values, convert to float32
                result[idx] = pd.to_numeric(row[feature_cols], errors='coerce').fillna(0).values.astype(np.float32)
                valid_rows += 1

        print(f"✓ {station_name}: {valid_rows} valid rows, {len(feature_cols)} features")
        return station_name, result, feature_cols

    except Exception as e:
        print(f"✗ Error processing {csv_file}: {e}")
        return None, None, None

def main():
    # Configuration
    csv_folder = "/root/autodl-tmp/合并数据_去PRMSL"
    output_folder = "./processed_data"

    print("🚀 FINAL ULTRA-FAST PREPROCESSING")
    print("=" * 50)

    # Get all CSV files
    station_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    print(f"📁 Found {len(station_files)} CSV files")

    # Create time index and mapping
    start_time = datetime(2022, 1, 1, 0, 0)
    end_time = datetime(2024, 12, 31, 23, 0)
    time_index = pd.date_range(start=start_time, end=end_time, freq='H')
    time_index_map = {dt: i for i, dt in enumerate(time_index)}

    print(f"📅 Time range: {start_time} to {end_time}")
    print(f"⏱️  Total time steps: {len(time_index)}")

    # Process all files
    all_results = []
    station_names = []
    feature_cols_list = []

    print("\n🔄 Processing files...")

    # Use ThreadPoolExecutor for IO-bound operations
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_file, f, time_index_map) for f in station_files]

        # Collect results
        for i, future in enumerate(futures):
            station_name, result, cols = future.result()
            if result is not None:
                all_results.append(result)
                station_names.append(station_name)
                feature_cols_list.append(cols)
            print(f"Progress: {i+1}/{len(station_files)} files processed", end='\r')

    print(f"\n✅ Successfully processed {len(all_results)} files")

    if not all_results:
        print("❌ No valid data processed!")
        return

    # Validate feature consistency
    feature_counts = [len(cols) for cols in feature_cols_list]
    if len(set(feature_counts)) > 1:
        print(f"⚠️  Warning: Inconsistent feature counts: {set(feature_counts)}")
        # Use the most common feature count
        most_common = max(set(feature_counts), key=feature_counts.count)
        print(f"📊 Using most common feature count: {most_common}")

        # Filter to consistent stations
        filtered_results = []
        filtered_names = []
        for result, name, cols in zip(all_results, station_names, feature_cols_list):
            if len(cols) == most_common:
                filtered_results.append(result)
                filtered_names.append(name)

        all_results = filtered_results
        station_names = filtered_names
        feature_cols_list = [feature_cols_list[i] for i in range(len(all_results))]

    # Stack all station data
    print("\n📚 Stacking all station data...")
    final_data = np.stack(all_results, axis=1)  # Shape: (time, stations, features)

    print(f"📊 Final data shape: {final_data.shape}")
    print(f"   Expected: ({len(time_index)}, {len(all_results)}, {len(feature_cols_list[0])})")

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Save main data file
    output_file = os.path.join(output_folder, 'processed_data.npy')
    np.save(output_file, final_data)
    print(f"💾 Saved processed data to: {output_file}")

    # Save station information
    station_info_file = os.path.join(output_folder, 'station_info.txt')
    with open(station_info_file, 'w', encoding='utf-8') as f:
        f.write("# Station information\n")
        f.write("# Index\tStationName\tCity\tLongitude\tLatitude\n")
        for i, name in enumerate(station_names):
            f.write(f"{i}\t{name}\tunknown\t0\t0\n")
    print(f"💾 Saved station info to: {station_info_file}")

    # Save feature mapping
    feature_mapping_file = os.path.join(output_folder, 'feature_mapping.txt')
    with open(feature_mapping_file, 'w', encoding='utf-8') as f:
        f.write("# Feature mapping\n")
        f.write(f"# Total features: {len(feature_cols_list[0])}\n")
        f.write("# Format: Index\tFeatureName\tDescription\n")

        for i, col in enumerate(feature_cols_list[0]):
            f.write(f"{i}\t{col}\tFeature_{i}\n")
    print(f"💾 Saved feature mapping to: {feature_mapping_file}")

    print("\n🎉 PROCESSING COMPLETED SUCCESSFULLY!")
    print(f"📈 Processed {len(all_results)} stations")
    print(f"📐 Final shape: {final_data.shape}")
    print(f"⚡ Ready for training with: python train.py")

if __name__ == "__main__":
    main()