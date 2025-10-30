#!/usr/bin/env python3
"""
Ultra-fast data preprocessing - minimal operations for maximum speed
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import glob
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

def process_single_file(csv_file, time_index):
    """Process one file with minimal operations"""
    try:
        station_name = os.path.basename(csv_file).replace('.csv', '')

        # Fast CSV reading - skip validation
        df = pd.read_csv(csv_file, engine='c', dtype='float32', na_values=['', 'NA', 'N/A'])

        # Convert datetime - fast method
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed', errors='coerce')
        df = df.dropna(subset=['DateTime'])

        # Create mapping from datetime to index
        time_to_idx = {dt: i for i, dt in enumerate(time_index)}

        # Get feature columns (skip first 5: DateTime, City, Station_name, Lon, Lat)
        feature_cols = df.columns[5:].tolist()
        if 'DateTime' in feature_cols:
            feature_cols.remove('DateTime')

        # Create result array - pre-filled with zeros
        result = np.zeros((len(time_index), len(feature_cols)), dtype=np.float32)

        # Fill data - vectorized operation
        for _, row in df.iterrows():
            dt = row['DateTime']
            if dt in time_to_idx:
                idx = time_to_idx[dt]
                result[idx] = row[feature_cols].fillna(0).values.astype(np.float32)

        return station_name, result, feature_cols

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None, None, None

def main():
    csv_folder = "/root/autodl-tmp/合并数据_去PRMSL"
    output_folder = "./processed_data"

    print("=== ULTRA FAST PREPROCESSING ===")

    # Get files
    station_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    print(f"Found {len(station_files)} files")

    # Create time index
    start_time = datetime(2022, 1, 1, 0, 0)
    end_time = datetime(2024, 12, 31, 23, 0)
    time_index = pd.date_range(start=start_time, end=end_time, freq='H')
    print(f"Time range: {start_time} to {end_time}")
    print(f"Time steps: {len(time_index)}")

    # Process files
    all_results = []
    station_names = []
    all_feature_cols = None

    # Use thread pool for IO-bound operations
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_file, f, time_index) for f in station_files]

        for future in futures:
            station_name, result, feature_cols = future.result()
            if result is not None:
                all_results.append(result)
                station_names.append(station_name)
                if all_feature_cols is None:
                    all_feature_cols = feature_cols
                    print(f"Features per station: {len(feature_cols)}")

        print(f"Successfully processed {len(all_results)} files")

    if not all_results:
        print("No valid data processed!")
        return

    # Stack results
    print("Stacking all station data...")
    final_data = np.stack(all_results, axis=1)  # (time, stations, features)
    print(f"Final shape: {final_data.shape}")

    # Save results
    os.makedirs(output_folder, exist_ok=True)

    # Save main data
    output_file = os.path.join(output_folder, 'processed_data.npy')
    np.save(output_file, final_data)
    print(f"✅ Saved to: {output_file}")

    # Save station names
    station_file = os.path.join(output_folder, 'station_info.txt')
    with open(station_file, 'w') as f:
        for i, name in enumerate(station_names):
            f.write(f"{i}\t{name}\tunknown\t0\t0\n")
    print(f"✅ Saved station info to: {station_file}")

    # Save feature mapping
    feature_file = os.path.join(output_folder, 'feature_mapping.txt')
    with open(feature_file, 'w') as f:
        f.write(f"# {len(all_feature_cols)} features\n")
        for i, col in enumerate(all_feature_cols):
            f.write(f"{i}\t{col}\t\n")
    print(f"✅ Saved feature mapping to: {feature_file}")

    print("🚀 PROCESSING COMPLETED!")

if __name__ == "__main__":
    main()