#!/usr/bin/env python3
"""
Fast data preprocessing script for PM2.5-GNN project
Optimized for speed with vectorized operations
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore')

class FastPreprocessor:
    def __init__(self, csv_folder, output_folder):
        self.csv_folder = csv_folder
        self.output_folder = output_folder
        self.station_files = glob.glob(os.path.join(csv_folder, "*.csv"))

        self.start_time = datetime(2022, 1, 1, 0, 0)
        self.end_time = datetime(2024, 12, 31, 23, 0)
        self.time_index = pd.date_range(start=self.start_time, end=self.end_time, freq='H')
        self.time_steps = len(self.time_index)

        print(f"Fast preprocessing with {len(self.station_files)} files")
        print(f"Time steps: {self.time_steps}")

    def parse_datetime(self, datetime_str):
        """Parse datetime string from CSV"""
        formats_to_try = [
            "%Y-%m-%d %H:%M:%S",  # Your format
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M",
        ]
        for fmt in formats_to_try:
            try:
                return datetime.strptime(datetime_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse datetime: {datetime_str}")

    def process_one_file_fast(self, csv_file):
        """Process a single CSV file with optimized logic"""
        station_name = os.path.basename(csv_file).replace('.csv', '')

        # Read CSV with optimized settings
        df = pd.read_csv(csv_file, dtype={'City': str, 'Station_name': str})

        # Parse datetime efficiently
        df['DateTime_parsed'] = pd.to_datetime(df['DateTime'], format='mixed')
        df = df.dropna(subset=['DateTime_parsed'])
        df = df.sort_values('DateTime_parsed')

        # Get all columns in order
        all_cols = df.columns.tolist()

        # Find data columns (exclude first 5 columns: DateTime, City, Station_name, Lon, Lat)
        data_cols = all_cols[5:]

        print(f"Station {station_name}: {len(data_cols)} features")

        # Create a sparse representation: only store actual data
        station_data_sparse = []

        for _, row in df.iterrows():
            dt = row['DateTime_parsed']
            if dt in self.time_index:
                idx = self.time_index.get_loc(dt)
                # Convert row to list, only keep data columns
                data_values = [row[col] if not pd.isna(row[col]) else 0.0 for col in data_cols]
                station_data_sparse.append((idx, data_values))

        return station_name, station_data_sparse, data_cols

    def preprocess_all_fast(self):
        """Process all files with optimized approach"""
        print("Loading station data (fast mode)...")

        all_station_data = []
        all_station_names = []
        all_data_cols = None

        # Process files sequentially but optimized
        for csv_file in tqdm(self.station_files):
            station_name, sparse_data, data_cols = self.process_one_file_fast(csv_file)
            all_station_data.append(sparse_data)
            all_station_names.append(station_name)

            if all_data_cols is None:
                all_data_cols = data_cols
                print(f"Total features: {len(data_cols)}")

        print("Creating final data array...")

        # Create final array
        num_stations = len(all_station_names)
        num_features = len(all_data_cols)

        # Initialize with zeros (faster than NaN)
        final_data = np.zeros((self.time_steps, num_stations, num_features), dtype=np.float32)

        # Fill in data
        for station_idx, sparse_data in enumerate(all_station_data):
            for time_idx, values in sparse_data:
                if time_idx < self.time_steps:
                    final_data[time_idx, station_idx, :] = values

        print(f"Final data shape: {final_data.shape}")

        # Save results
        os.makedirs(self.output_folder, exist_ok=True)

        # Save main data
        output_file = os.path.join(self.output_folder, 'processed_data.npy')
        np.save(output_file, final_data)
        print(f"Saved processed data to: {output_file}")

        # Save station info
        station_info_file = os.path.join(self.output_folder, 'station_info.txt')
        with open(station_info_file, 'w', encoding='utf-8') as f:
            f.write("# Station information\n")
            f.write("# Index\tStationName\n")
            for i, name in enumerate(all_station_names):
                f.write(f"{i}\t{name}\n")
        print(f"Saved station info to: {station_info_file}")

        # Save feature mapping
        feature_mapping_file = os.path.join(self.output_folder, 'feature_mapping.txt')
        with open(feature_mapping_file, 'w', encoding='utf-8') as f:
            f.write("# Feature mapping\n")
            f.write(f"# Total features: {len(all_data_cols)}\n")
            for i, col in enumerate(all_data_cols):
                f.write(f"{i}\t{col}\tFeature_{i}\n")
        print(f"Saved feature mapping to: {feature_mapping_file}")

        return final_data, all_station_names

def main():
    csv_folder = "/root/autodl-tmp/合并数据_去PRMSL"
    output_folder = "./processed_data"

    print("Starting fast data preprocessing...")

    preprocessor = FastPreprocessor(csv_folder, output_folder)
    all_data, station_names = preprocessor.preprocess_all_fast()

    print("Fast data preprocessing completed!")
    print(f"Processed {len(station_names)} stations")
    print(f"Data shape: {all_data.shape}")

if __name__ == "__main__":
    main()