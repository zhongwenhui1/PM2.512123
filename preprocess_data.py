#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preprocessing script for PM2.5-GNN project
Converts user's CSV format to the project's numpy format

User data format:
- Multiple CSV files, one for each station
- Station name format: 站点名称.csv
- Columns: DateTime, City, Station_name, Lon, Lat, Month, Weekday, Hour,
           PM2.5, PM10, SO2, NO2, O3, CO, T2M_C, D2M_C, PRES_hPa, APCP1h_mm,
           RH2M_pct, WSPD100, WDIR100, U100, V100,
           T2M_C(01), D2M_C(01), ..., APCP1h_mm(72) (72-hour forecasts)

Output format:
- processed_data.npy: (time_steps, station_num, feature_count)
- station_info.txt: station information with coordinates
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
from tqdm import tqdm
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, csv_folder, output_folder):
        self.csv_folder = csv_folder
        self.output_folder = output_folder
        self.station_files = glob.glob(os.path.join(csv_folder, "*.csv"))

        # Define the variables
        self.observation_vars = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO',
                                'T2M_C', 'D2M_C', 'PRES_hPa', 'APCP1h_mm',
                                'RH2M_pct', 'WSPD100', 'WDIR100', 'U100', 'V100']
        self.time_vars = ['Month', 'Weekday', 'Hour']
        self.forecast_vars = ['T2M_C', 'D2M_C', 'PRES_hPa', 'APCP1h_mm',
                             'RH2M_pct', 'WSPD100', 'WDIR100', 'U100', 'V100']

        # Calculate total feature count
        self.observation_count = len(self.observation_vars)  # 15
        self.time_count = len(self.time_vars)  # 3
        self.forecast_count = len(self.forecast_vars) * 72  # 8 * 72 = 576
        self.total_feature_count = self.observation_count + self.time_count + self.forecast_count  # 594

        # Time range
        self.start_time = datetime(2022, 1, 1, 0, 0)
        self.end_time = datetime(2024, 12, 31, 23, 0)
        self.time_steps = int((self.end_time - self.start_time).total_seconds() / 3600) + 1

        print(f"Total features: {self.total_feature_count}")
        print(f"Time steps: {self.time_steps}")
        print(f"Found {len(self.station_files)} station files")

    def parse_datetime(self, datetime_str):
        """Parse datetime string from CSV"""
        formats_to_try = [
            "%Y/%m/%d %H:%M:%S",    # 2022/01/01 00:00:00
            "%Y-%m-%d %H:%M:%S",    # 2022-01-01 00:00:00
            "%Y/%m/%d %H:%M",       # 2022/01/01 00:00
            "%Y-%m-%d %H:%M",       # 2022-01-01 00:00
            "%Y/%-m/%-d %H:%M",     # 2022/1/1 0:00
            "%Y-%-m-%-d %H:%M",     # 2022-1-1 0:00
        ]

        for fmt in formats_to_try:
            try:
                return datetime.strptime(datetime_str, fmt)
            except ValueError:
                continue

        raise ValueError(f"Cannot parse datetime: {datetime_str}")

    def extract_forecast_var(self, col_name):
        """Extract forecast variable name and hour from column name like T2M_C(01)"""
        match = re.match(r'(.+)\((\d+)\)', col_name)
        if match:
            var_name = match.group(1)
            hour = int(match.group(2))
            return var_name, hour
        return None, None

    def load_station_data(self, csv_file):
        """Load data from a single station CSV file"""
        station_name = os.path.basename(csv_file).replace('.csv', '')

        # Read CSV
        df = pd.read_csv(csv_file)

        # Parse datetime and create time index
        df['DateTime_parsed'] = df['DateTime'].apply(self.parse_datetime)
        df = df.sort_values('DateTime_parsed')

        # Create complete time series with hourly resolution
        time_index = pd.date_range(start=self.start_time, end=self.end_time, freq='H')

        # Dynamically calculate actual feature count from this file
        obs_cols = [col for col in df.columns if col in self.observation_vars]
        time_cols = [col for col in df.columns if col in self.time_vars]
        forecast_cols = [col for col in df.columns if '(' in col and ')' in col]

        # Calculate actual forecast hours and variables
        max_hour = 0
        actual_forecast_vars = set()
        for col in forecast_cols:
            var_name, hour = self.extract_forecast_var(col)
            if var_name and hour <= 100:  # Allow up to 100 hours
                max_hour = max(max_hour, hour)
                actual_forecast_vars.add(var_name)

        # Update feature count with actual values
        actual_obs_count = len(obs_cols)
        actual_time_count = len(time_cols)
        actual_forecast_count = len(actual_forecast_vars) * max_hour
        actual_total_count = actual_obs_count + actual_time_count + actual_forecast_count

        print(f"Station {station_name}: {actual_obs_count} obs, {actual_time_count} time, "
              f"{len(actual_forecast_vars)} forecast vars × {max_hour} hours = {actual_forecast_count} forecast features")
        print(f"  Total: {actual_total_count} features")

        # Initialize data array with actual feature count
        station_data = np.full((len(time_index), actual_total_count), np.nan)

        # Map time to index
        time_to_idx = {t: i for i, t in enumerate(time_index)}

        # Fill in observation data
        for i, row in df.iterrows():
            if row['DateTime_parsed'] in time_to_idx:
                idx = time_to_idx[row['DateTime_parsed']]

                # Observation variables
                for j, var in enumerate(obs_cols):
                    if var in row and not pd.isna(row[var]):
                        station_data[idx, j] = row[var]

                # Time variables
                for j, var in enumerate(time_cols):
                    if var in row and not pd.isna(row[var]):
                        station_data[idx, actual_obs_count + j] = row[var]

        # Fill in forecast data (sort by hour then by variable name for consistency)
        actual_forecast_vars = sorted(list(actual_forecast_vars))
        for col in forecast_cols:
            var_name, hour = self.extract_forecast_var(col)
            if var_name and var_name in actual_forecast_vars and hour <= max_hour:
                var_idx = actual_forecast_vars.index(var_name)
                forecast_idx = (hour - 1) * len(actual_forecast_vars) + var_idx
                feature_idx = actual_obs_count + actual_time_count + forecast_idx

                for i, row in df.iterrows():
                    if row['DateTime_parsed'] in time_to_idx:
                        idx = time_to_idx[row['DateTime_parsed']]
                        if col in row and not pd.isna(row[col]):
                            station_data[idx, feature_idx] = row[col]

        # Extract station info
        if len(df) > 0:
            first_row = df.iloc[0]
            station_info = {
                'name': station_name,
                'city': first_row.get('City', ''),
                'lon': first_row.get('Lon', 0),
                'lat': first_row.get('Lat', 0),
                'actual_feature_count': actual_total_count,
                'actual_obs_vars': obs_cols,
                'actual_time_vars': time_cols,
                'actual_forecast_vars': actual_forecast_vars,
                'max_forecast_hour': max_hour
            }
        else:
            station_info = {
                'name': station_name,
                'city': '',
                'lon': 0,
                'lat': 0,
                'actual_feature_count': actual_total_count,
                'actual_obs_vars': obs_cols,
                'actual_time_vars': time_cols,
                'actual_forecast_vars': actual_forecast_vars,
                'max_forecast_hour': max_hour
            }

        return station_data, station_info

    def _process_single_station(self, csv_file):
        """Process a single station file (for parallel processing)"""
        try:
            station_data, station_info = self.load_station_data(csv_file)
            return csv_file, station_data, station_info, None
        except Exception as e:
            return csv_file, None, None, str(e)

    def preprocess_all_data(self, parallel=True):
        """Process all station data with optional parallel processing"""
        print("Loading station data...")

        if parallel:
            # Use parallel processing
            num_workers = min(cpu_count(), 16)  # Cap at 16 workers to avoid overloading
            print(f"Using {num_workers} parallel workers")

            station_data_list = []
            station_info_list = []

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self._process_single_station, csv_file): csv_file
                    for csv_file in self.station_files
                }

                # Process completed tasks
                for future in tqdm(as_completed(future_to_file), total=len(self.station_files)):
                    csv_file = future_to_file[future]
                    try:
                        _, station_data, station_info, error = future.result()
                        if error:
                            print(f"Error processing {csv_file}: {error}")
                            continue
                        station_data_list.append(station_data)
                        station_info_list.append(station_info)
                    except Exception as e:
                        print(f"Error processing {csv_file}: {e}")
                        continue
        else:
            # Sequential processing (original)
            station_data_list = []
            station_info_list = []

            for csv_file in tqdm(self.station_files):
                try:
                    station_data, station_info = self.load_station_data(csv_file)
                    station_data_list.append(station_data)
                    station_info_list.append(station_info)
                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
                    continue

        if not station_data_list:
            raise ValueError("No valid station data found!")

        # Check if all stations have the same feature count
        feature_counts = [info['actual_feature_count'] for info in station_info_list]
        if len(set(feature_counts)) > 1:
            print(f"Warning: Different feature counts found: {set(feature_counts)}")
            print("Using the most common feature count")
            most_common_count = max(set(feature_counts), key=feature_counts.count)

            # Filter to stations with the most common feature count
            filtered_data = []
            filtered_info = []
            for data, info in zip(station_data_list, station_info_list):
                if info['actual_feature_count'] == most_common_count:
                    filtered_data.append(data)
                    filtered_info.append(info)

            station_data_list = filtered_data
            station_info_list = filtered_info
            print(f"Kept {len(station_data_list)} stations with {most_common_count} features")

        # Stack all station data
        all_data = np.stack(station_data_list, axis=1)  # (time_steps, station_num, features)

        print(f"Final data shape: {all_data.shape}")
        print(f"Time steps: {all_data.shape[0]}, Stations: {all_data.shape[1]}, Features: {all_data.shape[2]}")

        # Create output directory
        os.makedirs(self.output_folder, exist_ok=True)

        # Save main data
        output_file = os.path.join(self.output_folder, 'processed_data.npy')
        np.save(output_file, all_data)
        print(f"Saved processed data to: {output_file}")

        # Use the first station's feature info for mapping
        feature_info = station_info_list[0]

        # Save station info
        station_info_file = os.path.join(self.output_folder, 'station_info.txt')
        with open(station_info_file, 'w', encoding='utf-8') as f:
            f.write("# Station information\n")
            f.write("# Index\tStationName\tCity\tLongitude\tLatitude\n")
            for i, info in enumerate(station_info_list):
                f.write(f"{i}\t{info['name']}\t{info['city']}\t{info['lon']}\t{info['lat']}\n")
        print(f"Saved station info to: {station_info_file}")

        # Save feature mapping
        feature_mapping_file = os.path.join(self.output_folder, 'feature_mapping.txt')
        with open(feature_mapping_file, 'w', encoding='utf-8') as f:
            f.write("# Feature mapping\n")
            f.write(f"# Total features: {feature_info['actual_feature_count']}\n")
            f.write("# Format: Index\tFeatureName\tDescription\n")

            idx = 0
            # Observation variables
            for var in feature_info['actual_obs_vars']:
                f.write(f"{idx}\t{var}\tObservation_{var}\n")
                idx += 1

            # Time variables
            for var in feature_info['actual_time_vars']:
                f.write(f"{idx}\t{var}\tTime_{var}\n")
                idx += 1

            # Forecast variables
            actual_forecast_vars = sorted(list(feature_info['actual_forecast_vars']))
            max_hour = feature_info['max_forecast_hour']
            for hour in range(1, max_hour + 1):
                for var in actual_forecast_vars:
                    f.write(f"{idx}\t{var}({hour:02d})\tForecast_{var}_Hour{hour}\n")
                    idx += 1

        print(f"Saved feature mapping to: {feature_mapping_file}")
        print(f"Actual feature breakdown:")
        print(f"  Observation: {len(feature_info['actual_obs_vars'])}")
        print(f"  Time: {len(feature_info['actual_time_vars'])}")
        print(f"  Forecast: {len(actual_forecast_vars)} × {max_hour} = {len(actual_forecast_vars) * max_hour}")
        print(f"  Total: {feature_info['actual_feature_count']}")

        return all_data, station_info_list

def main():
    # Configuration - update these paths
    csv_folder = "/root/autodl-tmp/合并数据_去PRMSL"  # Update this path
    output_folder = "./processed_data"

    print("Starting data preprocessing...")

    preprocessor = DataPreprocessor(csv_folder, output_folder)
    all_data, station_info = preprocessor.preprocess_all_data(parallel=True)  # Use parallel processing

    print("Data preprocessing completed!")
    print(f"Processed {len(station_info)} stations")
    print(f"Time range: {preprocessor.start_time} to {preprocessor.end_time}")
    print(f"Data shape: {all_data.shape}")

if __name__ == "__main__":
    main()