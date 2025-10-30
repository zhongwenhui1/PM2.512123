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
        try:
            return datetime.strptime(datetime_str, "%Y/%m/%d %H:%M")
        except ValueError:
            try:
                # Try alternative format
                return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            except ValueError:
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

        # Initialize data array for this station
        station_data = np.full((len(time_index), self.total_feature_count), np.nan)

        # Map time to index
        time_to_idx = {t: i for i, t in enumerate(time_index)}

        # Fill in observation data
        for i, row in df.iterrows():
            if row['DateTime_parsed'] in time_to_idx:
                idx = time_to_idx[row['DateTime_parsed']]

                # Observation variables
                for j, var in enumerate(self.observation_vars):
                    if var in row and not pd.isna(row[var]):
                        station_data[idx, j] = row[var]

                # Time variables
                for j, var in enumerate(self.time_vars):
                    if var in row and not pd.isna(row[var]):
                        station_data[idx, self.observation_count + j] = row[var]

        # Fill in forecast data
        forecast_cols = [col for col in df.columns if '(' in col and ')' in col]

        for col in forecast_cols:
            var_name, hour = self.extract_forecast_var(col)
            if var_name and hour <= 72:  # We only need up to 72 hours
                var_idx = self.forecast_vars.index(var_name)
                forecast_idx = (hour - 1) * len(self.forecast_vars) + var_idx
                feature_idx = self.observation_count + self.time_count + forecast_idx

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
                'lat': first_row.get('Lat', 0)
            }
        else:
            station_info = {
                'name': station_name,
                'city': '',
                'lon': 0,
                'lat': 0
            }

        return station_data, station_info

    def preprocess_all_data(self):
        """Process all station data"""
        print("Loading station data...")

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

        # Stack all station data
        all_data = np.stack(station_data_list, axis=1)  # (time_steps, station_num, features)

        print(f"Data shape: {all_data.shape}")
        print(f"Expected shape: ({self.time_steps}, {len(station_data_list)}, {self.total_feature_count})")

        # Create output directory
        os.makedirs(self.output_folder, exist_ok=True)

        # Save main data
        output_file = os.path.join(self.output_folder, 'processed_data.npy')
        np.save(output_file, all_data)
        print(f"Saved processed data to: {output_file}")

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
            f.write("# Total features: 594\n")
            f.write("# Format: Index\tFeatureName\tDescription\n")

            idx = 0
            # Observation variables
            for var in self.observation_vars:
                f.write(f"{idx}\t{var}\tObservation_{var}\n")
                idx += 1

            # Time variables
            for var in self.time_vars:
                f.write(f"{idx}\t{var}\tTime_{var}\n")
                idx += 1

            # Forecast variables
            for hour in range(1, 73):
                for var in self.forecast_vars:
                    f.write(f"{idx}\t{var}({hour:02d})\tForecast_{var}_Hour{hour}\n")
                    idx += 1

        print(f"Saved feature mapping to: {feature_mapping_file}")

        return all_data, station_info_list

def main():
    # Configuration - update these paths
    csv_folder = "/root/autodl-tmp/合并数据_去PRMSL"  # Update this path
    output_folder = "./processed_data"

    print("Starting data preprocessing...")

    preprocessor = DataPreprocessor(csv_folder, output_folder)
    all_data, station_info = preprocessor.preprocess_all_data()

    print("Data preprocessing completed!")
    print(f"Processed {len(station_info)} stations")
    print(f"Time range: {preprocessor.start_time} to {preprocessor.end_time}")
    print(f"Data shape: {all_data.shape}")

if __name__ == "__main__":
    main()