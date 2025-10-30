#!/usr/bin/env python3
"""
32-Core 60GB Server Optimized Preprocessing
- 26 parallel processes
- Batch processing for optimal memory usage
- Real-time logging and progress monitoring
- Handles 206 CSV files with 666 features each
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [PID %(process)d] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('preprocess.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ServerPreprocessor:
    def __init__(self, csv_folder, output_folder):
        self.csv_folder = csv_folder
        self.output_folder = output_folder
        self.station_files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))

        # Time configuration
        self.start_time = datetime(2022, 1, 1, 0, 0)
        self.end_time = datetime(2024, 12, 31, 23, 0)
        self.time_index = pd.date_range(start=self.start_time, end=self.end_time, freq='H')
        self.time_index_map = {dt: i for i, dt in enumerate(self.time_index)}

        # Server optimization settings
        self.num_processes = 26
        self.batch_size = 52  # 206 files / 4 batches ≈ 52 files per batch

        logger.info(f"🚀 SERVER OPTIMIZED PREPROCESSING")
        logger.info(f"📁 CSV files: {len(self.station_files)}")
        logger.info(f"⏱️  Time range: {self.start_time} to {self.end_time}")
        logger.info(f"📊 Time steps: {len(self.time_index)}")
        logger.info(f"🔧 Using {self.num_processes} processes, batch size: {self.batch_size}")

    def process_single_file(self, args):
        """Process single CSV file - optimized for server performance"""
        file_idx, csv_file, global_time_map = args
        process_start = time.time()

        try:
            station_name = os.path.basename(csv_file).replace('.csv', '')

            # Fast CSV reading with optimized parameters
            logger.info(f"[Worker] Processing {station_name} (file {file_idx+1})")

            # Read CSV with memory-efficient settings
            df = pd.read_csv(
                csv_file,
                engine='c',
                dtype={
                    'City': 'category',
                    'Station_name': 'category'
                },
                na_values=['', 'NA', 'N/A', 'null', 'NULL']
            )

            # Convert DateTime efficiently
            df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed', errors='coerce')
            df = df.dropna(subset=['DateTime'])

            # Get feature columns (skip first 5 columns)
            feature_cols = df.columns[5:].tolist()

            # Remove any datetime-related columns that might have been added
            feature_cols = [col for col in feature_cols if not col.endswith('_parsed')]

            # Verify feature count
            if len(feature_cols) != 666:
                logger.warning(f"[Worker] {station_name}: {len(feature_cols)} features (expected 666)")

            # Initialize result array with zeros (float32 for memory efficiency)
            result = np.zeros((len(global_time_map), len(feature_cols)), dtype=np.float32)

            # Process data efficiently
            valid_rows = 0
            for _, row in df.iterrows():
                dt = row['DateTime']
                if dt in global_time_map:
                    idx = global_time_map[dt]
                    # Convert to numeric, handle NaN
                    values = pd.to_numeric(row[feature_cols], errors='coerce').fillna(0).values
                    result[idx] = values.astype(np.float32)
                    valid_rows += 1

            process_time = time.time() - process_start
            logger.info(f"[Worker] ✓ {station_name}: {valid_rows} rows, {len(feature_cols)} features, {process_time:.2f}s")

            return {
                'success': True,
                'station_name': station_name,
                'data': result,
                'features': feature_cols,
                'file_idx': file_idx,
                'valid_rows': valid_rows,
                'process_time': process_time
            }

        except Exception as e:
            logger.error(f"[Worker] ✗ Error processing {csv_file}: {e}")
            return {
                'success': False,
                'station_name': station_name,
                'error': str(e),
                'file_idx': file_idx
            }

    def process_batch(self, batch_files, batch_num, total_batches, global_time_map):
        """Process a batch of files with multiple processes"""
        batch_start = time.time()

        logger.info(f"📦 Processing BATCH {batch_num}/{total_batches} ({len(batch_files)} files)")

        # Prepare arguments for multiprocessing
        args_list = [(i, file, global_time_map) for i, file in enumerate(batch_files)]

        # Process with ProcessPoolExecutor
        results = []
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, args): args[0]
                for args in args_list
            }

            # Collect results with progress tracking
            completed = 0
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)
                completed += 1

                if result['success']:
                    logger.info(f"[Progress] Batch {batch_num}: {completed}/{len(batch_files)} completed - {result['station_name']}")
                else:
                    logger.warning(f"[Progress] Batch {batch_num}: {completed}/{len(batch_files)} completed - FAILED")

        # Sort results by original file index
        results.sort(key=lambda x: x['file_idx'])

        batch_time = time.time() - batch_start
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        logger.info(f"✅ Batch {batch_num} completed in {batch_time:.2f}s")
        logger.info(f"   Success: {len(successful)}, Failed: {len(failed)}")

        if failed:
            logger.warning(f"Failed files: {[f['station_name'] for f in failed]}")

        return successful, failed

    def save_batch_results(self, batch_results, batch_num):
        """Save intermediate batch results"""
        if not batch_results:
            return None

        batch_start = time.time()

        # Sort by original file index
        batch_results.sort(key=lambda x: x['file_idx'])

        # Stack data
        batch_data = np.stack([r['data'] for r in batch_results], axis=1)
        batch_names = [r['station_name'] for r in batch_results]
        batch_features = batch_results[0]['features']

        # Save batch file
        batch_file = os.path.join(self.output_folder, f'batch_{batch_num}.npy')
        np.save(batch_file, batch_data)

        # Save batch metadata
        metadata_file = os.path.join(self.output_folder, f'batch_{batch_num}_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f"Batch {batch_num}\n")
            f.write(f"Stations: {len(batch_names)}\n")
            f.write(f"Features: {len(batch_features)}\n")
            f.write(f"Shape: {batch_data.shape}\n")
            f.write(f"Stations: {','.join(batch_names)}\n")

        save_time = time.time() - batch_start
        logger.info(f"💾 Saved batch {batch_num} in {save_time:.2f}s")

        return {
            'data': batch_data,
            'names': batch_names,
            'features': batch_features,
            'shape': batch_data.shape
        }

    def combine_batches(self, batch_results_list):
        """Combine all batch results into final dataset"""
        logger.info("🔗 Combining all batches...")
        combine_start = time.time()

        # Combine all data
        all_data = []
        all_names = []
        all_features = None

        for batch_result in batch_results_list:
            if batch_result:
                all_data.append(batch_result['data'])
                all_names.extend(batch_result['names'])
                if all_features is None:
                    all_features = batch_result['features']

        # Concatenate along station axis
        final_data = np.concatenate(all_data, axis=1)

        combine_time = time.time() - combine_start
        logger.info(f"🔗 Combined data in {combine_time:.2f}s")
        logger.info(f"📊 Final shape: {final_data.shape}")

        return final_data, all_names, all_features

    def save_final_results(self, final_data, station_names, features):
        """Save final results in required format"""
        logger.info("💾 Saving final results...")
        save_start = time.time()

        # Save main data file
        output_file = os.path.join(self.output_folder, 'processed_data.npy')
        np.save(output_file, final_data)
        logger.info(f"✅ Saved processed data: {output_file}")

        # Save station information
        station_info_file = os.path.join(self.output_folder, 'station_info.txt')
        with open(station_info_file, 'w', encoding='utf-8') as f:
            f.write("# Station information\n")
            f.write("# Index\tStationName\tCity\tLongitude\tLatitude\n")
            for i, name in enumerate(station_names):
                f.write(f"{i}\t{name}\tunknown\t0\t0\n")
        logger.info(f"✅ Saved station info: {station_info_file}")

        # Save feature mapping
        feature_mapping_file = os.path.join(self.output_folder, 'feature_mapping.txt')
        with open(feature_mapping_file, 'w', encoding='utf-8') as f:
            f.write("# Feature mapping\n")
            f.write(f"# Total features: {len(features)}\n")
            f.write("# Format: Index\tFeatureName\tDescription\n")
            for i, feature in enumerate(features):
                f.write(f"{i}\t{feature}\tFeature_{i}\n")
        logger.info(f"✅ Saved feature mapping: {feature_mapping_file}")

        save_time = time.time() - save_start
        logger.info(f"💾 All files saved in {save_time:.2f}s")

    def process_all(self):
        """Main processing function"""
        total_start = time.time()

        logger.info("🚀 STARTING SERVER-OPTIMIZED PROCESSING")

        # Create output directory
        os.makedirs(self.output_folder, exist_ok=True)

        # Split files into batches
        batches = []
        for i in range(0, len(self.station_files), self.batch_size):
            batch = self.station_files[i:i + self.batch_size]
            batches.append(batch)

        total_batches = len(batches)
        logger.info(f"📦 Total batches: {total_batches}")

        # Process each batch
        all_batch_results = []
        total_failed = []

        for batch_num, batch_files in enumerate(batches, 1):
            batch_successful, batch_failed = self.process_batch(
                batch_files, batch_num, total_batches, self.time_index_map
            )

            # Save batch results
            batch_result = self.save_batch_results(batch_successful, batch_num)
            all_batch_results.append(batch_result)
            total_failed.extend(batch_failed)

            # Memory cleanup
            del batch_successful, batch_failed, batch_result

        # Combine all batches
        final_data, station_names, features = self.combine_batches(all_batch_results)

        # Save final results
        self.save_final_results(final_data, station_names, features)

        # Final summary
        total_time = time.time() - total_start
        total_processed = len(self.station_files) - len(total_failed)

        logger.info("🎉 PROCESSING COMPLETED!")
        logger.info(f"📊 Total files: {len(self.station_files)}")
        logger.info(f"✅ Successful: {total_processed}")
        logger.info(f"❌ Failed: {len(total_failed)}")
        logger.info(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f"🚀 Average speed: {total_processed/total_time:.2f} files/second")
        logger.info(f"💾 Output ready for: python train.py")

def main():
    # Configuration
    csv_folder = "/root/autodl-tmp/合并数据_去PRMSL"
    output_folder = "./processed_data"

    try:
        preprocessor = ServerPreprocessor(csv_folder, output_folder)
        preprocessor.process_all()
    except KeyboardInterrupt:
        logger.info("⚠️  Processing interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()