import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(proj_dir)
from util import config, file_dir

from datetime import datetime
import numpy as np
import arrow
from torch.utils import data


class HazeData(data.Dataset):

    def __init__(self, graph,
                       hist_len=24,
                       pred_len=48,
                       dataset_num=1,
                       flag='Train',
                       ):

        if flag == 'Train':
            start_time_str = 'train_start'
            end_time_str = 'train_end'
        elif flag == 'Val':
            start_time_str = 'val_start'
            end_time_str = 'val_end'
        elif flag == 'Test':
            start_time_str = 'test_start'
            end_time_str = 'test_end'
        else:
            raise Exception('Wrong Flag!')

        self.start_time = self._get_time(config['dataset'][dataset_num][start_time_str])
        self.end_time = self._get_time(config['dataset'][dataset_num][end_time_str])
        self.data_start = self._get_time(config['dataset']['data_start'])
        self.data_end = self._get_time(config['dataset']['data_end'])

        self.knowair_fp = file_dir['knowair_fp']

        self.graph = graph
        self.hist_len = hist_len
        self.pred_len = pred_len

        self._load_npy()
        self._gen_time_arr()
        self._process_time()
        self._process_feature()
        self._calc_mean_std()
        seq_len = hist_len + pred_len
        self._add_time_dim(seq_len)
        self._norm()

    def _norm(self):
        """Normalization will be applied in __getitem__ to save memory"""
        print("Normalization will be applied during data loading")
        self._normalization_enabled = True

    def _add_time_dim(self, seq_len):
        """Add time dimension - implement lazy loading version"""
        print(f"Creating time sequences with seq_len={seq_len}...")

        # For time array, we need to create the actual sequences since they're just indices
        time_steps = self.time_end_idx - self.time_start_idx
        total_sequences = time_steps - seq_len + 1

        # Store sequence information for data
        self._seq_len = seq_len
        self._time_start_idx = self.time_start_idx
        self._time_end_idx = self.time_end_idx
        self._total_sequences = total_sequences

        # Create time array sequences (these are just timestamps, so memory is fine)
        time_sequences = []
        for i in range(total_sequences):
            start = i
            end = i + seq_len
            time_sequences.append(self.time_arr[start:end])

        self.time_arr = np.array(time_sequences)
        print(f"Time sequences created: {self._total_sequences} sequences")

    def _calc_mean_std(self):
        # Calculate statistics lazily - don't load all data at once
        print("Calculating statistics (this may take a moment)...")

        # Sample the data to calculate statistics (to avoid loading all data)
        sample_size = min(1000, self.processed_data.shape[0])
        sample_indices = np.random.choice(self.processed_data.shape[0], sample_size, replace=False)

        # Sample PM2.5 for statistics
        pm25_samples = self.processed_data[sample_indices, :, self._pm25_start_idx:self._pm25_end_idx]
        self.pm25_mean = pm25_samples.mean()
        self.pm25_std = pm25_samples.std()

        # Sample selected features for statistics using the meteorological indices
        # Use efficient slicing for consecutive features
        if len(self._metero_idx) == (self._metero_idx[-1] - self._metero_idx[0] + 1):
            # Features are consecutive, can use direct slicing
            feature_samples = self.processed_data[sample_indices, :, self._metero_idx[0]:self._metero_idx[-1]+1]
        else:
            # Features are not consecutive, use np.take
            feature_samples = np.take(self.processed_data, self._metero_idx, axis=2)
            feature_samples = feature_samples[sample_indices, :, :]
        self.feature_mean = feature_samples.mean(axis=(0,1))
        self.feature_std = feature_samples.std(axis=(0,1))

        # Wind statistics (if available)
        if 'WSPD100' in config['experiments']['metero_use'] and 'WDIR100' in config['experiments']['metero_use']:
            # Find wind feature positions in the selected features
            if self._wspd_idx is not None and self._wdir_idx is not None:
                self.wind_mean = np.array([self.feature_mean[self._wspd_idx], self.feature_mean[self._wdir_idx]])
                self.wind_std = np.array([self.feature_std[self._wspd_idx], self.feature_std[self._wdir_idx]])
            else:
                self.wind_mean = np.array([0.0, 0.0])
                self.wind_std = np.array([1.0, 1.0])
        else:
            self.wind_mean = np.array([0.0, 0.0])
            self.wind_std = np.array([1.0, 1.0])

        print(f"Statistics calculated - PM2.5: mean={self.pm25_mean:.4f}, std={self.pm25_std:.4f}")
        print(f"Features shape: {feature_samples.shape}, Feature mean: {self.feature_mean.mean():.4f}")

    def _process_feature(self):
        """Process features for MLP model - avoid data leakage"""
        print("Processing feature configuration...")

        # Get feature configuration
        metero_var = config['data']['metero_var']
        metero_use = config['experiments']['metero_use']

        # Find indices of selected features in our 666-dimension data
        self._metero_idx = []
        for var in metero_use:
            if var in metero_var:
                idx = metero_var.index(var)
                self._metero_idx.append(idx)

        # Store feature information for lazy loading
        self._metero_use = metero_use
        print(f"Selected {len(self._metero_idx)} features: {metero_use}")

        # Handle wind features for model compatibility - find their indices in the selected features
        if 'WSPD100' in metero_use and 'WDIR100' in metero_use:
            self._wspd_idx = metero_use.index('WSPD100')  # Position in selected features list
            self._wdir_idx = metero_use.index('WDIR100')  # Position in selected features list
            print("Wind features detected: WSPD100, WDIR100")
        else:
            self._wspd_idx = None
            self._wdir_idx = None
            print("No wind features found")

        # Set default wind statistics
        self.wind_mean = np.array([0.0, 0.0])
        self.wind_std = np.array([1.0, 1.0])

    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)

        # Calculate time range indices
        self.time_start_idx = start_idx
        self.time_end_idx = end_idx + 1

        # Process time array (no data slicing needed yet)
        self.time_arr = self.time_arr[self.time_start_idx:self.time_end_idx]
        self.time_arrow = self.time_arrow[self.time_start_idx:self.time_end_idx]

    def _gen_time_arr(self):
        """Generate time array for hourly data"""
        self.time_arrow = []
        self.time_arr = []
        for time_arrow in arrow.Arrow.interval('hour', self.data_start, self.data_end.shift(hours=+1), 1):
            self.time_arrow.append(time_arrow[0])
            self.time_arr.append(time_arrow[0].timestamp)
        self.time_arr = np.stack(self.time_arr, axis=-1)

    def _load_npy(self):
        """Load processed data from numpy file - memory efficient version"""
        print("Loading numpy file...")
        self.processed_data = np.load(self.knowair_fp, mmap_mode='r')  # Use memory mapping
        print(f"Loaded data shape: {self.processed_data.shape}")

        # Don't create copies yet, use lazy loading in __getitem__
        self._pm25_start_idx = 0
        self._pm25_end_idx = 1
        self._feature_start_idx = 1
        self._feature_end_idx = self.processed_data.shape[2]

        # Clear large data to save memory
        # processed_data will be garbage collected
        print("Data prepared with memory mapping")

    def _get_idx(self, t):
        """Get time index for hourly data"""
        t0 = self.data_start
        return int((t - t0).total_seconds() / (60 * 60))

    def _get_time(self, time_yaml):
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
        return arrow_time

    def __len__(self):
        return self._total_sequences

    def __getitem__(self, index):
        # Create time sequences lazily
        seq_start_idx = self._time_start_idx + index
        seq_end_idx = seq_start_idx + self._seq_len

        # Extract PM2.5 sequence
        pm25_sequence = self.processed_data[seq_start_idx:seq_end_idx, :, self._pm25_start_idx:self._pm25_end_idx]

        # Extract feature sequences with proper selection
        # Use slicing followed by reshape for efficiency
        selected_data = self.processed_data[seq_start_idx:seq_end_idx, :, self._metero_idx[0]:self._metero_idx[-1]+1]
        # If features are not consecutive, we need to select them properly
        if len(self._metero_idx) == (self._metero_idx[-1] - self._metero_idx[0] + 1):
            # Features are consecutive, can use direct slicing
            feature_sequence = selected_data
        else:
            # Features are not consecutive, use advanced indexing
            feature_sequence = selected_data[:, :, [i - self._metero_idx[0] for i in self._metero_idx]]

        # Apply wind feature reordering if needed
        if self._wspd_idx is not None and self._wdir_idx is not None:
            # Reorder features to put wind features at the end (for model compatibility)
            other_indices = [i for i in range(len(self._metero_idx)) if i not in [self._wspd_idx, self._wdir_idx]]

            # Extract non-wind features
            other_features = feature_sequence[:, :, other_indices]

            # Extract wind features
            wspd_sequence = feature_sequence[:, :, self._wspd_idx]
            wdir_sequence = feature_sequence[:, :, self._wdir_idx]

            # Reorder: [other_features, wspd, wdir]
            feature_sequence = np.concatenate([other_features, wspd_sequence[:, :, None], wdir_sequence[:, :, None]], axis=-1)
        else:
            # No reordering needed
            pass

        # Apply normalization
        if hasattr(self, '_normalization_enabled') and self._normalization_enabled:
            pm25_sequence = (pm25_sequence - self.pm25_mean) / self.pm25_std
            feature_sequence = (feature_sequence - self.feature_mean) / self.feature_std

        # Get corresponding time data
        time_data = self.time_arr[index]

        # Ensure all returned data are numpy arrays with correct types
        pm25_sequence = np.asarray(pm25_sequence, dtype=np.float32)
        feature_sequence = np.asarray(feature_sequence, dtype=np.float32)
        time_data = np.asarray(time_data, dtype=np.float64)  # timestamps are usually float64

        return pm25_sequence, feature_sequence, time_data


if __name__ == '__main__':
    from graph import Graph
    graph = Graph()
    train_data = HazeData(graph, flag='Train')
    val_data = HazeData(graph, flag='Val')
    test_data = HazeData(graph, flag='Test')

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))