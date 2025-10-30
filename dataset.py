import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
from util import config, file_dir

from datetime import datetime
import numpy as np
import arrow
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils import data
import pandas as pd


class HazeData(data.Dataset):

    def __init__(self, graph,
                       hist_len=1,
                       pred_len=24,
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

        self._load_npy()
        self._gen_time_arr()
        self._process_time()
        self._process_feature()
        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)
        self._calc_mean_std()
        seq_len = hist_len + pred_len
        self._add_time_dim(seq_len)
        self._norm()

    def _norm(self):
        self.feature = (self.feature - self.feature_mean) / self.feature_std
        self.pm25 = (self.pm25 - self.pm25_mean) / self.pm25_std

    def _add_time_dim(self, seq_len):

        def _add_t(arr, seq_len):
            t_len = arr.shape[0]
            assert t_len > seq_len
            arr_ts = []
            for i in range(seq_len, t_len):
                arr_t = arr[i-seq_len:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts

        self.pm25 = _add_t(self.pm25, seq_len)
        self.feature = _add_t(self.feature, seq_len)
        self.time_arr = _add_t(self.time_arr, seq_len)

    def _calc_mean_std(self):
        self.feature_mean = self.feature.mean(axis=(0,1))
        self.feature_std = self.feature.std(axis=(0,1))
        self.wind_mean = self.feature_mean[-2:]
        self.wind_std = self.feature_std[-2:]
        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()

    def _process_feature(self):
        """Process features: select variables and handle forecast data"""
        # Get configuration
        metero_var = config['data']['metero_var']
        metero_use = config['experiments']['metero_use']

        # Select observation variables (excluding PM2.5 which is already extracted)
        obs_vars = [var for var in metero_use if var != 'PM2.5']
        metero_idx = [metero_var.index(var) - 1 for var in obs_vars]  # -1 because PM2.5 is removed
        self.feature = self.feature[:,:,metero_idx]

        # Handle wind speed and direction if available
        # Check if WSPD100 and WDIR100 are in selected variables
        if 'WSPD100' in obs_vars and 'WDIR100' in obs_vars:
            # Extract wind speed and direction for model compatibility
            wspd_idx = obs_vars.index('WSPD100')
            wdir_idx = obs_vars.index('WDIR100')

            speed = self.feature[:, :, wspd_idx]
            direc = self.feature[:, :, wdir_idx]

            # For model compatibility, ensure wind features are at the end
            # Remove them from current position and add to the end
            other_features = np.delete(self.feature, [wspd_idx, wdir_idx], axis=2)
            self.feature = np.concatenate([other_features, speed[:, :, None], direc[:, :, None]], axis=-1)

            # Update wind mean and std for model compatibility
            self.wind_mean = np.array([speed.mean(), direc.mean()])
            self.wind_std = np.array([speed.std(), direc.std()])
        else:
            # If no wind data, set default values
            self.wind_mean = np.array([0.0, 0.0])
            self.wind_std = np.array([1.0, 1.0])

        # Note: Time features (Month, Weekday, Hour) are already in the data
        # No need to add them dynamically like in the original code

    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        self.pm25 = self.pm25[start_idx: end_idx+1, :]
        self.feature = self.feature[start_idx: end_idx+1, :]
        self.time_arr = self.time_arr[start_idx: end_idx+1]
        self.time_arrow = self.time_arrow[start_idx: end_idx + 1]

    def _gen_time_arr(self):
        """Generate time array for hourly data (changed from 3-hourly to hourly)"""
        self.time_arrow = []
        self.time_arr = []
        for time_arrow in arrow.Arrow.interval('hour', self.data_start, self.data_end.shift(hours=+1), 1):
            self.time_arrow.append(time_arrow[0])
            self.time_arr.append(time_arrow[0].timestamp)
        self.time_arr = np.stack(self.time_arr, axis=-1)

    def _load_npy(self):
        """Load processed data from numpy file"""
        self.processed_data = np.load(self.knowair_fp)

        # Extract PM2.5 (first feature in our processed data)
        self.pm25 = self.processed_data[:, :, 0:1]

        # Extract other features (all except PM2.5)
        self.feature = self.processed_data[:, :, 1:]

    def _get_idx(self, t):
        """Get time index for hourly data (changed from 3-hourly to hourly)"""
        t0 = self.data_start
        return int((t.timestamp - t0.timestamp) / (60 * 60))

    def _get_time(self, time_yaml):
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
        return arrow_time

    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        return self.pm25[index], self.feature[index], self.time_arr[index]

if __name__ == '__main__':
    from graph import Graph
    graph = Graph()
    train_data = HazeData(graph, flag='Train')
    val_data = HazeData(graph, flag='Val')
    test_data = HazeData(graph, flag='Test')

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))
