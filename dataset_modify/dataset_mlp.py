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
        """Process features for MLP model - avoid data leakage"""

        # Get feature configuration
        metero_var = config['data']['metero_var']
        metero_use = config['experiments']['metero_use']

        # Find indices of selected features in our 666-dimension data
        metero_idx = []
        for var in metero_use:
            if var in metero_var:
                idx = metero_var.index(var)
                metero_idx.append(idx)

        # Extract selected features
        self.feature = self.feature[:,:,metero_idx]

        # Check wind features for model compatibility
        if 'WSPD100' in metero_use and 'WDIR100' in metero_use:
            wspd_idx = metero_use.index('WSPD100')
            wdir_idx = metero_use.index('WDIR100')

            # Extract wind features
            speed = self.feature[:, :, wspd_idx]
            direc = self.feature[:, :, wdir_idx]

            # Remove wind features from current position
            other_indices = [i for i in range(len(metero_use)) if i not in [wspd_idx, wdir_idx]]
            other_features = self.feature[:, :, other_indices]

            # Add wind features at the end (for model compatibility)
            self.feature = np.concatenate([other_features, speed[:, :, None], direc[:, :, None]], axis=-1)

            # Update wind statistics
            self.wind_mean = np.array([speed.mean(), direc.mean()])
            self.wind_std = np.array([speed.std(), direc.std()])
        else:
            # Default wind statistics if not available
            self.wind_mean = np.array([0.0, 0.0])
            self.wind_std = np.array([1.0, 1.0])

    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        self.pm25 = self.pm25[start_idx: end_idx+1, :]
        self.feature = self.feature[start_idx: end_idx+1, :]
        self.time_arr = self.time_arr[start_idx: end_idx+1]
        self.time_arrow = self.time_arrow[start_idx: end_idx + 1]

    def _gen_time_arr(self):
        """Generate time array for hourly data"""
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
        """Get time index for hourly data"""
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