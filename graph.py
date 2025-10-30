import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
import numpy as np
import torch
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as mpcalc
from bresenham import bresenham
import pandas as pd

# File paths - update to use processed data
city_fp = os.path.join(proj_dir, 'processed_data/station_info.txt')
altitude_fp = os.path.join(proj_dir, 'data/altitude.npy')
blocked_edges_fp = '/root/autodl-tmp/blocked_edges_1200.csv'


class Graph():
    def __init__(self):
        self.dist_thres = 3  # Distance threshold in degrees
        self.use_blocked_edges = True  # Use blocked edges file
        self.use_altitude = False  # Disable altitude-based filtering since we have blocked edges

        self.altitude = self._load_altitude()
        self.nodes = self._gen_nodes()
        self.node_attr = self._add_node_attr()
        self.node_num = len(self.nodes)
        self.blocked_edges = self._load_blocked_edges()
        self.edge_index, self.edge_attr = self._gen_edges()
        self.edge_num = self.edge_index.shape[1]
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]

    def _load_altitude(self):
        """Load altitude data"""
        if os.path.isfile(altitude_fp):
            return np.load(altitude_fp)
        else:
            print(f"Warning: Altitude file not found at {altitude_fp}")
            return None

    def _load_blocked_edges(self):
        """Load blocked edges from CSV file"""
        blocked_edges = set()
        if self.use_blocked_edges and os.path.isfile(blocked_edges_fp):
            try:
                df = pd.read_csv(blocked_edges_fp)
                for _, row in df.iterrows():
                    src_name = row['src_name']
                    dst_name = row['dst_name']
                    blocked = row['blocked']
                    if blocked == 1:
                        blocked_edges.add((src_name, dst_name))
                print(f"Loaded {len(blocked_edges)} blocked edges")
            except Exception as e:
                print(f"Error loading blocked edges: {e}")
                print("Will use distance-based filtering only")
        else:
            print(f"Blocked edges file not found: {blocked_edges_fp}")
        return blocked_edges

    def _lonlat2xy(self, lon, lat, is_aliti):
        if is_aliti:
            lon_l = 100.0
            lon_r = 128.0
            lat_u = 48.0
            lat_d = 16.0
            res = 0.05
        else:
            lon_l = 103.0
            lon_r = 122.0
            lat_u = 42.0
            lat_d = 28.0
            res = 0.125
        x = np.int64(np.round((lon - lon_l - res / 2) / res))
        y = np.int64(np.round((lat_u + res / 2 - lat) / res))
        return x, y

    def _gen_nodes(self):
        """Generate nodes from station info file"""
        nodes = OrderedDict()
        with open(city_fp, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.startswith('#'):  # Skip comments
                    continue
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 5:
                    idx = int(parts[0])
                    station_name = parts[1]
                    city = parts[2]
                    lon = float(parts[3])
                    lat = float(parts[4])

                    # Calculate altitude if available
                    altitude = 0.0
                    if self.altitude is not None:
                        try:
                            x, y = self._lonlat2xy(lon, lat, True)
                            if 0 <= y < self.altitude.shape[0] and 0 <= x < self.altitude.shape[1]:
                                altitude = self.altitude[y, x]
                        except:
                            altitude = 0.0

                    nodes.update({idx: {
                        'city': city,
                        'station_name': station_name,
                        'altitude': altitude,
                        'lon': lon,
                        'lat': lat
                    }})
        return nodes

    def _add_node_attr(self):
        node_attr = []
        altitude_arr = []
        for i in self.nodes:
            altitude = self.nodes[i]['altitude']
            altitude_arr.append(altitude)
        altitude_arr = np.stack(altitude_arr)
        node_attr = np.stack([altitude_arr], axis=-1)
        return node_attr

    def traverse_graph(self):
        lons = []
        lats = []
        citys = []
        idx = []
        for i in self.nodes:
            idx.append(i)
            city = self.nodes[i]['city']
            lon, lat = self.nodes[i]['lon'], self.nodes[i]['lat']
            lons.append(lon)
            lats.append(lat)
            citys.append(city)
        return idx, citys, lons, lats

    def gen_lines(self):

        lines = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            lines.append(([src_lon, dest_lon], [src_lat, dest_lat]))

        return lines

    def _gen_edges(self):
        """Generate edges with distance filtering and blocked edges"""
        coords = []
        for i in self.nodes:
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])

        # Calculate great circle distances using geopy
        dist_matrix = np.zeros((self.node_num, self.node_num))
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i != j:
                    src_location = (self.nodes[i]['lat'], self.nodes[i]['lon'])
                    dest_location = (self.nodes[j]['lat'], self.nodes[j]['lon'])
                    dist_km = geodesic(src_location, dest_location).kilometers
                    # Convert to degrees (approximately)
                    dist_matrix[i, j] = dist_km / 111.0  # 1 degree ≈ 111 km

        # Apply distance threshold
        adj = (dist_matrix <= self.dist_thres).astype(np.uint8)
        assert adj.shape == dist_matrix.shape
        dist = dist_matrix * adj
        edge_index, dist = dense_to_sparse(torch.tensor(dist))
        edge_index, dist = edge_index.numpy(), dist.numpy()

        # Filter edges based on blocked edges
        filtered_edges = []
        filtered_dist = []
        direc_arr = []
        dist_kilometer = []

        for i in range(edge_index.shape[1]):
            src_idx, dest_idx = edge_index[0, i], edge_index[1, i]
            src_name = self.nodes[src_idx]['station_name']
            dest_name = self.nodes[dest_idx]['station_name']

            # Check if this edge is blocked
            if (src_name, dest_name) not in self.blocked_edges:
                src_lat, src_lon = self.nodes[src_idx]['lat'], self.nodes[src_idx]['lon']
                dest_lat, dest_lon = self.nodes[dest_idx]['lat'], self.nodes[dest_idx]['lon']
                src_location = (src_lat, src_lon)
                dest_location = (dest_lat, dest_lon)
                dist_km = geodesic(src_location, dest_location).kilometers
                v, u = src_lat - dest_lat, src_lon - dest_lon

                u = u * units.meter / units.second
                v = v * units.meter / units.second
                direc = mpcalc.wind_direction(u, v)._magnitude

                filtered_edges.append([src_idx, dest_idx])
                filtered_dist.append(dist[i])
                direc_arr.append(direc)
                dist_kilometer.append(dist_km)

        if len(filtered_edges) == 0:
            print("Warning: No edges remain after filtering!")
            edge_index = np.array([[0], [0]])
            attr = np.array([[0.0, 0.0]])
        else:
            edge_index = np.array(filtered_edges).T
            direc_arr = np.stack(direc_arr)
            dist_arr = np.stack(dist_kilometer)
            attr = np.stack([dist_arr, direc_arr], axis=-1)

        print(f"Generated {len(filtered_edges)} edges after filtering")
        return edge_index, attr

    def _update_edges(self):
        edge_index = []
        edge_attr = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_x, src_y = self._lonlat2xy(src_lon, src_lat, True)
            dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat, True)
            points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1,0))
            altitude_points = self.altitude[points[0], points[1]]
            altitude_src = self.altitude[src_y, src_x]
            altitude_dest = self.altitude[dest_y, dest_x]
            if np.sum(altitude_points - altitude_src > self.alti_thres) < 3 and \
               np.sum(altitude_points - altitude_dest > self.alti_thres) < 3:
                edge_index.append(self.edge_index[:,i])
                edge_attr.append(self.edge_attr[i])

        self.edge_index = np.stack(edge_index, axis=1)
        self.edge_attr = np.stack(edge_attr, axis=0)


if __name__ == '__main__':
    graph = Graph()