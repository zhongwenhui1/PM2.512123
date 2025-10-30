#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('dataset_modify')
sys.path.append('model')

import torch
from dataset_mlp import HazeData
from graph import Graph
from model.MLP import MLP
from util import config

def test_mlp_training_setup():
    """Test the MLP training setup without actually training"""
    print("Testing MLP training setup...")

    try:
        # Create graph and dataset
        graph = Graph()

        # Configuration
        hist_len = config['train']['hist_len']
        pred_len = config['train']['pred_len']
        dataset_num = config['experiments']['dataset_num']

        print(f"Configuration: hist_len={hist_len}, pred_len={pred_len}")

        # Create a small dataset for testing
        print("Creating dataset...")
        train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')

        # Calculate dimensions
        in_dim = len(train_data._metero_idx)
        city_num = train_data.graph.node_num

        print(f"Dataset created successfully!")
        print(f"Dataset length: {len(train_data)}")
        print(f"Input dimension: {in_dim}")
        print(f"Number of cities: {city_num}")

        # Create model
        print("Creating MLP model...")
        model = MLP(hist_len, pred_len, in_dim)
        print(f"Model created: {model}")

        # Test getting a sample
        print("Testing data sample...")
        sample = train_data[0]
        pm25_seq, feature_seq, time_data = sample

        print(f"PM2.5 sequence shape: {pm25_seq.shape}")
        print(f"Feature sequence shape: {feature_seq.shape}")
        print(f"Time data shape: {time_data.shape}")

        # Convert to tensors
        pm25_tensor = torch.FloatTensor(pm25_seq).unsqueeze(0)  # Add batch dimension
        feature_tensor = torch.FloatTensor(feature_seq).unsqueeze(0)

        print(f"PM2.5 tensor shape: {pm25_tensor.shape}")
        print(f"Feature tensor shape: {feature_tensor.shape}")

        # Test model forward pass
        print("Testing model forward pass...")
        pm25_hist = pm25_tensor[:, :hist_len]
        feature = feature_tensor

        print(f"PM2.5 hist shape: {pm25_hist.shape}")
        print(f"Feature shape: {feature.shape}")

        with torch.no_grad():
            prediction = model(pm25_hist, feature)
            print(f"Prediction shape: {prediction.shape}")

        print("SUCCESS: MLP training setup works correctly!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mlp_training_setup()