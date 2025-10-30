#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append('dataset_modify')

import numpy as np
from util import config, file_dir

def test_memeory_mapping():
    """Test the memory mapping feature extraction"""
    print("Testing memory mapping feature extraction...")

    # Load data with memory mapping
    knowair_fp = file_dir['knowair_fp']
    print(f"Loading data from: {knowair_fp}")

    try:
        processed_data = np.load(knowair_fp, mmap_mode='r')
        print(f"Data loaded successfully. Shape: {processed_data.shape}")

        # Get feature configuration
        metero_var = config['data']['metero_var']
        metero_use = config['experiments']['metero_use']

        print(f"Available features ({len(metero_var)}): {metero_var}")
        print(f"Selected features ({len(metero_use)}): {metero_use}")

        # Find indices of selected features
        metero_idx = []
        for var in metero_use:
            if var in metero_var:
                idx = metero_var.index(var)
                metero_idx.append(idx)

        print(f"Feature indices: {metero_idx}")

        # Test feature extraction with small sample
        sample_size = min(10, processed_data.shape[0])
        sample_indices = np.random.choice(processed_data.shape[0], sample_size, replace=False)

        print(f"Testing feature extraction with {sample_size} samples...")

        # Test consecutive feature optimization
        if len(metero_idx) == (metero_idx[-1] - metero_idx[0] + 1):
            print("Features are consecutive - using direct slicing")
            feature_samples = processed_data[sample_indices, :, metero_idx[0]:metero_idx[-1]+1]
        else:
            print("Features are not consecutive - using advanced indexing")
            feature_samples = np.take(processed_data, metero_idx, axis=2)
            feature_samples = feature_samples[sample_indices, :, :]

        print(f"Feature samples shape: {feature_samples.shape}")
        print(f"Feature mean: {feature_samples.mean(axis=(0,1))}")
        print(f"Feature std: {feature_samples.std(axis=(0,1))}")

        print("SUCCESS: Memory mapping feature extraction works!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_memeory_mapping()