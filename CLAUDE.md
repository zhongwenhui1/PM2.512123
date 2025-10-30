# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PM2.5-GNN is a domain knowledge enhanced graph neural network for PM2.5 forecasting. The project implements multiple neural network models including the main PM25_GNN model that incorporates meteorological data and geographic relationships between cities.

## Key Architecture Components

- **Graph System** (`graph.py`): Creates a graph representation of cities with edges based on geographic distance and altitude thresholds. Handles edge attributes (distance, direction) and node attributes (city coordinates, altitude).

- **Dataset Pipeline** (`dataset.py`): `HazeData` class loads and processes the KnowAir dataset, handling train/validation/test splits with configurable time windows. Integrates PM2.5 measurements with meteorological features.

- **Model Library** (`model/`): Multiple forecasting models:
  - `PM25_GNN.py`: Main graph neural network with wind dynamics and edge-aware message passing
  - `PM25_GNN_nosub.py`: Variant without subgraph connections
  - `GC_LSTM.py`, `LSTM.py`, `GRU.py`: Temporal sequence models
  - `MLP.py`: Baseline feedforward network
  - `nodesFC_GRU.py`: Node-wise fully connected GRU

- **Training Framework** (`train.py`): Main training loop with experiment repetition, early stopping, and comprehensive metrics (RMSE, MAE, CSI, POD, FAR).

## Configuration

All settings are in `config.yaml`:
- **Model Selection**: Uncomment one of the available models (MLP, LSTM, GRU, GC_LSTM, nodesFC_GRU, PM25_GNN, PM25_GNN_nosub)
- **Dataset Configuration**: Choose dataset number (1, 2, or 3) for different train/val/test time periods
- **Meteorological Variables**: Configure which weather features to use from the available 18 variables
- **File Paths**: Update `filepath` section with your server paths for KnowAir dataset and results directory

## Common Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

The training script will:
- Create result directories automatically based on timestamp and experiment index
- Save best models during training with early stopping
- Generate prediction/label/time arrays if `save_npy: True`
- Run multiple experiments (configurable via `exp_repeat`) and compute mean/std metrics

## Dataset Requirements

- **KnowAir Dataset**: Download from Google Drive or Baidu Yun as specified in README.md
- **Expected Format**: `.npy` file containing PM2.5 and meteorological data for all cities
- **Time Coverage**: 2015-2018 data (KnowAir-V2 extends to 2023 with additional variables)

## Model Input/Output

- **Input**: Historical PM2.5 values (`hist_len` hours) + meteorological features
- **Output**: PM2.5 forecasts for next `pred_len` hours (default 24 hours)
- **Graph Structure**: Cities as nodes, edges based on geographic proximity with wind direction weighting