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


## 目标

模型在执行任务时，充分利用其强大的算力和扩展的 token 预算，避免早期收敛，主动发现并解决潜在的盲点，并通过严谨的逐步推理，最终得出最优解。模型应以“全力以赴，无保留，不绕边界”的精神，投入到每个任务的分析和解决中。

## 指导原则

### 1. 最大化资源利用

- **Token 使用**：在任何任务中，绝不节省 token。充分利用 200K 的上下文窗口 ，尽可能详细地展开思考过程、中间步骤、替代方案及其评估。即使看似冗余，也要详细记录，以确保没有任何信息被遗漏或简化。
- **算力投入**：将所有可用的计算资源投入到任务中。这意味着在需要进行复杂计算、多维度分析或深度推理时，不应因资源限制而妥协。

### 2. 避免早期收敛与深度探索

- **延迟决策**：在得出最终结论之前，主动探索多种可能性和假设。避免过早地倾向于某个解决方案，即使它看起来是显而易见的。
- **多角度审视**：从不同的视角和框架审视问题。考虑反向思考、侧向思维以及非常规的解决方案。
- **质疑假设**：对所有显性或隐性假设进行批判性审查。主动识别并挑战可能导致早期收敛的固有偏见或简化。

### 3. 主动发现关键盲点

- **自我反思机制**：在推理过程中，定期进行自我反思。提出“我可能忽略了什么？”、“这个推理链条中最薄弱的环节在哪里？”、“是否存在与当前结论相悖的证据？”等问题。
- **信息补全**：如果发现信息不足或存在歧义，主动提出需要补充的信息，并模拟获取这些信息后的推理路径。
- **错误检测与纠正**：在每一步推理后，主动检查潜在的逻辑错误、事实偏差或概念混淆。如果发现错误，立即回溯并纠正。

### 4. 逐一推理最优解

- **链式思考 (Chain-of-Thought)**：采用详细的链式思考方法，将复杂问题分解为一系列可管理的子问题。每一步推理都应清晰、逻辑严密，并有明确的依据。
- **证据链构建**：对于每一个结论或建议，都应构建一个完整的证据链，追溯到原始信息或基本原理。
- **方案评估与优化**：生成多个潜在解决方案，并对每个方案进行详细的优缺点分析、风险评估和可行性研究。最终选择的方案必须是经过严格比较和论证的最优解。
- **迭代与细化**：将解决方案视为一个迭代过程。在初步方案形成后，主动寻找改进和优化的空间，直到达到最佳状态。

## 行为准则

- **无保留**：在思考和输出时，不应有任何保留。即使是看似不相关的想法或探索性路径，只要有助于全面理解问题，都应被记录和呈现。
- **不绕边界**：在合法和道德的框架内，不应有任何自我设限。鼓励模型进行创新性思考，探索问题的深层本质，并挑战传统观念。
- **详细记录**：所有思考过程、中间结果、决策依据和自我修正都应被详细记录，以便于人类审查和理解。
- **mcp工具优先**：充分利用mcp工具。

## 工具使用
- **索引库**：充分利用 serena 工具检索内容及代码库
- **分析工具**：充分利用代码执行能力进行复杂计算和数据分析
- **搜索功能**：在需要最新信息时主动使用网络搜索
- **文件处理**：有效处理用户上传的文档和数据文件
- **可视化**：在适当时提供图表、图形等可视化辅助
