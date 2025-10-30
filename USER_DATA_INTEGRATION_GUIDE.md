# 用户数据集成指南

## 概述

本文档说明如何将你的206个站点CSV数据集成到PM2.5-GNN项目中。项目已经过修改以支持你的数据格式，包括观测数据、时间特征和72小时预报数据。

## 数据格式要求

### 输入CSV文件格式
- **位置**: 包含所有站点CSV文件的文件夹
- **命名**: `站点名称.csv` (例如: `万安.csv`)
- **编码**: UTF-8
- **时间范围**: 2022-01-01 00:00 到 2024-12-31 23:00 (逐小时)

### CSV列结构
```
DateTime, City, Station_name, Lon, Lat, Month, Weekday, Hour,
PM2.5, PM10, SO2, NO2, O3, CO, T2M_C, D2M_C, PRES_hPa, APCP1h_mm,
RH2M_pct, WSPD100, WDIR100, U100, V100,
T2M_C(01), D2M_C(01), ..., APCP1h_mm(72)
```

**详细说明**:
1. **基础列**: DateTime, City, Station_name, Lon, Lat, Month, Weekday, Hour
2. **观测变量**: PM2.5, PM10, SO2, NO2, O3, CO, T2M_C, D2M_C, PRES_hPa, APCP1h_mm, RH2M_pct, WSPD100, WDIR100, U100, V100 (15个)
3. **预报变量**: 8个变量 × 72小时 = 576个
   - 格式: `变量名(小时数)`，如 `T2M_C(01)` 表示第1小时的气温预报
   - 变量: T2M_C, D2M_C, PRES_hPa, APCP1h_mm, RH2M_pct, WSPD100, WDIR100, U100, V100

## 集成步骤

### 第1步: 数据预处理 (一次性)

1. **更新预处理脚本路径**:
   编辑 `preprocess_data.py` 第251行:
   ```python
   csv_folder = "/path/to/your/csv/folder"  # 修改为你的CSV文件夹路径
   ```

2. **运行预处理**:
   ```bash
   python preprocess_data.py
   ```

3. **输出文件**:
   - `processed_data/processed_data.npy` - 主数据文件 (T, 206, 594)
   - `processed_data/station_info.txt` - 站点信息
   - `processed_data/feature_mapping.txt` - 特征映射说明

### 第2步: 配置项目

1. **确认阻断表路径**:
   检查 `graph.py` 第19行的路径是否正确:
   ```python
   blocked_edges_fp = '/root/autodl-tmp/blocked_edges_1200.csv'
   ```

2. **调整训练参数** (可选):
   编辑 `config.yaml`:
   - `hist_len`: 历史数据长度 (默认24小时)
   - `pred_len`: 预测长度 (默认48小时)
   - `forecast.use_hours`: 使用的预报数据长度 (48或72小时)

### 第3步: 训练模型

```bash
python train.py
```

## 关键修改说明

### 数据加载 (dataset.py)
- 支持逐小时数据 (原始为3小时间隔)
- 自动提取PM2.5作为目标变量
- 风速风向处理保持原始格式 (WSPD100, WDIR100)
- 时间特征直接使用，不再动态添加

### 图结构 (graph.py)
- 使用你提供的站点信息文件
- 集成有向阻断表过滤连接
- 基于大圆距离的距离阈值过滤
- 保持原始的边属性计算逻辑

### 模型兼容性
- **无需修改模型代码**: 所有模型 (PM25_GNN, GC_LSTM, LSTM等) 自动适配输入维度
- **预报数据使用**: 在预测时刻自动使用对应时间的预报数据更新图状态
- **特征维度**: 模型根据配置自动调整输入层维度

## 配置选项

### 可配置参数 (config.yaml)

```yaml
experiments:
  metero_use: [可选择的特征列表]  # 选择使用的特征
  model: PM25_GNN                 # 选择模型

train:
  hist_len: 24                    # 历史数据长度
  pred_len: 48                    # 预测长度

forecast:
  use_hours: 48                   # 使用48小时预报数据
  forecast_vars: [预报变量列表]    # 8个预报变量
```

### 支持的模型
- `PM25_GNN`: 主要图神经网络模型 (推荐)
- `PM25_GNN_nosub`: 无子图连接的变体
- `GC_LSTM`: 图卷积LSTM
- `LSTM`: 标准LSTM
- `GRU`: GRU模型
- `nodesFC_GRU`: 节点全连接GRU
- `MLP`: 多层感知机

## 故障排除

### 常见问题

1. **数据加载失败**:
   - 检查CSV文件路径和命名
   - 确认时间格式和列名正确

2. **阻断表加载失败**:
   - 检查 `blocked_edges_fp` 路径
   - 确认阻断表列名: `src_name`, `dst_name`, `blocked`

3. **内存不足**:
   - 减少批量大小 (`batch_size`)
   - 减少历史长度 (`hist_len`)

4. **训练不收敛**:
   - 调整学习率 (`lr`)
   - 增加训练轮数 (`epochs`)
   - 检查数据质量和缺失值

### 验证数据完整性

```python
import numpy as np

# 检查预处理后的数据
data = np.load('processed_data/processed_data.npy')
print(f"数据形状: {data.shape}")  # 应该显示 (26304, 206, 594)
print(f"时间步数: {data.shape[0]}")  # 应该是 26304 (3年 × 365天 × 24小时)
print(f"站点数: {data.shape[1]}")    # 应该是 206
print(f"特征数: {data.shape[2]}")    # 应该是 594
```

## 性能建议

1. **数据预处理**: 预处理只需要运行一次，后续训练直接使用npy文件
2. **预报数据**: 根据预测需求选择48或72小时预报，减少不必要的数据使用
3. **批处理**: 根据GPU内存调整批量大小
4. **模型选择**: PM25_GNN和GC_LSTM能最好利用预报数据和图结构信息

## 支持的功能

✅ **完全支持**:
- 206个站点的逐时数据
- 594个特征 (观测+时间+预报)
- 有向阻断表过滤
- 48/72小时预报数据
- 所有原始模型
- 动态图结构更新

✅ **保持兼容**:
- 原始训练逻辑
- 模型算法结构
- 评估指标计算
- 结果保存格式