# 面向PM2.5精准预测的自适应多尺度PINN框架

## 项目简介

本项目实现了一种面向PM2.5精准预测的自适应多尺度物理信息神经网络（PINN）框架，结合时空特征融合与可信AI方法，用于空气质量（PM2.5）的精准预测。该框架能够有效融合气象数据、地理信息和历史PM2.5浓度数据，通过物理约束提高预测的准确性和可靠性（Explainable AI-Driven Adaptive Multiscale PINN Framework(EAM-PINN) for Accurate PM2.5 Prediction with Spatiotemporal Feature Fusio）。

## 主要功能

- 实现了自适应多尺度PINN模型，融合物理约束和深度学习
- 支持多种主流深度学习模型对比（LSTM、GRU、Transformer、CNN-LSTM）
- 完整的数据处理流程，支持自定义时间序列长度
- 实时训练监控和收敛曲线可视化
- 多指标模型评估（MSE、RMSE、MAE、R²）
- 模型性能对比和可视化报告生成
- 简洁的命令行接口，方便用户使用
- **多源数据融合**：支持移动巡航、激光雷达、小微站等数据源
- **领域知识特征工程**：基于气象学原理的特征交互
- **模块化架构**：可独立调用数据处理、模型训练、可视化模块

## 目录结构

```
PINN_PM25/
├── data/                          # 数据目录
│   ├── pm25_data_template.csv     # 基础气象数据模板
│   ├── pm25_mobile_template.csv   # 移动巡航数据模板
│   ├── pm25_lidar_template.csv    # 激光雷达数据模板
│   └── pm25_microsite_template.csv # 小微站数据模板
├── models/                        # 模型目录
│   ├── adaptive_ms_pinn.py        # 自适应多尺度PINN模型
│   └── baseline_models.py         # 基线模型（LSTM、GRU等）
├── utils/                         # 工具模块
│   ├── data_processor.py          # 多源数据处理模块
│   ├── trainer.py                 # 训练和评估模块
│   ├── visualization.py           # 可视化模块
│   └── feature_fusion.py          # 特征融合模块
├── scripts/                       # 脚本文件
│   └── run_model.py               # 命令行运行脚本
├── test_new_features.py           # 新功能测试脚本
├── requirements.txt               # Python依赖包
├── config_unified.yaml            # 统一配置文件，支持单数据源和多数据源
└── README.md                      # 项目说明文档
```

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- PyYAML
- Argparse

## 快速开始

### 1. 环境配置

**推荐使用虚拟环境：**

```bash
# 创建虚拟环境
python -m venv pinn_env

# 激活虚拟环境（Windows）
pinn_env\Scripts\activate

# 安装依赖（推荐使用清华镜像源）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 2. 验证安装

```bash
# 测试多源数据处理功能
python test_new_features.py

# 测试基础模型运行
python scripts/run_model.py --data_path data/pm25_data_template.csv --model_type lstm --epochs 5 --seq_len 12
```

## 数据格式与模板

### 基础气象数据模板

项目提供了多种数据模板，支持多源数据融合：

**基础气象数据模板 (`data/pm25_data_template.csv`)**：

| 字段名         | 类型     | 描述                 |
|--------------|----------|----------------------|
| datetime     | 字符串   | 时间戳（YYYY-MM-DD HH:MM:SS） |
| pm25         | 数值     | PM2.5浓度（μg/m³）   |
| temperature  | 数值     | 温度（℃）            |
| humidity     | 数值     | 湿度（%）            |
| pressure     | 数值     | 气压（hPa）          |
| wind_speed   | 数值     | 风速（m/s）          |
| wind_direction | 数值    | 风向（度）           |
| rainfall     | 数值     | 降雨量（mm）         |
| dew_point    | 数值     | 露点温度（℃）        |
| visibility   | 数值     | 能见度（km）         |

### 多源数据模板

**移动巡航数据模板 (`data/pm25_mobile_template.csv`)**：
- `latitude`, `longitude`: 经纬度坐标
- `device_id`: 设备标识
- `pm25`: PM2.5浓度
- 支持空间过滤和时间对齐

**激光雷达数据模板 (`data/pm25_lidar_template.csv`)**：
- `pm25_est`: 激光雷达估计的PM2.5浓度
- `vertical_profile`: 垂直剖面数据
- 自动重命名为标准pm25列

**小微站数据模板 (`data/pm25_microsite_template.csv`)**：
- 多站点PM2.5浓度数据
- 支持多站点数据融合

### 数据准备

1. **单数据源**：使用基础气象数据模板
2. **多数据源**：准备多个数据文件，使用字典格式配置
3. **时间格式**：统一为YYYY-MM-DD HH:MM:SS
4. **缺失值处理**：支持自动填充和插值

## 使用方法

### 1. 命令行接口

项目提供了简洁的命令行接口，支持单数据源和多数据源模式：

**单数据源模式（推荐新手使用）**：
```bash
python scripts/run_model.py --data_path data/pm25_data_template.csv --model_type lstm --epochs 50 --seq_len 24 --visualize
```

**多数据源模式（高级功能）**：
```bash
# 使用统一配置文件，支持1个或多个数据源
python scripts/run_model.py --config_path config_unified.yaml --model_type adaptive_ms_pinn --epochs 100
```

### 2. 主要命令行参数

| 参数名             | 类型   | 默认值 | 描述                               |
|------------------|--------|--------|----------------------------------|
| `--data_path`      | 字符串 | 必填   | 数据文件路径（CSV格式）              |
| `--config_path`    | 字符串 | config_unified.yaml | 配置文件路径                |
| `--model_type`     | 字符串 | all    | 模型类型：lstm, gru, transformer, cnn_lstm, adaptive_ms_pinn, all |
| `--epochs`         | 整数   | 100    | 训练轮次                           |
| `--batch_size`     | 整数   | 32     | 批量大小                           |
| `--seq_len`        | 整数   | 24     | 序列长度（时间步数）                |
| `--lr`             | 浮点数 | 0.001  | 学习率                             |
| `--visualize`      | 标志   | False  | 生成可视化结果                     |
| `--save_figures`   | 标志   | False  | 保存可视化图片                     |
| `--generate_report`| 标志   | False  | 生成性能报告                       |

### 3. 统一配置文件 (config_unified.yaml)

为了解决配置文件重复问题，项目提供了统一的配置文件 `config_unified.yaml`，支持单数据源和多数据源场景，包含所有参数的详细说明。

#### 配置文件结构

```yaml
# PM2.5预测模型统一配置文件
# 支持单数据源和多数据源场景

# =========================== 全局参数 ===========================
model_type: adaptive_ms_pinn  # 模型类型：adaptive_ms_pinn, lstm, transformer, gru
seed: 42                       # 随机种子，确保结果可复现
device: auto                   # 设备选择：auto（自动检测）, cuda, cpu

# =========================== 数据参数 ===========================
data:
  # 基础数据参数
  seq_len: 24                  # 序列长度，用于构建时间序列数据，单位：小时
  batch_size: 32               # 批量大小，影响训练速度和内存占用
  time_resolution: "1h"         # 数据时间分辨率，可选：1h, 3h, 6h, 12h, 24h
  test_split: 0.2              # 测试集比例
  val_split: 0.1               # 验证集比例
  
  # 空间位置参数
  target_latitude: 39.9042     # 目标监测点纬度（用于空间特征计算）
  target_longitude: 116.4074   # 目标监测点经度（用于空间特征计算）
  
  # 数据源配置
  # 支持1个或多个数据源，每个数据源类型对应不同的处理逻辑
  # 数据源类型：meteorological（气象）, mobile（移动走航）, lidar（激光雷达）, microsite（小微站）
  data_sources:
    # 示例1：单个数据源
    # meteorological: "data/pm25_data.csv"
    
    # 示例2：多个数据源
    meteorological: "data/pm25_data_template.csv"  # 气象数据
    mobile: "data/pm25_data_template.csv"          # 移动走航数据
    lidar: "data/pm25_data_template.csv"           # 激光雷达数据
    microsite: "data/pm25_data_template.csv"        # 小微站数据

# =========================== 模型参数 ===========================
model:
  # 通用模型参数
  hidden_dim: 128              # 隐藏层维度，影响模型表达能力
  num_layers: 3                # 网络层数，影响模型复杂度
  dropout: 0.2                 # Dropout率，防止过拟合
  weight_decay: 0.0001         # L2正则化系数，防止过拟合
  
  # 自适应多尺度PINN特有参数
  num_adaptive_layers: 3       # 自适应层数量，控制自适应特征学习能力
  alpha: 0.1                   # 物理损失权重，平衡数据损失和物理约束损失
  input_dim: 25                # 输入特征维度（可自动调整）
  
  # Transformer相关参数（部分模型使用）
  d_model: 128                 # Transformer模型维度
  nhead: 4                     # 注意力头数量
  num_encoder_layers: 2        # Transformer编码器层数

# =========================== 训练参数 ===========================
training:
  epochs: 100                  # 训练轮次
  lr: 0.001                    # 学习率，影响模型收敛速度
  patience: 10                 # 早停机制patience值，连续10轮验证集损失不下降则停止训练
  lr_scheduler: true           # 是否启用学习率调度器
  early_stopping: true         # 是否启用早停机制
  
  # 自适应反馈机制
  adaptive_feedback: true      # 是否启用自适应反馈机制
  # 自适应反馈机制会根据数据源数量和质量自动调整：
  # 1. 数据处理策略
  # 2. 特征工程方法
  # 3. 模型复杂度
  # 4. 训练参数

# =========================== 特征工程参数 ===========================
feature_engineering:
  auto_scale: true             # 是否自动缩放特征（标准化/归一化）
  feature_selection: true      # 是否启用特征选择，自动筛选重要特征
  temporal_features: true      # 是否生成时间特征（小时、星期、月份等）
  spatial_features: true       # 是否生成空间特征（距离、方向等）
  interaction_features: true   # 是否生成特征交互项
  rolling_features: true       # 是否生成滚动统计特征
  
  # 数据源特有特征处理
  meteorological_features: true  # 是否处理气象数据特有特征
  mobile_features: true         # 是否处理移动走航数据特有特征
  lidar_features: true          # 是否处理激光雷达数据特有特征
  microsite_features: true      # 是否处理小微站数据特有特征

# =========================== 输出参数 ===========================
output:
  save_model: true             # 是否保存训练后的模型
  save_figures: true           # 是否保存可视化结果
  save_metrics: true           # 是否保存评估指标
  model_dir: "models/saved"     # 模型保存目录
  figures_dir: "figures"        # 可视化结果保存目录
  metrics_dir: "metrics"        # 评估指标保存目录
  log_level: "INFO"            # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
```

#### 自适应反馈机制

项目实现了自适应反馈机制，根据数据源数量和类型自动调整处理策略：

| 数据源数量 | 特征工程策略 | 数据处理参数 |
|------------|--------------|--------------|
| 1个        | 简化特征工程 | 基本特征保留 |
| 2个        | 适度特征工程 | 增加交互特征 |
| 3个及以上  | 完整特征工程 | 完整特征处理 |

**数据源类型适配**：
- 包含移动走航数据：增强空间特征处理
- 包含激光雷达数据：增强垂直剖面特征处理
- 包含小微站数据：增强多站点融合特征处理

#### 单数据源与多数据源支持

统一配置文件支持1个或多个数据源，通过调整`data_sources`字段即可：

**单数据源示例**：
```yaml
data_sources:
  meteorological: "data/pm25_data.csv"
```

**多数据源示例**：
```yaml
data_sources:
  meteorological: "data/pm25_data.csv"
  mobile: "data/mobile_data.csv"
  lidar: "data/lidar_data.csv"
```


## 模块化使用指南

### 1. 独立使用数据处理模块

```python
import sys
sys.path.append('path/to/PINN_PM25')

from utils.data_processor import DataProcessor
import yaml

# 加载配置
with open('config_unified.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建数据处理器
processor = DataProcessor(config)

# 加载单数据源
data = processor.load_data('data/pm25_data_template.csv')

# 加载多数据源
data_sources = {
    'meteorological': 'data/pm25_data_template.csv',
    'microsite': 'data/pm25_microsite_template.csv'
}
data = processor.load_data(data_sources)

# 数据预处理
features, target = processor.preprocess_data(data)

# 提取时空特征序列
X, y = processor.extract_spatiotemporal_features(features, target, seq_len=24)

# 划分数据集
X_train, X_val, X_test, y_train, y_val, y_test = processor.split_dataset(X, y)

# 创建数据加载器
train_loader, val_loader, test_loader = processor.create_dataloaders(
    X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32
)
```

### 2. 独立使用训练模块

```python
from utils.trainer import ModelTrainer

# 创建训练器
trainer = ModelTrainer(config)

# 准备数据
trainer.prepare_data('data/pm25_data_template.csv')

# 构建模型
trainer.build_model('lstm')

# 训练模型
trainer.train(epochs=100)

# 评估模型
metrics, preds, labels, history = trainer.evaluate()

print(f"模型性能: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
```

### 3. 独立使用可视化模块

```python
from utils.visualization import Visualizer

# 创建可视化器
visualizer = Visualizer()

# 绘制训练曲线
visualizer.plot_training_curve(history, 'LSTM Model')

# 绘制预测结果
visualizer.plot_predictions(preds, labels, 'LSTM Predictions')

# 绘制模型对比
model_results = {
    'LSTM': {'metrics': metrics1, 'predictions': preds1, 'labels': labels1},
    'GRU': {'metrics': metrics2, 'predictions': preds2, 'labels': labels2}
}
visualizer.plot_model_comparison(model_results)

# 保存图片
visualizer.save_figures('figures/')
```

### 4. 在外部项目中引用

```python
# 在你的项目中引用PM2.5预测模块
import sys
sys.path.append('/path/to/PINN_PM25')

from utils.data_processor import DataProcessor
from utils.trainer import ModelTrainer
from utils.visualization import Visualizer

# 配置你的数据路径和参数
config = {
    'seq_len': 24,
    'batch_size': 32,
    'time_resolution': '1h',
    'target_latitude': 39.9042,
    'target_longitude': 116.4074
}

# 使用模块进行PM2.5预测
processor = DataProcessor(config)
data = processor.load_data('your_pm25_data.csv')
features, target = processor.preprocess_data(data)

# 继续你的分析流程...
```

## 实际使用示例

### 示例1：快速测试运行

```bash
# 激活虚拟环境
pinn_env\Scripts\activate

# 使用示例数据运行LSTM模型
python scripts/run_model.py --data_path data/pm25_data_template.csv --model_type lstm --epochs 10 --seq_len 12 --visualize

# 运行自适应多尺度PINN模型
python scripts/run_model.py --data_path data/pm25_data_template.csv --model_type adaptive_ms_pinn --epochs 20 --seq_len 24 --generate_report
```

### 示例2：多数据源配置

使用统一配置文件 `config_unified.yaml`，修改其中的 `data_sources` 字段：

```yaml
data_sources:
  meteorological: "data/pm25_data_template.csv"
  mobile: "data/pm25_mobile_template.csv"
  lidar: "data/pm25_lidar_template.csv"
  microsite: "data/pm25_microsite_template.csv"
```

运行多数据源模型：
```bash
python scripts/run_model.py --config_path config_unified.yaml --model_type adaptive_ms_pinn --epochs 50 --save_figures
```

### 示例3：批量运行所有模型

```bash
# 运行所有模型进行对比
python scripts/run_model.py --data_path data/pm25_data_template.csv --model_type all --epochs 30 --seq_len 12 --generate_report --save_figures
```

## 故障排除

### 常见问题及解决方案

1. **AttributeError: 'str' object has no attribute 'items'**
   - **原因**：数据路径参数类型错误
   - **解决**：确保使用正确的数据路径格式，支持字符串或字典

2. **ZeroDivisionError: division by zero**
   - **原因**：验证集为空
   - **解决**：减少序列长度或增加数据量，系统会自动处理

3. **ModuleNotFoundError: No module named 'utils'**
   - **原因**：Python路径未正确设置
   - **解决**：添加项目路径到sys.path

4. **数据加载失败**
   - **原因**：文件路径错误或格式不匹配
   - **解决**：检查文件路径和数据格式模板

### 性能优化建议

1. **数据量较少时**：
   - 使用较小的序列长度（如12-24）
   - 减少批量大小（如16-32）
   - 增加早停耐心值

2. **数据量充足时**：
   - 使用较长的序列长度（如48-72）
   - 增加批量大小（如64-128）
   - 使用多数据源融合

3. **模型选择**：
   - 小数据集：LSTM或GRU
   - 大数据集：Transformer或自适应多尺度PINN

## 技术支持

如有问题或建议，请检查：
1. 确保所有依赖包已正确安装
2. 验证数据格式是否符合模板要求
3. 查看控制台输出的错误信息
4. 参考 `test_new_features.py` 进行功能测试

## 更新日志

- **v1.1**：新增多源数据融合功能
- **v1.0**：基础PM2.5预测框架发布

---

**注意**：本项目为研究用途，实际部署时请根据具体需求调整参数配置。

### 配置文件

用户也可以通过修改 `config_unified.yaml` 文件来调整模型参数，配置文件包含了所有模型的详细参数设置。

### 示例用法

1. 运行所有模型进行对比：

```bash
python scripts/run_model.py --data_path data/pm25_data.csv --visualize --save_figures --generate_report
```

2. 仅运行自适应多尺度PINN模型：

```bash
python scripts/run_model.py --data_path data/pm25_data.csv --model_type adaptive_ms_pinn --epochs 200 --batch_size 64
```

## 模型说明

### 1. 自适应多尺度PINN模型

自适应多尺度PINN模型是本项目的核心，主要特点包括：

- **多尺度特征提取**：使用不同尺度的卷积核提取时空特征
- **自适应网络结构**：根据输入特征自动调整网络深度和宽度
- **物理约束**：融合大气扩散方程等物理知识，提高预测可靠性
- **注意力机制**：自动学习特征重要性，增强关键信息的权重

### 2. 基线模型

为了验证PINN模型的性能，项目实现了以下主流深度学习模型作为对比：

- **LSTM**：长短期记忆网络，适合时间序列预测
- **GRU**：门控循环单元，LSTM的简化版本
- **Transformer**：基于自注意力机制的序列模型
- **CNN-LSTM**：卷积神经网络与LSTM的结合，用于提取时空特征

## 结果解释

### 评估指标

项目使用以下指标评估模型性能：

- **MSE（均方误差）**：预测值与真实值差值的平方的平均值
- **RMSE（均方根误差）**：MSE的平方根，反映预测的整体误差
- **MAE（平均绝对误差）**：预测值与真实值差值的绝对值的平均值
- **R²（决定系数）**：反映模型对数据的解释能力，范围[-∞, 1]

### 结果文件

模型运行完成后，结果将保存到以下位置：

- **模型文件**：`models/saved/best_model.pt`
- **可视化结果**：`figures/` 目录下的各种图表
- **性能报告**：`figures/model_performance_report.html` 和 `figures/model_performance_report.csv`

## 可视化功能

项目提供了丰富的可视化功能，包括：

1. **训练收敛曲线**：展示训练和验证损失的变化
2. **预测结果对比**：真实值与预测值的时间序列对比
3. **散点图**：直观展示预测值与真实值的相关性
4. **误差分布**：预测误差的直方图和核密度估计
5. **模型性能对比**：不同模型的指标柱状图
6. **雷达图**：多指标综合性能对比
7. **HTML报告**：包含所有结果的交互式报告

## 安装依赖

项目依赖可以通过以下命令安装：

```bash
pip install numpy pandas torch scikit-learn matplotlib seaborn pyyaml
```

## 系统要求

- **操作系统**：Windows、Linux或macOS
- **Python版本**：3.8及以上
- **GPU支持**：推荐使用GPU加速（可选，支持CUDA）
- **内存**：建议8GB以上

## 许可证

本项目采用MIT许可证，详情请见LICENSE文件。

## 注意事项

1. 首次运行时，模型会自动创建必要的目录结构
2. 数据文件必须包含`datetime`和`pm25`字段
3. 建议使用至少3个月以上的历史数据以获得更好的预测效果
4. 可以通过调整`seq_len`参数来改变时间序列长度，影响预测精度
5. 物理损失权重`alpha`可以根据实际数据进行调整

## 扩展功能

项目支持以下扩展功能：

- 自定义特征工程：用户可以在`data_processor.py`中添加自定义特征
- 新模型集成：可以在`baseline_models.py`中添加新的对比模型
- 自定义评估指标：可以在`trainer.py`中添加新的评估指标
- 分布式训练：支持多GPU分布式训练（需要修改配置）

## 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 项目地址：[GitHub链接]
- 邮箱：[联系人邮箱]

---

**致谢**：感谢所有为本项目做出贡献的研究人员和开发者！
