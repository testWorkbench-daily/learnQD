# PRD: LSTM价格方向预测器

## 1. 产品定义

**一句话描述**: 读取期货M1 Bar数据CSV，输出每个时间点的LSTM预测信号CSV

```
输入: nq_m1_bars.csv (OHLCV)
      ↓
   [LSTM模型]
      ↓
输出: nq_m1_signal_L60_H5.csv (预测信号)
                       ↑   ↑
              lookback=60  horizon=5
```

---

## 2. 核心参数

| 参数 | 含义 | 默认值 | 说明 |
|-----|------|-------|------|
| `lookback` | 输入窗口长度 | 60 | 用过去60个bar预测 |
| `horizon` | 预测时间跨度 | 5 | 预测未来5个bar的方向 |

**输出文件命名规则**: `{原文件名}_signal_L{lookback}_H{horizon}.csv`

---

## 3. 信号定义

```
信号值 ∈ [-1, +1]

+1.0 ──── 强烈看涨（预测未来horizon个bar大概率上涨）
+0.5 ──── 温和看涨
 0.0 ──── 中性
-0.5 ──── 温和看跌
-1.0 ──── 强烈看跌
```

**执行策略建议**:
- signal > +0.5 → 考虑做多
- signal < -0.5 → 考虑做空
- |signal| < 0.3 → 观望

---

## 4. 数据流程

```
┌─────────────────────────────────────────────────────────────┐
│                     Phase 1: 数据准备                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入CSV                特征工程                标签构造    │
│  ┌─────────┐         ┌──────────┐          ┌──────────┐   │
│  │ts_event │         │ returns  │          │ 未来收益  │   │
│  │open     │───────> │ volatility│ ──────> │ 方向标签  │   │
│  │high     │         │ momentum │          │ [-1,+1]  │   │
│  │low      │         │ volume_chg│          └──────────┘   │
│  │close    │         └──────────┘                          │
│  │volume   │                                               │
│  └─────────┘                                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Phase 2: 模型训练                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   时序划分              LSTM训练              模型保存       │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐     │
│  │Train 70% │──────> │ 前向传播  │──────> │model.pt  │     │
│  │Val   15% │        │ 反向传播  │        │scaler.pkl│     │
│  │Test  15% │        │ 早停验证  │        └──────────┘     │
│  └──────────┘        └──────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Phase 3: 信号生成                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   滚动窗口推理                              输出CSV          │
│  ┌────────────────────────────┐         ┌──────────────┐  │
│  │ Bar 60 → 预测 → signal[60] │         │ts_event      │  │
│  │ Bar 61 → 预测 → signal[61] │ ──────> │signal        │  │
│  │ Bar 62 → 预测 → signal[62] │         │              │  │
│  │ ...                        │         └──────────────┘  │
│  └────────────────────────────┘                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 输入输出规格

### 5.1 输入CSV格式

```csv
ts_event,open,high,low,close,volume
2010-06-06 22:00:00,1831.75,1831.75,1828.75,1829.0,117
2010-06-06 22:01:00,1828.5,1831.0,1828.5,1830.5,99
...
```

**要求**:
- 必须包含列: `ts_event`, `open`, `high`, `low`, `close`, `volume`
- 按时间升序排列
- 无缺失值（或程序自动处理）

### 5.2 输出CSV格式

```csv
ts_event,signal
2010-06-06 23:00:00,0.234
2010-06-06 23:01:00,0.456
2010-06-06 23:02:00,-0.123
...
```

**说明**:
- 前`lookback-1`行无输出（数据不足）
- 最后`horizon`行无标签（无法验证），但仍输出预测值
- `signal`保留3位小数

---

## 6. 特征工程

### 6.1 使用的特征（共8维）

| # | 特征名 | 计算公式 | 含义 |
|---|-------|---------|------|
| 1 | returns | `(close - close[-1]) / close[-1]` | 收益率 |
| 2 | log_returns | `log(close / close[-1])` | 对数收益 |
| 3 | high_low_pct | `(high - low) / close` | 振幅 |
| 4 | close_position | `(close - low) / (high - low + 1e-8)` | K线位置 |
| 5 | volume_change | `volume / SMA(volume, 20) - 1` | 量比 |
| 6 | returns_ma5 | `SMA(returns, 5)` | 短期动量 |
| 7 | returns_ma20 | `SMA(returns, 20)` | 中期动量 |
| 8 | volatility | `STD(returns, 20)` | 波动率 |

### 6.2 标准化方法

每个特征独立做Z-score标准化：
```
x_normalized = (x - mean) / std
```

**注意**: 使用**训练集**的mean/std，应用到验证集和测试集（防止泄露）

---

## 7. 标签构造

```python
# 未来horizon个bar的累计收益
future_return = (close[t+horizon] - close[t]) / close[t]

# 转换为[-1, +1]信号
label = tanh(future_return / scale)

# scale参数控制敏感度，默认0.005 (0.5%)
# 即：收益率±0.5%对应信号约±0.46
```

**可视化理解**:
```
future_return:  -2%    -1%   -0.5%    0   +0.5%   +1%    +2%
                 │      │      │      │      │      │      │
    tanh映射:  -0.96  -0.76  -0.46   0.0   +0.46  +0.76  +0.96
                 │      │      │      │      │      │      │
       信号:   强空    空    弱空   中性   弱多    多    强多
```

---

## 8. LSTM模型结构

```
输入: [batch_size, lookback, 8]   # 8个特征
       │
       ▼
┌─────────────────────────┐
│    LayerNorm            │
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│   LSTM Layer            │
│   input_size=8          │
│   hidden_size=32        │
│   num_layers=2          │
│   dropout=0.2           │
│   bidirectional=False   │  ← 单向，防止未来泄露
└───────────┬─────────────┘
            │
            │ 取最后时间步: hidden[-1]
            ▼
┌─────────────────────────┐
│   Linear(32 → 16)       │
│   ReLU                  │
│   Dropout(0.2)          │
│   Linear(16 → 1)        │
│   Tanh                  │  ← 输出[-1, +1]
└───────────┬─────────────┘
            │
            ▼
      signal ∈ [-1, +1]
```

**参数量**: 约 8K（非常轻量）

---

## 9. 训练配置

```yaml
# 数据划分（严格按时间顺序）
train_ratio: 0.70      # 前70%训练
val_ratio: 0.15        # 中间15%验证
test_ratio: 0.15       # 最后15%测试

# 训练参数
batch_size: 64
learning_rate: 0.001
epochs: 100
early_stopping_patience: 10   # 验证loss连续10轮不降则停止

# 损失函数
loss: MSE              # 均方误差

# 优化器
optimizer: Adam
weight_decay: 0.0001   # L2正则化
```

---

## 10. 使用方式

### 10.1 命令行接口

```bash
# 基本使用
python lstm_predictor.py --input nq_m1.csv

# 指定参数
python lstm_predictor.py \
    --input nq_m1.csv \
    --lookback 60 \
    --horizon 5 \
    --output_dir ./output

# 仅推理（使用已训练模型）
python lstm_predictor.py \
    --input nq_m1.csv \
    --model_path models/lstm_L60_H5.pt \
    --inference_only
```

### 10.2 输出文件

```
output/
├── nq_m1_signal_L60_H5.csv      # 预测信号
├── lstm_L60_H5.pt               # 模型权重
├── scaler_L60_H5.pkl            # 标准化参数
└── training_log_L60_H5.json     # 训练日志
```

---

## 11. 评估指标

训练完成后自动输出以下指标（在测试集上）：

| 指标 | 含义 | 期望值 |
|-----|------|-------|
| **IC** | 预测值与实际收益的相关系数 | > 0.03 |
| **Direction Accuracy** | 预测方向正确率 | > 52% |
| **MSE** | 均方误差 | 越小越好 |

```
============ 测试集评估 ============
IC (Information Coefficient): 0.051
方向准确率: 53.2%
MSE: 0.0823
===================================
```

---

## 12. 注意事项

### 12.1 防止未来泄露

| 检查点 | 做法 |
|-------|------|
| 数据划分 | 严格按时间，不shuffle |
| 特征标准化 | 仅用训练集统计量 |
| LSTM方向 | 单向，不用双向 |
| 标签构造 | shift正确（向未来看） |

### 12.2 关于预测效果的预期

金融市场预测本质上很难，合理的预期：

- IC > 0.05 已经是很好的信号
- 方向准确率 55% 以上可以盈利（考虑手续费后）
- 单独使用ML信号风险大，建议结合其他指标

### 12.3 数据量建议

| 数据量 | 效果 |
|-------|------|
| < 10万条 | 容易过拟合，谨慎使用 |
| 10-50万条 | 可用，建议简化模型 |
| > 50万条 | 较好，可尝试更复杂模型 |

你的数据从2010年开始，M1频率约有百万级bar，数据量充足。

---

## 附录: 核心代码结构预览

```python
# lstm_predictor.py 主要结构

class FeatureEngine:
    """特征工程"""
    def transform(self, df) -> np.ndarray:
        # 计算8个特征
        pass

class LSTMModel(nn.Module):
    """LSTM模型"""
    def forward(self, x) -> torch.Tensor:
        # 输入[B, L, 8] → 输出[B, 1]
        pass

class Trainer:
    """训练器"""
    def train(self, train_loader, val_loader):
        # 训练循环 + 早停
        pass

class Predictor:
    """预测器"""
    def predict(self, df) -> pd.DataFrame:
        # 生成信号CSV
        pass

def main():
    args = parse_args()
    
    # 1. 加载数据
    df = pd.read_csv(args.input)
    
    # 2. 特征工程
    features, labels = FeatureEngine().transform(df)
    
    # 3. 训练（或加载已有模型）
    if not args.inference_only:
        model = Trainer().train(features, labels)
    else:
        model = load_model(args.model_path)
    
    # 4. 生成信号
    signals = Predictor(model).predict(df)
    
    # 5. 保存结果
    output_name = f"{args.input.stem}_signal_L{args.lookback}_H{args.horizon}.csv"
    signals.to_csv(output_name, index=False)
```

---

*文档结束 - 下一步可以开始写具体实现代码*
