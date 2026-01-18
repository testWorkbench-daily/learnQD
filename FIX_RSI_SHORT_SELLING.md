# RSI策略做空功能修复指南

## 问题诊断

### 当前状况
- portfolio_rank3_combo 组合策略中，RSI子策略（34%权重）**只做多不做空**
- 导致做空信号强度不足：
  - 理论最大做空强度：-1.0（所有策略看空）
  - 实际最大做空强度：-0.66（RSI不参与做空）
  - 影响：无法达到-3手强烈做空（需要<=-0.70）

### 验证结果
- 诊断脚本显示：做空占比44.8%，但从未出现-2或-3手空头
- 仓位轨迹：大部分时间0-2手多头，做空时只有-1手

---

## 修复方案

### 修复文件
`/Users/hong/PycharmProjects/prepareQD/atoms/portfolio_rank3_combo.py`

### 修改位置
行117-136: `_simulate_rsi_reversal()` 方法

### 修改前（当前代码）
```python
def _simulate_rsi_reversal(self):
    """模拟RSI反转策略的虚拟持仓

    原始逻辑：
    - 空仓时：RSI<30买入
    - 持仓时：RSI>70平仓
    """
    if len(self.rsi) == 0:
        return self.virtual_pos_rsi

    rsi_value = self.rsi[0]

    if self.virtual_pos_rsi == 0:  # 虚拟空仓
        if rsi_value < 30:
            self.virtual_pos_rsi = 1  # 虚拟买入
    else:  # 虚拟持仓中
        if rsi_value > 70:
            self.virtual_pos_rsi = 0  # 虚拟平仓

    return self.virtual_pos_rsi
```

### 修改后（支持做空）
```python
def _simulate_rsi_reversal(self):
    """模拟RSI反转策略的虚拟持仓

    改进逻辑：
    - 空仓时：RSI<30买入，RSI>70卖空
    - 多头持仓时：RSI>70平仓（或转空）
    - 空头持仓时：RSI<30平仓（或转多）
    """
    if len(self.rsi) == 0:
        return self.virtual_pos_rsi

    rsi_value = self.rsi[0]

    if self.virtual_pos_rsi == 0:  # 虚拟空仓
        if rsi_value < 30:
            self.virtual_pos_rsi = 1  # 虚拟买入
        elif rsi_value > 70:
            self.virtual_pos_rsi = -1  # 虚拟卖空（新增）

    elif self.virtual_pos_rsi > 0:  # 虚拟多头持仓
        if rsi_value > 70:
            self.virtual_pos_rsi = 0  # 虚拟平仓
            # 可选：激进版本直接转空
            # self.virtual_pos_rsi = -1

    elif self.virtual_pos_rsi < 0:  # 虚拟空头持仓（新增）
        if rsi_value < 30:
            self.virtual_pos_rsi = 0  # 虚拟平空
            # 可选：激进版本直接转多
            # self.virtual_pos_rsi = 1

    return self.virtual_pos_rsi
```

---

## 预期改进

### 做空信号强度对比

**修改前**（极端看空情况）:
```
Triple MA:     -1 × 0.3366 = -0.3366
RSI:            0 × 0.3401 =  0.0000  ← 不参与做空
Vol Breakout:  -1 × 0.0843 = -0.0843
Vol Regime:    -1 × 0.2390 = -0.2390
-----------------------------------------
总计:                        -0.6599
触发手数: -2手（中度做空）
```

**修改后**（极端看空情况）:
```
Triple MA:     -1 × 0.3366 = -0.3366
RSI:           -1 × 0.3401 = -0.3401  ← 参与做空
Vol Breakout:  -1 × 0.0843 = -0.0843
Vol Regime:    -1 × 0.2390 = -0.2390
-----------------------------------------
总计:                        -1.0000
触发手数: -3手（强烈做空）✓
```

### 预期效果

1. **做空能力增强**：
   - 可以达到-3手强烈做空
   - 充分利用RSI的34%权重

2. **熊市表现改善**：
   - 在下跌行情中能够通过做空获利
   - 相对基准的超额收益应该提升

3. **整体夏普比率可能提升**：
   - 牛熊市场都能获利
   - 回撤可能减小

---

## 实施步骤

### 1. 备份原文件
```bash
cp atoms/portfolio_rank3_combo.py atoms/portfolio_rank3_combo.py.backup
```

### 2. 修改代码
按照上述"修改后"的代码更新 `_simulate_rsi_reversal()` 方法

### 3. 测试对比
```bash
# 测试修改前（使用备份）
python bt_main.py --atom portfolio_rank3_combo --timeframe d1 --start 2024-01-01 --end 2024-12-31

# 修改后测试
python bt_main.py --atom portfolio_rank3_combo --timeframe d1 --start 2024-01-01 --end 2024-12-31

# 对比牛市期（例如2024年1-6月）
python bt_main.py --atom portfolio_rank3_combo --timeframe d1 --start 2024-01-01 --end 2024-06-30

# 对比熊市期（例如2024年7-12月，如果有下跌）
python bt_main.py --atom portfolio_rank3_combo --timeframe d1 --start 2024-07-01 --end 2024-12-31
```

### 4. 验证改进
运行诊断脚本：
```bash
python diagnose_short_selling.py
```

检查：
- 是否出现-2或-3手空头
- 做空期间的盈亏表现
- 整体夏普比率变化

---

## 其他注意事项

### 1. 是否需要同步修改原始RSI策略？
**建议**：是的，也修改原始的 `atoms/rsi_reversal.py`，使其支持做空。

**原因**：
- 保持一致性
- 原始策略也能在熊市中表现更好
- 如果重新运行优化，会得到更好的组合

### 2. 激进版本 vs 保守版本

**保守版本**（推荐）:
```python
elif self.virtual_pos_rsi > 0:  # 多头持仓
    if rsi_value > 70:
        self.virtual_pos_rsi = 0  # 先平仓
```
- 平仓后等待新信号
- 减少频繁反转

**激进版本**:
```python
elif self.virtual_pos_rsi > 0:  # 多头持仓
    if rsi_value > 70:
        self.virtual_pos_rsi = -1  # 直接转空
```
- RSI>70时直接从多头转空头
- 交易更频繁，可能增加成本
- 捕捉反转更快

### 3. Backtrader卖空的限制

Backtrader默认允许卖空，但需要注意：
- 不考虑借券成本
- 不考虑保证金要求
- 实盘交易需要额外配置

如果要更真实模拟，可以在broker中设置：
```python
cerebro.broker.set_shortcash(True)  # 卖空收到的钱不能用于再投资
```

---

## 检查清单

- [ ] 备份原文件
- [ ] 修改 `_simulate_rsi_reversal()` 方法
- [ ] 运行回测测试
- [ ] 运行诊断脚本验证
- [ ] 对比牛熊市表现
- [ ] 检查夏普比率变化
- [ ] （可选）修改原始RSI策略
- [ ] （可选）重新运行组合优化

---

## 预期结果

修复后，你应该看到：
1. ✅ 做空手数能达到-2或-3手
2. ✅ 熊市期间表现改善
3. ✅ 整体夏普比率提升
4. ✅ 相对基准的超额收益增加

如果修复后效果不明显，可能需要：
- 检查熊市识别逻辑
- 调整阈值参数
- 考虑市场环境是否真的有明显熊市期
