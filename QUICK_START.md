# NQ期货回测系统 - 快速开始

## 🚀 快速使用

### 基本用法

```bash
# 默认：2024年全年，1分钟K线
python backtrader_demo.py

# 5分钟K线
python backtrader_demo.py --timeframe m5

# 1小时K线，指定时间范围
python backtrader_demo.py --timeframe h1 --start 2024-01-01 --end 2024-06-30

# 日线，长期回测
python backtrader_demo.py --timeframe d1 --start 2020-01-01 --end 2024-12-31

# 保存交易记录到CSV
python backtrader_demo.py --timeframe h1 --start 2024-01-01 --end 2024-06-30 --save-trades

# 生成可视化图表
python backtrader_demo.py --timeframe d1 --start 2024-01-01 --end 2024-12-31 --plot

# 完整功能：保存记录+生成图表
python backtrader_demo.py --timeframe h4 --start 2024-01-01 --end 2024-06-30 --save-trades --plot
```

## 📊 K线周期对照表

| 参数 | 周期 | 数据量 | 适用场景 | 策略参数 |
|------|------|--------|----------|----------|
| `m1` | 1分钟 | 最多 | 高频/超短线 | 快20/慢60 |
| `m5` | 5分钟 | 较多 | 日内短线 | 快20/慢60 |
| `m15` | 15分钟 | 中等 | 日内波段 | 快15/慢45 |
| `m30` | 30分钟 | 中等 | 日内波段 | 快10/慢30 |
| `h1` | 1小时 | 较少 | 短期趋势 | 快10/慢30 |
| `h4` | 4小时 | 少 | 中期趋势 | 快5/慢15 |
| `d1` | 日线 | 最少 | 长期趋势 | 快5/慢20 |

## 📈 推荐配置

### 日内交易（当天开平仓）
```bash
python backtrader_demo.py --timeframe m5 --start 2024-01-01 --end 2024-12-31
```

### 短线交易（持仓数天）
```bash
python backtrader_demo.py --timeframe h1 --start 2023-01-01 --end 2024-12-31
```

### 中长线交易（持仓数周/数月）
```bash
python backtrader_demo.py --timeframe d1 --start 2020-01-01 --end 2024-12-31
```

## 🔧 命令行参数完整列表

```bash
python backtrader_demo.py \
  --start 2024-01-01 \          # 开始日期（YYYY-MM-DD）
  --end 2024-12-31 \            # 结束日期（YYYY-MM-DD）
  --timeframe h1 \              # K线周期（m1/m5/m15/m30/h1/h4/d1）
  --data nq_m1_cleaned.csv \    # 数据文件路径
  --save-trades \               # 保存交易记录到CSV
  --plot                        # 生成可视化图表
```

### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--start` | 日期 | 回测开始日期 | 2024-01-01 |
| `--end` | 日期 | 回测结束日期 | 2024-12-31 |
| `--timeframe` | 选项 | K线周期 | m1 |
| `--data` | 路径 | 数据文件路径 | nq_m1_cleaned.csv |
| `--save-trades` | 开关 | 保存交易记录 | False |
| `--plot` | 开关 | 生成可视化图表 | False |

## 📊 交易记录文件说明

使用 `--save-trades` 参数后，会在 `backtest_results/` 目录下生成CSV文件。

### 文件命名格式
```
trades_{timeframe}_{start_date}_{end_date}_{timestamp}.csv
```

例如: `trades_h1_20241201_20241215_20251221_132546.csv`

### CSV文件包含的字段

| 字段 | 说明 |
|------|------|
| `trade_id` | 交易序号 |
| `datetime` | 交易时间 |
| `type` | 交易类型（BUY/SELL） |
| `price` | 成交价格 |
| `size` | 交易数量 |
| `value` | 交易价值 |
| `commission` | 手续费 |
| `sma_fast` | 快线数值 |
| `sma_slow` | 慢线数值 |
| `rsi` | RSI指标 |
| `macd` | MACD数值 |
| `macd_signal` | MACD信号线 |
| `atr` | ATR波动率 |
| `portfolio_value` | 账户总价值 |
| `cash` | 现金余额 |
| `position` | 持仓数量 |
| `pnl` | 单笔盈亏（仅SELL时） |
| `pnl_percent` | 盈亏百分比（仅SELL时） |

### 用途
- ✅ 详细分析每笔交易的进出场点位
- ✅ 回溯交易时的技术指标状态
- ✅ 统计盈亏分布
- ✅ 优化策略参数
- ✅ 导入Excel进行进一步分析

## 📈 可视化图表说明

使用 `--plot` 参数后，会生成交互式图表窗口，包含：

### 图表内容
- 📊 **K线图**: 显示价格走势（红涨绿跌）
- 📈 **成交量**: K线下方显示成交量柱状图
- 📉 **技术指标**: 
  - SMA快线/慢线（主图）
  - RSI相对强弱指标（副图）
  - MACD指标（副图）
  - 布林带（主图）
- 🔴 **买入信号**: 向上箭头标记
- 🟢 **卖出信号**: 向下箭头标记
- 💰 **资金曲线**: 显示账户价值变化
- 📉 **回撤曲线**: 显示最大回撤情况

### 图表交互
- 🔍 可以缩放查看细节
- 📍 可以移动时间轴
- 💾 可以保存为图片

### 前置要求
需要安装matplotlib：
```bash
pip install matplotlib
```

## 💡 常见问题

### Q: 如何查看所有可用参数？
```bash
python backtrader_demo.py --help
```

### Q: 如何修改初始资金？
编辑 `backtrader_demo.py` 文件：
```python
cerebro.broker.setcash(100000)  # 改为你想要的金额
```

### Q: 如何修改手续费？
编辑 `backtrader_demo.py` 文件：
```python
cerebro.broker.setcommission(commission=0.001)  # 0.1%
```

### Q: 数据文件在哪里？
- 原始数据：`nq_m1_all_backtrader.csv`（包含脏数据）
- 清洗后：`nq_m1_cleaned.csv`（推荐使用）

### Q: 如何重新清洗数据？
```bash
python clean_nq_data.py
```

## 📝 输出说明

回测结束后会显示：
- ✅ 初始/最终资金
- ✅ 总收益和收益率
- ✅ 交易次数和胜率
- ✅ 平均盈利/亏损
- ✅ 最大回撤
- ✅ 夏普比率（如果有足够数据）

## ⚠️ 注意事项

1. **数据量**：较小的K线周期（m1、m5）数据量大，运行时间长
2. **内存**：建议8GB以上内存
3. **时间范围**：建议先用短时间测试，确认没问题再运行长时间
4. **策略优化**：当前策略仅供演示，实际使用需优化
5. **风险控制**：回测结果不代表实盘表现，请谨慎使用

## 🎯 性能参考

| K线周期 | 1年数据运行时间 | 内存占用 |
|---------|----------------|----------|
| m1 | 2-3分钟 | ~2GB |
| m5 | 30-60秒 | ~1GB |
| h1 | 10-20秒 | ~500MB |
| d1 | 5-10秒 | ~200MB |

## 📚 相关文件

- `backtrader_demo.py` - 主程序（已删除所有模拟数据）
- `clean_nq_data.py` - 数据清洗脚本
- `nq_m1_cleaned.csv` - 清洗后的数据（210万行）
- `README_backtrader.md` - 详细说明文档
- `QUICK_START.md` - 本文件（快速开始）

## 🌟 示例命令

```bash
# 示例1：快速测试（默认参数）
python backtrader_demo.py

# 示例2：测试2024年上半年，5分钟K线，保存交易记录
python backtrader_demo.py --timeframe m5 --start 2024-01-01 --end 2024-06-30 --save-trades

# 示例3：测试2023年全年，1小时K线，生成图表
python backtrader_demo.py --timeframe h1 --start 2023-01-01 --end 2023-12-31 --plot

# 示例4：测试过去3年，日线K线，保存记录+生成图表
python backtrader_demo.py --timeframe d1 --start 2022-01-01 --end 2024-12-31 --save-trades --plot

# 示例5：测试单个月，15分钟K线，完整分析
python backtrader_demo.py --timeframe m15 --start 2024-06-01 --end 2024-06-30 --save-trades --plot

# 示例6：测试最近2周，小时K线，保存记录
python backtrader_demo.py --timeframe h1 --start 2024-12-01 --end 2024-12-15 --save-trades

# 示例7：测试4小时K线，中期趋势
python backtrader_demo.py --timeframe h4 --start 2024-01-01 --end 2024-12-31 --save-trades

# 示例8：测试30分钟K线，短期波段
python backtrader_demo.py --timeframe m30 --start 2024-11-01 --end 2024-12-31 --plot
```

## 📂 输出文件位置

- **交易记录CSV**: `backtest_results/trades_*.csv`
- **程序输出**: 直接显示在终端

## 🎯 最佳实践

1. **先用短时间测试**: 建议先测试1-2周数据，确认策略正常
2. **保存重要记录**: 对于关键测试，使用 `--save-trades` 保存详细记录
3. **定期查看图表**: 使用 `--plot` 直观了解策略表现
4. **对比不同周期**: 同一时间段测试不同K线周期，找到最优配置
5. **记录测试结果**: 建议在笔记中记录每次测试的参数和结果

祝交易顺利！📈

