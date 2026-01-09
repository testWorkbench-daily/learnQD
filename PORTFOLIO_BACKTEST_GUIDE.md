# 策略组合回测指南

## 概述

本指南介绍如何使用策略组合回测功能，通过组合多个低相关性的策略来分散风险、提高收益稳定性。

## 完整工作流程

### 步骤1: 运行所有单个策略回测

首先需要运行所有策略的单独回测，生成每个策略的 `daily_values` 数据：

```bash
# 方法1: 使用批量脚本
bash run_all_strategies_2024.sh

# 方法2: 手动运行特定策略
python bt_main.py --atom sma_cross --start 2024-01-01 --end 2024-12-31 --timeframe d1
python bt_main.py --atom rsi_reversal --start 2024-01-01 --end 2024-12-31 --timeframe d1
python bt_main.py --atom macd_trend --start 2024-01-01 --end 2024-12-31 --timeframe d1
# ... 更多策略
```

**重要提示**: 确保所有策略都使用相同的：
- 开始日期 (--start)
- 结束日期 (--end)
- 时间周期 (--timeframe)

这样生成的 `daily_values_*.csv` 文件才能正确对齐。

### 步骤2: 分析策略相关性

运行相关性分析，找出低相关性的策略组合：

```bash
python analyze_correlation.py \
  --start 20240101 \
  --end 20241231 \
  --timeframe d1 \
  --threshold 0.3 \
  --max-strategies 4 \
  --results-dir backtest_results
```

**参数说明**:
- `--start`: 开始日期，格式 YYYYMMDD
- `--end`: 结束日期，格式 YYYYMMDD
- `--timeframe`: 时间周期 (m1/m5/m15/m30/h1/h4/d1)
- `--threshold`: 相关性阈值，推荐0.3（相关性低于此值的策略会被推荐）
- `--max-strategies`: 每个组合最多包含的策略数量
- `--results-dir`: 结果保存目录

**输出文件**:
- `correlation_matrix_d1_20240101_20241231.csv` - 相关性矩阵
- `correlation_heatmap_d1_20240101_20241231.png` - 相关性热力图
- `recommended_portfolios_d1_20240101_20241231.csv` - **推荐的策略组合** ⭐

**推荐组合CSV格式**:
```csv
portfolio_id,num_strategies,strategies,equal_weight
1,4,turtle_sys1,vol_regime_standard,vwap_rev_1_5,cci_20_100,0.25
2,2,adx_14_25,boll_mr_20_2,0.5
3,2,donchian_20_10,rsi_reversal,0.5
...
```

### 步骤3: 回测策略组合

使用生成的推荐组合进行回测：

```bash
# 回测所有推荐的组合
python portfolio_backtest.py \
  --portfolio-file backtest_results/recommended_portfolios_d1_20240101_20241231.csv \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --timeframe d1

# 只回测特定ID的组合
python portfolio_backtest.py \
  --portfolio-file backtest_results/recommended_portfolios_d1_20240101_20241231.csv \
  --portfolio-id 1 \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --timeframe d1
```

**输出示例**:
```
================================================================================
回测组合 1: ['turtle_sys1', 'vol_regime_standard', 'vwap_rev_1_5', 'cci_20_100']
权重: [0.25, 0.25, 0.25, 0.25]
--------------------------------------------------------------------------------

回测结果:
  最终资金: $115,234.56
  收益率: 15.23%
  夏普比率: 1.45
  最大回撤: -8.32%
  交易次数: 156 (所有子策略之和)
  胜率: N/A (组合策略)

每日价值已保存: backtest_results/daily_values_portfolio_1_d1_20240101_20241231.csv
```

**生成的文件**:
- `daily_values_portfolio_1_d1_20240101_20241231.csv` - 组合的每日价值数据
- `daily_values_portfolio_2_d1_20240101_20241231.csv` - 第二个组合的数据
- ...

### 步骤4: 对比分析

回测完成后，脚本会自动生成对比表：

```
================================================================================
组合策略对比
================================================================================
组合名称                      收益率      夏普    回撤   策略数
--------------------------------------------------------------------------------
portfolio_4strats             15.23%     1.45   -8.32%        4
portfolio_2strats             12.45%     1.32  -10.21%        2
portfolio_2strats             18.67%     1.68   -6.54%        2
```

## 工作原理

### 组合回测的实现方式

`portfolio_backtest.py` 不会重新运行backtrader回测，而是采用更高效的方法：

1. **读取已有数据**: 加载各个策略已经生成的 `daily_values_*.csv` 文件
2. **加权平均**: 按照权重对各策略的累积收益率进行加权平均
3. **计算指标**: 基于组合的价值序列计算收益率、夏普比率、最大回撤等指标
4. **保存结果**: 生成组合的 `daily_values` 文件，格式与单个策略一致

**数学公式**:
```
组合累积收益率 = w1 * 策略1累积收益率 + w2 * 策略2累积收益率 + ...
组合价值 = 初始资金 * (1 + 组合累积收益率)
```

### 为什么不直接运行多策略？

Backtrader的多策略功能比较复杂，而且：
- 难以实现资金按权重分配
- 子策略间可能互相干扰
- 性能开销大

使用事后计算的方法：
- 更快速、更简单
- 结果准确（假设策略间无资金竞争）
- 可以灵活调整权重而无需重新回测

## 常见问题

### Q1: 为什么提示"未找到策略数据文件"？

**原因**: 组合中的某个策略还没有被单独回测过。

**解决方法**: 运行该策略的单独回测：
```bash
python bt_main.py --atom <策略名> --start 2024-01-01 --end 2024-12-31 --timeframe d1
```

### Q2: 日期格式不一致怎么办？

**问题**: `analyze_correlation.py` 使用 YYYYMMDD 格式，而 `portfolio_backtest.py` 使用 YYYY-MM-DD 格式。

**答案**: 这是正常的，脚本会自动转换。只要确保日期范围一致即可。

### Q3: 可以自定义策略权重吗？

**当前版本**: 暂不支持，所有组合都使用等权重 (1/n)。

**未来计划**: 可以修改 `portfolio_backtest.py` 添加自定义权重支持。

### Q4: 组合回测的交易次数是怎么计算的？

组合的"交易次数"是所有子策略交易次数的总和。这只是一个参考指标，因为组合策略实际上不会产生这么多交易（子策略会共享资金）。

### Q5: 为什么组合的胜率显示 N/A？

因为组合策略没有真实的交易记录，无法计算胜率。组合的表现应该看整体收益率和风险指标。

## 高级用法

### 手动创建策略组合

你可以手动创建 `custom_portfolios.csv` 文件：

```csv
portfolio_id,num_strategies,strategies,equal_weight
1,3,sma_cross,rsi_reversal,macd_trend,0.333333
2,2,turtle_sys1,donchian_20_10,0.5
```

然后回测：
```bash
python portfolio_backtest.py --portfolio-file custom_portfolios.csv --start 2024-01-01 --end 2024-12-31
```

### 批量测试不同时间段

```bash
# 2024年上半年
python analyze_correlation.py --start 20240101 --end 20240630 --timeframe d1
python portfolio_backtest.py --portfolio-file backtest_results/recommended_portfolios_d1_20240101_20240630.csv

# 2024年下半年
python analyze_correlation.py --start 20240701 --end 20241231 --timeframe d1
python portfolio_backtest.py --portfolio-file backtest_results/recommended_portfolios_d1_20240701_20241231.csv
```

### 多时间周期分析

```bash
# 日线
python analyze_correlation.py --start 20240101 --end 20241231 --timeframe d1

# 4小时线
python analyze_correlation.py --start 20240101 --end 20241231 --timeframe h4

# 小时线
python analyze_correlation.py --start 20240101 --end 20241231 --timeframe h1
```

## 最佳实践

1. **足够的数据量**: 确保至少有20+个策略的回测数据，这样相关性分析才有意义

2. **合理的阈值**:
   - 0.3: 较严格，推荐的策略相关性很低
   - 0.5: 中等，平衡相关性和可选策略数量
   - 0.7: 较宽松，可能包含相关性偏高的策略

3. **策略数量**:
   - 2-3个: 简单，易于理解
   - 4-5个: 分散效果好
   - 6+个: 过度分散，收益可能被摊薄

4. **定期重新评估**: 策略的相关性会随市场环境变化，建议定期（如每季度）重新分析

5. **out-of-sample测试**:
   - 在历史数据上找出组合（如2024年）
   - 在新数据上验证组合表现（如2025年）

## 示例：完整工作流程

```bash
# 步骤1: 运行所有策略回测（2024年数据）
bash run_all_strategies_2024.sh

# 步骤2: 分析相关性，生成推荐组合
python analyze_correlation.py \
  --start 20240101 \
  --end 20241231 \
  --timeframe d1 \
  --threshold 0.3 \
  --max-strategies 4

# 步骤3: 回测所有推荐的组合
python portfolio_backtest.py \
  --portfolio-file backtest_results/recommended_portfolios_d1_20240101_20241231.csv \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --timeframe d1

# 步骤4: 查看结果
ls -lh backtest_results/daily_values_portfolio_*
cat backtest_results/recommended_portfolios_d1_20240101_20241231.csv
```

## 输出文件说明

### correlation_matrix_*.csv
相关性矩阵，展示所有策略两两之间的皮尔逊相关系数。

### correlation_heatmap_*.png
相关性热力图，直观显示策略间的相关性：
- 绿色：正相关
- 黄色：无相关
- 红色：负相关

### recommended_portfolios_*.csv
推荐的策略组合配置文件，包含：
- portfolio_id: 组合编号
- num_strategies: 策略数量
- strategies: 策略名称列表（逗号分隔）
- equal_weight: 每个策略的权重

### daily_values_portfolio_*.csv
组合的每日价值数据，包含：
- datetime: 日期时间
- portfolio_value: 组合总价值
- daily_return: 每日收益率
- cumulative_return: 累积收益率

格式与单个策略的 `daily_values` 完全一致，可以用于进一步分析。

## 总结

策略组合回测是提高交易系统稳健性的重要工具：
- ✅ 分散风险，降低回撤
- ✅ 提高收益稳定性
- ✅ 发现互补的策略组合
- ✅ 科学的资金管理方法

通过本指南，你可以系统地分析策略相关性，构建和回测低相关性的策略组合，从而提高整体交易系统的表现。
