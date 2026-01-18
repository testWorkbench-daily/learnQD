# 1. **数据准备**：
确保已有清洗后的NQ数据
   - 文件：`nq_m1_forward_adjusted.csv`

# 2. **目录结构**：
```
prepareQD/
├── atoms/                    # 策略atom目录
│   ├── portfolios/          # 组合策略目录
│   └── *.py                 # 单策略文件
├── backtest_results/        # 回测结果目录
│   └── rolling_validation/  # 滚动验证结果
├── bt_main.py              # 主运行入口
├── portfolio_optimizer.py   # 组合优化器
├── portfolio_backtest.py    # 组合回测工具
└── rolling_portfolio_validator.py  # 滚动验证工具
```

# 3. 快速批量回测（推荐）

使用脚本批量运行所有策略：

```bash
# 回测2020-2024年全部历史数据
bash run_all_strategies_2020_2024.sh
```

# 4. 单个策略回测

如果只想测试特定策略：

```bash
# 查看可用策略列表
python bt_main.py --list

# 回测单个策略
python bt_main.py --atom sma_cross --timeframe d1 --start 2020-01-01 --end 2024-12-31
```

# 5. 生成的文件

每个策略会生成：
- `daily_values_{策略名}_{周期}_{开始日期}_{结束日期}.csv`

示例：
```
backtest_results/daily_values_sma_cross_d1_20200101_20241231.csv
backtest_results/daily_values_rsi_reversal_d1_20200101_20241231.csv
...
```
# 6. 运行相关性分析

```bash
python analyze_correlation.py \
  --start 20200101 \
  --end 20241231 \
  --timeframe d1 \
  --threshold 0.3 \
  --max-strategies 4 \
  --results-dir backtest_results
```

**参数说明**：
- `--threshold 0.3`：相关性阈值（推荐0.3，越低越严格）
- `--max-strategies 4`：组合中最多策略数（2-4个）
- `--min-sharpe 0.5`：最低夏普要求（可选）

# 7. 生成的文件

```
backtest_results/
├── correlation_matrix_d1_20200101_20241231.csv      # 相关性矩阵
├── correlation_heatmap_d1_20200101_20241231.png     # 热力图
└── recommended_portfolios_d1_20200101_20241231.csv  # 推荐组合
```

# 8. 查看推荐结果

```bash
# 查看推荐的前10个组合
head -n 11 backtest_results/recommended_portfolios_d1_20200101_20241231.csv | column -t -s,
```

# 9. 回测所有推荐组合

```bash
python portfolio_backtest.py \
  --portfolio-file backtest_results/recommended_portfolios_d1_20200101_20241231.csv
```

# 10. 回测特定组合

```bash
# 只回测排名前5的组合
python portfolio_backtest.py \
  --portfolio-file backtest_results/recommended_portfolios_d1_20200101_20241231.csv \
  --portfolio-ids 1 2 3 4 5

# 回测单个组合
python portfolio_backtest.py \
  --portfolio-file backtest_results/recommended_portfolios_d1_20200101_20241231.csv \
  --portfolio-id 1
```

# 11. 查看回测结果

组合回测结果会直接在终端显示：

```
组合 #1: sma_cross,rsi_reversal
  权重: 0.5000, 0.5000
  收益率: 15.23%
  夏普比率: 1.85
  最大回撤: -3.45%
```

**重要提示**：
- 组合回测不会重新运行子策略
- 而是加载各子策略的daily_values进行加权计算
- 确保步骤1已完成，所有子策略的daily_values文件存在

---

# 12. 滚动窗口验证

### 目的
识别在不同时间窗口都表现稳健的组合（避免过拟合）。

# 13. **年度滚动窗口**（推荐新手使用）：

```bash
python rolling_portfolio_validator.py \
  --timeframe d1 \
  --window-months 12 \
  --step-months 12 \
  --top-n 10 \
  --workers auto \
  --sorting-mode threshold \
  --min-recommend-freq 0.6
```

⏱️ **预计耗时**：45秒-2分钟（取决于CPU核心数）

# 14. **季度滚动窗口**（更精细）：

```bash
python rolling_portfolio_validator.py \
  --timeframe d1 \
  --window-months 12 \
  --step-months 3 \
  --top-n 10 \
  --workers auto \
  --sorting-mode threshold \
  --min-recommend-freq 0.8
```

# 15. 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--window-months` | 每个窗口的长度（月） | 12（一年） |
| `--step-months` | 滑动步长（月） | 12（无重叠）或 3（季度滚动） |
| `--top-n` | 每个窗口保留Top N组合 | 10 |
| `--workers` | 并行进程数 | auto（自动检测CPU核心数） |
| `--sorting-mode` | 排序模式 | `threshold`（阈值筛选）或 `robustness`（综合评分） |
| `--min-recommend-freq` | 推荐频率阈值（0-1） | 0.6（60%）或 0.8（80%） |

# 16. 排序模式说明

#### threshold 模式（阈值筛选法）
- 先筛选：推荐频率 ≥ 阈值
- 再排序：按平均夏普降序
- **适用场景**：优先要求稳定性，牺牲部分收益

#### robustness 模式（综合评分法）
- 使用加权评分公式：
  ```
  Score = 0.3×平均夏普 + 0.25×穿越率 + 0.25×推荐频率
          - 0.15×夏普标准差 - 0.05×最差夏普惩罚
  ```
- **适用场景**：平衡收益和稳定性

# 17. 生成的文件

```
backtest_results/rolling_validation/
├── rolling_window_summary.csv            # 每个窗口的汇总统计
├── robust_portfolios_ranking.csv         # 稳健组合排名（⭐包含权重信息）
└── window_details.csv                    # 所有窗口的详细数据
```

# 18. 查看结果

**终端输出示例**：

```
【Top 5 稳健组合】（阈值筛选法: 推荐频率≥60%, 按平均夏普排序）
---------------------------------------------------------------
排名  策略组成                推荐频率  平均夏普  最差夏普  最佳夏普
1     sma_cross,rsi_reversal   80%      1.85     1.45     2.20
2     macd_trend,triple_ma     60%      1.72     1.30     2.05
...

【穿越能力分析】（阈值: 夏普>0.5）
- 在所有5个窗口都被推荐的组合: 2个 ⭐⭐⭐
- 在≥80%窗口被推荐的组合: 5个 ⭐⭐
- 在≥60%窗口被推荐的组合: 8个 ⭐
```

**查看CSV文件**：

```bash
# 查看前10个稳健组合
head -n 11 backtest_results/rolling_validation/robust_portfolios_ranking.csv | column -t -s,
```

---

# 19. 生成策略Atom

### 目的
将稳健组合转换为可运行的策略代码。

**手动创建组合策略**（Claude Code辅助）：

1. 查看稳健组合排名：
```bash
head -n 5 backtest_results/rolling_validation/robust_portfolios_ranking.csv
```

2. 告诉Claude Code要生成哪个排名的策略：
```
"请为ranking.csv的排名X的策略生成一个真实可运行的atom"
```

3. Claude Code会：
   - 读取子策略的代码逻辑
   - 理解各子策略的信号计算方式
   - 生成包含完整信号加权逻辑的策略代码
   - 自动注册到bt_main.py

**优点**：
- ✅ 策略可以直接运行回测
- ✅ 信号逻辑完整，可实盘交易
- ✅ 可自定义调整参数

**缺点**：
- ⚠️ 需要人工确认每个策略
- ⚠️ 批量生成较慢

# 20. 生成的文件

```
atoms/portfolios/
├── proj0113_top0001.py    # 排名1的组合
├── proj0113_top0002.py    # 排名2的组合
├── proj0113_top0003.py    # 排名3的组合
...
```
---

# 20. 运行组合策略

### 6.1 方式A：运行真实策略（bt_main.py）

适用于方式A生成的策略：

```bash
# 运行单个组合策略
python bt_main.py --atom portfolio_rank3_combo --timeframe d1

# 指定时间范围
python bt_main.py --atom portfolio_rank3_combo \
  --timeframe d1 \
  --start 2020-01-01 \
  --end 2024-12-31

# 不保存结果，快速测试
python bt_main.py --atom portfolio_rank3_combo \
  --timeframe d1 \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --no-save --no-plot
```

# 21. 查看策略列表

```bash
# 查看所有可用策略
python bt_main.py --list

# 查看组合策略
python bt_main.py --list | grep portfolio
```

---

# 推荐工作流程

### 场景1：快速验证大量组合

**目标**：从132个策略中快速找到表现最好的组合

**流程**：
1. ✅ 步骤1：批量回测所有单策略（一次性，1-2小时）
2. ✅ 步骤2：相关性分析（1分钟）
3. ✅ 步骤3：组合回测（30秒）
4. ✅ 步骤4：滚动窗口验证（2分钟）
5. ⚠️ 步骤5：跳过atom生成，直接使用portfolio_backtest.py
6. ✅ 步骤6：回测稳健组合

**总耗时**：1-2小时（大部分时间在步骤1）

### 场景2：精选少量策略用于实盘

**目标**：生成2-3个真实可运行的策略用于实盘交易

**流程**：
1. ✅ 步骤1：批量回测所有单策略（如已完成可跳过）
2. ✅ 步骤2：相关性分析
3. ✅ 步骤3：组合回测
4. ✅ 步骤4：滚动窗口验证，使用严格阈值（0.8）
5. ✅ 步骤5：手动生成真实策略（通过Claude Code，每个5分钟）
6. ✅ 步骤6：全时间段回测验证

**总耗时**：20-30分钟（假设步骤1已完成）

### 场景3：定期更新组合

**目标**：每月/每季度更新稳健组合

**流程**：
1. ✅ 步骤1：增量回测新数据（只需回测最近1个月，5分钟）
2. ✅ 步骤4：滚动窗口验证（包含新数据）
3. ✅ 比对新旧排名，决定是否调整策略

**总耗时**：10分钟

---

# 文件清单

### 核心脚本
- `bt_main.py` - 主运行入口
- `bt_runner.py` - 回测引擎
- `analyze_correlation.py` - 相关性分析工具
- `portfolio_backtest.py` - 组合回测工具
- `portfolio_optimizer.py` - 组合优化器
- `rolling_portfolio_validator.py` - 滚动窗口验证工具
- `generate_portfolio_atoms.py` - Atom代码生成器

### 批处理脚本
- `run_all_strategies_2024.sh` - 批量回测2024年数据
- `run_all_strategies_2020_2024.sh` - 批量回测2020-2024年数据

### 数据文件
- `nq_m1_forward_adjusted.csv` - NQ期货1分钟数据（前向调整）

### 结果目录
- `backtest_results/` - 所有回测结果
- `backtest_results/rolling_validation/` - 滚动验证结果
- `atoms/portfolios/` - 组合策略atom

---

## 参考资料

### 相关文档
- `README.md` - 项目总览
- `CLAUDE.md` - Claude Code使用指南
- `USAGE.md` - 详细使用说明

### 关键概念
- **Strategy Atom**: 策略原子，封装策略逻辑的可复用单元
- **Daily Values**: 每日账户价值，用于计算收益率和相关性
- **Correlation Threshold**: 相关性阈值，控制组合多样性
- **Rolling Window**: 滚动窗口，用于测试策略稳健性
- **Recommendation Frequency**: 推荐频率，策略在多个窗口中被推荐的比例
- **Sharpe Ratio**: 夏普比率，衡量风险调整后收益

---

## 更新日志

### v2.0 (2026-01-14)
- ✅ 新增：滚动窗口验证步骤
- ✅ 改进：rolling_portfolio_validator.py直接输出包含权重的CSV
- ✅ 改进：无需export_robust_portfolios.py中间步骤
- ✅ 新增：两种atom生成方式（真实策略 vs 模板代码）
- ✅ 优化：threshold和robustness两种排序模式

### v1.0 (2026-01-11)
- ✅ 初始版本：基础工作流程

---

**文档维护**: Claude Code
**最后更新**: 2026-01-14
