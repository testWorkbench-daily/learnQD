# 回测基准对比指标指南

## 已实现的指标 ✓

### 1. 收益指标
- **收益率** (Return): 期间总收益百分比
- **年化收益率** (Annualized Return): 换算为年化的收益率，便于跨期比较
  - 计算公式: `收益率 × (252 / 交易天数)`

### 2. 风险指标
- **年化波动率** (Annualized Volatility): 衡量收益的波动程度
  - 计算公式: `日收益率标准差 × √252 × 100%`
  - 意义: 波动率越低，风险越小

- **最大回撤** (Max Drawdown): 从峰值到谷底的最大跌幅
  - 意义: 衡量最坏情况下的损失

### 3. 风险调整收益指标
- **夏普比率** (Sharpe Ratio): 每单位风险的超额收益
  - 计算公式: `(收益率 - 无风险利率) / 波动率`
  - 意义: 值越高，风险调整后收益越好
  - 一般标准: >1为良好，>2为优秀，>3为卓越

- **卡尔玛比率** (Calmar Ratio): 年化收益与最大回撤的比值
  - 计算公式: `年化收益率 / 最大回撤`
  - 意义: 衡量承受回撤风险获得的收益
  - 一般标准: >3为良好，>5为优秀

### 4. 交易统计
- **交易次数** (Total Trades): 总交易次数
- **胜率** (Win Rate): 盈利交易占比

---

## 推荐添加的指标

### 高优先级 (强烈推荐)

#### 1. 索提诺比率 (Sortino Ratio) ⭐⭐⭐
- **定义**: 类似夏普比率，但只考虑下行波动率
- **计算公式**: `(收益率 - 无风险利率) / 下行标准差`
- **优势**:
  - 更合理：只惩罚下行波动，不惩罚上行波动
  - 对于不对称收益分布的策略更准确
- **实现难度**: 中等
- **实现方法**:
```python
# 只计算负收益的标准差
negative_returns = [r for r in returns if r < 0]
downside_std = np.std(negative_returns, ddof=1) * np.sqrt(252)
sortino = annualized_return / downside_std
```

#### 2. 信息比率 (Information Ratio) ⭐⭐⭐
- **定义**: 相对于基准的超额收益与跟踪误差的比值
- **计算公式**: `超额收益均值 / 超额收益标准差`
- **优势**:
  - 直接衡量相对基准的表现
  - 评估主动管理的价值
- **实现难度**: 中等
- **实现方法**:
```python
# 计算策略收益 - 基准收益的时间序列
excess_returns = strategy_returns - benchmark_returns
information_ratio = excess_returns.mean() / excess_returns.std()
```

#### 3. 盈亏比 (Profit Factor) ⭐⭐⭐
- **定义**: 总盈利 / 总亏损
- **优势**:
  - 直观反映策略的盈利能力
  - 考虑了盈利和亏损的绝对金额
- **实现难度**: 简单
- **实现方法**:
```python
wins = [t['pnlcomm'] for t in trades if t['pnlcomm'] > 0]
losses = [abs(t['pnlcomm']) for t in trades if t['pnlcomm'] < 0]
profit_factor = sum(wins) / sum(losses) if sum(losses) > 0 else 0
```
- **一般标准**: >1.5为良好，>2.0为优秀

---

### 中优先级 (建议添加)

#### 4. 最大连续亏损 (Max Consecutive Losses) ⭐⭐
- **定义**: 最多连续亏损的交易次数
- **优势**:
  - 衡量心理承受压力
  - 评估策略稳定性
- **实现难度**: 简单

#### 5. 平均持仓时间 (Average Holding Period) ⭐⭐
- **定义**: 平均每笔交易的持仓时长
- **优势**:
  - 了解策略的交易频率特性
  - 评估资金利用效率
- **实现难度**: 中等（需要记录开仓平仓时间）

#### 6. 风险价值 (VaR - Value at Risk) ⭐⭐
- **定义**: 在给定置信水平下，一定时期内可能发生的最大损失
- **计算**: 95%置信水平的VaR = 收益率分布的5%分位数
- **优势**:
  - 金融行业标准风险指标
  - 直观理解尾部风险
- **实现难度**: 简单

#### 7. 贝塔系数 (Beta) ⭐⭐
- **定义**: 策略收益相对于基准收益的敏感度
- **计算公式**: `Cov(策略收益, 基准收益) / Var(基准收益)`
- **优势**:
  - 理解策略与市场的关系
  - β>1表示波动大于市场，β<1表示波动小于市场
- **实现难度**: 中等

#### 8. 阿尔法 (Alpha) ⭐⭐
- **定义**: 策略相对于基准的超额收益（调整贝塔后）
- **计算公式**: `策略收益 - (无风险利率 + β × (基准收益 - 无风险利率))`
- **优势**:
  - 衡量主动管理创造的价值
  - 金融行业标准指标
- **实现难度**: 中等

---

### 低优先级 (可选)

#### 9. Omega比率 ⭐
- **定义**: 收益分布中高于阈值的部分与低于阈值的部分的比值
- **优势**: 考虑了收益分布的所有矩
- **实现难度**: 复杂

#### 10. 平均盈亏比 (Average Win/Loss Ratio) ⭐
- **定义**: 平均盈利交易金额 / 平均亏损交易金额
- **实现难度**: 简单

#### 11. 回撤恢复时间 (Drawdown Recovery Time) ⭐
- **定义**: 从回撤谷底恢复到新高的平均时间
- **实现难度**: 中等

#### 12. 尾部比率 (Tail Ratio) ⭐
- **定义**: 右尾（极端盈利）与左尾（极端亏损）的比值
- **实现难度**: 中等

---

## 推荐实施顺序

### 第一阶段（已完成）✓
1. 收益率
2. 年化收益率
3. 年化波动率
4. 夏普比率
5. 卡尔玛比率
6. 最大回撤

### 第二阶段（强烈推荐）
1. **索提诺比率** - 改进的风险调整收益指标
2. **信息比率** - 相对基准的表现
3. **盈亏比** - 盈利能力直观指标

### 第三阶段（进一步完善）
4. 贝塔和阿尔法
5. 最大连续亏损
6. 平均持仓时间
7. VaR

---

## 对比表格建议格式

当前输出格式：
```
基准对比 (买入并持有):
指标              策略          基准          超额
--------------------------------------------------------
收益率            12.42%        4.35%        8.07%
年化收益率        13.50%        4.72%        8.78%
年化波动率        15.20%        18.50%      -3.30%
夏普比率           1.90         1.21         0.68
卡尔玛比率         6.22         1.61         4.61
最大回撤           2.17%        2.94%       -0.77%
```

建议扩展格式（第二阶段）：
```
基准对比 (买入并持有):
指标              策略          基准          超额        解释
------------------------------------------------------------------
收益率            12.42%        4.35%        8.07%
年化收益率        13.50%        4.72%        8.78%
年化波动率        15.20%        18.50%      -3.30%      ✓ 风险更低
夏普比率           1.90         1.21         0.68       ✓ 优秀
索提诺比率         2.45         1.50         0.95       ✓ 下行风险更低
卡尔玛比率         6.22         1.61         4.61       ✓ 优秀
信息比率           0.82          -            -         ✓ 主动管理价值
最大回撤           2.17%        2.94%       -0.77%      ✓ 回撤更小
盈亏比             2.34          -            -         ✓ 良好
```

---

## 代码实现示例

### 索提诺比率
```python
def _calculate_sortino(self, strat, target_return=0.0) -> float:
    """计算索提诺比率"""
    try:
        import numpy as np
        daily_recorder = strat.analyzers.dailyvaluerecorder.get_analysis()
        daily_values = daily_recorder.get('daily_values', [])

        if len(daily_values) < 2:
            return 0.0

        returns = [dv['daily_return'] for dv in daily_values if dv['daily_return'] != 0]

        # 只计算低于目标收益的收益率（下行偏差）
        downside_returns = [r - target_return for r in returns if r < target_return]

        if len(downside_returns) < 2:
            return 0.0

        # 计算下行标准差并年化
        downside_std = np.std(downside_returns, ddof=1) * np.sqrt(252)

        # 计算年化收益
        mean_return = np.mean(returns)
        annualized_return = mean_return * 252

        sortino = (annualized_return - target_return) / downside_std if downside_std > 0 else 0.0

        return sortino
    except Exception as e:
        print(f'  [索提诺比率计算失败: {e}]')
        return 0.0
```

### 信息比率
```python
def _calculate_information_ratio(self, strategy_values, benchmark_values) -> float:
    """计算信息比率"""
    try:
        import numpy as np

        # 确保长度相同
        min_len = min(len(strategy_values), len(benchmark_values))
        strat_vals = strategy_values[:min_len]
        bench_vals = benchmark_values[:min_len]

        # 计算收益率
        strat_returns = np.diff(strat_vals) / strat_vals[:-1]
        bench_returns = np.diff(bench_vals) / bench_vals[:-1]

        # 超额收益
        excess_returns = strat_returns - bench_returns

        # 信息比率
        if len(excess_returns) > 1 and np.std(excess_returns) > 0:
            information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            return information_ratio

        return 0.0
    except Exception as e:
        print(f'  [信息比率计算失败: {e}]')
        return 0.0
```

### 盈亏比
```python
def _calculate_profit_factor(self, strat) -> float:
    """计算盈亏比"""
    try:
        recorder = strat.analyzers.traderecorder.get_analysis()
        trades = recorder.get('trades', [])

        if not trades:
            return 0.0

        wins = [t['pnlcomm'] for t in trades if t['pnlcomm'] > 0]
        losses = [abs(t['pnlcomm']) for t in trades if t['pnlcomm'] < 0]

        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 0

        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        return profit_factor
    except Exception as e:
        print(f'  [盈亏比计算失败: {e}]')
        return 0.0
```

---

## 总结

### 当前状态
✅ 已实现6个核心指标，可以满足基本的策略评估需求

### 下一步建议
优先实现：
1. **索提诺比率** - 更准确的风险调整收益指标
2. **信息比率** - 直接评估相对基准的价值
3. **盈亏比** - 最直观的盈利能力指标

这三个指标实现难度适中，但能显著提升分析深度。

### 长期规划
根据使用需求，逐步添加贝塔、阿尔法等高级指标，构建完整的策略评估体系。
