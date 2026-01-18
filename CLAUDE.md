# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative trading backtesting system for NQ futures (NASDAQ-100 index futures) built on Backtrader. The system uses a modular "Strategy Atom" architecture where each trading strategy is encapsulated as a reusable, composable unit.

**Key characteristics:**
- Data: 1-minute NQ futures data (2020-2025, ~2.1M rows)
- Architecture: Strategy Atom pattern with centralized Runner
- Strategies: 132+ registered trading strategies across 6 categories
- Multi-timeframe: Supports m1/m5/m15/m30/h1/h4/d1 via resampling

## Core Architecture

### Strategy Atom Pattern

The codebase uses a unique "Atom" pattern where each strategy is:
1. **Self-contained**: Includes Strategy logic, Sizer, and custom Indicators/Analyzers
2. **Declarative**: Strategy defined via `strategy_cls()` method returning a Backtrader Strategy class
3. **Composable**: Multiple atoms can be compared or combined

**Key base class: `bt_base.py::StrategyAtom`**
- `strategy_cls()` - Returns the Strategy class (required)
- `sizer_cls()` - Returns position sizing logic (optional)
- `indicators()` - Returns custom indicators (optional)
- `analyzers()` - Returns custom analyzers (optional)

**Execution flow:**
```
bt_main.py (CLI) → Runner (bt_runner.py) → StrategyAtom → Backtrader
```

### File Organization

**Core framework:**
- `bt_base.py` - Base classes (StrategyAtom, BaseStrategy, Sizers, Analyzers)
- `bt_runner.py` - Execution engine that handles data loading, backtest execution, result collection
- `bt_main.py` - CLI entry point with ATOMS registry

**Strategy library (`atoms/` directory):**
- 21 strategy modules organized by type
- Each module exports 3-10 parameter variants
- `atoms/__init__.py` centralizes all exports

**Data processing:**
- `forward_adjust.py` - Adjusts prices for contract rollovers
- `quick_fix_data.py` - Cleans and validates market data
- Data files: `nq_m1_forward_adjusted.csv` (primary), `nq_m1_cleaned.csv` (legacy)

**Utilities:**
- `plot_utils.py` - Plotting/visualization module (274 lines)
- `analyze_correlation.py` - Strategy correlation analysis

## Common Commands

### Running Backtests

```bash
# Single strategy with default settings
python bt_main.py

# Specify strategy and timeframe
python bt_main.py --atom sma_cross --timeframe h1

# Custom date range
python bt_main.py --atom turtle_sys1 --start 2024-01-01 --end 2024-12-31

# Compare all strategies
python bt_main.py --compare --start 2024-01-01 --end 2024-12-31

# List available strategies
python bt_main.py --list

# Disable output (no plots/trades)
python bt_main.py --atom rsi_reversal --no-save --no-plot
```

### Batch Testing

```bash
# Run all typical strategies for 2024
bash run_all_strategies_2024.sh

# Run all strategies on full historical period (2020-2024)
bash run_all_strategies_2020_2024.sh

# Analyze strategy correlation and save recommended portfolios
python analyze_correlation.py --start 20240101 --end 20241231 --timeframe d1 --threshold 0.3 --max-strategies 4

# Backtest recommended portfolio combinations
python portfolio_backtest.py --portfolio-file backtest_results/recommended_portfolios_d1_20240101_20241231.csv

# Backtest specific portfolio by ID
python portfolio_backtest.py --portfolio-file backtest_results/recommended_portfolios_d1_20240101_20241231.csv --portfolio-id 1

# Rolling window validation - identify robust portfolios across time periods
python rolling_portfolio_validator.py --timeframe d1 --window-months 12 --step-months 12 --workers auto

# Portfolio atom validation - compare actual execution vs theoretical testing
python rolling_atom_validator.py --atom portfolio_rank3_combo --timeframe d1 --window-months 12 --step-months 12
```

### Benchmark Testing

```bash
# Run buy-and-hold benchmark
python bt_main.py --atom buy_and_hold --start 2024-01-01 --end 2024-12-31 --timeframe d1

# Compare single strategy to benchmark
python compare_to_benchmark.py --strategy sma_cross --timeframe d1 --start 20240101 --end 20241231

# Compare multiple strategies to benchmark
python compare_to_benchmark.py --strategies sma_cross rsi_reversal macd_trend --timeframe d1 --start 20240101 --end 20241231

# Compare all backtested strategies to benchmark
python compare_to_benchmark.py --all --timeframe d1 --start 20240101 --end 20241231

# Export comparison to CSV
python compare_to_benchmark.py --all --timeframe d1 --start 20240101 --end 20241231 --output benchmark_comparison.csv
```

**Important:** The benchmark must be run first before comparison. The comparison tool reads daily values CSV files from both the benchmark and strategies.

### Data Processing

```bash
# Clean raw data
python quick_fix_data.py

# Apply forward adjustment for contract rollovers
python forward_adjust.py
```

## Strategy Development

### Creating a New Strategy Atom

1. **Create file in `atoms/` directory** (e.g., `atoms/my_strategy.py`)

2. **Define the Atom class:**
```python
from bt_base import StrategyAtom, BaseStrategy
import backtrader as bt

class MyStrategyAtom(StrategyAtom):
    name = "my_strategy"
    params = {'period': 20, 'threshold': 2.0}

    def strategy_cls(self):
        period = self.params['period']
        threshold = self.params['threshold']

        class Strategy(BaseStrategy):
            def __init__(self):
                super().__init__()
                self.indicator = bt.ind.SMA(period=period)

            def next(self):
                if self.order:
                    return
                # Trading logic here
                pass

        return Strategy
```

3. **Create parameter variants:**
```python
# Conservative variant
class MyStrategy_Conservative(MyStrategyAtom):
    name = "my_strategy_conservative"
    params = {'period': 30, 'threshold': 3.0}
```

4. **Register in `atoms/__init__.py`:**
```python
from atoms.my_strategy import MyStrategyAtom, MyStrategy_Conservative
```

5. **Register in `bt_main.py` ATOMS dict:**
```python
ATOMS = {
    'my_strategy': MyStrategyAtom,
    'my_strategy_conservative': MyStrategy_Conservative,
}
```

### Best Practices

- **Always inherit from `BaseStrategy`**: Provides trade recording, PnL tracking, and logging
- **Use `self.order` guard**: Check `if self.order: return` to prevent overlapping orders
- **Position tracking**: Use `self.position` to check if currently in a trade
- **Timeframe awareness**: Strategies run on resampled data based on `--timeframe` arg
- **Parameter encapsulation**: Pass params from Atom to Strategy via closure

## System Behavior

### Data Loading & Resampling

The Runner automatically handles resampling:
- Source data: Always 1-minute bars
- Resampling: Applied based on `--timeframe` parameter
- Date filtering: Applied via `--start` and `--end`

**Sharpe ratio calculation:** Dynamically adjusts statistics period based on timeframe:
- m1/m5 → hourly returns
- m15/m30 → 4-hour returns
- h1/h4 → daily returns
- d1 → weekly returns

### Trade Recording

**Two levels of recording:**
1. **Trade-level** (via `BaseStrategy.get_trade_records()`):
   - Every buy/sell execution
   - Includes price, size, commission, portfolio value, PnL
   - Saved to `backtest_results/trades_{name}_{timeframe}_{start}_{end}.csv`

2. **Daily-level** (via `DailyValueRecorder`):
   - Portfolio value per bar
   - Daily returns, cumulative returns
   - Saved to `backtest_results/daily_values_{name}_{timeframe}_{start}_{end}.csv`
   - Used for correlation analysis

### Position Sizing

Default: Fixed 1 contract (`bt.sizers.FixedSize, stake=1`)

To customize, override `sizer_cls()` in Atom:
```python
def sizer_cls(self):
    from bt_base import PercentSizer
    return PercentSizer  # Uses 10% of capital per trade
```

## Strategy Categories

The system includes 7 categories of strategies (133 total):

1. **Trend Following** (17 strategies): SMA cross, MACD, ADX, Triple MA
2. **Breakout** (47 strategies): Donchian, Keltner, ATR, Volatility, New High/Low, ORB, Turtle
3. **Mean Reversion** (17 strategies): RSI, Bollinger MR, VWAP, CCI
4. **Volatility** (15 strategies): Constant Vol, Vol Expansion, Vol Regime
5. **Intraday** (13 strategies): Intraday Momentum, Intraday Reversal
6. **Classic Systems** (6 strategies): Turtle System 1/2
7. **Benchmark** (1 strategy): Buy and Hold

See `bt_main.py::ATOMS` dictionary for complete list.

## Data Format

**Input CSV format** (`nq_m1_forward_adjusted.csv`):
```
datetime,open,high,low,close,volume
2020-01-02 09:30:00,9120.25,9125.50,9118.75,9122.00,1234
```

- Column order matters (used by GenericCSVData)
- Datetime format: `%Y-%m-%d %H:%M:%S`
- No missing values allowed
- Forward-adjusted for contract rollovers

## Testing & Validation

**No formal test suite exists.** Validation is done via:
1. Running known strategies on historical data
2. Comparing results with expected behavior
3. Using `--compare` mode to ensure strategies execute without errors

When modifying core framework (`bt_base.py`, `bt_runner.py`):
- Test with at least 3 diverse strategies (e.g., sma_cross, turtle_sys1, rsi_reversal)
- Verify trade recording, PnL calculation, and Sharpe ratio
- Check both single-run and compare mode

## Portfolio Backtesting

The system supports backtesting strategy portfolios (combinations of multiple strategies) through a two-step process:

### Step 1: Analyze Correlation and Generate Recommendations

```bash
python analyze_correlation.py --start 20240101 --end 20241231 --timeframe d1 --threshold 0.3 --max-strategies 4 --results-dir backtest_results
```

This generates:
- `correlation_matrix_*.csv` - Correlation matrix between all strategies
- `correlation_heatmap_*.png` - Visual heatmap
- `recommended_portfolios_*.csv` - Recommended low-correlation portfolio combinations

The recommended portfolios CSV contains:
- `portfolio_id` - Unique portfolio identifier
- `num_strategies` - Number of strategies in portfolio
- `strategies` - Comma-separated strategy names
- `equal_weight` - Equal weight per strategy (1/n)

### Step 2: Backtest Portfolio Combinations

```bash
python portfolio_backtest.py --portfolio-file backtest_results/recommended_portfolios_d1_20240101_20241231.csv
```

**Important implementation note:** Portfolio backtesting does NOT re-run strategies. Instead, it:
1. Loads `daily_values_*.csv` files for each strategy in the portfolio
2. Calculates weighted average portfolio returns from individual strategy returns
3. Computes portfolio metrics (return, Sharpe, drawdown) from the aggregated returns
4. Saves portfolio daily values for further analysis

**Prerequisites:** All constituent strategies must have been backtested individually first, with their `daily_values_*.csv` files present in the results directory.

**Output format:** Identical to single strategy backtests:
- Console output matches `bt_runner.py` format
- Generates `daily_values_portfolio_*_{timeframe}_{start}_{end}.csv`
- Comparison table if multiple portfolios tested

## Rolling Portfolio Validation

The system includes a rolling window validation tool to identify portfolios that are robust across different time periods.

### Purpose

Instead of testing a portfolio on a single time period, rolling validation:
1. Runs the optimizer on multiple time windows (e.g., each year: 2020, 2021, 2022, 2023, 2024)
2. Tracks which portfolio combinations are recommended across multiple windows
3. Identifies portfolios that "penetrate through" different market conditions with consistent high Sharpe ratios

**Use case:** Before leveraging a high-Sharpe portfolio, verify it maintains performance across different time periods, not just lucky in one specific period.

### Prerequisites

All strategies must be backtested on the full historical period first:

```bash
# One-time setup: Run all strategies on 2020-2024 (takes 1-2 hours)
bash run_all_strategies_2020_2024.sh
```

This generates `daily_values_{strategy}_d1_20200101_20241231.csv` files for all strategies.

### Running Rolling Validation

**Recommended configuration (annual windows):**
```bash
python rolling_portfolio_validator.py \
  --timeframe d1 \
  --window-months 12 \
  --step-months 12 \
  --top-n 10 \
  --workers auto
```

**Fine-grained configuration (quarterly rolling):**
```bash
python rolling_portfolio_validator.py \
  --timeframe d1 \
  --window-months 12 \
  --step-months 3 \
  --top-n 10 \
  --workers 8
```

### How It Works

1. **Generate Windows**: Creates time windows based on `--window-months` and `--step-months`
   - Example: 12-month windows with 12-month steps → 5 windows (2020, 2021, 2022, 2023, 2024)

2. **Per-Window Optimization**: For each window:
   - Calls `portfolio_optimizer.optimize_programmatically()` with that window's date range
   - Gets Top N recommended portfolios with their Sharpe ratios

3. **Robustness Analysis**: Across all windows:
   - **Recommendation Frequency**: How often a portfolio appears across windows
   - **Average Sharpe**: Mean Sharpe across all windows where it appeared
   - **Sharpe Std Dev**: Stability of Sharpe across windows (lower is better)
   - **Worst Sharpe**: Minimum Sharpe across windows (risk indicator)
   - **Penetration Rate**: % of windows where Sharpe > threshold (default 0.5)
   - **Robustness Score**: Weighted combination of above metrics

4. **Output**: Generates 3 CSV files:
   - `rolling_window_summary.csv` - Per-window statistics
   - `robust_portfolios_ranking.csv` - **Core output**: Portfolios ranked by robustness score
   - `window_details.csv` - All portfolios from all windows with detailed metrics

### Robustness Score Formula

```
Score = 0.3 × avg_sharpe +
        0.25 × penetration_rate +
        0.25 × recommend_freq -
        0.15 × sharpe_std -
        0.05 × abs(worst_sharpe if < 0)
```

**Interpretation:**
- High score → Portfolio consistently performs well across different time periods
- Top-ranked portfolios are ideal candidates for leveraged trading

### Performance

- **Annual windows** (5 windows): ~45 seconds with 4-core parallel execution
- **Quarterly rolling** (17 windows): ~1.5 minutes with 8-core parallel execution

### Example Output

```
【Top 5 稳健组合】（按稳健评分降序）
排名  策略组成                      推荐频率  平均夏普  夏普标准差  最差夏普  最佳夏普  穿越率    稳健评分
1     sma_cross,rsi_reversal       100%     1.68      0.18       1.45     1.92     100%     0.92
2     turtle_sys1,macd_trend        80%     1.54      0.25       1.22     1.85     100%     0.85
...

→ Portfolio #1 appeared in ALL 5 windows with Sharpe always > 1.4 ⭐⭐⭐
```

## Portfolio Atom Validation

The system includes a validation tool to compare actual portfolio atom execution vs theoretical testing results.

### Purpose

When a portfolio combination is identified as robust through rolling validation, the next step is to validate how it performs when executed as a single-account portfolio atom vs theoretical multi-account testing.

**Two execution approaches:**
1. **Theoretical Testing** (`portfolio_backtest.py`): Assumes multiple independent accounts, each running a single strategy, then calculates weighted average of returns
2. **Actual Execution** (Portfolio Atom): Single account running a combined strategy using virtual position simulation

**Key differences:**
- **Signal Aggregation**: Single account requires aggregated signals to exceed a threshold before trading
- **Trade Timing**: Actual execution may have different entry/exit times due to signal weighting
- **Trade Frequency**: Signal cancellation may reduce total trades in single-account execution
- **Costs**: Single account has one transaction cost vs sum of individual costs in theory

### Prerequisites

Both the portfolio atom and all constituent strategies must have backtest data:

```bash
# Ensure all constituent strategies have been backtested on full historical period
bash run_all_strategies_2020_2024.sh

# The portfolio atom must also be registered and backtested
python bt_main.py --atom portfolio_rank3_combo --start 2020-01-01 --end 2024-12-31 --timeframe d1
```

### Running Atom Validation

**Basic usage:**
```bash
python rolling_atom_validator.py --atom portfolio_rank3_combo --timeframe d1
```

**Custom window configuration:**
```bash
python rolling_atom_validator.py \
  --atom portfolio_rank3_combo \
  --timeframe d1 \
  --window-months 12 \
  --step-months 6
```

**Parallel execution:**
```bash
python rolling_atom_validator.py \
  --atom portfolio_rank3_combo \
  --workers auto
```

### How It Works

1. **Generate Windows**: Creates rolling time windows (e.g., 5 annual windows: 2020-2024)

2. **For Each Window**:
   - **Actual Execution**: Runs the portfolio atom backtest using `Runner`
   - **Theoretical Testing**: Calculates weighted average using `backtest_portfolio_from_daily_values()`
   - **Compare Metrics**: Calculates differences across 4 key metrics

3. **Generate Report**: Produces detailed CSV and terminal output with:
   - Per-window comparison
   - Aggregate statistics
   - Key insights and recommendations

### Comparison Metrics

**1. Return Percentage**: Absolute return difference
- Identifies revenue impact of single-account execution

**2. Sharpe Ratio**: Risk-adjusted return difference
- Shows if risk/reward profile changes

**3. Trade Frequency**: Number of trades difference
- Critical for cost analysis (fewer trades = lower costs)

**4. Maximum Drawdown**: Risk characteristic difference
- Indicates if single-account execution is more/less risky

### Output Files

**Detailed CSV**: `backtest_results/atom_validation/atom_vs_theory_comparison_{atom_name}_{timeframe}.csv`
- Columns: window details, actual metrics, theory metrics, differences (absolute & percentage)

**Summary Report**: `backtest_results/atom_validation/atom_vs_theory_summary_{atom_name}_{timeframe}.txt`
- Aggregate statistics and recommendations

**Terminal Output**: Per-window analysis with insights like:
```
【窗口1】2020-01-01 to 2020-12-31
  实际执行: 收益率=15.2%, 夏普=1.85, 交易次数=45, 最大回撤=-3.2%
  理论测试: 收益率=16.8%, 夏普=1.92, 交易次数=120, 最大回撤=-3.5%
  差异分析:
    ✓ 收益率: -1.6% (-9.5%)   理论更优
    ✓ 夏普比率: -0.07 (-3.6%)  理论更优
    ✓ 交易频率: -75 (-62.5%)   实际更少（成本更低）
    ✓ 最大回撤: +0.3% (-8.6%)  实际更好（风险更低）

【汇总统计】跨5个窗口
  - 收益率差异: -2.1% (实际比理论低2.1%)
  - 夏普差异: -0.05 (实际比理论低0.05)
  - 交易频率差异: -68次/年 (实际减少57%交易)
  - 回撤差异: +0.2% (实际回撤更小)

建议:
  ✓ 该组合适合单账户实际执行
  ✓ 交易成本节省可能抵消部分收益下降
```

### Adding New Portfolio Atoms

To validate a new portfolio atom, add its configuration to `ATOM_CONFIGS` in `rolling_atom_validator.py`:

```python
ATOM_CONFIGS = {
    'portfolio_rank3_combo': {
        'strategies': ['vol_breakout_aggressive', 'vol_regime_long', 'triple_ma', 'rsi_reversal'],
        'weights': [0.0843, 0.2390, 0.3366, 0.3401],
        'description': 'Robust ranking #3: Volatility + Trend + Mean Reversion',
    },
    'your_new_portfolio': {
        'strategies': ['strategy1', 'strategy2', 'strategy3'],
        'weights': [0.33, 0.33, 0.34],
        'description': 'Your portfolio description',
    },
}
```

### Performance

- **Per window**: 2-5 minutes (includes full backtest execution)
- **5 annual windows (serial)**: ~10-25 minutes
- **5 annual windows (parallel, 4 cores)**: ~5-10 minutes

### Interpretation Guide

**Good alignment** (實際 ≈ 理論):
- Return difference: < ±3%
- Sharpe difference: < ±0.15
- Trade frequency: -20% to +20%
- Max drawdown: < ±1%

**Acceptable with benefits** (實際 < 理論 but reduced trades):
- Return slightly lower (-5% to -2%)
- Sharpe slightly lower (-0.2 to -0.1)
- **Trade frequency significantly lower** (-30% or more)
- Drawdown similar or better

**Requires investigation** (大差異):
- Return difference: > ±10%
- Sharpe difference: > ±0.3
- Trade frequency: > ±50%
- Max drawdown: > ±3%

## Known Issues & Limitations

- **Sharpe ratio fallback**: If Backtrader's native Sharpe calculation fails (returns None), Runner falls back to manual calculation from account values
- **Memory usage**: 1-minute data for 5+ years requires ~2GB+ RAM
- **No parameter optimization**: System doesn't support grid search or optimization (by design)
- **Single instrument**: Only supports one data feed per backtest
- **No transaction costs**: Default commission is 0 (set explicitly if needed)

## Language & Documentation

The codebase primarily uses Chinese for:
- Comments in code
- CLI output messages
- Documentation files (README, USAGE, etc.)

However, code identifiers (variables, functions, classes) use English.
