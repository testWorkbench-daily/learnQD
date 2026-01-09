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

# Analyze strategy correlation and save recommended portfolios
python analyze_correlation.py --start 20240101 --end 20241231 --timeframe d1 --threshold 0.3 --max-strategies 4

# Backtest recommended portfolio combinations
python portfolio_backtest.py --portfolio-file backtest_results/recommended_portfolios_d1_20240101_20241231.csv

# Backtest specific portfolio by ID
python portfolio_backtest.py --portfolio-file backtest_results/recommended_portfolios_d1_20240101_20241231.csv --portfolio-id 1
```

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

The system includes 6 categories of strategies (132 total):

1. **Trend Following** (17 strategies): SMA cross, MACD, ADX, Triple MA
2. **Breakout** (47 strategies): Donchian, Keltner, ATR, Volatility, New High/Low, ORB, Turtle
3. **Mean Reversion** (17 strategies): RSI, Bollinger MR, VWAP, CCI
4. **Volatility** (15 strategies): Constant Vol, Vol Expansion, Vol Regime
5. **Intraday** (13 strategies): Intraday Momentum, Intraday Reversal
6. **Classic Systems** (6 strategies): Turtle System 1/2

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
