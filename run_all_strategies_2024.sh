#!/bin/bash
# 批量运行所有典型策略回测 (2024年数据)
# 用法: bash run_all_strategies_2024.sh

# 设置统一的时间范围和时间周期
START="2024-01-01"
END="2024-12-31"
TIMEFRAME="d1"

echo "================================================================"
echo "批量运行策略回测"
echo "时间范围: $START 至 $END"
echo "时间周期: $TIMEFRAME"
echo "================================================================"
echo ""

# 创建结果目录
mkdir -p backtest_results

# ============================================================================
# 1. 均线类策略 (Moving Average Strategies)
# ============================================================================
echo ">>> 1. 均线类策略"

# 简单双均线
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom sma_cross

# 快速双均线 (5/20)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom sma_5_20

# 中期双均线 (10/30)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom sma_10_30

# 三重均线标准版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom triple_ma

# 三重均线 (8/21/55 - 斐波那契)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom triple_ma_8_21_55

# ============================================================================
# 2. 震荡指标策略 (Oscillator Strategies)
# ============================================================================
echo ""
echo ">>> 2. 震荡指标策略"

# RSI反转
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom rsi_reversal

# MACD趋势
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom macd_trend

# CCI通道标准版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom cci_channel

# CCI通道严格版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom cci_strict

# ============================================================================
# 3. 布林带策略 (Bollinger Bands Strategies)
# ============================================================================
echo ""
echo ">>> 3. 布林带策略"

# 布林带突破
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom boll_breakout

# 布林带均值回归标准版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom boll_mr

# 布林带均值回归 (20/2)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom boll_mr_20_2

# 布林带均值回归严格版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom boll_mr_strict

# ============================================================================
# 4. 通道突破策略 (Channel Breakout Strategies)
# ============================================================================
echo ""
echo ">>> 4. 通道突破策略"

# Keltner通道标准版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom keltner_channel

# Keltner通道紧密版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom keltner_tight

# 唐奇安通道标准版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom donchian_channel

# 唐奇安通道 (20/10)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom donchian_20_10

# 唐奇安通道海龟系统1
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom donchian_turtle_sys1

# ============================================================================
# 5. 波动率策略 (Volatility Strategies)
# ============================================================================
echo ""
echo ">>> 5. 波动率策略"

# ATR突破标准版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom atr_breakout

# ATR突破激进版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom atr_breakout_aggressive

# 波动率突破标准版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom vol_breakout

# 波动率突破 (14/2)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom vol_breakout_14_2

# 波动率区制标准版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom vol_regime_standard

# 波动率扩张标准版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom vol_expansion_standard

# 恒定波动率 (10%)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom const_vol_10

# 恒定波动率 (15%)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom const_vol_15

# ============================================================================
# 6. 趋势强度策略 (Trend Strength Strategies)
# ============================================================================
echo ""
echo ">>> 6. 趋势强度策略"

# ADX趋势标准版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom adx_trend

# ADX趋势 (14/25)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom adx_14_25

# ADX趋势 (14/30 - 更严格)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom adx_14_30

# ============================================================================
# 7. 日内策略 (Intraday Strategies)
# ============================================================================
echo ""
echo ">>> 7. 日内策略"

# 开盘区间突破 (30分钟)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom orb_30min

# 开盘区间突破 (60分钟)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom orb_60min

# 日内动量 (1.0%)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom intraday_mom_1_0

# 日内动量激进版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom intraday_mom_aggressive

# 日内反转 (1.5%)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom intraday_rev_1_5

# VWAP回归标准版
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom vwap_reversion

# VWAP回归 (1.5%)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom vwap_rev_1_5

# ============================================================================
# 8. 经典策略 (Classic Strategies)
# ============================================================================
echo ""
echo ">>> 8. 经典策略"

# 海龟交易系统1
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom turtle_sys1

# 海龟交易系统2
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom turtle_sys2

# N日新高新低 (20日)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom new_hl_20

# N日新高新低 (50日)
python bt_main.py --start $START --end $END --timeframe $TIMEFRAME --atom new_hl_50

# ============================================================================
echo ""
echo "================================================================"
echo "所有策略回测完成!"
echo "================================================================"
echo ""
echo "接下来运行相关性分析:"
echo "python analyze_correlation.py --start 20240101 --end 20241231 --timeframe d1"
