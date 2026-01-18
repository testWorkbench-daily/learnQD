#!/bin/bash
# 批量运行所有策略回测 (2020-2024完整历史周期)
# 按策略类型使用合适的timeframe，支持并发执行
# 用法: bash run_all_strategies_2020_2024.sh

# ============================================================================
# 配置参数
# ============================================================================
START="2020-01-01"
END="2024-12-31"
PARALLEL_JOBS=8  # 并发执行数量

echo "================================================================"
echo "批量运行策略回测（完整历史周期）"
echo "时间范围: $START 至 $END"
echo "并发数量: $PARALLEL_JOBS"
echo "================================================================"
echo ""

# 创建结果目录
mkdir -p backtest_results

# 并发控制函数
run_with_limit() {
    local timeframe=$1
    local strategy=$2

    # 等待直到有可用的并发槽位
    while [ $(jobs -r | wc -l) -ge $PARALLEL_JOBS ]; do
        sleep 0.5
    done

    # 后台运行策略
    python bt_main.py --start $START --end $END --timeframe $timeframe --atom $strategy --no-plot &
}

# ============================================================================
# 1. 日内策略 - m5 (27个)
# ============================================================================
echo ">>> 1. 日内策略 (m5 timeframe)"

INTRADAY_M5=(
    "orb" "orb_15min" "orb_30min" "orb_60min" "orb_30min_no_close" "orb_45min" "orb_aggressive" "orb_conservative"
    "intraday_mom" "intraday_mom_0_5" "intraday_mom_1_0" "intraday_mom_1_5" "intraday_mom_2_0" "intraday_mom_0_3"
    "intraday_mom_aggressive" "intraday_mom_conservative" "intraday_mom_moderate"
    "intraday_reversal" "intraday_rev_1_5" "intraday_rev_1_0" "intraday_rev_2_0" "intraday_rev_aggressive" "intraday_rev_conservative"
    "vwap_reversion" "vwap_rev_1_0" "vwap_rev_1_5" "vwap_rev_2_0" "vwap_rev_aggressive" "vwap_rev_conservative"
)

for strategy in "${INTRADAY_M5[@]}"; do
    run_with_limit "m5" "$strategy"
done

wait  # 等待所有日内策略完成
echo "日内策略 (m5) 完成"
echo ""

# ============================================================================
# 2. 趋势跟踪策略 - h1 (16个)
# ============================================================================
echo ">>> 2. 趋势跟踪策略 (h1 timeframe)"

TREND_H1=(
    "sma_cross" "sma_5_20" "sma_10_30" "sma_20_60"
    "triple_ma" "triple_ma_5_20_50" "triple_ma_10_30_60" "triple_ma_8_21_55" "triple_ma_12_26_52"
    "adx_trend" "adx_14_25" "adx_14_30" "adx_14_20" "adx_21_25" "adx_10_25"
    "macd_trend"
)

for strategy in "${TREND_H1[@]}"; do
    run_with_limit "h1" "$strategy"
done

wait  # 等待所有趋势策略完成
echo "趋势跟踪策略 (h1) 完成"
echo ""

# ============================================================================
# 3. 均值回归策略 - h1 (13个)
# ============================================================================
echo ">>> 3. 均值回归策略 (h1 timeframe)"

MEANREV_H1=(
    "rsi_reversal"
    "boll_mr" "boll_mr_20_2" "boll_mr_20_2_5" "boll_mr_20_1_5" "boll_mr_30_2" "boll_mr_10_2" "boll_mr_strict"
    "cci_channel" "cci_20_100" "cci_20_150" "cci_20_80" "cci_14_100" "cci_30_100" "cci_strict"
)

for strategy in "${MEANREV_H1[@]}"; do
    run_with_limit "h1" "$strategy"
done

wait  # 等待所有均值回归策略完成
echo "均值回归策略 (h1) 完成"
echo ""

# ============================================================================
# 4. 突破策略 - d1 (46个)
# ============================================================================
echo ">>> 4. 突破策略 (d1 timeframe)"

BREAKOUT_D1=(
    # Donchian系列 (10个)
    "donchian_channel" "donchian_20_10" "donchian_55_20" "donchian_20_20" "donchian_10_5" "donchian_5_3"
    "donchian_40_15" "donchian_turtle_sys1" "donchian_turtle_sys2" "donchian_aggressive" "donchian_conservative"
    # Keltner系列 (8个)
    "keltner_channel" "keltner_20_10_1_5" "keltner_20_10_2_0" "keltner_20_10_1_0" "keltner_20_14_1_5"
    "keltner_30_10_1_5" "keltner_10_10_1_5" "keltner_tight"
    # ATR系列 (9个)
    "atr_breakout" "atr_breakout_20_14_2" "atr_breakout_20_14_3" "atr_breakout_20_14_1_5" "atr_breakout_20_10_2"
    "atr_breakout_50_14_2" "atr_breakout_10_14_2" "atr_breakout_aggressive" "atr_breakout_conservative"
    # VolBreakout系列 (8个)
    "vol_breakout" "vol_breakout_14_2" "vol_breakout_14_2_5" "vol_breakout_14_1_5" "vol_breakout_20_2"
    "vol_breakout_10_2" "vol_breakout_10_3" "vol_breakout_aggressive" "vol_breakout_conservative"
    # NewHighLow系列 (9个)
    "new_hl" "new_hl_20" "new_hl_50" "new_hl_100" "new_hl_250" "new_hl_10" "new_hl_5" "new_hl_aggressive" "new_hl_conservative"
    # Bollinger突破 (1个)
    "boll_breakout"
)

for strategy in "${BREAKOUT_D1[@]}"; do
    run_with_limit "d1" "$strategy"
done

wait  # 等待所有突破策略完成
echo "突破策略 (d1) 完成"
echo ""

# ============================================================================
# 5. 波动率策略 - d1 (15个)
# ============================================================================
echo ">>> 5. 波动率策略 (d1 timeframe)"

VOLATILITY_D1=(
    "const_vol" "const_vol_10" "const_vol_15" "const_vol_20" "const_vol_conservative" "const_vol_aggressive"
    "vol_expansion" "vol_expansion_standard" "vol_expansion_sensitive" "vol_expansion_conservative" "vol_expansion_short" "vol_expansion_long"
    "vol_regime" "vol_regime_standard" "vol_regime_sensitive" "vol_regime_conservative" "vol_regime_short" "vol_regime_long"
)

for strategy in "${VOLATILITY_D1[@]}"; do
    run_with_limit "d1" "$strategy"
done

wait  # 等待所有波动率策略完成
echo "波动率策略 (d1) 完成"
echo ""

# ============================================================================
# 6. 经典系统 - d1 (6个)
# ============================================================================
echo ">>> 6. 经典系统 (d1 timeframe)"

CLASSIC_D1=(
    "turtle" "turtle_sys1" "turtle_sys1_conservative" "turtle_sys1_aggressive" "turtle_sys2" "turtle_es" "turtle_mnq"
)

for strategy in "${CLASSIC_D1[@]}"; do
    run_with_limit "d1" "$strategy"
done

wait  # 等待所有经典策略完成
echo "经典系统 (d1) 完成"
echo ""

# ============================================================================
echo "================================================================"
echo "所有策略回测完成!"
echo "================================================================"
echo ""
echo "统计信息:"
echo "  - 日内策略 (m5): ${#INTRADAY_M5[@]} 个"
echo "  - 趋势策略 (h1): ${#TREND_H1[@]} 个"
echo "  - 均值回归 (h1): ${#MEANREV_H1[@]} 个"
echo "  - 突破策略 (d1): ${#BREAKOUT_D1[@]} 个"
echo "  - 波动率策略 (d1): ${#VOLATILITY_D1[@]} 个"
echo "  - 经典系统 (d1): ${#CLASSIC_D1[@]} 个"
echo ""
echo "接下来运行滚动验证（按timeframe分别运行）:"
echo "  python rolling_portfolio_validator.py --timeframe m5 --window-months 12 --step-months 12 --workers auto"
echo "  python rolling_portfolio_validator.py --timeframe h1 --window-months 12 --step-months 12 --workers auto"
echo "  python rolling_portfolio_validator.py --timeframe d1 --window-months 12 --step-months 12 --workers auto"
