#!/bin/bash
# 批量运行所有策略回测 - 每个策略测试所有兼容的timeframe
# NQ期货回测系统
# 用法: bash run_all_strategies_2020_2024.sh [START_DATE] [END_DATE] [PARALLEL_JOBS]
# 示例: bash run_all_strategies_2020_2024.sh 2018-01-01 2019-01-01 4

# ============================================================================
# 配置参数（支持命令行传入）
# ============================================================================
START="${1:-2011-01-01}"           # 默认值：2018-01-01
END="${2:-2025-12-01}"             # 默认值：2025-01-01
PARALLEL_JOBS="${3:-}"            # 默认值：4

echo "================================================================"
echo "批量运行策略回测（全timeframe测试）"
echo "时间范围: $START 至 $END"
echo "并发数量: $PARALLEL_JOBS"
echo "================================================================"
echo ""

# 创建结果目录
mkdir -p backtest_results

# 定义所有timeframe
ALL_TIMEFRAMES=("m1" "m15" "h1" "h4" "d1")
INTRADAY_TIMEFRAMES=("m1" "m15")  # 日内策略只能用这些
# ALL_TIMEFRAMES=("m1" "m5" "m15" "m30" "h1" "h4" "d1")
# INTRADAY_TIMEFRAMES=("m1" "m5" "m15" "m30")  # 日内策略只能用这些

# 并发控制函数
run_with_limit() {
    local timeframe=$1
    local strategy=$2

    # 等待直到有可用的并发槽位
    while [ $(jobs -r | wc -l) -ge $PARALLEL_JOBS ]; do
        sleep 0.5
    done

    # 后台运行策略
    echo "  启动: $strategy @ $timeframe"
    python bt_main.py --start $START --end $END --timeframe $timeframe --atom $strategy --no-plot &
}

# ============================================================================
# 1. 日内策略 - 测试 m1/m5/m15/m30 四个周期
# ============================================================================
echo "================================================================"
echo "【1/3】日内策略 - 测试 m1/m5/m15/m30 四个周期"
echo "================================================================"

# ORB系列 (8个)
INTRADAY_STRATEGIES=(
    "orb" "orb_15min" "orb_30min" "orb_60min"
    "orb_30min_no_close" "orb_45min" "orb_aggressive" "orb_conservative"
)

# Intraday Momentum系列 (9个)
INTRADAY_STRATEGIES+=(
    "intraday_mom" "intraday_mom_0_5" "intraday_mom_1_0" "intraday_mom_1_5"
    "intraday_mom_2_0" "intraday_mom_0_3" "intraday_mom_aggressive"
    "intraday_mom_conservative" "intraday_mom_moderate"
)

# Intraday Reversal系列 (6个)
INTRADAY_STRATEGIES+=(
    "intraday_reversal" "intraday_rev_1_5" "intraday_rev_1_0" "intraday_rev_2_0"
    "intraday_rev_aggressive" "intraday_rev_conservative"
)

# VWAP Reversion系列 (6个)
INTRADAY_STRATEGIES+=(
    "vwap_reversion" "vwap_rev_1_0" "vwap_rev_1_5" "vwap_rev_2_0"
    "vwap_rev_aggressive" "vwap_rev_conservative"
)

echo "日内策略总数: ${#INTRADAY_STRATEGIES[@]} 个"
echo "每个策略测试 4 个timeframe (m1/m5/m15/m30)"
echo "总任务数: $((${#INTRADAY_STRATEGIES[@]} * 4))"
echo ""

for strategy in "${INTRADAY_STRATEGIES[@]}"; do
    for timeframe in "${INTRADAY_TIMEFRAMES[@]}"; do
        run_with_limit "$timeframe" "$strategy"
    done
done

wait  # 等待所有日内策略完成
echo ""
echo "✓ 日内策略全部完成"
echo ""

# ============================================================================
# 2. 非日内策略 - 测试所有7个周期 (m1/m5/m15/m30/h1/h4/d1)
# ============================================================================
echo "================================================================"
echo "【2/3】非日内策略 - 测试所有7个周期"
echo "================================================================"

NON_INTRADAY_STRATEGIES=()

# 趋势跟踪策略 (16个)
NON_INTRADAY_STRATEGIES+=(
    "sma_cross" "sma_5_20" "sma_10_30" "sma_20_60"
    "triple_ma" "triple_ma_5_20_50" "triple_ma_10_30_60" "triple_ma_8_21_55" "triple_ma_12_26_52"
    "adx_trend" "adx_14_25" "adx_14_30" "adx_14_20" "adx_21_25" "adx_10_25"
    "macd_trend"
)

# 均值回归策略 (14个)
NON_INTRADAY_STRATEGIES+=(
    "rsi_reversal"
    "boll_mr" "boll_mr_20_2" "boll_mr_20_2_5" "boll_mr_20_1_5" "boll_mr_30_2" "boll_mr_10_2" "boll_mr_strict"
    "cci_channel" "cci_20_100" "cci_20_150" "cci_20_80" "cci_14_100" "cci_30_100" "cci_strict"
)

# 突破策略 (47个)
# Donchian系列 (11个)
NON_INTRADAY_STRATEGIES+=(
    "donchian_channel" "donchian_20_10" "donchian_55_20" "donchian_20_20" "donchian_10_5" "donchian_5_3"
    "donchian_40_15" "donchian_turtle_sys1" "donchian_turtle_sys2" "donchian_aggressive" "donchian_conservative"
)

# Keltner系列 (8个)
NON_INTRADAY_STRATEGIES+=(
    "keltner_channel" "keltner_20_10_1_5" "keltner_20_10_2_0" "keltner_20_10_1_0" "keltner_20_14_1_5"
    "keltner_30_10_1_5" "keltner_10_10_1_5" "keltner_tight"
)

# ATR系列 (9个)
NON_INTRADAY_STRATEGIES+=(
    "atr_breakout" "atr_breakout_20_14_2" "atr_breakout_20_14_3" "atr_breakout_20_14_1_5" "atr_breakout_20_10_2"
    "atr_breakout_50_14_2" "atr_breakout_10_14_2" "atr_breakout_aggressive" "atr_breakout_conservative"
)

# VolBreakout系列 (9个)
NON_INTRADAY_STRATEGIES+=(
    "vol_breakout" "vol_breakout_14_2" "vol_breakout_14_2_5" "vol_breakout_14_1_5" "vol_breakout_20_2"
    "vol_breakout_10_2" "vol_breakout_10_3" "vol_breakout_aggressive" "vol_breakout_conservative"
)

# NewHighLow系列 (9个)
NON_INTRADAY_STRATEGIES+=(
    "new_hl" "new_hl_20" "new_hl_50" "new_hl_100" "new_hl_250" "new_hl_10" "new_hl_5" "new_hl_aggressive" "new_hl_conservative"
)

# Bollinger突破 (1个)
NON_INTRADAY_STRATEGIES+=("boll_breakout")

# 波动率策略 (18个)
NON_INTRADAY_STRATEGIES+=(
    "const_vol" "const_vol_10" "const_vol_15" "const_vol_20" "const_vol_conservative" "const_vol_aggressive"
    "vol_expansion" "vol_expansion_standard" "vol_expansion_sensitive" "vol_expansion_conservative" "vol_expansion_short" "vol_expansion_long"
    "vol_regime" "vol_regime_standard" "vol_regime_sensitive" "vol_regime_conservative" "vol_regime_short" "vol_regime_long"
)

# 经典系统 (7个)
NON_INTRADAY_STRATEGIES+=(
    "turtle" "turtle_sys1" "turtle_sys1_conservative" "turtle_sys1_aggressive" "turtle_sys2" "turtle_es" "turtle_mnq"
)

echo "非日内策略总数: ${#NON_INTRADAY_STRATEGIES[@]} 个"
echo "每个策略测试 7 个timeframe (m1/m5/m15/m30/h1/h4/d1)"
echo "总任务数: $((${#NON_INTRADAY_STRATEGIES[@]} * 7))"
echo ""

for strategy in "${NON_INTRADAY_STRATEGIES[@]}"; do
    for timeframe in "${ALL_TIMEFRAMES[@]}"; do
        run_with_limit "$timeframe" "$strategy"
    done
done

wait  # 等待所有非日内策略完成
echo ""
echo "✓ 非日内策略全部完成"
echo ""

# ============================================================================
# 3. 基准策略 - 测试所有7个周期
# ============================================================================
echo "================================================================"
echo "【3/3】基准策略 - Buy & Hold"
echo "================================================================"

for timeframe in "${ALL_TIMEFRAMES[@]}"; do
    run_with_limit "$timeframe" "buy_and_hold"
done

wait
echo ""
echo "✓ 基准策略完成"
echo ""

# ============================================================================
# 完成统计
# ============================================================================
echo "================================================================"
echo "所有策略回测完成！"
echo "================================================================"
echo ""
echo "【统计信息】"
echo "  - 日内策略: ${#INTRADAY_STRATEGIES[@]} 个 × 4 timeframes = $((${#INTRADAY_STRATEGIES[@]} * 4)) 个任务"
echo "  - 非日内策略: ${#NON_INTRADAY_STRATEGIES[@]} 个 × 7 timeframes = $((${#NON_INTRADAY_STRATEGIES[@]} * 7)) 个任务"
echo "  - 基准策略: 1 个 × 7 timeframes = 7 个任务"
echo "  - 总任务数: $((${#INTRADAY_STRATEGIES[@]} * 4 + ${#NON_INTRADAY_STRATEGIES[@]} * 7 + 7))"
echo ""
echo "【结果文件】"
echo "  所有结果已保存到: backtest_results/"
echo "  - trades_*.csv - 交易记录"
echo "  - daily_values_*.csv - 每日收益（用于组合优化）"
echo ""
echo "【下一步】"
echo "  可以按timeframe运行滚动验证分析最优组合："
echo "  python rolling_portfolio_validator.py --timeframe m1 --window-months 12 --step-months 12 --workers auto"
echo "  python rolling_portfolio_validator.py --timeframe m5 --window-months 12 --step-months 12 --workers auto"
echo "  python rolling_portfolio_validator.py --timeframe m15 --window-months 12 --step-months 12 --workers auto"
echo "  python rolling_portfolio_validator.py --timeframe m30 --window-months 12 --step-months 12 --workers auto"
echo "  python rolling_portfolio_validator.py --timeframe h1 --window-months 12 --step-months 12 --workers auto"
echo "  python rolling_portfolio_validator.py --timeframe h4 --window-months 12 --step-months 12 --workers auto"
echo "  python rolling_portfolio_validator.py --timeframe d1 --window-months 12 --step-months 12 --workers auto"
echo ""
