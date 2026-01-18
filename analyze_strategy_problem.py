#!/usr/bin/env python
"""分析策略落后基准的根本原因"""
import pandas as pd
import numpy as np

print("="*80)
print("策略落后基准的根本原因分析")
print("="*80)

# 读取2023年交易记录
trades = pd.read_csv('backtest_results/trades_portfolio_rank3_combo_d1_20230101_20240101.csv')
trades['datetime'] = pd.to_datetime(trades['datetime'])

# 读取每日价值
daily_values = pd.read_csv('backtest_results/daily_values_portfolio_rank3_combo_d1_20230101_20240101.csv')
daily_values['datetime'] = pd.to_datetime(daily_values['datetime'])

# 读取市场数据
market = pd.read_csv('data/nq_m1_forward_adjusted.csv')
market['datetime'] = pd.to_datetime(market['ts_event'])
market_daily = market.set_index('datetime').resample('D').agg({
    'close': 'last'
}).dropna()

# 筛选2023年
market_2023 = market_daily[(market_daily.index >= '2023-01-01') & (market_daily.index < '2024-01-01')]

print("\n问题1: 仓位利用率")
print("-"*80)

# 计算平均仓位
trades_with_position = []
current_position = 0

for idx, trade in trades.iterrows():
    if trade['type'] == 'BUY':
        current_position += abs(trade['size'])
    else:
        current_position -= abs(trade['size'])

    trades_with_position.append({
        'datetime': trade['datetime'],
        'position': current_position,
        'portfolio_value': trade['portfolio_value']
    })

if trades_with_position:
    avg_position = np.mean([t['position'] for t in trades_with_position])
    max_position = max([t['position'] for t in trades_with_position])

    print(f"平均仓位: {avg_position:.2f} 手")
    print(f"最大仓位: {max_position} 手")
    print(f"仓位利用率: {avg_position/max_position*100:.1f}%" if max_position > 0 else "N/A")

    # 基准对比：买入持有是6手（$100,000 / ~$15,000）
    buy_hold_position = 6
    print(f"\n基准仓位: {buy_hold_position} 手（买入持有）")
    print(f"策略仓位不足: {(buy_hold_position - avg_position) / buy_hold_position * 100:.1f}%")

print("\n问题2: 交易频率")
print("-"*80)

total_trades = len(trades)
trading_days = len(market_2023)

print(f"总交易次数: {total_trades}")
print(f"交易日数: {trading_days}")
print(f"平均每月交易: {total_trades / 12:.1f} 次")

# 计算交易成本（假设每次1点滑点）
slippage_per_trade = 1  # 1点滑点
total_slippage_cost = total_trades * slippage_per_trade * avg_position
print(f"\n假设每次1点滑点:")
print(f"  总滑点成本: ${total_slippage_cost:.2f}")
print(f"  占初始资金: {total_slippage_cost/100000*100:.2f}%")

print("\n问题3: 持仓时长分析")
print("-"*80)

# 分析每次持仓的时长
holding_periods = []
entry_date = None
entry_type = None

for idx, trade in trades.iterrows():
    if trade['type'] == 'BUY' and current_position == 0:
        entry_date = trade['datetime']
        entry_type = 'LONG'
    elif trade['type'] == 'SELL' and entry_date:
        exit_date = trade['datetime']
        days_held = (exit_date - entry_date).days
        holding_periods.append(days_held)
        entry_date = None

if holding_periods:
    print(f"平均持仓天数: {np.mean(holding_periods):.1f} 天")
    print(f"中位数持仓: {np.median(holding_periods):.1f} 天")
    print(f"最长持仓: {max(holding_periods)} 天")
    print(f"最短持仓: {min(holding_periods)} 天")

    print(f"\n⚠️  对比：买入持有={trading_days}天")
    print(f"⚠️  策略平均持仓仅为基准的 {np.mean(holding_periods)/trading_days*100:.1f}%")

print("\n问题4: 错失的涨幅")
print("-"*80)

# 计算策略每日持仓暴露度
strategy_exposure = []

for idx, row in daily_values.iterrows():
    dt = row['datetime']

    # 找到这一天的持仓
    trades_before = trades[trades['datetime'] <= dt]
    if len(trades_before) > 0:
        position = 0
        for _, t in trades_before.iterrows():
            if t['type'] == 'BUY':
                position += abs(t['size'])
            else:
                position -= abs(t['size'])

        # 计算暴露度（相对于买入持有）
        exposure_pct = position / buy_hold_position if buy_hold_position > 0 else 0
        strategy_exposure.append(exposure_pct)

if strategy_exposure:
    avg_exposure = np.mean(strategy_exposure) * 100
    print(f"平均市场暴露度: {avg_exposure:.1f}%")
    print(f"基准暴露度: 100%")
    print(f"\n结论: 策略只有基准的{avg_exposure:.1f}%市场暴露")
    print(f"      理论收益上限 ≈ 基准收益 × {avg_exposure:.1f}%")
    print(f"      = 47.58% × {avg_exposure/100:.2f} = {47.58 * avg_exposure/100:.2f}%")

print("\n"+"="*80)
print("总结：策略落后基准的三大原因")
print("="*80)
print("\n1. 仓位不足：平均仓位远低于买入持有")
print("2. 频繁交易：交易成本（滑点+手续费）侵蚀收益")
print("3. 持仓时间短：在上涨市场中频繁换手，错失趋势")
print("\n这是一个【防守型策略】：")
print("  优势：波动小、回撤小、夏普高（在某些时期）")
print("  劣势：收益低、跟不上牛市")
print("\n如果你的目标是追求绝对收益，这个策略不适合。")
print("如果你的目标是稳定收益、低回撤，可以接受。")
