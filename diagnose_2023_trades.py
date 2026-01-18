#!/usr/bin/env python
"""诊断2023年策略交易问题"""
import pandas as pd
import numpy as np

# 读取2023年交易记录
trades_df = pd.read_csv('backtest_results/trades_portfolio_rank3_combo_d1_20230101_20240101.csv')
trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])

# 读取市场数据
market_df = pd.read_csv('data/nq_m1_forward_adjusted.csv')
market_df['datetime'] = pd.to_datetime(market_df['ts_event'])
market_df = market_df.set_index('datetime').resample('D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last'
}).dropna()

# 筛选2023年数据
market_2023 = market_df[(market_df.index >= '2023-01-01') & (market_df.index < '2024-01-01')]

print("="*80)
print("2023年策略交易诊断")
print("="*80)

# 市场概况
start_price = market_2023['close'].iloc[0]
end_price = market_2023['close'].iloc[-1]
market_return = (end_price / start_price - 1) * 100

print(f"\n市场表现:")
print(f"  起始价格: {start_price:.2f}")
print(f"  结束价格: {end_price:.2f}")
print(f"  市场涨幅: {market_return:.2f}%")
print(f"  买入持有收益: 假设$100,000 → ${100000 * (1 + market_return/100):.2f}")

# 策略表现
final_value = trades_df['portfolio_value'].iloc[-1]
print(f"\n策略表现:")
print(f"  最终资金: ${final_value:.2f}")
print(f"  策略收益: {(final_value/100000 - 1)*100:.2f}%")

# 错失的机会
print(f"\n错失机会分析:")
print(f"  如果买入持有: +{market_return:.2f}%")
print(f"  策略实际收益: +{(final_value/100000 - 1)*100:.2f}%")
print(f"  差距: {market_return - (final_value/100000 - 1)*100:.2f}%")

# 分析交易类型
print(f"\n交易统计:")
print(f"  总交易次数: {len(trades_df)}")
buy_trades = trades_df[trades_df['type'] == 'BUY']
sell_trades = trades_df[trades_df['type'] == 'SELL']
print(f"  买入次数: {len(buy_trades)}")
print(f"  卖出次数: {len(sell_trades)}")

# 分析每次卖出后的市场表现
print(f"\n卖出后市场表现分析（策略是否卖早了？）:")
print("-"*80)

for idx, sell_trade in sell_trades.iterrows():
    sell_date = sell_trade['datetime']
    sell_price = sell_trade['price']

    # 找到下一次买入
    future_buys = buy_trades[buy_trades['datetime'] > sell_date]

    if len(future_buys) > 0:
        next_buy = future_buys.iloc[0]
        next_buy_date = next_buy['datetime']
        next_buy_price = next_buy['price']

        # 计算这段时间市场的最高价
        period_market = market_2023[(market_2023.index > sell_date) & (market_2023.index <= next_buy_date)]
        if len(period_market) > 0:
            max_price_in_period = period_market['high'].max()
            missed_gain = (max_price_in_period / sell_price - 1) * 100

            print(f"{sell_date.strftime('%Y-%m-%d')}: 卖出@{sell_price:.2f}")
            print(f"  → 下次买入: {next_buy_date.strftime('%Y-%m-%d')}@{next_buy_price:.2f} " +
                  f"({(next_buy_price/sell_price - 1)*100:+.2f}%)")
            print(f"  → 期间最高: {max_price_in_period:.2f} (错失{missed_gain:.2f}%上涨)")

            if missed_gain > 3:
                print(f"  ⚠️  卖早了！错过了{missed_gain:.2f}%的涨幅")
            print()

# 计算持仓时间占比
total_days = len(market_2023)
print(f"\n持仓分析:")
print(f"  总交易日: {total_days}")

# 简化计算：估算空仓时间
# 假设每次SELL之后到下次BUY之前是空仓
empty_position_days = 0

for idx, sell_trade in sell_trades.iterrows():
    sell_date = sell_trade['datetime']
    future_buys = buy_trades[buy_trades['datetime'] > sell_date]

    if len(future_buys) > 0:
        next_buy_date = future_buys.iloc[0]['datetime']
        days_empty = len(market_2023[(market_2023.index > sell_date) & (market_2023.index <= next_buy_date)])
        empty_position_days += days_empty

holding_days = total_days - empty_position_days
print(f"  持仓天数: 约{holding_days} ({holding_days/total_days*100:.1f}%)")
print(f"  空仓天数: 约{empty_position_days} ({empty_position_days/total_days*100:.1f}%)")

if empty_position_days / total_days > 0.3:
    print(f"  ⚠️  空仓时间超过30%，在上涨市场中错失大量机会")

# 总结
print(f"\n{'='*80}")
print("问题总结")
print(f"{'='*80}")
print(f"\n1. 市场类型: 强劲单边上涨市场 (+53%)")
print(f"2. 策略问题: 频繁交易，过早卖出")
print(f"3. 根本原因: 组合策略中包含均值回归成分，在趋势市场中过早获利了结")
print(f"4. 表现差距: 策略只赚1.92%，而市场涨了53.35%")
