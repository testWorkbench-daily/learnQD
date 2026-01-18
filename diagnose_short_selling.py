#!/usr/bin/env python
"""诊断卖空功能"""
import pandas as pd

# 读取交易记录
df = pd.read_csv('backtest_results/trades_portfolio_rank3_combo_d1_20231111_20241105.csv')

print("=" * 80)
print("卖空交易诊断")
print("=" * 80)

# 分析所有SELL交易
sell_trades = df[df['type'] == 'SELL'].copy()

print(f"\n总交易次数: {len(df)}")
print(f"SELL交易次数: {len(sell_trades)}")
print(f"BUY交易次数: {len(df[df['type'] == 'BUY'])}")

# 检查是否有真正的做空（size为负）
sell_trades['is_short'] = sell_trades['size'] < 0

short_sells = sell_trades[sell_trades['is_short']]
print(f"\n真正的做空交易（从0或多头变空头）: {len(short_sells)}")

if len(short_sells) > 0:
    print("\n做空交易详情:")
    print("-" * 80)
    for idx, row in short_sells.iterrows():
        print(f"时间: {row['datetime']}")
        print(f"  价格: {row['price']:.2f}")
        print(f"  手数: {row['size']}")
        print(f"  现金: {row['cash']:.2f}")
        print(f"  盈亏: {row['pnl']:.2f}")
        print()

# 检查仓位变化
print("\n仓位变化轨迹:")
print("-" * 80)
print(f"{'时间':<12} {'操作':<6} {'手数':>6} {'累计仓位':>10} {'现金':>12} {'盈亏':>10}")
print("-" * 80)

position = 0
for idx, row in df.iterrows():
    if row['type'] == 'BUY':
        position += abs(row['size'])
    else:  # SELL
        position -= abs(row['size'])

    print(f"{row['datetime'][:10]:<12} {row['type']:<6} {row['size']:>6} {position:>10} {row['cash']:>12.2f} {row['pnl']:>10.2f}")

# 统计卖空期间的表现
print("\n" + "=" * 80)
print("卖空期间统计")
print("=" * 80)

short_periods = []
in_short = False
short_entry_idx = None

for idx, row in df.iterrows():
    if row['type'] == 'SELL' and row['size'] < 0 and not in_short:
        # 进入空头
        in_short = True
        short_entry_idx = idx
    elif row['type'] == 'BUY' and in_short:
        # 退出空头
        if short_entry_idx is not None:
            entry = df.iloc[short_entry_idx]
            exit_trade = row
            pnl = exit_trade['pnl']
            short_periods.append({
                'entry_date': entry['datetime'],
                'exit_date': exit_trade['datetime'],
                'entry_price': entry['price'],
                'exit_price': exit_trade['price'],
                'pnl': pnl
            })
        in_short = False
        short_entry_idx = None

if len(short_periods) > 0:
    print(f"\n共有 {len(short_periods)} 个卖空期间:")
    for i, period in enumerate(short_periods, 1):
        print(f"\n期间 {i}:")
        print(f"  开仓: {period['entry_date'][:10]} @ {period['entry_price']:.2f}")
        print(f"  平仓: {period['exit_date'][:10]} @ {period['exit_price']:.2f}")
        print(f"  盈亏: {period['pnl']:.2f}")
else:
    print("\n未发现完整的做空期间（可能还在持仓中）")

# 计算多空占比
total_bars = len(df)
# 简化统计：统计多头和空头交易次数
long_trades = df[(df['type'] == 'BUY') & (df['size'] > 0)]
short_trades = df[(df['type'] == 'SELL') & (df['size'] < 0)]

print("\n" + "=" * 80)
print("多空比例")
print("=" * 80)
print(f"多头开仓次数: {len(long_trades)}")
print(f"空头开仓次数: {len(short_trades)}")
print(f"做空占比: {len(short_trades) / (len(long_trades) + len(short_trades)) * 100:.1f}%")
