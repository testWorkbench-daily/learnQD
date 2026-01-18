#!/usr/bin/env python
"""分析不同时间段的市场环境特征"""
import pandas as pd
import numpy as np
import datetime

def analyze_period(start_date, end_date, period_name):
    """分析指定时间段的市场特征"""
    # 读取数据
    df = pd.read_csv('data/nq_m1_forward_adjusted.csv')
    df['datetime'] = pd.to_datetime(df['ts_event'])

    # 按日聚合
    df_daily = df.set_index('datetime').resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # 筛选时间段
    mask = (df_daily.index >= start_date) & (df_daily.index < end_date)
    period_data = df_daily[mask].copy()

    if len(period_data) == 0:
        print(f"\n{period_name}: 无数据")
        return None

    # 计算关键指标
    start_price = period_data['close'].iloc[0]
    end_price = period_data['close'].iloc[-1]
    total_return = (end_price / start_price - 1) * 100

    # 计算日收益率
    period_data['daily_return'] = period_data['close'].pct_change()
    daily_returns = period_data['daily_return'].dropna()

    # 波动率
    volatility = daily_returns.std() * np.sqrt(252) * 100

    # 趋势强度（累计收益的稳定性）
    cumulative_returns = (1 + daily_returns).cumprod()
    max_cumulative = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - max_cumulative) / max_cumulative
    max_drawdown = drawdowns.min() * 100

    # 正负收益天数
    positive_days = (daily_returns > 0).sum()
    negative_days = (daily_returns < 0).sum()
    total_days = len(daily_returns)

    # 趋势性指标：计算价格相对于移动平均的位置
    period_data['ma20'] = period_data['close'].rolling(20).mean()
    period_data['ma50'] = period_data['close'].rolling(50).mean()

    # 价格在均线上方的天数
    above_ma20 = (period_data['close'] > period_data['ma20']).sum()
    above_ma50 = (period_data['close'] > period_data['ma50']).sum()

    # 连续上涨/下跌天数
    streak = 0
    max_up_streak = 0
    max_down_streak = 0
    current_streak = 0

    for ret in daily_returns:
        if ret > 0:
            if current_streak >= 0:
                current_streak += 1
            else:
                current_streak = 1
            max_up_streak = max(max_up_streak, current_streak)
        elif ret < 0:
            if current_streak <= 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_down_streak = max(max_down_streak, abs(current_streak))

    # 价格区间震荡幅度
    price_range = (period_data['high'].max() - period_data['low'].min()) / start_price * 100

    # 判断市场类型
    if total_return > 10 and max_drawdown > -5:
        market_type = "强势单边上涨"
    elif total_return > 5 and max_drawdown > -10:
        market_type = "温和上涨"
    elif total_return < -10:
        market_type = "单边下跌"
    elif abs(total_return) < 5 and price_range > 20:
        market_type = "宽幅震荡"
    elif abs(total_return) < 5:
        market_type = "窄幅震荡"
    else:
        market_type = "混合趋势"

    print(f"\n{'='*80}")
    print(f"{period_name} ({start_date} ~ {end_date})")
    print(f"{'='*80}")
    print(f"\n市场类型: {market_type}")
    print(f"\n价格走势:")
    print(f"  起始价格: {start_price:.2f}")
    print(f"  结束价格: {end_price:.2f}")
    print(f"  总收益率: {total_return:.2f}%")
    print(f"  价格区间: {price_range:.2f}%")

    print(f"\n波动性指标:")
    print(f"  年化波动率: {volatility:.2f}%")
    print(f"  最大回撤: {max_drawdown:.2f}%")

    print(f"\n趋势性指标:")
    print(f"  正收益天数: {positive_days} ({positive_days/total_days*100:.1f}%)")
    print(f"  负收益天数: {negative_days} ({negative_days/total_days*100:.1f}%)")
    print(f"  价格在MA20上方: {above_ma20}/{len(period_data)} 天 ({above_ma20/len(period_data)*100:.1f}%)")
    print(f"  价格在MA50上方: {above_ma50}/{len(period_data)} 天 ({above_ma50/len(period_data)*100:.1f}%)")
    print(f"  最长连涨天数: {max_up_streak}")
    print(f"  最长连跌天数: {max_down_streak}")

    return {
        'period_name': period_name,
        'market_type': market_type,
        'total_return': total_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'positive_pct': positive_days/total_days*100,
        'above_ma20_pct': above_ma20/len(period_data)*100,
        'above_ma50_pct': above_ma50/len(period_data)*100,
    }

if __name__ == '__main__':
    print("\n分析两个时间段的市场环境差异\n")

    # 分析2023年（表现差）
    result1 = analyze_period('2023-01-01', '2024-01-01', '2023年（策略表现差）')

    # 分析2023年11月-2024年11月（表现好）
    result2 = analyze_period('2023-11-11', '2024-11-05', '2023-11-11至2024-11-05（策略表现好）')

    # 对比分析
    if result1 and result2:
        print(f"\n{'='*80}")
        print("对比总结")
        print(f"{'='*80}")
        print(f"\n市场类型:")
        print(f"  2023年: {result1['market_type']}")
        print(f"  好的时期: {result2['market_type']}")

        print(f"\n关键差异:")
        print(f"  总收益率: {result1['total_return']:.2f}% vs {result2['total_return']:.2f}%")
        print(f"  波动率: {result1['volatility']:.2f}% vs {result2['volatility']:.2f}%")
        print(f"  最大回撤: {result1['max_drawdown']:.2f}% vs {result2['max_drawdown']:.2f}%")
        print(f"  正收益天数占比: {result1['positive_pct']:.1f}% vs {result2['positive_pct']:.1f}%")
        print(f"  价格在MA20上方: {result1['above_ma20_pct']:.1f}% vs {result2['above_ma20_pct']:.1f}%")

        print(f"\n结论:")
        if result1['above_ma20_pct'] > 70 and result1['total_return'] > 10:
            print("  ⚠️  2023年是强势单边上涨市场")
            print("  ⚠️  策略在单边趋势市场中频繁交易，错过大趋势")
            print("  ⚠️  策略更适合震荡市或温和趋势市")

        if result2['volatility'] > result1['volatility'] * 1.3:
            print("  ✓  好的时期波动率更高，策略在波动市场中表现更好")
