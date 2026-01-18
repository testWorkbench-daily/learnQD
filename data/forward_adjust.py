#!/usr/bin/env python
"""
NQ期货前复权处理 (Ratio法)

处理季度换期（rollover）造成的跳空问题：
- 换期日：3、6、9、12月的第三个周五
- 计算换期日最后一根K线与下一根K线的比例
- 将换期日之前的数据进行累计比例调整（前复权）

Ratio法原理：
- 最新数据保持不变
- ratio = next_open / close
- 旧数据 * ratio，使得调整后的 close * ratio = next_open，消除缺口
- 多个换期点时累乘ratio

用法:
    python forward_adjust.py
    python forward_adjust.py --input ./nq_m1_all_backtrader.csv --output ./nq_m1_forward_adjusted.csv
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
import argparse


def get_third_friday(year: int, month: int) -> datetime:
    """获取指定年月的第三个周五"""
    first_day = datetime(year, month, 1)
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    third_friday = first_friday + timedelta(days=14)
    return third_friday


def get_rollover_dates(start_year: int, end_year: int) -> List[datetime]:
    """获取所有换期日（季度月的第三个周五）"""
    rollover_dates = []
    quarterly_months = [3, 6, 9, 12]
    
    for year in range(start_year, end_year + 1):
        for month in quarterly_months:
            third_friday = get_third_friday(year, month)
            rollover_dates.append(third_friday)
    
    return sorted(rollover_dates)


def find_rollover_gaps(df: pd.DataFrame, rollover_dates: List[datetime]) -> List[dict]:
    """
    找到每个换期日的跳空缺口，计算ratio
    """
    gaps = []
    
    for rollover_date in rollover_dates:
        rollover_day_start = rollover_date.replace(hour=0, minute=0, second=0)
        rollover_day_end = rollover_date.replace(hour=23, minute=59, second=59)
        
        rollover_day_mask = (df['datetime'] >= rollover_day_start) & (df['datetime'] <= rollover_day_end)
        rollover_day_bars = df[rollover_day_mask]
        
        if rollover_day_bars.empty:
            continue
        
        last_bar_idx = rollover_day_bars.index[-1]
        last_bar = df.loc[last_bar_idx]
        
        next_bars = df[df.index > last_bar_idx]
        if next_bars.empty:
            continue
        
        next_bar = next_bars.iloc[0]
        
        # ratio = next_open / close
        ratio = next_bar['open'] / last_bar['close']
        gap = next_bar['open'] - last_bar['close']
        
        # 只记录显著缺口（>10点，相对变化<1%）
        relative_gap = abs(gap) / last_bar['close'] * 100
        if abs(gap) > 10 and relative_gap < 1.0:
            gaps.append({
                'date': rollover_date,
                'ratio': ratio,
                'gap': gap,
                'split_idx': last_bar_idx,
                'close': last_bar['close'],
                'next_open': next_bar['open'],
            })
            print(f"  {rollover_date.strftime('%Y-%m-%d')}: "
                  f"{last_bar['close']:.2f} -> {next_bar['open']:.2f}, "
                  f"ratio={ratio:.6f}, gap={gap:+.2f}")
    
    return gaps


def forward_adjust(df: pd.DataFrame, gaps: List[dict]) -> pd.DataFrame:
    """
    执行前复权 (Ratio法)
    
    策略：
    - 从新到旧累乘ratio
    - 每个时间段乘以该时间点之后的累计ratio
    - 最新数据不调整 (ratio=1)
    """
    df = df.copy()
    price_cols = ['open', 'high', 'low', 'close']
    
    # 按日期升序排列（从旧到新）
    gaps_asc = sorted(gaps, key=lambda x: x['date'])
    n = len(gaps_asc)
    
    # 累乘ratio（从后往前）
    # 时间段i的调整ratio = 该时间点之后所有ratio的乘积
    cum_ratio = [1.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        cum_ratio[i] = cum_ratio[i + 1] * gaps_asc[i]['ratio']
    
    print(f"\n各时间段调整比例 (Ratio法):")
    
    # 应用调整
    prev_idx = -1
    for i, g in enumerate(gaps_asc):
        adjustment_ratio = cum_ratio[i]  # 该时间点之后的累计ratio
        
        # 创建mask
        if prev_idx == -1:
            mask = df.index <= g['split_idx']
        else:
            mask = (df.index > prev_idx) & (df.index <= g['split_idx'])
        
        print(f"  时间段 {i+1} [{prev_idx+1}:{g['split_idx']}]: ×{adjustment_ratio:.6f} ({g['date'].strftime('%Y-%m-%d')}之前)")
        
        # 应用调整：乘以累计ratio
        for col in price_cols:
            df.loc[mask, col] = df.loc[mask, col] * adjustment_ratio
        
        prev_idx = g['split_idx']
    
    # 最后一段（最新数据）不调整
    print(f"  时间段 {n+1} [{prev_idx+1}:end]: ×1.000000 (最新)")
    
    return df


def verify_adjustment(df: pd.DataFrame, gaps: List[dict]) -> bool:
    """验证调整结果"""
    print(f"\n验证结果:")
    all_fixed = True
    
    for g in gaps:
        split_idx = g['split_idx']
        if split_idx >= len(df) - 1:
            continue
        
        adjusted_close = df.loc[split_idx, 'close']
        next_open = df.loc[split_idx + 1, 'open']
        new_gap = next_open - adjusted_close
        new_ratio = next_open / adjusted_close if adjusted_close != 0 else 0
        
        # Ratio法验证：ratio应该接近1
        if abs(new_ratio - 1.0) < 0.0001:
            print(f"  ✅ {g['date'].strftime('%Y-%m-%d')}: {adjusted_close:.2f} -> {next_open:.2f}, ratio={new_ratio:.6f}")
        else:
            print(f"  ⚠️ {g['date'].strftime('%Y-%m-%d')}: {adjusted_close:.2f} -> {next_open:.2f}, ratio={new_ratio:.6f}")
            all_fixed = False
    
    return all_fixed


def main():
    parser = argparse.ArgumentParser(description='NQ期货前复权处理 (Ratio法)')
    parser.add_argument('--input', default='./nq_m1_cleaned.csv',
                        help='输入文件')
    parser.add_argument('--output', default='./nq_m1_forward_adjusted.csv',
                        help='输出文件')
    parser.add_argument('--start-year', type=int, default=2010, help='开始年份')
    parser.add_argument('--end-year', type=int, default=2025, help='结束年份')
    parser.add_argument('--dry-run', action='store_true', help='只检测缺口，不执行调整')
    args = parser.parse_args()
    
    print("=" * 60)
    print("NQ期货前复权处理 (Ratio法)")
    print("=" * 60)
    
    # 读取数据
    print(f"\n1. 读取数据: {args.input}")
    df = pd.read_csv(args.input, parse_dates=['ts_event'])
    df = df.rename(columns={'ts_event': 'datetime'})
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"  数据量: {len(df):,} 条")
    print(f"  时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    
    # 获取换期日
    print(f"\n3. 获取换期日...")
    rollover_dates = get_rollover_dates(args.start_year, args.end_year)
    
    min_date = df['datetime'].min()
    max_date = df['datetime'].max()
    rollover_dates = [d for d in rollover_dates 
                      if d >= min_date.replace(tzinfo=None) and d <= max_date.replace(tzinfo=None)]
    print(f"  有效换期日: {len(rollover_dates)} 个")
    
    # 找到跳空缺口
    print(f"\n4. 检测跳空缺口...")
    gaps = find_rollover_gaps(df, rollover_dates)
    print(f"\n  发现 {len(gaps)} 个有效缺口")
    
    if not gaps:
        print("\n没有需要调整的缺口")
        return
    
    # 计算累计ratio
    total_ratio = 1.0
    for g in gaps:
        total_ratio *= g['ratio']
    print(f"  累计ratio: {total_ratio:.6f}")
    
    if args.dry_run:
        print("\n[Dry Run] 不执行实际调整")
        return
    
    # 执行前复权
    print(f"\n5. 执行前复权 (Ratio法)...")
    df_adjusted = forward_adjust(df, gaps)
    
    # 显示调整前后对比
    print(f"\n6. 调整前后对比:")
    print(f"  原始: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    print(f"  调整: {df_adjusted['close'].min():.2f} ~ {df_adjusted['close'].max():.2f}")
    
    # 验证
    verify_adjustment(df_adjusted, gaps)
    
    # 保存结果
    print(f"\n7. 保存: {args.output}")
    df_adjusted = df_adjusted.rename(columns={'datetime': 'ts_event'})
    float_cols = df_adjusted.select_dtypes(include='float').columns
    df_adjusted[float_cols] = df_adjusted[float_cols].round(2)
    df_adjusted.to_csv(args.output, index=False)
    print(f"\n✅ 完成")


if __name__ == '__main__':
    main()
