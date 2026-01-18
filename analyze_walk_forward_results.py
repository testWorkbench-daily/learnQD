#!/usr/bin/env python3
"""
Walk-Forward结果分析工具
"""
import pandas as pd
import numpy as np

# 读取结果
summary = pd.read_csv('backtest_results/walk_forward/window_summary_d1.csv')
details = pd.read_csv('backtest_results/walk_forward/walk_forward_details_d1.csv')
robustness = pd.read_csv('backtest_results/walk_forward/portfolio_robustness_d1.csv')

print("\n" + "="*80)
print("Walk-Forward验证结果分析")
print("="*80)

# 1. 窗口级别的过拟合分析
print("\n【1. 各窗口的过拟合情况】")
print(f"{'窗口':<6} {'训练夏普':<12} {'测试夏普':<12} {'衰减':<10} {'过拟合%':<12} {'返回%':<10}")
print("-"*80)
for _, row in summary.iterrows():
    wid = int(row['window_id'])
    train = row['train_sharpe']
    test = row['test_sharpe']
    decay = train - test
    oi = row['overfitting_index'] * 100
    ret = row['test_return_pct']
    print(f"{wid:<6} {train:<12.2f} {test:<12.2f} {decay:<10.2f} {oi:<11.1f}% {ret:<10.2f}%")

# 2. 过拟合统计
print("\n【2. 过拟合统计】")
oi_values = summary['overfitting_index'].dropna()
print(f"平均过拟合指数: {oi_values.mean():.2%}")
print(f"最小过拟合指数: {oi_values.min():.2%}")
print(f"最大过拟合指数: {oi_values.max():.2%}")
print(f"标准差: {oi_values.std():.2%}")

# 3. 测试期收益统计
print("\n【3. 测试期表现】")
ret_values = summary['test_return_pct'].dropna()
print(f"平均收益率: {ret_values.mean():.2f}%")
print(f"最大收益率: {ret_values.max():.2f}%")
print(f"最小收益率: {ret_values.min():.2f}%")
print(f"负收益窗口数: {(ret_values < 0).sum()} / {len(ret_values)}")

# 4. 组合层级分析
print("\n【4. 组合稳健性Top10】")
print(f"{'排名':<6} {'过拟合%':<12} {'平均测试夏普':<15} {'出现频率':<10} {'策略':<40}")
print("-"*80)
for idx, row in robustness.head(10).iterrows():
    oi = row['overfitting_index'] * 100
    sharpe = row['avg_test_sharpe']
    freq = int(row['frequency']) if pd.notna(row['frequency']) else 0
    strats = row['strategies'][:40]
    print(f"{idx+1:<6} {oi:<11.1f}% {sharpe:<15.2f} {freq:<10} {strats:<40}")

# 5. 关键发现
print("\n【5. 关键发现】")
best_oi = robustness['overfitting_index'].min()
best_sharpe = robustness['avg_test_sharpe'].max()
print(f"最佳过拟合指数: {best_oi:.2%}")
print(f"最佳平均测试夏普: {best_sharpe:.2f}")

if best_oi < 0.20:
    print("✅ 发现了好的组合 (过拟合指数 < 20%)")
elif best_oi < 0.30:
    print("⚠️  找到可接受的组合 (过拟合指数 < 30%)")
else:
    print("❌ 没有找到稳健组合 (所有组合过拟合指数 > 30%)")

# 6. 样本外表现
print("\n【6. 样本外表现问题】")
print(f"正收益窗口: {(summary['test_return_pct'] > 0).sum()} / {len(summary)}")
print(f"负夏普窗口: {(summary['test_sharpe'] < 0).sum()} / {len(summary)}")

if (summary['test_sharpe'] < 0).sum() > 0:
    print("⚠️  警告: 部分窗口的测试夏普为负（策略完全失效）")

# 7. 推荐
print("\n【7. 诊断结论】")
print("""
问题根源:
1. 策略学习了市场特定时期的规律，难以跨越环境变化
2. 2020-2024包含极端市场环境变化:
   - 2020-2021: 疫情后牛市
   - 2022: 加息熊市
   - 2023-2024: 反弹+震荡
3. 在这些环境下，同一策略组合的表现差异巨大

建议:
1. 使用更长的训练期（24个月而非12个月）
2. 提高质量筛选门槛
3. 考虑基于市场环境的动态配置
4. 避免使用过拟合指数>30%的组合
""")

print("="*80)
