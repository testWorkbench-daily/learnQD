#!/usr/bin/env python
"""
前视偏差诊断工具

验证相关性分析和组合优化是否存在前视偏差(Look-Ahead Bias):
- 检查相关性矩阵是否只使用了窗口内数据
- 对比不同窗口的相关性是否有显著差异
- 验证数据过滤是否正确执行

用法:
    # 诊断指定窗口
    python diagnose_lookahead_bias.py \
      --timeframe d1 \
      --window 20200101-20201231

    # 对比多个窗口
    python diagnose_lookahead_bias.py \
      --timeframe d1 \
      --windows 20200101-20201231,20210101-20211231,20220101-20221231
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from analyze_correlation import StrategyCorrelationAnalyzer


def diagnose_window(
    start_date: str,
    end_date: str,
    timeframe: str,
    data_start_date: str,
    data_end_date: str,
    results_dir: str = 'backtest_results'
) -> Dict:
    """
    诊断指定窗口的前视偏差

    Args:
        start_date: 分析窗口开始日期 (YYYYMMDD)
        end_date: 分析窗口结束日期 (YYYYMMDD)
        timeframe: 时间周期
        data_start_date: 数据文件开始日期 (YYYYMMDD)
        data_end_date: 数据文件结束日期 (YYYYMMDD)
        results_dir: 结果目录

    Returns:
        诊断结果字典
    """
    print(f"\n{'='*80}")
    print(f"诊断窗口: {start_date} ~ {end_date}")
    print(f"数据文件: {data_start_date} ~ {data_end_date}")
    print(f"{'='*80}")

    # 创建分析器
    analyzer = StrategyCorrelationAnalyzer(
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        results_dir=results_dir,
        data_start_date=data_start_date,
        data_end_date=data_end_date
    )

    # 加载数据
    analyzer.load_strategy_returns()

    if not analyzer.daily_returns:
        print(f"  ✗ 未找到策略数据")
        return None

    # 验证1: 检查数据点数量是否符合窗口大小
    print(f"\n【验证1: 数据点数量】")
    start_dt = pd.to_datetime(start_date, format='%Y%m%d')
    end_dt = pd.to_datetime(end_date, format='%Y%m%d')
    expected_days = (end_dt - start_dt).days + 1

    for strategy, returns in analyzer.daily_returns.items():
        actual_days = len(returns)
        print(f"  {strategy}: {actual_days} 个数据点 (预期约 {expected_days} 天)")

        # 检查日期范围
        min_date = pd.to_datetime(returns.index.min())
        max_date = pd.to_datetime(returns.index.max())

        if min_date < start_dt or max_date > end_dt:
            print(f"    ⚠️  警告: 数据超出窗口范围!")
            print(f"       数据范围: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
            print(f"       窗口范围: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}")
            return {
                'window': f"{start_date}-{end_date}",
                'has_lookahead_bias': True,
                'reason': f"{strategy} 数据超出窗口范围"
            }

    print(f"  ✓ 所有策略数据都在窗口范围内")

    # 验证2: 计算相关性矩阵
    print(f"\n【验证2: 相关性矩阵】")
    corr_matrix = analyzer.calculate_correlation_matrix()

    if corr_matrix.empty:
        print(f"  ✗ 相关性矩阵计算失败")
        return None

    # 计算相关性统计
    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    avg_corr = corr_values.mean()
    std_corr = corr_values.std()
    min_corr = corr_values.min()
    max_corr = corr_values.max()

    print(f"  平均相关性: {avg_corr:.3f}")
    print(f"  相关性标准差: {std_corr:.3f}")
    print(f"  相关性范围: [{min_corr:.3f}, {max_corr:.3f}]")

    return {
        'window': f"{start_date}-{end_date}",
        'has_lookahead_bias': False,
        'num_strategies': len(analyzer.daily_returns),
        'avg_correlation': avg_corr,
        'corr_std': std_corr,
        'corr_min': min_corr,
        'corr_max': max_corr,
        'avg_data_points': np.mean([len(r) for r in analyzer.daily_returns.values()])
    }


def compare_windows(
    windows: List[Tuple[str, str]],
    timeframe: str,
    data_start_date: str,
    data_end_date: str,
    results_dir: str = 'backtest_results'
):
    """
    对比多个窗口的相关性差异

    前视偏差的标志:
    - 不同窗口的相关性矩阵几乎相同（因为都用了全量数据）
    - 相关性变化很小

    正常情况:
    - 不同窗口的相关性有明显差异（反映不同市场环境）
    """
    print(f"\n{'='*80}")
    print(f"对比 {len(windows)} 个窗口的相关性差异")
    print(f"{'='*80}")

    results = []

    for start_date, end_date in windows:
        result = diagnose_window(
            start_date, end_date, timeframe,
            data_start_date, data_end_date, results_dir
        )
        if result:
            results.append(result)

    if len(results) < 2:
        print("\n至少需要2个窗口才能进行对比")
        return

    # 创建对比表
    print(f"\n{'='*80}")
    print(f"窗口对比")
    print(f"{'='*80}")
    print(f"{'窗口':<25} {'策略数':<8} {'平均相关性':<12} {'相关性标准差':<14} {'数据点数':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['window']:<25} {r['num_strategies']:<8} {r['avg_correlation']:>11.3f} "
              f"{r['corr_std']:>13.3f} {r['avg_data_points']:>9.0f}")

    # 计算相关性变化
    avg_corrs = [r['avg_correlation'] for r in results]
    corr_variation = np.std(avg_corrs)

    print(f"\n相关性跨窗口变化 (标准差): {corr_variation:.3f}")

    # 判断
    if corr_variation < 0.05:
        print(f"\n⚠️  警告: 相关性变化很小 ({corr_variation:.3f} < 0.05)")
        print(f"   这可能表明存在前视偏差 - 所有窗口都使用了相同的数据")
        print(f"   正常情况下,不同市场时期的相关性应该有明显差异")
    else:
        print(f"\n✓ 相关性变化合理 ({corr_variation:.3f} >= 0.05)")
        print(f"  不同窗口的相关性有差异,符合预期")

    # 保存结果
    output_file = Path(results_dir) / f'lookahead_bias_diagnosis_{timeframe}.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"\n诊断结果已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='前视偏差诊断工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--timeframe', default='d1', help='时间周期 (默认: d1)')
    parser.add_argument('--window', type=str, help='单个窗口 (格式: YYYYMMDD-YYYYMMDD)')
    parser.add_argument('--windows', type=str, help='多个窗口,逗号分隔 (格式: YYYYMMDD-YYYYMMDD,YYYYMMDD-YYYYMMDD,...)')
    parser.add_argument('--data-start', default='20200101', help='数据文件开始日期 (YYYYMMDD, 默认: 20200101)')
    parser.add_argument('--data-end', default='20241231', help='数据文件结束日期 (YYYYMMDD, 默认: 20241231)')
    parser.add_argument('--results-dir', default='backtest_results', help='结果目录 (默认: backtest_results)')

    args = parser.parse_args()

    if not args.window and not args.windows:
        print("错误: 必须指定 --window 或 --windows")
        parser.print_help()
        return

    # 解析窗口
    windows = []

    if args.window:
        parts = args.window.split('-')
        if len(parts) != 2:
            print(f"错误: 窗口格式错误: {args.window}")
            print(f"正确格式: YYYYMMDD-YYYYMMDD")
            return
        windows.append((parts[0], parts[1]))

    if args.windows:
        for window_str in args.windows.split(','):
            parts = window_str.strip().split('-')
            if len(parts) != 2:
                print(f"错误: 窗口格式错误: {window_str}")
                continue
            windows.append((parts[0], parts[1]))

    if len(windows) == 0:
        print("错误: 没有有效的窗口")
        return

    # 运行诊断
    if len(windows) == 1:
        # 单窗口诊断
        start_date, end_date = windows[0]
        result = diagnose_window(
            start_date, end_date, args.timeframe,
            args.data_start, args.data_end, args.results_dir
        )

        if result:
            if result['has_lookahead_bias']:
                print(f"\n⚠️  检测到前视偏差!")
                print(f"   原因: {result['reason']}")
            else:
                print(f"\n✓ 未检测到前视偏差")
    else:
        # 多窗口对比
        compare_windows(
            windows, args.timeframe,
            args.data_start, args.data_end, args.results_dir
        )

    print("\n诊断完成!")


if __name__ == '__main__':
    main()
