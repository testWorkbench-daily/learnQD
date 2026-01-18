#!/usr/bin/env python
"""
Atom权重配置对比工具

对比Portfolio Atom的固定权重配置 vs 滚动验证推荐的权重配置:
- 提取Atom代码中的硬编码权重
- 对比滚动验证每个窗口的最优权重
- 计算权重偏差和影响

用法:
    # 对比portfolio_rank3_combo的权重
    python compare_atom_weights.py \
      --atom portfolio_rank3_combo \
      --rolling-results backtest_results/rolling_validation/robust_portfolios_ranking.csv

    # 对比walk-forward结果
    python compare_atom_weights.py \
      --atom portfolio_rank3_combo \
      --walk-forward-results backtest_results/walk_forward/portfolio_robustness_d1.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import re


# Atom权重配置注册表
ATOM_CONFIGS = {
    'portfolio_rank3_combo': {
        'strategies': ['vol_breakout_aggressive', 'vol_regime_long', 'triple_ma', 'rsi_reversal'],
        'weights': [0.0843, 0.2390, 0.3366, 0.3401],
        'description': '稳健排名 #3: 波动率突破 + 波动率择时 + 三重均线 + RSI反转'
    },
    # 可以添加更多Atom配置
}


def load_atom_config(atom_name: str) -> Dict:
    """
    加载Atom的权重配置

    Args:
        atom_name: Atom名称

    Returns:
        配置字典 {strategies: [...], weights: [...], description: '...'}
    """
    if atom_name not in ATOM_CONFIGS:
        print(f"错误: 未找到Atom配置: {atom_name}")
        print(f"可用的Atom: {list(ATOM_CONFIGS.keys())}")
        return None

    return ATOM_CONFIGS[atom_name]


def parse_rolling_results(rolling_file: str, atom_config: Dict) -> List[Dict]:
    """
    解析滚动验证结果,提取匹配Atom组合的权重

    Args:
        rolling_file: 滚动验证结果文件 (robust_portfolios_ranking.csv)
        atom_config: Atom配置

    Returns:
        窗口权重列表
    """
    if not Path(rolling_file).exists():
        print(f"错误: 文件不存在: {rolling_file}")
        return []

    df = pd.read_csv(rolling_file)

    # Atom的策略组成 (排序后用于匹配)
    atom_strategies_set = set(atom_config['strategies'])

    matching_rows = []

    for _, row in df.iterrows():
        # 解析策略列表
        if 'strategies' in row:
            row_strategies = set(str(row['strategies']).split(','))
        elif 'strategy_combo' in row:
            row_strategies = set(str(row['strategy_combo']).split(','))
        else:
            continue

        # 检查是否匹配
        if row_strategies == atom_strategies_set:
            matching_rows.append(row.to_dict())

    return matching_rows


def parse_walk_forward_results(wf_file: str, atom_config: Dict) -> List[Dict]:
    """
    解析walk-forward结果

    Args:
        wf_file: walk-forward结果文件
        atom_config: Atom配置

    Returns:
        窗口配置列表
    """
    if not Path(wf_file).exists():
        print(f"错误: 文件不存在: {wf_file}")
        return []

    df = pd.read_csv(wf_file)

    atom_strategies_set = set(atom_config['strategies'])

    matching_rows = []

    for _, row in df.iterrows():
        if 'strategies' in row:
            row_strategies = set(str(row['strategies']).split(','))

            if row_strategies == atom_strategies_set:
                matching_rows.append(row.to_dict())

    return matching_rows


def compare_weights(
    atom_name: str,
    atom_config: Dict,
    rolling_results: List[Dict] = None,
    wf_results: List[Dict] = None
):
    """
    对比Atom固定权重 vs 推荐权重

    Args:
        atom_name: Atom名称
        atom_config: Atom配置
        rolling_results: 滚动验证结果列表
        wf_results: Walk-forward结果列表
    """
    print("\n" + "=" * 100)
    print(f"Atom权重配置对比: {atom_name}")
    print("=" * 100)
    print(f"描述: {atom_config['description']}")
    print(f"策略: {', '.join(atom_config['strategies'])}")
    print(f"固定权重: {atom_config['weights']}")
    print("-" * 100)

    # Atom固定权重
    atom_weights = np.array(atom_config['weights'])

    # 收集所有推荐权重
    all_recommended_weights = []
    sources = []

    if rolling_results:
        for result in rolling_results:
            # 解析权重(可能在不同列)
            if 'weights' in result and pd.notna(result['weights']):
                weights_str = str(result['weights'])
                weights = [float(w) for w in weights_str.split(',')]
                all_recommended_weights.append(weights)
                sources.append('rolling_validation')

    if wf_results:
        for result in wf_results:
            if 'weights' in result and pd.notna(result['weights']):
                weights_str = str(result['weights'])
                weights = [float(w) for w in weights_str.split(',')]
                all_recommended_weights.append(weights)
                sources.append('walk_forward')

    if not all_recommended_weights:
        print("\n未找到匹配的推荐权重配置")
        print("可能原因:")
        print("  1. 该Atom的策略组合未出现在推荐结果中")
        print("  2. 结果文件格式不匹配")
        return

    # 计算统计
    recommended_weights_array = np.array(all_recommended_weights)
    mean_weights = recommended_weights_array.mean(axis=0)
    std_weights = recommended_weights_array.std(axis=0)
    min_weights = recommended_weights_array.min(axis=0)
    max_weights = recommended_weights_array.max(axis=0)

    # 打印对比
    print(f"\n找到 {len(all_recommended_weights)} 个推荐权重配置")
    print("\n权重对比:")
    print(f"{'策略':<30} {'Atom固定':<12} {'推荐平均':<12} {'推荐范围':<25} {'偏差':<10}")
    print("-" * 100)

    for i, strategy in enumerate(atom_config['strategies']):
        deviation = atom_weights[i] - mean_weights[i]
        deviation_pct = (deviation / mean_weights[i]) * 100 if mean_weights[i] > 0 else 0

        print(f"{strategy:<30} {atom_weights[i]*100:>10.2f}% {mean_weights[i]*100:>10.2f}% "
              f"{min_weights[i]*100:>6.1f}%-{max_weights[i]*100:>6.1f}% "
              f"{deviation_pct:>8.1f}%")

    # 计算总体偏差
    total_deviation = np.abs(atom_weights - mean_weights).sum()
    mse = ((atom_weights - mean_weights) ** 2).mean()

    print("\n总体偏差:")
    print(f"  L1偏差 (绝对值和): {total_deviation:.4f}")
    print(f"  均方误差 (MSE): {mse:.6f}")
    print(f"  均方根误差 (RMSE): {np.sqrt(mse):.6f}")

    # 判断
    if total_deviation < 0.05:
        print(f"\n✓ Atom权重与推荐权重非常接近 (偏差 < 5%)")
    elif total_deviation < 0.15:
        print(f"\n⚠️  Atom权重与推荐权重有一定偏差 (偏差 5%-15%)")
        print(f"   建议: 考虑调整Atom权重以更接近推荐平均值")
    else:
        print(f"\n⚠️  Atom权重与推荐权重偏差较大 (偏差 > 15%)")
        print(f"   建议: 重新评估Atom的权重配置")

    # 打印建议的权重配置
    print(f"\n建议的权重配置 (基于推荐平均):")
    print(f"weights = {{")
    for strategy, weight in zip(atom_config['strategies'], mean_weights):
        print(f"    '{strategy}': {weight:.4f},")
    print(f"}}")

    # 保存对比报告
    output_file = Path('backtest_results') / f'weight_comparison_{atom_name}.csv'
    comparison_df = pd.DataFrame({
        'strategy': atom_config['strategies'],
        'atom_weight': atom_weights,
        'recommended_mean': mean_weights,
        'recommended_std': std_weights,
        'recommended_min': min_weights,
        'recommended_max': max_weights,
        'deviation_pct': ((atom_weights - mean_weights) / mean_weights) * 100
    })
    comparison_df.to_csv(output_file, index=False)
    print(f"\n对比报告已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Atom权重配置对比工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--atom', required=True, help='Atom名称 (如: portfolio_rank3_combo)')
    parser.add_argument('--rolling-results', type=str, help='滚动验证结果文件 (robust_portfolios_ranking.csv)')
    parser.add_argument('--walk-forward-results', type=str, help='Walk-forward结果文件 (portfolio_robustness_*.csv)')

    args = parser.parse_args()

    # 加载Atom配置
    atom_config = load_atom_config(args.atom)
    if not atom_config:
        return

    # 解析结果
    rolling_results = None
    wf_results = None

    if args.rolling_results:
        print(f"\n加载滚动验证结果: {args.rolling_results}")
        rolling_results = parse_rolling_results(args.rolling_results, atom_config)
        print(f"  找到 {len(rolling_results)} 个匹配配置")

    if args.walk_forward_results:
        print(f"\n加载Walk-forward结果: {args.walk_forward_results}")
        wf_results = parse_walk_forward_results(args.walk_forward_results, atom_config)
        print(f"  找到 {len(wf_results)} 个匹配配置")

    if not rolling_results and not wf_results:
        print("\n错误: 至少需要提供 --rolling-results 或 --walk-forward-results")
        return

    # 对比权重
    compare_weights(args.atom, atom_config, rolling_results, wf_results)

    print("\n对比完成!")


if __name__ == '__main__':
    main()
