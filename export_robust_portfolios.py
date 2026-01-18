#!/usr/bin/env python3
"""
导出稳健组合的详细配置信息

从滚动验证结果中提取稳健组合的策略、权重、表现指标，
生成一个便于使用的CSV文件，用于后续生成组合策略代码。

用法:
    python export_robust_portfolios.py --timeframe d1 --top-n 20
    python export_robust_portfolios.py --timeframe d1 --top-n 10 --weight-method max_sharpe
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_robust_portfolios(timeframe: str, validation_dir: str = 'backtest_results/rolling_validation'):
    """加载稳健组合排名"""
    ranking_file = Path(validation_dir) / 'robust_portfolios_ranking.csv'

    if not ranking_file.exists():
        raise FileNotFoundError(f"未找到稳健组合排名文件: {ranking_file}")

    df = pd.read_csv(ranking_file)
    print(f"加载了 {len(df)} 个稳健组合")
    return df


def load_window_details(timeframe: str, validation_dir: str = 'backtest_results/rolling_validation'):
    """加载所有窗口的组合详情"""
    details_file = Path(validation_dir) / 'window_details.csv'

    if not details_file.exists():
        raise FileNotFoundError(f"未找到窗口详情文件: {details_file}")

    df = pd.read_csv(details_file)
    print(f"加载了 {len(df)} 条窗口组合记录")
    return df


def normalize_strategy_string(strategies_str):
    """标准化策略字符串（排序）"""
    strategies = sorted(str(strategies_str).split(','))
    return ','.join(strategies)


def select_best_config(group, weight_method_preference=None):
    """
    从多个窗口的配置中选择"最佳"代表配置

    策略:
    1. 如果指定了weight_method，优先选择该方法
    2. 否则选择夏普最高的配置
    3. 如果有多个相同夏普，选择第一个
    """
    if weight_method_preference:
        filtered = group[group['weight_method'] == weight_method_preference]
        if len(filtered) > 0:
            group = filtered

    # 选择夏普最高的
    best_idx = group['expected_sharpe'].idxmax()
    return group.loc[best_idx]


def parse_weights(weights_str, strategies_str):
    """
    解析权重字符串，返回{策略: 权重}字典

    Args:
        weights_str: "0.3587,0.2807,0.1907,0.1699"
        strategies_str: "bollinger_mean_reversion,rsi_reversal,triple_ma_8_21_55,keltner_10_10_1_5"

    Returns:
        {'bollinger_mean_reversion': 0.3587, 'rsi_reversal': 0.2807, ...}
    """
    strategies = [s.strip() for s in strategies_str.split(',')]
    weights = [float(w.strip()) for w in weights_str.split(',')]

    if len(strategies) != len(weights):
        raise ValueError(f"策略数({len(strategies)})与权重数({len(weights)})不匹配")

    return dict(zip(strategies, weights))


def export_robust_portfolios(
    timeframe: str,
    top_n: int = 20,
    weight_method: str = None,
    validation_dir: str = 'backtest_results/rolling_validation',
    output_file: str = None
):
    """
    导出稳健组合的详细配置

    Args:
        timeframe: 时间周期
        top_n: 导出前N个稳健组合
        weight_method: 优先选择的权重方法 (max_sharpe, risk_parity, sharpe_weighted等)
        validation_dir: 验证结果目录
        output_file: 输出文件路径，如不指定则自动生成
    """
    # 加载数据
    robust_df = load_robust_portfolios(timeframe, validation_dir)
    window_df = load_window_details(timeframe, validation_dir)

    # 标准化策略字符串
    robust_df['combo_id_normalized'] = robust_df['strategies'].apply(normalize_strategy_string)
    window_df['combo_id_normalized'] = window_df['strategies'].apply(normalize_strategy_string)

    # 只取前N个
    robust_df = robust_df.head(top_n)

    print(f"\n处理前 {len(robust_df)} 个稳健组合...")

    # 为每个稳健组合找到最佳配置
    export_list = []

    for idx, row in robust_df.iterrows():
        combo_id = row['combo_id_normalized']

        # 找到该组合在所有窗口中的出现
        matches = window_df[window_df['combo_id_normalized'] == combo_id]

        if len(matches) == 0:
            print(f"  警告: 组合 {row['rank']} 在window_details中未找到匹配")
            continue

        # 选择最佳配置
        best_config = select_best_config(matches, weight_method)

        # 解析权重
        weight_dict = parse_weights(best_config['weights'], best_config['strategies'])

        # 构建导出记录
        export_record = {
            'rank': row['rank'],
            'combo_id': combo_id,
            'num_strategies': row['num_strategies'],
            'strategies': best_config['strategies'],  # 保留原始顺序
            'weights': best_config['weights'],
            'weight_method': best_config['weight_method'],

            # 稳健性指标（来自robust_portfolios_ranking）
            'recommend_count': row['recommend_count'],
            'recommend_freq': row['recommend_freq'],
            'avg_sharpe': row['avg_sharpe'],
            'sharpe_std': row['sharpe_std'],
            'worst_sharpe': row['worst_sharpe'],
            'best_sharpe': row['best_sharpe'],
            'penetration_rate': row['penetration_rate'],
            'robustness_score': row['robustness_score'],

            # 最佳配置的表现（来自window_details）
            'config_expected_sharpe': best_config['expected_sharpe'],
            'config_expected_return': best_config['expected_return'],
            'config_expected_max_dd': best_config['expected_max_dd'],
            'config_window': f"{best_config['window_start']}-{best_config['window_end']}"
        }

        # 添加各策略的权重（分列）
        for i, (strategy, weight) in enumerate(weight_dict.items(), 1):
            export_record[f'strategy_{i}'] = strategy
            export_record[f'weight_{i}'] = weight

        export_list.append(export_record)

    # 创建DataFrame
    export_df = pd.DataFrame(export_list)

    # 生成输出文件名
    if output_file is None:
        method_suffix = f"_{weight_method}" if weight_method else ""
        output_file = Path(validation_dir) / f'robust_portfolios_export_{timeframe}{method_suffix}_top{top_n}.csv'

    # 保存
    export_df.to_csv(output_file, index=False)
    print(f"\n✓ 导出完成: {output_file}")
    print(f"  共 {len(export_df)} 个组合")

    # 打印预览
    print(f"\n前3个组合预览:")
    print("=" * 100)
    for idx, row in export_df.head(3).iterrows():
        print(f"\n排名 {row['rank']}: {row['strategies']}")
        print(f"  权重: {row['weights']} ({row['weight_method']})")
        print(f"  稳健性评分: {row['robustness_score']:.3f}")
        print(f"  平均夏普: {row['avg_sharpe']:.3f} (标准差: {row['sharpe_std']:.3f})")
        print(f"  推荐频率: {row['recommend_freq']:.1%} ({row['recommend_count']}次)")
        print(f"  最佳配置夏普: {row['config_expected_sharpe']:.3f} (窗口: {row['config_window']})")

    return export_df


def main():
    parser = argparse.ArgumentParser(description='导出稳健组合的详细配置')
    parser.add_argument('--timeframe', type=str, default='d1',
                        help='时间周期 (d1, h1, m5)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='导出前N个稳健组合 (默认20)')
    parser.add_argument('--weight-method', type=str, default=None,
                        choices=['max_sharpe', 'risk_parity', 'sharpe_weighted', 'return_weighted', 'equal'],
                        help='优先选择的权重方法')
    parser.add_argument('--validation-dir', type=str, default='backtest_results/rolling_validation',
                        help='验证结果目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径')

    args = parser.parse_args()

    print("=" * 100)
    print("稳健组合配置导出工具")
    print("=" * 100)
    print(f"时间周期: {args.timeframe}")
    print(f"导出数量: Top {args.top_n}")
    if args.weight_method:
        print(f"权重方法: {args.weight_method}")
    print("=" * 100)
    print()

    try:
        export_robust_portfolios(
            timeframe=args.timeframe,
            top_n=args.top_n,
            weight_method=args.weight_method,
            validation_dir=args.validation_dir,
            output_file=args.output
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
