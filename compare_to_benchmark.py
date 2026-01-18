"""
基准对比分析工具
将策略回测结果与买入并持有基准进行对比
"""
import argparse
import glob
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_strategy_data(strategy_name: str, timeframe: str, start_date: str, end_date: str, results_dir: str = 'backtest_results') -> pd.DataFrame:
    """
    加载策略的每日价值数据

    Args:
        strategy_name: 策略名称
        timeframe: 时间周期 (d1, h1, etc.)
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        results_dir: 结果目录

    Returns:
        DataFrame with columns: datetime, portfolio_value, daily_return, cumulative_return
    """
    filename = f'{results_dir}/daily_values_{strategy_name}_{timeframe}_{start_date}_{end_date}.csv'

    if not os.path.exists(filename):
        raise FileNotFoundError(f'找不到策略数据文件: {filename}')

    df = pd.read_csv(filename)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def calculate_metrics(df: pd.DataFrame, initial_value: float = 100000) -> Dict[str, float]:
    """
    从daily_values计算核心指标

    Args:
        df: daily_values DataFrame
        initial_value: 初始资金

    Returns:
        字典包含: return_pct, annualized_return, sharpe, max_dd, volatility, calmar
    """
    if len(df) < 2:
        return {
            'return_pct': 0.0,
            'annualized_return': 0.0,
            'sharpe': 0.0,
            'max_dd': 0.0,
            'volatility': 0.0,
            'calmar': 0.0,
        }

    # 收益率
    final_value = df['portfolio_value'].iloc[-1]
    return_pct = (final_value / initial_value - 1) * 100

    # 年化收益
    trading_days = len(df)
    annualized_return = return_pct * (252 / trading_days) if trading_days > 0 else 0.0

    # 年化波动率
    returns = df['daily_return'].values[1:]  # 排除第一天
    volatility = np.std(returns, ddof=1) * np.sqrt(252) * 100 if len(returns) > 1 else 0.0

    # 夏普比率
    if len(returns) > 1 and np.std(returns, ddof=1) > 0:
        sharpe = (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # 最大回撤
    values = df['portfolio_value'].values
    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max
    max_dd = drawdown.min() * 100

    # 卡尔玛比率
    calmar = annualized_return / abs(max_dd) if max_dd < 0 else 0.0

    return {
        'return_pct': return_pct,
        'annualized_return': annualized_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'volatility': volatility,
        'calmar': calmar,
    }


def calculate_tracking_error(strategy_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> float:
    """
    计算跟踪误差（策略收益与基准收益的标准差）

    Args:
        strategy_df: 策略的daily_values
        benchmark_df: 基准的daily_values

    Returns:
        年化跟踪误差（百分比）
    """
    # 确保两个DataFrame有相同的日期
    strategy_returns = strategy_df['daily_return'].values[1:]
    benchmark_returns = benchmark_df['daily_return'].values[1:]

    # 计算超额收益
    min_len = min(len(strategy_returns), len(benchmark_returns))
    if min_len < 2:
        return 0.0

    excess_returns = strategy_returns[:min_len] - benchmark_returns[:min_len]

    # 计算跟踪误差并年化
    tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(252) * 100

    return tracking_error


def compare_metrics(strategy_metrics: Dict[str, float], benchmark_metrics: Dict[str, float], strategy_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> Dict[str, float]:
    """
    计算超额指标

    Args:
        strategy_metrics: 策略指标
        benchmark_metrics: 基准指标
        strategy_df: 策略daily_values
        benchmark_df: 基准daily_values

    Returns:
        字典包含: excess_return, information_ratio
    """
    excess_return = strategy_metrics['return_pct'] - benchmark_metrics['return_pct']

    # 信息比率 = (策略收益 - 基准收益) / 跟踪误差
    tracking_error = calculate_tracking_error(strategy_df, benchmark_df)
    information_ratio = excess_return / tracking_error if tracking_error > 0 else 0.0

    return {
        'excess_return': excess_return,
        'excess_ann_return': strategy_metrics['annualized_return'] - benchmark_metrics['annualized_return'],
        'excess_sharpe': strategy_metrics['sharpe'] - benchmark_metrics['sharpe'],
        'excess_calmar': strategy_metrics['calmar'] - benchmark_metrics['calmar'],
        'dd_diff': strategy_metrics['max_dd'] - benchmark_metrics['max_dd'],
        'vol_diff': strategy_metrics['volatility'] - benchmark_metrics['volatility'],
        'information_ratio': information_ratio,
        'tracking_error': tracking_error,
    }


def discover_all_strategies(timeframe: str, start_date: str, end_date: str, results_dir: str = 'backtest_results') -> List[str]:
    """
    扫描backtest_results目录，找到所有已回测的策略

    Args:
        timeframe: 时间周期
        start_date: 开始日期
        end_date: 结束日期
        results_dir: 结果目录

    Returns:
        策略名称列表
    """
    pattern = f'{results_dir}/daily_values_*_{timeframe}_{start_date}_{end_date}.csv'
    files = glob.glob(pattern)

    strategies = []
    for file in files:
        # 提取策略名: daily_values_<name>_<tf>_<start>_<end>.csv
        basename = Path(file).stem  # 去掉.csv
        parts = basename.split('_')

        # 找到timeframe的位置
        try:
            tf_index = parts.index(timeframe)
            # 策略名是从daily_values后到timeframe前的所有部分
            name_parts = parts[2:tf_index]
            name = '_'.join(name_parts)

            if name != 'buy-and-hold' and name not in strategies:
                strategies.append(name)
        except (ValueError, IndexError):
            continue

    return sorted(strategies)


def print_comparison_report(results: List[Dict[str, Any]], benchmark_metrics: Dict[str, float], timeframe: str, start_date: str, end_date: str):
    """
    打印对比报告到终端

    Args:
        results: 对比结果列表
        benchmark_metrics: 基准指标
        timeframe: 时间周期
        start_date: 开始日期
        end_date: 结束日期
    """
    print('\n' + '=' * 100)
    print('基准对比报告')
    print('=' * 100)
    print(f'基准策略: buy_and_hold')
    print(f'对比周期: {timeframe} ({start_date} ~ {end_date})')
    print()

    # 策略对比表
    print('策略对比:')
    print(f'{"策略名称":<25} {"收益率":>10} {"夏普":>8} {"回撤":>8} {"超额收益":>10} {"信息比率":>10}')
    print('-' * 100)

    for r in results:
        print(f'{r["strategy"]:<25} {r["return_pct"]:>9.2f}% {r["sharpe"]:>8.2f} {r["max_dd"]:>7.2f}% {r["excess_return"]:>9.2f}% {r["information_ratio"]:>10.2f}')

    print('-' * 100)
    print(f'{"buy_and_hold (基准)":<25} {benchmark_metrics["return_pct"]:>9.2f}% {benchmark_metrics["sharpe"]:>8.2f} {benchmark_metrics["max_dd"]:>7.2f}% {"0.00":>9}% {"1.00":>10}')

    # 关键洞察
    print('\n关键洞察:')

    # 统计跑赢基准的策略
    sharpe_better = [r for r in results if r['sharpe'] > benchmark_metrics['sharpe']]
    return_better = [r for r in results if r['return_pct'] > benchmark_metrics['return_pct']]

    if sharpe_better:
        print(f'  ✓ {len(sharpe_better)}/{len(results)} 个策略跑赢基准的风险调整收益（夏普比率）')
        best_sharpe = max(sharpe_better, key=lambda x: x['sharpe'])
        print(f'  ✓ {best_sharpe["strategy"]} 表现最佳：夏普{best_sharpe["sharpe"]:.2f} vs 基准{benchmark_metrics["sharpe"]:.2f}')
    else:
        print(f'  ⚠ 所有策略的夏普比率均低于基准')

    if return_better:
        print(f'  ✓ {len(return_better)}/{len(results)} 个策略绝对收益跑赢基准')
    else:
        print(f'  ⚠ 所有策略绝对收益均低于基准')
        if results:
            min_excess = min(results, key=lambda x: x['excess_return'])['excess_return']
            max_excess = max(results, key=lambda x: x['excess_return'])['excess_return']
            print(f'  ⚠ 超额收益范围: {min_excess:.1f}% ~ {max_excess:.1f}%')


def save_comparison_csv(results: List[Dict[str, Any]], benchmark_metrics: Dict[str, float], output_file: str):
    """
    保存对比结果到CSV

    Args:
        results: 对比结果列表
        benchmark_metrics: 基准指标
        output_file: 输出文件路径
    """
    # 合并策略结果和基准
    all_results = results + [{
        'strategy': 'buy_and_hold',
        **benchmark_metrics,
        'excess_return': 0.0,
        'excess_ann_return': 0.0,
        'excess_sharpe': 0.0,
        'excess_calmar': 0.0,
        'dd_diff': 0.0,
        'vol_diff': 0.0,
        'information_ratio': 1.0,
        'tracking_error': 0.0,
    }]

    df = pd.DataFrame(all_results)

    # 选择列顺序
    columns = [
        'strategy', 'return_pct', 'annualized_return', 'sharpe', 'max_dd', 'volatility', 'calmar',
        'excess_return', 'excess_ann_return', 'excess_sharpe', 'excess_calmar',
        'dd_diff', 'vol_diff', 'information_ratio', 'tracking_error'
    ]

    df = df[columns]
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f'\n对比结果已保存: {output_file}')


def main():
    parser = argparse.ArgumentParser(description='基准对比分析工具')
    parser.add_argument('--strategy', type=str, help='单个策略名称')
    parser.add_argument('--strategies', nargs='+', help='多个策略名称（空格分隔）')
    parser.add_argument('--all', action='store_true', help='对比所有已回测的策略')
    parser.add_argument('--timeframe', default='d1', help='时间周期 (默认: d1)')
    parser.add_argument('--start', required=True, help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end', required=True, help='结束日期 (YYYYMMDD)')
    parser.add_argument('--output', type=str, help='输出CSV文件路径（可选）')
    parser.add_argument('--results-dir', default='backtest_results', help='回测结果目录 (默认: backtest_results)')

    args = parser.parse_args()

    # 确定要对比的策略列表
    if args.all:
        print('正在扫描已回测的策略...')
        strategies = discover_all_strategies(args.timeframe, args.start, args.end, args.results_dir)
        if not strategies:
            print(f'错误: 未找到任何策略数据文件 (timeframe={args.timeframe}, start={args.start}, end={args.end})')
            return
        print(f'找到 {len(strategies)} 个策略')
    elif args.strategies:
        strategies = args.strategies
    elif args.strategy:
        strategies = [args.strategy]
    else:
        parser.error('必须指定 --strategy, --strategies, 或 --all')

    # 加载基准数据
    try:
        print(f'\n加载基准数据: buy_and_hold')
        benchmark_df = load_strategy_data('buy_and_hold', args.timeframe, args.start, args.end, args.results_dir)
        benchmark_metrics = calculate_metrics(benchmark_df)
    except FileNotFoundError as e:
        print(f'错误: {e}')
        print('\n请先运行基准策略:')
        print(f'  python bt_main.py --atom buy_and_hold --start {args.start[:4]}-{args.start[4:6]}-{args.start[6:]} --end {args.end[:4]}-{args.end[4:6]}-{args.end[6:]} --timeframe {args.timeframe}')
        return

    # 逐个对比策略
    results = []
    failed_strategies = []

    for strategy in strategies:
        try:
            print(f'加载策略数据: {strategy}')
            strategy_df = load_strategy_data(strategy, args.timeframe, args.start, args.end, args.results_dir)
            strategy_metrics = calculate_metrics(strategy_df)
            comparison = compare_metrics(strategy_metrics, benchmark_metrics, strategy_df, benchmark_df)

            results.append({
                'strategy': strategy,
                **strategy_metrics,
                **comparison
            })
        except FileNotFoundError:
            failed_strategies.append(strategy)
            print(f'  ⚠ 跳过（找不到数据文件）')

    if not results:
        print('\n错误: 没有成功加载任何策略数据')
        return

    # 生成报告
    print_comparison_report(results, benchmark_metrics, args.timeframe, args.start, args.end)

    # 保存CSV（可选）
    if args.output:
        save_comparison_csv(results, benchmark_metrics, args.output)

    # 如果有失败的策略，提示
    if failed_strategies:
        print(f'\n⚠ 警告: 以下策略未找到数据文件:')
        for s in failed_strategies:
            print(f'  - {s}')


if __name__ == '__main__':
    main()
