#!/usr/bin/env python3
"""
投资组合滚动窗口稳健性验证器（简化版 - 一致性分析）

功能：
- 在不同时间窗口（如每年）运行优化器
- 分析哪些组合能在多个时期保持高夏普
- 识别真正穿越周期的稳健组合

用法示例：
    python rolling_portfolio_validator.py --timeframe d1 --window-months 12 --step-months 12 --workers auto

作者：Claude Code
日期：2026-01-11
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import multiprocessing as mp

import pandas as pd
import numpy as np

# 导入优化器
from portfolio_optimizer import optimize_programmatically

warnings.filterwarnings('ignore')


class RollingPortfolioValidator:
    """投资组合滚动窗口稳健性验证器（一致性分析版本）"""

    def __init__(
        self,
        timeframe: str = 'd1',
        window_months: int = 12,
        step_months: int = 12,
        results_dir: str = 'backtest_results',
        output_dir: str = 'backtest_results/rolling_validation',
        n_workers: int = 1,
        penetration_threshold: float = 0.5,
        min_recommend_freq: float = 0.0,
        sorting_mode: str = 'robustness',
        min_sharpe: float = 0.5,
        min_return: float = 1.0,
        max_drawdown: float = -10.0,
        correlation_threshold: float = 0.3,
        min_strategies: int = 2,
        max_strategies: int = 4
    ):
        """初始化"""
        self.timeframe = timeframe
        self.window_months = window_months
        self.step_months = step_months
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.n_workers = n_workers
        self.penetration_threshold = penetration_threshold
        self.min_recommend_freq = min_recommend_freq
        self.sorting_mode = sorting_mode
        self._degraded_mode = False  # 降级模式标记

        # 优化器参数
        self.optimizer_params = {
            'min_sharpe': min_sharpe,
            'min_return': min_return,
            'max_drawdown': max_drawdown,
            'correlation_threshold': correlation_threshold,
            'min_strategies': min_strategies,
            'max_strategies': max_strategies
        }

        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 检测数据范围
        self.data_start, self.data_end = self._detect_data_range()

    def _detect_data_range(self) -> Tuple[str, str]:
        """检测可用数据的时间范围"""
        import glob

        pattern = f"{self.results_dir}/daily_values_*_{self.timeframe}_*.csv"
        files = glob.glob(pattern)

        if not files:
            print(f"警告: 未找到任何 daily_values 文件")
            return '20200101', '20241231'

        # 从文件名中提取日期
        dates = []
        for file in files:
            parts = Path(file).stem.split('_')
            if len(parts) >= 5:
                try:
                    start_date = parts[-2]
                    end_date = parts[-1]
                    dates.append((start_date, end_date))
                except:
                    continue

        if not dates:
            return '20200101', '20241231'

        all_starts = [d[0] for d in dates]
        all_ends = [d[1] for d in dates]

        return min(all_starts), max(all_ends)

    def generate_windows(self) -> List[Dict]:
        """生成所有滚动窗口"""
        windows = []
        window_id = 1

        start = datetime.strptime(self.data_start, '%Y%m%d')
        end = datetime.strptime(self.data_end, '%Y%m%d')

        current_start = start
        while True:
            window_end = current_start + timedelta(days=30 * self.window_months)

            if window_end > end:
                break

            windows.append({
                'window_id': window_id,
                'start_date': current_start.strftime('%Y%m%d'),
                'end_date': window_end.strftime('%Y%m%d')
            })

            window_id += 1
            current_start = current_start + timedelta(days=30 * self.step_months)

        return windows

    def run_window_optimization(self, window: Dict, top_n: int = 10) -> pd.DataFrame:
        """在指定窗口运行优化器"""
        start_date = window['start_date']
        end_date = window['end_date']

        try:
            portfolios_df = optimize_programmatically(
                start_date=start_date,
                end_date=end_date,
                timeframe=self.timeframe,
                **self.optimizer_params,
                results_dir=self.results_dir,
                quiet=True,
                data_start_date=self.data_start,  # 使用完整数据范围匹配文件名
                data_end_date=self.data_end       # 数据会被自动过滤到窗口范围
            )

            if portfolios_df is None or len(portfolios_df) == 0:
                return pd.DataFrame()

            portfolios_df = portfolios_df.head(top_n).copy()
            portfolios_df['window_id'] = window['window_id']
            portfolios_df['window_start'] = start_date
            portfolios_df['window_end'] = end_date

            return portfolios_df

        except Exception as e:
            print(f"  错误: {str(e)}")
            return pd.DataFrame()

    def calculate_robustness_metrics(self, all_windows_results: List[pd.DataFrame]) -> pd.DataFrame:
        """计算稳健性指标"""
        all_portfolios = pd.concat(all_windows_results, ignore_index=True)

        if len(all_portfolios) == 0:
            return pd.DataFrame()

        def make_combo_id(strategies_str):
            if pd.isna(strategies_str):
                return 'unknown'
            strategies = sorted(str(strategies_str).split(','))
            return ','.join(strategies)

        all_portfolios['combo_id'] = all_portfolios['strategies'].apply(make_combo_id)

        total_windows = all_portfolios['window_id'].nunique()

        robustness_list = []
        for combo_id, group in all_portfolios.groupby('combo_id'):
            # 使用正确的列名：expected_sharpe（来自portfolio_optimizer）
            sharpes = group['expected_sharpe'].values

            recommend_count = len(group)
            recommend_freq = recommend_count / total_windows
            avg_sharpe = np.mean(sharpes)
            sharpe_std = np.std(sharpes)
            worst_sharpe = np.min(sharpes)
            best_sharpe = np.max(sharpes)
            penetration_count = np.sum(sharpes > self.penetration_threshold)
            penetration_rate = penetration_count / recommend_count

            worst_penalty = abs(worst_sharpe) if worst_sharpe < 0 else 0
            robustness_score = (
                0.3 * avg_sharpe +
                0.25 * penetration_rate +
                0.25 * recommend_freq -
                0.15 * sharpe_std -
                0.05 * worst_penalty
            )

            strategies = group.iloc[0]['strategies']
            num_strategies = len(str(strategies).split(','))

            # 🆕 选择最佳权重配置（夏普最高的那个）
            best_config_idx = group['expected_sharpe'].idxmax()
            best_config = group.loc[best_config_idx]

            # 🆕 解析权重信息
            weights_str = best_config.get('weights', '')
            weight_method = best_config.get('weight_method', '')

            robustness_record = {
                'combo_id': combo_id,
                'num_strategies': num_strategies,
                'strategies': strategies,
                'weights': weights_str,
                'weight_method': weight_method,
                'recommend_count': recommend_count,
                'recommend_freq': recommend_freq,
                'avg_sharpe': avg_sharpe,
                'sharpe_std': sharpe_std,
                'worst_sharpe': worst_sharpe,
                'best_sharpe': best_sharpe,
                'penetration_rate': penetration_rate,
                'robustness_score': robustness_score,
                'config_expected_sharpe': best_config.get('expected_sharpe', 0),
                'config_expected_return': best_config.get('expected_return', 0),
                'config_expected_max_dd': best_config.get('expected_max_dd', 0),
                'config_window': f"{best_config.get('window_start', '')}-{best_config.get('window_end', '')}"
            }

            # 🆕 添加分列的策略和权重（便于generate_portfolio_atoms使用）
            if weights_str:
                strategies_list = [s.strip() for s in str(strategies).split(',')]
                weights_list = [float(w.strip()) for w in str(weights_str).split(',')]

                for i, (strat, weight) in enumerate(zip(strategies_list, weights_list), 1):
                    robustness_record[f'strategy_{i}'] = strat
                    robustness_record[f'weight_{i}'] = weight

            robustness_list.append(robustness_record)

        robustness_df = pd.DataFrame(robustness_list)

        # 根据排序模式选择不同逻辑
        if self.sorting_mode == 'threshold':
            # 阈值筛选法：先筛选推荐频率，再按平均夏普排序
            print(f"\n【筛选模式】阈值筛选法")
            print(f"  推荐频率阈值: {self.min_recommend_freq*100:.0f}%")

            # 第1步：按推荐频率筛选
            filtered_df = robustness_df[robustness_df['recommend_freq'] >= self.min_recommend_freq]
            print(f"  筛选前组合数: {len(robustness_df)}")
            print(f"  筛选后组合数: {len(filtered_df)}")

            if len(filtered_df) == 0:
                print(f"  ⚠️  警告: 没有组合满足推荐频率≥{self.min_recommend_freq*100:.0f}%")
                print(f"  降级策略: 按推荐频率降序排列，显示最接近阈值的组合")
                # 降级：按推荐频率降序，再按平均夏普排序
                filtered_df = robustness_df.sort_values(['recommend_freq', 'avg_sharpe'], ascending=[False, False])
                # 标记降级状态
                self._degraded_mode = True
            else:
                # 第2步：按平均夏普排序
                filtered_df = filtered_df.sort_values('avg_sharpe', ascending=False)
                self._degraded_mode = False

            filtered_df = filtered_df.reset_index(drop=True)
            filtered_df['rank'] = range(1, len(filtered_df) + 1)
            return filtered_df

        elif self.sorting_mode == 'robustness':
            # 综合评分法（原有逻辑）
            print(f"\n【筛选模式】综合评分法（robustness_score）")
            robustness_df = robustness_df.sort_values('robustness_score', ascending=False)
            robustness_df.reset_index(drop=True, inplace=True)
            robustness_df['rank'] = range(1, len(robustness_df) + 1)
            self._degraded_mode = False
            return robustness_df

        else:
            raise ValueError(f"未知的排序模式: {self.sorting_mode}")

    def run_rolling_validation(self, top_n_per_window: int = 10, use_parallel: bool = True) -> Dict:
        """运行完整的滚动验证流程"""
        print("=" * 80)
        print("投资组合滚动窗口稳健性验证（一致性分析）")
        print("=" * 80)
        print(f"配置:")
        print(f"  时间周期: {self.timeframe}")
        print(f"  窗口长度: {self.window_months}个月")
        print(f"  滑动步长: {self.step_months}个月")
        print(f"  数据范围: {self.data_start[:4]}-{self.data_start[4:6]} 至 {self.data_end[:4]}-{self.data_end[4:6]}")
        print("=" * 80)
        print("")

        print("生成窗口...")
        windows = self.generate_windows()
        print(f"  共生成 {len(windows)} 个滚动窗口")
        print("")

        if len(windows) == 0:
            print("错误: 无法生成窗口")
            return {'summary': pd.DataFrame(), 'robust_ranking': pd.DataFrame(), 'all_windows': []}

        start_time = time.time()

        if use_parallel and self.n_workers > 1:
            all_results = self._run_parallel(windows, top_n_per_window)
        else:
            all_results = self._run_serial(windows, top_n_per_window)

        elapsed = time.time() - start_time

        print("")
        print(f"执行完成！总耗时: {elapsed:.0f}秒")
        print("")

        robustness_df = self.calculate_robustness_metrics(all_results)

        summary_list = []
        for i, window in enumerate(windows):
            window_result = all_results[i]
            if len(window_result) > 0:
                summary_list.append({
                    'window_id': window['window_id'],
                    'start_date': window['start_date'],
                    'end_date': window['end_date'],
                    'num_portfolios': len(window_result),
                    'best_sharpe': window_result['expected_sharpe'].max(),
                    'avg_sharpe': window_result['expected_sharpe'].mean()
                })

        summary_df = pd.DataFrame(summary_list)

        return {
            'summary': summary_df,
            'robust_ranking': robustness_df,
            'all_windows': all_results
        }

    def _run_serial(self, windows: List[Dict], top_n_per_window: int) -> List[pd.DataFrame]:
        """串行执行"""
        all_results = []

        for i, window in enumerate(windows):
            print(f"窗口 {i+1}/{len(windows)}: {window['start_date'][:4]}-{window['start_date'][4:6]} ~ {window['end_date'][:4]}-{window['end_date'][4:6]}")
            print(f"  运行优化器...")

            result = self.run_window_optimization(window, top_n_per_window)

            if len(result) > 0:
                print(f"  生成 {len(result)} 个推荐组合")
                best = result.iloc[0]
                print(f"  最佳组合: {best.get('strategies', 'N/A')} (夏普{best.get('expected_sharpe', 0):.2f})")
            else:
                print(f"  警告: 未生成任何组合")

            all_results.append(result)
            print("")

        return all_results

    def _run_parallel(self, windows: List[Dict], top_n_per_window: int) -> List[pd.DataFrame]:
        """并行执行"""
        print(f"启动并行执行（{self.n_workers}核）...")
        print("")

        args_list = [(self, window, top_n_per_window, i+1, len(windows)) for i, window in enumerate(windows)]

        with mp.Pool(processes=self.n_workers) as pool:
            all_results = pool.starmap(_run_window_worker, args_list)

        return all_results

    def generate_report(self, results: Dict):
        """生成详细报告"""
        summary_df = results['summary']
        robustness_df = results['robust_ranking']

        print("=" * 80)
        print("稳健性分析结果")
        print("=" * 80)
        print("")

        if len(robustness_df) == 0:
            print("未找到任何有效组合")
            return

        if self.sorting_mode == 'threshold':
            if hasattr(self, '_degraded_mode') and self._degraded_mode:
                print(f"【Top 5 稳健组合】（降级模式: 按推荐频率降序，无组合满足≥{self.min_recommend_freq*100:.0f}%）")
            else:
                print(f"【Top 5 稳健组合】（阈值筛选法: 推荐频率≥{self.min_recommend_freq*100:.0f}%, 按平均夏普排序）")
        else:
            print("【Top 5 稳健组合】（综合评分法: 按robustness_score排序）")
        print("-" * 120)
        print(f"{'排名':<6} {'策略组成':<35} {'推荐频率':<10} {'平均夏普':<10} "
              f"{'夏普标准差':<12} {'最差夏普':<10} {'最佳夏普':<10} {'穿越率':<10} {'稳健评分':<10}")
        print("-" * 120)

        top_5 = robustness_df.head(5)
        for _, row in top_5.iterrows():
            print(f"{row['rank']:<6} {row['strategies'][:33]:<35} "
                  f"{row['recommend_freq']:>8.0%}  {row['avg_sharpe']:>10.2f} "
                  f"{row['sharpe_std']:>12.3f} {row['worst_sharpe']:>10.2f} "
                  f"{row['best_sharpe']:>10.2f} {row['penetration_rate']:>8.0%}  {row['robustness_score']:>10.3f}")

        print("")

        total_windows = len(summary_df)
        print(f"【穿越能力分析】（阈值: 夏普>{self.penetration_threshold}）")

        all_windows_combos = robustness_df[robustness_df['recommend_count'] == total_windows]
        high_freq_combos = robustness_df[robustness_df['recommend_count'] >= total_windows * 0.8]
        mid_freq_combos = robustness_df[robustness_df['recommend_count'] >= total_windows * 0.6]

        print(f"- 在所有{total_windows}个窗口都被推荐的组合: {len(all_windows_combos)}个 ⭐⭐⭐")
        print(f"- 在≥80%窗口被推荐的组合: {len(high_freq_combos)}个 ⭐⭐")
        print(f"- 在≥60%窗口被推荐的组合: {len(mid_freq_combos)}个 ⭐")

        full_penetration = robustness_df[robustness_df['penetration_rate'] == 1.0]
        print(f"- 所有出现窗口都保持夏普>{self.penetration_threshold}的组合: {len(full_penetration)}个 ✓")
        print("")

        if len(robustness_df) > 0:
            best = robustness_df.iloc[0]
            print(f"【最稳健组合详细分析】")
            print(f"组合: {best['strategies']}")
            print(f"  推荐频率: {best['recommend_freq']:.0%} ({best['recommend_count']}/{total_windows}个窗口)")
            print(f"  夏普稳定性: 标准差{best['sharpe_std']:.3f}")
            print(f"  平均夏普: {best['avg_sharpe']:.2f}")
            print(f"  最差夏普: {best['worst_sharpe']:.2f}")
            print(f"  最佳夏普: {best['best_sharpe']:.2f}")
            print(f"  穿越率: {best['penetration_rate']:.0%}")
            print("")

            if best['recommend_count'] == total_windows and best['penetration_rate'] == 1.0:
                print(f"  → 这个组合真正穿越了所有{total_windows}个时间窗口！ ⭐⭐⭐")
            print("")

        summary_path = f"{self.output_dir}/rolling_window_summary.csv"
        robustness_path = f"{self.output_dir}/robust_portfolios_ranking.csv"

        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        robustness_df.to_csv(robustness_path, index=False, encoding='utf-8-sig')

        all_windows_df = pd.concat(results['all_windows'], ignore_index=True)
        if len(all_windows_df) > 0:
            details_path = f"{self.output_dir}/window_details.csv"
            all_windows_df.to_csv(details_path, index=False, encoding='utf-8-sig')
            print(f"结果已保存:")
            print(f"  - {summary_path}")
            print(f"  - {robustness_path} (包含权重信息，可直接用于generate_portfolio_atoms)")
            print(f"  - {details_path}")
        else:
            print(f"结果已保存:")
            print(f"  - {summary_path}")
            print(f"  - {robustness_path} (包含权重信息，可直接用于generate_portfolio_atoms)")
        print("")


def _run_window_worker(validator, window, top_n, window_index, total_windows):
    """Worker function for parallel execution"""
    print(f"  [{window_index}/{total_windows}] 窗口 {window['window_id']}: "
          f"{window['start_date'][:4]}-{window['start_date'][4:6]} ~ "
          f"{window['end_date'][:4]}-{window['end_date'][4:6]}")
    return validator.run_window_optimization(window, top_n)


def main():
    parser = argparse.ArgumentParser(description='投资组合滚动窗口稳健性验证器（简化版）')

    parser.add_argument('--timeframe', type=str, default='d1', help='时间周期')
    parser.add_argument('--window-months', type=int, default=12, help='窗口长度（月）')
    parser.add_argument('--step-months', type=int, default=12, help='滑动步长（月）')
    parser.add_argument('--top-n', type=int, default=10, help='每个窗口保留Top N个组合')
    parser.add_argument('--workers', type=str, default='auto', help='并行进程数')

    parser.add_argument('--min-sharpe', type=float, default=0.5, help='夏普比率阈值')
    parser.add_argument('--min-return', type=float, default=1.0, help='收益率阈值')
    parser.add_argument('--max-drawdown', type=float, default=-10.0, help='最大回撤阈值')
    parser.add_argument('--correlation-threshold', type=float, default=0.3, help='相关性阈值')
    parser.add_argument('--min-strategies', type=int, default=2, help='组合最少策略数')
    parser.add_argument('--max-strategies', type=int, default=4, help='组合最多策略数')

    parser.add_argument('--results-dir', type=str, default='backtest_results', help='结果目录')
    parser.add_argument('--output-dir', type=str, default='backtest_results/rolling_validation', help='输出目录')
    parser.add_argument('--penetration-threshold', type=float, default=0.5, help='穿越率阈值')
    parser.add_argument('--min-recommend-freq', type=float, default=0.0,
                        help='推荐频率阈值（0-1），仅在threshold模式下生效，例如0.8表示≥80%%')
    parser.add_argument('--sorting-mode', type=str, default='robustness',
                        choices=['robustness', 'threshold'],
                        help='排序模式: robustness=综合评分法（默认）, threshold=阈值筛选法')

    args = parser.parse_args()

    if args.workers == 'auto':
        n_workers = mp.cpu_count()
    else:
        try:
            n_workers = int(args.workers)
        except:
            n_workers = 1

    validator = RollingPortfolioValidator(
        timeframe=args.timeframe,
        window_months=args.window_months,
        step_months=args.step_months,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        n_workers=n_workers,
        penetration_threshold=args.penetration_threshold,
        min_recommend_freq=args.min_recommend_freq,
        sorting_mode=args.sorting_mode,
        min_sharpe=args.min_sharpe,
        min_return=args.min_return,
        max_drawdown=args.max_drawdown,
        correlation_threshold=args.correlation_threshold,
        min_strategies=args.min_strategies,
        max_strategies=args.max_strategies
    )

    results = validator.run_rolling_validation(
        top_n_per_window=args.top_n,
        use_parallel=(n_workers > 1)
    )

    validator.generate_report(results)


if __name__ == '__main__':
    main()
