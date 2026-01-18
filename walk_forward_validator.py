#!/usr/bin/env python
"""
Walk-Forward 验证器 - 防止过拟合的关键工具

通过时间滚动窗口进行严格的训练期/测试期分离:
1. 在训练期优化参数和组合
2. 在测试期验证样本外表现
3. 对比训练期vs测试期指标,计算过拟合指数
4. 识别真正稳健的策略组合

用法:
    # 基本用法: 12月训练 + 6月测试, 每6月滚动一次
    python walk_forward_validator.py \
      --timeframe d1 \
      --train-months 12 \
      --test-months 6 \
      --step-months 6

    # 严格测试: 12月训练 + 12月测试, 不重叠
    python walk_forward_validator.py \
      --timeframe d1 \
      --train-months 12 \
      --test-months 12 \
      --step-months 12

    # 快速测试: 指定总体数据范围
    python walk_forward_validator.py \
      --data-start 20200101 \
      --data-end 20241231 \
      --train-months 12 \
      --test-months 6 \
      --step-months 6 \
      --top-n 5
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import datetime
from dateutil.relativedelta import relativedelta
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# 导入自定义模块
from portfolio_optimizer import optimize_programmatically
from portfolio_backtest_signal_weighted import SignalWeightedPortfolioBacktester


class WalkForwardValidator:
    """Walk-Forward验证器"""

    def __init__(
        self,
        data_start_date: str,
        data_end_date: str,
        timeframe: str = 'd1',
        train_months: int = 12,
        test_months: int = 6,
        step_months: int = 6,
        top_n: int = 10,
        correlation_threshold: float = 0.3,
        results_dir: str = 'backtest_results',
        use_signal_weighted: bool = True
    ):
        """
        初始化Walk-Forward验证器

        Args:
            data_start_date: 完整数据开始日期 (YYYYMMDD)
            data_end_date: 完整数据结束日期 (YYYYMMDD)
            timeframe: 时间周期
            train_months: 训练窗口大小(月)
            test_months: 测试窗口大小(月)
            step_months: 滚动步长(月)
            top_n: 每个窗口选择前N个组合
            correlation_threshold: 相关性阈值
            results_dir: 结果目录
            use_signal_weighted: 是否使用单账户信号加权回测(推荐True)
        """
        self.data_start_date = pd.to_datetime(data_start_date, format='%Y%m%d')
        self.data_end_date = pd.to_datetime(data_end_date, format='%Y%m%d')
        self.timeframe = timeframe
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.top_n = top_n
        self.correlation_threshold = correlation_threshold
        self.results_dir = results_dir
        self.use_signal_weighted = use_signal_weighted

        # 数据容器
        self.windows = []  # [(train_start, train_end, test_start, test_end), ...]
        self.train_results = []  # 训练期优化结果
        self.test_results = []  # 测试期验证结果

    def generate_windows(self) -> List[Tuple]:
        """
        生成训练/测试窗口

        Returns:
            [(train_start, train_end, test_start, test_end), ...]
        """
        print("\n" + "=" * 80)
        print("生成Walk-Forward窗口")
        print("=" * 80)
        print(f"数据范围: {self.data_start_date.strftime('%Y-%m-%d')} ~ {self.data_end_date.strftime('%Y-%m-%d')}")
        print(f"训练窗口: {self.train_months} 个月")
        print(f"测试窗口: {self.test_months} 个月")
        print(f"滚动步长: {self.step_months} 个月")
        print("-" * 80)

        windows = []
        current_start = self.data_start_date

        while True:
            # 训练期
            train_start = current_start
            train_end = train_start + relativedelta(months=self.train_months) - relativedelta(days=1)

            # 测试期
            test_start = train_end + relativedelta(days=1)
            test_end = test_start + relativedelta(months=self.test_months) - relativedelta(days=1)

            # 检查是否超出数据范围
            if test_end > self.data_end_date:
                break

            windows.append((train_start, train_end, test_start, test_end))

            print(f"窗口 {len(windows)}:")
            print(f"  训练期: {train_start.strftime('%Y-%m-%d')} ~ {train_end.strftime('%Y-%m-%d')}")
            print(f"  测试期: {test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')}")

            # 移动到下一个窗口
            current_start = current_start + relativedelta(months=self.step_months)

        print(f"\n共生成 {len(windows)} 个窗口")
        self.windows = windows
        return windows

    def run_train_optimization(self, window_id: int, train_start: pd.Timestamp, train_end: pd.Timestamp) -> pd.DataFrame:
        """
        在训练期运行优化器

        Args:
            window_id: 窗口编号
            train_start: 训练开始日期
            train_end: 训练结束日期

        Returns:
            优化结果DataFrame
        """
        train_start_str = train_start.strftime('%Y%m%d')
        train_end_str = train_end.strftime('%Y%m%d')

        # 数据文件的完整日期范围(用于匹配文件名)
        data_start_str = self.data_start_date.strftime('%Y%m%d')
        data_end_str = self.data_end_date.strftime('%Y%m%d')

        print(f"\n{'='*60}")
        print(f"窗口 {window_id} - 训练期优化")
        print(f"{'='*60}")
        print(f"训练范围: {train_start_str} ~ {train_end_str}")
        print(f"数据文件: {data_start_str} ~ {data_end_str}")

        try:
            # 调用优化器(传入训练期窗口和完整数据文件范围)
            portfolios_df = optimize_programmatically(
                start_date=train_start_str,
                end_date=train_end_str,
                timeframe=self.timeframe,
                min_sharpe=0.5,
                min_return=1.0,
                max_drawdown=-10.0,
                top_n_quality=20,
                correlation_threshold=self.correlation_threshold,
                min_strategies=2,
                max_strategies=4,
                weight_methods=['sharpe_weighted', 'risk_parity', 'max_sharpe'],
                max_portfolios=100,
                results_dir=self.results_dir,
                quiet=False,
                data_start_date=data_start_str,
                data_end_date=data_end_str
            )

            if portfolios_df.empty:
                print(f"  警告: 窗口 {window_id} 优化未找到组合")
                return pd.DataFrame()

            # 选择前N个
            top_portfolios = portfolios_df.head(self.top_n).copy()
            top_portfolios['window_id'] = window_id
            top_portfolios['train_start'] = train_start_str
            top_portfolios['train_end'] = train_end_str

            print(f"  ✓ 选择前 {len(top_portfolios)} 个组合")
            print(f"    最佳夏普: {top_portfolios.iloc[0]['expected_sharpe']:.2f}")

            return top_portfolios

        except Exception as e:
            print(f"  ✗ 窗口 {window_id} 训练期优化失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def run_test_validation(
        self,
        window_id: int,
        portfolio_config: Dict,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp
    ) -> Dict:
        """
        在测试期验证组合

        Args:
            window_id: 窗口编号
            portfolio_config: 组合配置(从训练期获得)
            test_start: 测试开始日期
            test_end: 测试结束日期

        Returns:
            测试结果字典
        """
        test_start_str = test_start.strftime('%Y%m%d')
        test_end_str = test_end.strftime('%Y%m%d')

        # 数据文件的完整日期范围
        data_start_str = self.data_start_date.strftime('%Y%m%d')
        data_end_str = self.data_end_date.strftime('%Y%m%d')

        # 解析配置
        strategies = portfolio_config['strategies'].split(',')
        weights = [float(w) for w in portfolio_config['weights'].split(',')]

        print(f"\n  测试组合: {portfolio_config['strategies']}")
        print(f"  权重: {portfolio_config['weights']}")
        print(f"  测试范围: {test_start_str} ~ {test_end_str}")

        if self.use_signal_weighted:
            # 使用单账户信号加权回测
            try:
                # 阈值配置(与portfolio_rank3_combo一致)
                threshold_config = {'high': 0.70, 'mid': 0.35, 'low': 0.05}

                backtester = SignalWeightedPortfolioBacktester(
                    strategies=strategies,
                    weights=weights,
                    threshold_config=threshold_config,
                    timeframe=self.timeframe,
                    start_date=data_start_str,  # 数据文件范围
                    end_date=data_end_str,
                    initial_cash=100000.0,
                    results_dir=self.results_dir
                )

                # 加载数据
                backtester.load_strategy_data()

                # 过滤到测试期窗口
                backtester.market_data = backtester.market_data[
                    (backtester.market_data['datetime'] >= test_start) &
                    (backtester.market_data['datetime'] <= test_end)
                ].reset_index(drop=True)

                # 运行回测
                result = backtester.run_backtest()

                metrics = result['metrics']

                return {
                    'window_id': window_id,
                    'portfolio_id': portfolio_config['portfolio_id'],
                    'test_start': test_start_str,
                    'test_end': test_end_str,
                    'test_return_pct': metrics['total_return_pct'],
                    'test_sharpe': metrics['sharpe_ratio'],
                    'test_max_dd_pct': metrics['max_drawdown_pct'],
                    'test_total_trades': metrics['total_trades'],
                    'method': 'signal_weighted'
                }

            except Exception as e:
                print(f"    ✗ 测试失败: {e}")
                import traceback
                traceback.print_exc()
                return None

        else:
            # 使用传统的多账户加权平均回测
            # TODO: 调用 portfolio_backtest.backtest_portfolio_from_daily_values()
            # 这里暂不实现,因为推荐使用signal_weighted
            print("    警告: 传统多账户回测暂未实现,请使用 --use-signal-weighted")
            return None

    def run_validation(self, workers: int = 1):
        """
        运行完整的Walk-Forward验证

        Args:
            workers: 并行工作进程数(1=串行)
        """
        if not self.windows:
            self.generate_windows()

        print("\n" + "=" * 80)
        print("开始Walk-Forward验证")
        print("=" * 80)
        print(f"窗口数量: {len(self.windows)}")
        print(f"每窗口选择前 {self.top_n} 个组合")
        print(f"回测方式: {'单账户信号加权' if self.use_signal_weighted else '多账户加权平均'}")
        print("=" * 80)

        all_train_results = []
        all_test_results = []

        for idx, (train_start, train_end, test_start, test_end) in enumerate(self.windows, 1):
            # 步骤1: 训练期优化
            train_portfolios = self.run_train_optimization(idx, train_start, train_end)

            if train_portfolios.empty:
                print(f"  跳过窗口 {idx}: 训练期优化失败")
                continue

            all_train_results.append(train_portfolios)

            # 步骤2: 测试期验证
            print(f"\n{'='*60}")
            print(f"窗口 {idx} - 测试期验证")
            print(f"{'='*60}")
            print(f"测试范围: {test_start.strftime('%Y%m%d')} ~ {test_end.strftime('%Y%m%d')}")

            for _, portfolio_config in train_portfolios.iterrows():
                test_result = self.run_test_validation(
                    idx,
                    portfolio_config.to_dict(),
                    test_start,
                    test_end
                )

                if test_result:
                    # 合并训练期指标
                    test_result.update({
                        'train_start': portfolio_config['train_start'],
                        'train_end': portfolio_config['train_end'],
                        'train_sharpe': portfolio_config['expected_sharpe'],
                        'train_return': portfolio_config['expected_return'],
                        'train_max_dd': portfolio_config['expected_max_dd'],
                        'strategies': portfolio_config['strategies'],
                        'weights': portfolio_config['weights'],
                        'weight_method': portfolio_config['weight_method']
                    })
                    all_test_results.append(test_result)

        # 保存结果
        self.train_results = pd.concat(all_train_results, ignore_index=True) if all_train_results else pd.DataFrame()
        self.test_results = pd.DataFrame(all_test_results) if all_test_results else pd.DataFrame()

        # 生成报告
        self.generate_reports()

    def generate_reports(self):
        """生成Walk-Forward分析报告"""
        if self.test_results.empty:
            print("\n警告: 没有测试结果,无法生成报告")
            return

        print("\n" + "=" * 80)
        print("生成Walk-Forward分析报告")
        print("=" * 80)

        # 计算过拟合指标
        self.test_results['sharpe_decay'] = self.test_results['train_sharpe'] - self.test_results['test_sharpe']
        self.test_results['sharpe_decay_pct'] = (self.test_results['sharpe_decay'] / self.test_results['train_sharpe']) * 100
        self.test_results['return_decay'] = self.test_results['train_return'] - self.test_results['test_return_pct'] / 100
        self.test_results['return_decay_pct'] = (self.test_results['return_decay'] / self.test_results['train_return']) * 100

        # 过拟合指数 (Overfitting Index)
        self.test_results['overfitting_index'] = self.test_results['sharpe_decay_pct'] / 100

        # 输出目录
        output_dir = Path(self.results_dir) / 'walk_forward'
        output_dir.mkdir(exist_ok=True)

        # 1. 窗口汇总
        window_summary = self.test_results.groupby('window_id').agg({
            'train_sharpe': 'mean',
            'test_sharpe': 'mean',
            'sharpe_decay': 'mean',
            'overfitting_index': 'mean',
            'test_return_pct': 'mean'
        }).reset_index()

        window_summary_file = output_dir / f'window_summary_{self.timeframe}.csv'
        window_summary.to_csv(window_summary_file, index=False)
        print(f"  ✓ 窗口汇总: {window_summary_file}")

        # 2. 组合稳健性排名
        # 对于每个组合(strategies组合),计算跨窗口的平均表现
        portfolio_robustness = self.test_results.groupby('strategies').agg({
            'test_sharpe': ['mean', 'std', 'min', 'max'],
            'overfitting_index': 'mean',
            'window_id': 'count'  # 出现频率
        }).reset_index()

        portfolio_robustness.columns = ['strategies', 'avg_test_sharpe', 'sharpe_std', 'min_sharpe', 'max_sharpe', 'overfitting_index', 'frequency']

        # 计算稳健性评分
        # Score = avg_sharpe - overfitting_penalty - volatility_penalty
        portfolio_robustness['robustness_score'] = (
            portfolio_robustness['avg_test_sharpe'] -
            portfolio_robustness['overfitting_index'] * 2 -
            portfolio_robustness['sharpe_std'] * 0.5
        )

        portfolio_robustness = portfolio_robustness.sort_values('robustness_score', ascending=False)

        robustness_file = output_dir / f'portfolio_robustness_{self.timeframe}.csv'
        portfolio_robustness.to_csv(robustness_file, index=False)
        print(f"  ✓ 组合稳健性: {robustness_file}")

        # 3. 详细结果
        detail_file = output_dir / f'walk_forward_details_{self.timeframe}.csv'
        self.test_results.to_csv(detail_file, index=False)
        print(f"  ✓ 详细结果: {detail_file}")

        # 4. 终端报告
        self._print_summary_report(window_summary, portfolio_robustness)

    def _print_summary_report(self, window_summary: pd.DataFrame, portfolio_robustness: pd.DataFrame):
        """打印汇总报告"""
        print("\n" + "=" * 100)
        print("Walk-Forward 验证报告")
        print("=" * 100)

        # 窗口汇总
        print("\n【窗口汇总】")
        print(f"{'窗口':<6} {'训练夏普':<10} {'测试夏普':<10} {'夏普衰减':<10} {'过拟合指数':<12} {'测试收益%':<12}")
        print("-" * 100)
        for _, row in window_summary.iterrows():
            print(f"{int(row['window_id']):<6} {row['train_sharpe']:>9.2f} {row['test_sharpe']:>9.2f} "
                  f"{row['sharpe_decay']:>9.2f} {row['overfitting_index']:>11.2%} {row['test_return_pct']:>11.2f}")

        # 整体统计
        print("\n【整体统计】")
        print(f"  平均训练期夏普: {window_summary['train_sharpe'].mean():.2f}")
        print(f"  平均测试期夏普: {window_summary['test_sharpe'].mean():.2f}")
        print(f"  平均夏普衰减: {window_summary['sharpe_decay'].mean():.2f}")
        print(f"  平均过拟合指数: {window_summary['overfitting_index'].mean():.2%}")

        # 稳健组合排名
        print("\n【前10个稳健组合】")
        print(f"{'排名':<4} {'平均测试夏普':<14} {'夏普标准差':<12} {'过拟合指数':<12} {'稳健评分':<10} {'出现次数':<8}")
        print("-" * 100)
        for idx, row in portfolio_robustness.head(10).iterrows():
            print(f"{idx+1:<4} {row['avg_test_sharpe']:>13.2f} {row['sharpe_std']:>11.2f} "
                  f"{row['overfitting_index']:>11.2%} {row['robustness_score']:>9.2f} {int(row['frequency']):>7}")
            # 打印策略组成
            strategies = row['strategies'].split(',')
            print(f"     策略: {', '.join(strategies[:3])}{'...' if len(strategies) > 3 else ''}")

        print("\n" + "=" * 100)
        print("评估标准:")
        print("  - 过拟合指数 < 0.1 : 优秀 (样本外衰减<10%)")
        print("  - 过拟合指数 < 0.2 : 良好 (衰减<20%)")
        print("  - 过拟合指数 < 0.3 : 可接受")
        print("  - 过拟合指数 >= 0.3 : 过拟合严重")
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description='Walk-Forward验证器 - 防止过拟合',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 数据范围
    parser.add_argument('--data-start', default='20200101', help='完整数据开始日期 (YYYYMMDD, 默认: 20200101)')
    parser.add_argument('--data-end', default='20241231', help='完整数据结束日期 (YYYYMMDD, 默认: 20241231)')

    # 窗口配置
    parser.add_argument('--timeframe', default='d1', help='时间周期 (默认: d1)')
    parser.add_argument('--train-months', type=int, default=12, help='训练窗口大小(月) (默认: 12)')
    parser.add_argument('--test-months', type=int, default=6, help='测试窗口大小(月) (默认: 6)')
    parser.add_argument('--step-months', type=int, default=6, help='滚动步长(月) (默认: 6)')

    # 优化参数
    parser.add_argument('--top-n', type=int, default=10, help='每个窗口选择前N个组合 (默认: 10)')
    parser.add_argument('--correlation-threshold', type=float, default=0.3, help='相关性阈值 (默认: 0.3)')

    # 回测方式
    parser.add_argument('--use-signal-weighted', action='store_true', default=True,
                       help='使用单账户信号加权回测 (默认: True, 推荐)')
    parser.add_argument('--no-signal-weighted', action='store_false', dest='use_signal_weighted',
                       help='使用传统多账户加权平均回测')

    # 其他
    parser.add_argument('--results-dir', default='backtest_results', help='结果目录 (默认: backtest_results)')
    parser.add_argument('--workers', type=int, default=1, help='并行工作进程数 (默认: 1, 串行执行)')

    args = parser.parse_args()

    # 创建验证器
    validator = WalkForwardValidator(
        data_start_date=args.data_start,
        data_end_date=args.data_end,
        timeframe=args.timeframe,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        top_n=args.top_n,
        correlation_threshold=args.correlation_threshold,
        results_dir=args.results_dir,
        use_signal_weighted=args.use_signal_weighted
    )

    # 运行验证
    validator.run_validation(workers=args.workers)

    print("\n" + "=" * 80)
    print("Walk-Forward验证完成!")
    print("=" * 80)
    print(f"\n结果已保存到: {args.results_dir}/walk_forward/")
    print("\n使用建议:")
    print("  1. 查看 portfolio_robustness_*.csv 选择稳健组合")
    print("  2. 选择 过拟合指数 < 0.2 的组合")
    print("  3. 优先考虑 出现频率 高的组合")
    print("  4. 避免使用 过拟合指数 > 0.3 的组合")


if __name__ == '__main__':
    main()
