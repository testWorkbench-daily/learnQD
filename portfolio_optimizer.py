#!/usr/bin/env python
"""
投资组合智能优化器

整合质量筛选、相关性分析、权重优化，生成高夏普高收益的投资组合

用法:
    python portfolio_optimizer.py --start 20240101 --end 20241231 --timeframe d1

特性:
- 三层优化架构：质量筛选 → 相关性分组 → 智能权重
- 5种权重方法：夏普加权、风险平价、最大夏普、收益加权、等权
- 输出格式兼容 portfolio_backtest.py
"""

import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

# 导入自定义模块
from strategy_quality_filter import QualityFilter
from weight_optimizer import WeightOptimizer
from analyze_correlation import StrategyCorrelationAnalyzer


class PortfolioOptimizer:
    """投资组合优化器"""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        timeframe: str,
        results_dir: str = 'backtest_results'
    ):
        """
        初始化

        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            timeframe: 时间周期 (如 d1, h1)
            results_dir: 回测结果目录
        """
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.results_dir = results_dir

        # 初始化子模块
        self.quality_filter = QualityFilter(results_dir, timeframe, start_date, end_date)
        self.corr_analyzer = StrategyCorrelationAnalyzer(start_date, end_date, timeframe, results_dir)

    def optimize(
        self,
        # 质量筛选参数
        min_sharpe: float = 0.5,
        min_return: float = 1.0,
        max_drawdown: float = -10.0,
        top_n_quality: int = 20,

        # 相关性参数
        correlation_threshold: float = 0.3,
        min_strategies_per_portfolio: int = 2,
        max_strategies_per_portfolio: int = 4,

        # 权重方法
        weight_methods: List[str] = None,

        # 输出控制
        max_portfolios: int = 50,

    ) -> pd.DataFrame:
        """
        主优化流程

        Returns:
            DataFrame包含推荐组合，格式兼容 portfolio_backtest.py
        """
        if weight_methods is None:
            weight_methods = ['sharpe_weighted', 'risk_parity', 'max_sharpe']

        print("\n" + "=" * 80)
        print("投资组合智能优化器")
        print("=" * 80)
        print(f"时间范围: {self.start_date} - {self.end_date}")
        print(f"时间周期: {self.timeframe}")
        print(f"结果目录: {self.results_dir}")
        print("=" * 80)

        # === 步骤1: 质量筛选 ===
        print("\n" + "=" * 80)
        print("步骤1: 策略质量筛选")
        print("=" * 80)

        high_quality_strategies = self.quality_filter.filter_strategies(
            min_sharpe=min_sharpe,
            min_return=min_return,
            max_drawdown=max_drawdown,
            top_n=top_n_quality
        )

        self.quality_filter.print_quality_report(high_quality_strategies)

        if len(high_quality_strategies) < min_strategies_per_portfolio:
            print(f"\n错误: 高质量策略数量不足 ({len(high_quality_strategies)} < {min_strategies_per_portfolio})")
            print("建议: 放宽筛选条件（降低 min_sharpe 或 min_return）")
            return pd.DataFrame()

        print(f"通过质量筛选: {len(high_quality_strategies)} 个策略")

        # === 步骤2: 相关性分析 ===
        print("\n" + "=" * 80)
        print("步骤2: 相关性分析")
        print("=" * 80)

        # 加载收益率数据
        self.corr_analyzer.load_strategy_returns()

        # 过滤出高质量策略
        filtered_returns = {
            k: v for k, v in self.corr_analyzer.daily_returns.items()
            if k in high_quality_strategies
        }

        if len(filtered_returns) < min_strategies_per_portfolio:
            print(f"\n错误: 可用策略收益率数据不足")
            return pd.DataFrame()

        self.corr_analyzer.daily_returns = filtered_returns

        # 计算相关性矩阵
        corr_matrix = self.corr_analyzer.calculate_correlation_matrix()

        if corr_matrix.empty:
            print("\n错误: 相关性矩阵计算失败")
            return pd.DataFrame()

        # 绘制热力图（仅高质量策略）
        heatmap_file = f'{self.results_dir}/correlation_heatmap_optimized_{self.timeframe}_{self.start_date}_{self.end_date}.png'
        self.corr_analyzer.plot_correlation_heatmap(corr_matrix, heatmap_file)

        # === 步骤3: 生成低相关组合 ===
        print("\n" + "=" * 80)
        print("步骤3: 生成低相关策略组合")
        print("=" * 80)

        combinations = self._generate_low_correlation_combinations(
            high_quality_strategies,
            corr_matrix,
            correlation_threshold,
            min_strategies_per_portfolio,
            max_strategies_per_portfolio,
            max_portfolios
        )

        print(f"生成 {len(combinations)} 个低相关组合候选")

        if len(combinations) == 0:
            print("\n警告: 未找到满足条件的低相关组合")
            print("建议: 放宽 correlation_threshold 或减少 max_strategies_per_portfolio")
            return pd.DataFrame()

        # === 步骤4: 权重优化 ===
        print("\n" + "=" * 80)
        print("步骤4: 权重优化")
        print("=" * 80)
        print(f"权重方法: {weight_methods}")

        portfolios = []
        portfolio_id = 1

        for combo in combinations:
            # 为每个组合尝试不同的权重方法
            for weight_method in weight_methods:
                try:
                    weights = self._calculate_optimal_weights(combo, weight_method)

                    # 计算组合预期指标
                    metrics = self._calculate_portfolio_metrics(combo, weights)

                    portfolios.append({
                        'portfolio_id': portfolio_id,
                        'num_strategies': len(combo),
                        'strategies': ','.join(combo),
                        'weight_method': weight_method,
                        'weights': ','.join([f'{w:.4f}' for w in weights]),
                        'expected_sharpe': metrics['sharpe_ratio'],
                        'expected_return': metrics['total_return'],
                        'expected_max_dd': metrics['max_drawdown']
                    })

                    portfolio_id += 1
                except Exception as e:
                    print(f"  警告: 组合 {combo} 使用 {weight_method} 失败: {e}")
                    continue

        if len(portfolios) == 0:
            print("\n错误: 没有成功生成任何组合")
            return pd.DataFrame()

        # 转为DataFrame
        portfolios_df = pd.DataFrame(portfolios)

        # 按预期夏普比率排序
        portfolios_df = portfolios_df.sort_values('expected_sharpe', ascending=False)

        # 限制数量
        portfolios_df = portfolios_df.head(max_portfolios)

        # 重新分配ID
        portfolios_df['portfolio_id'] = range(1, len(portfolios_df) + 1)

        # === 步骤5: 保存结果 ===
        output_file = f'{self.results_dir}/optimized_portfolios_{self.timeframe}_{self.start_date}_{self.end_date}.csv'
        portfolios_df.to_csv(output_file, index=False)

        print(f"\n优化组合已保存: {output_file}")
        print(f"共 {len(portfolios_df)} 个组合")

        # 打印前10个推荐组合
        self._print_top_portfolios(portfolios_df, top_n=10)

        return portfolios_df

    def _generate_low_correlation_combinations(
        self,
        strategies: List[str],
        corr_matrix: pd.DataFrame,
        threshold: float,
        min_size: int,
        max_size: int,
        max_combinations: int
    ) -> List[List[str]]:
        """
        生成低相关组合

        使用改进的贪心算法:
        1. 对于每个策略作为起点
        2. 迭代添加与已选策略相关性都<threshold的策略
        3. 生成不同大小的组合（min_size到max_size）
        """
        combinations = []

        # 按质量评分排序（高质量策略优先作为种子）
        quality_scores = self.quality_filter.all_metrics
        strategies_sorted = sorted(
            strategies,
            key=lambda s: quality_scores.get(s, {}).get('sharpe_ratio', 0),
            reverse=True
        )

        print(f"  策略排序（按夏普）: {strategies_sorted[:5]}...")

        # 尝试每个策略作为起点
        for seed in strategies_sorted:
            for target_size in range(min_size, max_size + 1):
                combo = self._build_combination_from_seed(
                    seed, target_size, strategies_sorted, corr_matrix, threshold
                )
                if combo and len(combo) >= min_size:
                    # 避免重复
                    combo_sorted = sorted(combo)
                    if combo_sorted not in [sorted(c) for c in combinations]:
                        combinations.append(combo)
                        if len(combinations) >= max_combinations * 3:  # 生成足够多的候选
                            break
            if len(combinations) >= max_combinations * 3:
                break

        # 打印前几个组合示例
        print(f"\n  组合示例（前5个）:")
        for i, combo in enumerate(combinations[:5], 1):
            print(f"    {i}. {combo}")

        return combinations

    def _build_combination_from_seed(
        self,
        seed: str,
        target_size: int,
        candidates: List[str],
        corr_matrix: pd.DataFrame,
        threshold: float
    ) -> List[str]:
        """从种子策略构建组合"""
        selected = [seed]

        for candidate in candidates:
            if candidate in selected:
                continue

            # 检查与已选策略的相关性
            try:
                corrs = [abs(corr_matrix.loc[candidate, s]) for s in selected]
                avg_corr = sum(corrs) / len(corrs)

                if avg_corr < threshold:
                    selected.append(candidate)
                    if len(selected) >= target_size:
                        break
            except KeyError:
                # 某些策略可能不在相关性矩阵中
                continue

        return selected if len(selected) >= 2 else []

    def _calculate_optimal_weights(
        self,
        strategies: List[str],
        method: str
    ) -> np.ndarray:
        """计算最优权重"""
        # 构建收益率DataFrame
        returns_df = pd.DataFrame({
            s: self.corr_analyzer.daily_returns[s]
            for s in strategies
            if s in self.corr_analyzer.daily_returns
        })

        if returns_df.empty:
            raise ValueError(f"策略 {strategies} 的收益率数据不可用")

        optimizer = WeightOptimizer(returns_df)
        weights = optimizer.calculate_weights(method)

        return weights

    def _calculate_portfolio_metrics(
        self,
        strategies: List[str],
        weights: np.ndarray
    ) -> Dict:
        """计算组合指标（预期值）"""
        # 复用 analyze_correlation.py 的方法
        weights_list = weights.tolist()
        metrics = self.corr_analyzer.calculate_portfolio_metrics(strategies, weights_list)

        return metrics

    def _print_top_portfolios(self, portfolios_df: pd.DataFrame, top_n: int = 10):
        """打印前N个推荐组合"""
        print("\n" + "=" * 100)
        print(f"前 {top_n} 个推荐组合（按预期夏普排序）")
        print("=" * 100)
        print(f"{'ID':<4} {'策略数':<6} {'权重方法':<18} {'预期夏普':<10} {'预期收益%':<12} {'预期回撤%':<12}")
        print("-" * 100)

        for idx, row in portfolios_df.head(top_n).iterrows():
            print(f"{row['portfolio_id']:<4} {row['num_strategies']:<6} "
                  f"{row['weight_method']:<18} {row['expected_sharpe']:<10.2f} "
                  f"{row['expected_return']*100:<12.2f} {row['expected_max_dd']*100:<12.2f}")

            # 打印策略和权重
            strats = row['strategies'].split(',')
            weights = [float(w) for w in row['weights'].split(',')]
            for s, w in zip(strats, weights):
                print(f"      - {s:<30} {w*100:>5.1f}%")
            print()

        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='智能投资组合优化器')

    # 必需参数
    parser.add_argument('--start', required=True, help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end', required=True, help='结束日期 (YYYYMMDD)')
    parser.add_argument('--timeframe', default='d1', help='时间周期 (默认: d1)')

    # 质量筛选
    parser.add_argument('--min-sharpe', type=float, default=0.5, help='最小夏普比率 (默认: 0.5)')
    parser.add_argument('--min-return', type=float, default=1.0, help='最小收益率%% (默认: 1.0)')
    parser.add_argument('--max-drawdown', type=float, default=-10.0, help='最大回撤%% (默认: -10.0)')
    parser.add_argument('--top-n-quality', type=int, default=20, help='选择质量评分前N个 (默认: 20)')

    # 相关性
    parser.add_argument('--correlation-threshold', type=float, default=0.3, help='相关性阈值 (默认: 0.3)')
    parser.add_argument('--min-strategies', type=int, default=2, help='组合最少策略数 (默认: 2)')
    parser.add_argument('--max-strategies', type=int, default=4, help='组合最多策略数 (默认: 4)')

    # 权重方法
    parser.add_argument('--weight-methods', nargs='+',
                       default=['sharpe_weighted', 'risk_parity', 'max_sharpe'],
                       help='权重方法列表 (默认: sharpe_weighted risk_parity max_sharpe)')

    # 输出
    parser.add_argument('--max-portfolios', type=int, default=50, help='最多生成组合数 (默认: 50)')
    parser.add_argument('--results-dir', default='backtest_results', help='结果目录 (默认: backtest_results)')

    args = parser.parse_args()

    # 创建优化器
    optimizer = PortfolioOptimizer(
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe,
        results_dir=args.results_dir
    )

    # 运行优化
    portfolios_df = optimizer.optimize(
        min_sharpe=args.min_sharpe,
        min_return=args.min_return,
        max_drawdown=args.max_drawdown,
        top_n_quality=args.top_n_quality,
        correlation_threshold=args.correlation_threshold,
        min_strategies_per_portfolio=args.min_strategies,
        max_strategies_per_portfolio=args.max_strategies,
        weight_methods=args.weight_methods,
        max_portfolios=args.max_portfolios
    )

    if not portfolios_df.empty:
        print("\n" + "=" * 80)
        print("优化完成！")
        print("=" * 80)
        print(f"\n可运行以下命令回测推荐组合:")
        print(f"  python portfolio_backtest.py \\")
        print(f"    --portfolio-file {args.results_dir}/optimized_portfolios_{args.timeframe}_{args.start}_{args.end}.csv")
        print()
    else:
        print("\n优化失败，请检查参数设置")


if __name__ == '__main__':
    main()
