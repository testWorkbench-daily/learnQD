#!/usr/bin/env python
"""
策略相关性分析工具

用法:
    # 分析指定时间段和timeframe的所有策略
    python analyze_correlation.py --start 20240101 --end 20241231 --timeframe d1

    # 指定相关性阈值来推荐策略组合
    python analyze_correlation.py --start 20240101 --end 20241231 --timeframe d1 --threshold 0.5
"""
import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from bt_runner import calculate_sharpe_ratio, calculate_annualized_return

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class StrategyCorrelationAnalyzer:
    """策略相关性分析器"""

    def __init__(self, start_date: str, end_date: str, timeframe: str, results_dir: str = 'backtest_results',
                 data_start_date: str = None, data_end_date: str = None):
        """
        初始化分析器

        Args:
            start_date: 开始日期 (格式: YYYYMMDD) - 窗口开始日期
            end_date: 结束日期 (格式: YYYYMMDD) - 窗口结束日期
            timeframe: 时间周期 (如: d1, h1, m1)
            results_dir: 回测结果目录
            data_start_date: 数据文件开始日期 (YYYYMMDD) - 用于匹配文件名，如不提供则使用start_date
            data_end_date: 数据文件结束日期 (YYYYMMDD) - 用于匹配文件名，如不提供则使用end_date
        """
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.results_dir = results_dir
        self.data_start_date = data_start_date if data_start_date else start_date
        self.data_end_date = data_end_date if data_end_date else end_date
        self.daily_returns = {}  # {strategy_name: daily_return_series}

    def load_strategy_returns(self) -> Dict[str, pd.Series]:
        """
        加载所有策略的每日收益率数据

        Returns:
            字典，键为策略名称，值为收益率序列
        """
        pattern = f"daily_values_*_{self.timeframe}_{self.data_start_date}_{self.data_end_date}.csv"
        files = list(Path(self.results_dir).glob(pattern))

        if not files:
            print(f"未找到匹配的文件: {pattern}")
            print(f"搜索目录: {self.results_dir}")
            return {}

        print(f"找到 {len(files)} 个策略的数据文件:")

        for file in files:
            # 从文件名提取策略名称
            # 格式: daily_values_{strategy}_{timeframe}_{start}_{end}.csv
            filename = file.stem
            parts = filename.split('_')
            # 移除 'daily', 'values', timeframe, start, end
            strategy_name = '_'.join(parts[2:-3])  # 策略名在中间部分

            try:
                df = pd.read_csv(file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')

                # CRITICAL: ALWAYS filter to analysis window to prevent look-ahead bias
                # Even if start_date == data_start_date, we must filter to ensure
                # correlation matrix only uses data from the specified window
                start_dt = pd.to_datetime(self.start_date, format='%Y%m%d')
                end_dt = pd.to_datetime(self.end_date, format='%Y%m%d')
                df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]

                if df.empty:
                    print(f"  ✗ {strategy_name}: 窗口内无数据 ({self.start_date} ~ {self.end_date})")
                    continue

                # 提取每日收益率
                returns = df['daily_return'].values
                dates = df['datetime'].values

                self.daily_returns[strategy_name] = pd.Series(returns, index=dates, name=strategy_name)
                print(f"  ✓ {strategy_name}: {len(returns)} 个数据点")

            except Exception as e:
                print(f"  ✗ 加载 {file.name} 失败: {e}")

        return self.daily_returns

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        计算策略之间的相关性矩阵

        Returns:
            相关性矩阵 DataFrame
        """
        if not self.daily_returns:
            print("错误: 没有可用的收益率数据")
            return pd.DataFrame()

        # 将所有策略的收益率合并到一个DataFrame
        returns_df = pd.DataFrame(self.daily_returns)

        # 计算皮尔逊相关系数
        corr_matrix = returns_df.corr(method='pearson')

        return corr_matrix

    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame, output_file: str = None):
        """
        绘制相关性热力图

        Args:
            corr_matrix: 相关性矩阵
            output_file: 输出文件路径 (可选)
        """
        if corr_matrix.empty:
            print("相关性矩阵为空，无法绘制热力图")
            return

        plt.figure(figsize=(16, 14))

        # 使用seaborn绘制热力图
        sns.heatmap(
            corr_matrix,
            annot=True,  # 显示数值
            fmt='.2f',   # 数值格式
            cmap='RdYlGn',  # 颜色映射：红色(负相关) -> 黄色(无相关) -> 绿色(正相关)
            center=0,    # 中心值为0
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot_kws={'fontsize': 8}  # 数值字体大小
        )

        plt.title(f'Strategy Correlation Matrix\n({self.start_date} to {self.end_date}, {self.timeframe})',
                  fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n相关性热力图已保存: {output_file}")
        else:
            plt.savefig(f'{self.results_dir}/correlation_heatmap_{self.timeframe}_{self.start_date}_{self.end_date}.png',
                       dpi=300, bbox_inches='tight')
            print(f"\n相关性热力图已保存: {self.results_dir}/correlation_heatmap_{self.timeframe}_{self.start_date}_{self.end_date}.png")

        plt.close()

    def find_low_correlation_pairs(self, corr_matrix: pd.DataFrame, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """
        找出相关性低于阈值的策略对

        Args:
            corr_matrix: 相关性矩阵
            threshold: 相关性阈值 (绝对值)

        Returns:
            列表，每个元素为 (strategy1, strategy2, correlation)
        """
        low_corr_pairs = []

        strategies = corr_matrix.columns.tolist()
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) < threshold:
                    low_corr_pairs.append((strategies[i], strategies[j], corr))

        # 按相关性绝对值排序（从低到高）
        low_corr_pairs.sort(key=lambda x: abs(x[2]))

        return low_corr_pairs

    def suggest_portfolio_combinations(self, corr_matrix: pd.DataFrame, max_strategies: int = 4, threshold: float = 0.3) -> List[List[str]]:
        """
        推荐低相关性的策略组合

        Args:
            corr_matrix: 相关性矩阵
            max_strategies: 组合中最多包含的策略数量
            threshold: 相关性阈值

        Returns:
            推荐的策略组合列表
        """
        print(f"\n推荐低相关性策略组合 (相关性阈值 < {threshold}):")
        print("=" * 80)

        strategies = corr_matrix.columns.tolist()
        all_combinations = []

        # 简单的贪心算法：从相关性最低的策略开始构建组合
        selected = []

        # 首先找出平均相关性最低的策略作为起点
        avg_corr = corr_matrix.abs().mean(axis=1)
        first_strategy = avg_corr.idxmin()
        selected.append(first_strategy)

        # 迭代添加与已选策略相关性都较低的策略
        while len(selected) < max_strategies and len(selected) < len(strategies):
            best_candidate = None
            best_avg_corr = float('inf')

            for candidate in strategies:
                if candidate in selected:
                    continue

                # 计算候选策略与已选策略的平均相关性
                corrs = [abs(corr_matrix.loc[candidate, s]) for s in selected]
                avg_corr_with_selected = np.mean(corrs)

                if avg_corr_with_selected < best_avg_corr and avg_corr_with_selected < threshold:
                    best_avg_corr = avg_corr_with_selected
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
            else:
                break  # 找不到满足条件的策略

        if len(selected) >= 2:
            all_combinations.append(selected)
            print(f"\n组合 1 ({len(selected)} 个策略):")
            for i, strategy in enumerate(selected, 1):
                print(f"  {i}. {strategy}")

            # 显示组合内相关性
            print(f"\n  组合内两两相关性:")
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    corr = corr_matrix.loc[selected[i], selected[j]]
                    print(f"    {selected[i]} <-> {selected[j]}: {corr:>6.3f}")
        else:
            print("  未找到满足条件的策略组合")

        # 找出所有相关性低的策略对，并构建更多组合
        low_corr_pairs = self.find_low_correlation_pairs(corr_matrix, threshold)

        if low_corr_pairs:
            print(f"\n所有低相关性策略对 (前20个):")
            print("-" * 80)
            for i, (s1, s2, corr) in enumerate(low_corr_pairs[:20], 1):
                print(f"  {i:2d}. {s1:<25} <-> {s2:<25} : {corr:>6.3f}")

            # 从前10个低相关性策略对中构建额外的组合
            for idx, (s1, s2, _) in enumerate(low_corr_pairs[:10], 2):
                combo = [s1, s2]
                if combo not in all_combinations and sorted(combo) not in [sorted(c) for c in all_combinations]:
                    all_combinations.append(combo)

        return all_combinations

    def calculate_portfolio_metrics(self, strategies: List[str], weights: List[float] = None) -> Dict:
        """
        计算策略组合的绩效指标

        Args:
            strategies: 策略名称列表
            weights: 权重列表 (如果为None，则等权重)

        Returns:
            包含组合绩效指标的字典
        """
        if weights is None:
            weights = [1.0 / len(strategies)] * len(strategies)

        if len(strategies) != len(weights):
            raise ValueError("策略数量与权重数量不匹配")

        # 获取策略收益率
        returns_list = [self.daily_returns[s] for s in strategies if s in self.daily_returns]

        if len(returns_list) != len(strategies):
            missing = set(strategies) - set(self.daily_returns.keys())
            raise ValueError(f"部分策略数据缺失: {missing}")

        # 合并收益率序列 (确保日期对齐)
        returns_df = pd.DataFrame({s: self.daily_returns[s] for s in strategies})
        returns_df = returns_df.fillna(0)  # 缺失值填0

        # 计算组合收益率
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # 计算绩效指标
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = calculate_annualized_return(total_return, len(portfolio_returns), periods_per_year=252)
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = calculate_sharpe_ratio(portfolio_returns.values, annualize=True, periods_per_year=252)

        # 最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
        }

    def print_summary(self, corr_matrix: pd.DataFrame):
        """打印分析摘要"""
        print("\n" + "=" * 80)
        print(f"策略相关性分析摘要")
        print("=" * 80)
        print(f"分析周期: {self.start_date} 至 {self.end_date}")
        print(f"时间框架: {self.timeframe}")
        print(f"策略数量: {len(self.daily_returns)}")

        if not corr_matrix.empty:
            # 统计相关性分布
            corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            print(f"\n相关性统计:")
            print(f"  平均相关性: {corr_values.mean():.3f}")
            print(f"  最大相关性: {corr_values.max():.3f}")
            print(f"  最小相关性: {corr_values.min():.3f}")
            print(f"  标准差: {corr_values.std():.3f}")

            # 找出高度相关的策略对
            high_corr_pairs = []
            strategies = corr_matrix.columns.tolist()
            for i in range(len(strategies)):
                for j in range(i + 1, len(strategies)):
                    corr = corr_matrix.iloc[i, j]
                    if corr > 0.7:  # 高度正相关
                        high_corr_pairs.append((strategies[i], strategies[j], corr))

            if high_corr_pairs:
                high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
                print(f"\n高度相关策略对 (相关性 > 0.7):")
                for s1, s2, corr in high_corr_pairs[:10]:
                    print(f"  {s1:<25} <-> {s2:<25} : {corr:.3f}")


def main():
    parser = argparse.ArgumentParser(description='策略相关性分析工具')
    parser.add_argument('--start', required=True, help='开始日期 (格式: YYYYMMDD)')
    parser.add_argument('--end', required=True, help='结束日期 (格式: YYYYMMDD)')
    parser.add_argument('--timeframe', default='d1', help='时间周期 (默认: d1)')
    parser.add_argument('--threshold', type=float, default=0.3, help='低相关性阈值 (默认: 0.3)')
    parser.add_argument('--results-dir', default='backtest_results', help='回测结果目录')
    parser.add_argument('--max-strategies', type=int, default=4, help='组合最多包含的策略数 (默认: 4)')

    args = parser.parse_args()

    # 创建分析器
    analyzer = StrategyCorrelationAnalyzer(
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe,
        results_dir=args.results_dir
    )

    # 加载数据
    print("正在加载策略收益率数据...")
    analyzer.load_strategy_returns()

    if not analyzer.daily_returns:
        print("\n错误: 没有找到任何策略数据")
        print(f"请先运行回测生成数据，确保文件名格式为: daily_values_*_{args.timeframe}_{args.start}_{args.end}.csv")
        return

    # 计算相关性矩阵
    print("\n正在计算相关性矩阵...")
    corr_matrix = analyzer.calculate_correlation_matrix()

    # 打印摘要
    analyzer.print_summary(corr_matrix)

    # 绘制热力图
    analyzer.plot_correlation_heatmap(corr_matrix)

    # 推荐策略组合
    combinations = analyzer.suggest_portfolio_combinations(
        corr_matrix,
        max_strategies=args.max_strategies,
        threshold=args.threshold
    )

    # 保存推荐的策略组合
    if combinations:
        combo_output = f'{args.results_dir}/recommended_portfolios_{args.timeframe}_{args.start}_{args.end}.csv'
        combo_data = []
        for idx, combo in enumerate(combinations, 1):
            combo_data.append({
                'portfolio_id': idx,
                'num_strategies': len(combo),
                'strategies': ','.join(combo),
                'equal_weight': 1.0 / len(combo)
            })
        combo_df = pd.DataFrame(combo_data)
        combo_df.to_csv(combo_output, index=False)
        print(f"\n推荐策略组合已保存: {combo_output}")
        print(f"共 {len(combinations)} 个组合")

    # 保存相关性矩阵
    output_csv = f'{args.results_dir}/correlation_matrix_{args.timeframe}_{args.start}_{args.end}.csv'
    corr_matrix.to_csv(output_csv)
    print(f"\n相关性矩阵已保存: {output_csv}")

    print("\n分析完成!")


if __name__ == '__main__':
    main()
