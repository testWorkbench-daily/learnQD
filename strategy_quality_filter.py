#!/usr/bin/env python
"""
策略质量筛选器

功能:
- 从 daily_values CSV 提取策略指标
- 计算质量评分（基于夏普、收益、回撤）
- 筛选高质量策略

用法:
    from strategy_quality_filter import QualityFilter

    filter = QualityFilter('backtest_results', 'd1', '20240101', '20241231')
    high_quality = filter.filter_strategies(min_sharpe=0.5, min_return=1.0, top_n=20)
    filter.print_quality_report(high_quality)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from bt_runner import calculate_sharpe_ratio, calculate_annualized_return


class StrategyMetrics:
    """从 daily_values 计算策略指标"""

    def __init__(self, daily_values_file: str, start_date: str = None, end_date: str = None):
        """
        初始化

        Args:
            daily_values_file: daily_values CSV 文件路径
                格式: datetime, portfolio_value, daily_return, cumulative_return
            start_date: 开始日期 (YYYYMMDD) - 可选，用于过滤数据
            end_date: 结束日期 (YYYYMMDD) - 可选，用于过滤数据
        """
        self.file_path = daily_values_file
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        self._load_data()

    def _load_data(self):
        """加载数据并过滤日期范围"""
        try:
            self.df = pd.read_csv(self.file_path)
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.df = self.df.sort_values('datetime')

            # 如果提供了日期范围，过滤数据
            if self.start_date:
                start_dt = pd.to_datetime(self.start_date, format='%Y%m%d')
                self.df = self.df[self.df['datetime'] >= start_dt]
            if self.end_date:
                end_dt = pd.to_datetime(self.end_date, format='%Y%m%d')
                self.df = self.df[self.df['datetime'] <= end_dt]

        except Exception as e:
            raise FileNotFoundError(f"无法加载文件 {self.file_path}: {e}")

    def calculate_metrics(self) -> Dict:
        """
        计算策略指标

        Returns:
            {
                'name': 'rsi_reversal',
                'total_return': 5.70,           # 总收益率 %
                'sharpe_ratio': 2.08,            # 夏普比率
                'max_drawdown': -2.15,           # 最大回撤 %
                'volatility': 12.5,              # 年化波动率 %
                'win_rate': 0.62,                # 胜率（正收益日占比）
                'final_value': 105696.62,
                'num_days': 252
            }
        """
        if self.df is None or len(self.df) == 0:
            raise ValueError("数据为空")

        # 从文件名提取策略名称
        # 格式: daily_values_{strategy}_{timeframe}_{start}_{end}.csv
        file_stem = Path(self.file_path).stem
        parts = file_stem.split('_')
        strategy_name = '_'.join(parts[2:-3])  # 策略名在中间部分

        # 基础指标
        initial_value = self.df['portfolio_value'].iloc[0]
        final_value = self.df['portfolio_value'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100  # 百分比

        # 夏普比率（年化，使用统一方法）
        daily_returns = self.df['daily_return'].values
        sharpe_ratio = calculate_sharpe_ratio(daily_returns, annualize=True, periods_per_year=252)

        # 波动率（年化）
        volatility = daily_returns.std() * np.sqrt(252) * 100  # 百分比

        # 最大回撤
        cumulative = self.df['portfolio_value'].values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100  # 百分比

        # 胜率（正收益日占比）
        positive_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        win_rate = positive_days / total_days if total_days > 0 else 0.0

        # 计算连续盈亏
        consecutive_wins, consecutive_losses = self._calculate_consecutive_streaks(daily_returns)
        max_consecutive_wins = consecutive_wins
        max_consecutive_losses = consecutive_losses

        # 计算盈亏比（平均盈利/平均亏损的绝对值）
        win_returns = daily_returns[daily_returns > 0]
        loss_returns = daily_returns[daily_returns < 0]
        avg_win = win_returns.mean() if len(win_returns) > 0 else 0.0
        avg_loss = abs(loss_returns.mean()) if len(loss_returns) > 0 else 0.0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        return {
            'name': strategy_name,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'profit_loss_ratio': profit_loss_ratio,
            'avg_win': avg_win * 100,  # 转为百分比
            'avg_loss': avg_loss * 100,  # 转为百分比
            'final_value': final_value,
            'num_days': total_days
        }

    def _calculate_consecutive_streaks(self, returns: np.ndarray) -> tuple:
        """
        计算最大连续盈利/亏损天数

        Args:
            returns: 收益率序列

        Returns:
            (最大连续盈利天数, 最大连续亏损天数)
        """
        if len(returns) == 0:
            return 0, 0

        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for ret in returns:
            if ret > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif ret < 0:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                # ret == 0，保持当前streak不变
                pass

        return max_win_streak, max_loss_streak


class QualityFilter:
    """质量评分和筛选"""

    def __init__(self, results_dir: str, timeframe: str, start_date: str, end_date: str,
                 price_data_file: str = './data/nq_m1_forward_adjusted.csv',
                 data_start_date: str = None, data_end_date: str = None):
        """
        初始化

        Args:
            results_dir: 回测结果目录
            timeframe: 时间周期 (如 d1, h1)
            start_date: 开始日期 (YYYYMMDD) - 窗口开始日期
            end_date: 结束日期 (YYYYMMDD) - 窗口结束日期
            price_data_file: 价格数据文件路径
            data_start_date: 数据文件开始日期 (YYYYMMDD) - 用于匹配文件名，如不提供则使用start_date
            data_end_date: 数据文件结束日期 (YYYYMMDD) - 用于匹配文件名，如不提供则使用end_date
        """
        self.results_dir = results_dir
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.price_data_file = price_data_file

        # 使用数据文件日期范围匹配文件，如果未提供则使用窗口日期
        file_start = data_start_date if data_start_date else start_date
        file_end = data_end_date if data_end_date else end_date
        self.pattern = f"daily_values_*_{timeframe}_{file_start}_{file_end}.csv"

        self.all_metrics = {}  # {strategy_name: metrics_dict}
        self.buy_hold_metrics = None  # 买入并持有基准

    def load_all_strategies(self) -> Dict[str, Dict]:
        """
        加载所有策略的指标

        Returns:
            {'rsi_reversal': {...metrics...}, 'sma_cross': {...metrics...}, ...}
        """
        files = list(Path(self.results_dir).glob(self.pattern))

        if not files:
            print(f"警告: 未找到匹配的文件: {self.pattern}")
            print(f"搜索目录: {self.results_dir}")
            return {}

        print(f"找到 {len(files)} 个策略的 daily_values 文件")

        for file in files:
            # 排除组合策略文件（portfolio_*）
            if 'portfolio_' in file.name:
                continue

            try:
                # 传递日期范围用于过滤数据
                metrics_calc = StrategyMetrics(str(file), self.start_date, self.end_date)
                metrics = metrics_calc.calculate_metrics()
                strategy_name = metrics['name']
                self.all_metrics[strategy_name] = metrics
            except Exception as e:
                print(f"  警告: 加载 {file.name} 失败: {e}")

        print(f"成功加载 {len(self.all_metrics)} 个策略的指标")

        # 计算买入并持有基准
        self.buy_hold_metrics = self._calculate_buy_hold_benchmark()

        return self.all_metrics

    def calculate_quality_score(self, metrics: Dict) -> float:
        """
        计算质量评分

        公式:
        score = 0.5 × sharpe_normalized
              + 0.3 × return_normalized
              + 0.2 × drawdown_penalty

        其中:
        - sharpe_normalized = (sharpe - min_sharpe) / (max_sharpe - min_sharpe)
        - return_normalized = (return - min_return) / (max_return - min_return)
        - drawdown_penalty = 1 - abs(max_dd) / worst_dd

        Returns:
            质量评分 [0, 1]
        """
        if not self.all_metrics:
            self.load_all_strategies()

        if len(self.all_metrics) == 0:
            return 0.0

        # 提取所有策略的指标
        all_sharpes = [m['sharpe_ratio'] for m in self.all_metrics.values()]
        all_returns = [m['total_return'] for m in self.all_metrics.values()]
        all_drawdowns = [m['max_drawdown'] for m in self.all_metrics.values()]

        min_sharpe = min(all_sharpes)
        max_sharpe = max(all_sharpes)
        min_return = min(all_returns)
        max_return = max(all_returns)
        worst_dd = min(all_drawdowns)  # 最差回撤（最负）

        # 归一化
        sharpe = metrics['sharpe_ratio']
        ret = metrics['total_return']
        dd = metrics['max_drawdown']

        # 避免除零
        sharpe_norm = (sharpe - min_sharpe) / (max_sharpe - min_sharpe) if max_sharpe != min_sharpe else 0.5
        return_norm = (ret - min_return) / (max_return - min_return) if max_return != min_return else 0.5
        dd_penalty = 1 - abs(dd) / abs(worst_dd) if worst_dd != 0 else 0.5

        # 加权计算
        score = 0.5 * sharpe_norm + 0.3 * return_norm + 0.2 * dd_penalty

        return max(0.0, min(1.0, score))  # 限制在 [0, 1]

    def filter_strategies(
        self,
        min_sharpe: float = 0.5,
        min_return: float = 1.0,
        max_drawdown: float = -10.0,
        top_n: int = 20
    ) -> List[str]:
        """
        筛选高质量策略

        Args:
            min_sharpe: 最小夏普比率
            min_return: 最小收益率 %
            max_drawdown: 最大允许回撤 %（负数）
            top_n: 返回评分最高的N个策略

        Returns:
            策略名称列表，按质量评分降序排列
        """
        if not self.all_metrics:
            self.load_all_strategies()

        # 1. 应用硬性筛选条件
        filtered = {}
        for name, metrics in self.all_metrics.items():
            if (metrics['sharpe_ratio'] >= min_sharpe and
                metrics['total_return'] >= min_return and
                metrics['max_drawdown'] >= max_drawdown):  # max_drawdown 是负数，所以用 >=
                filtered[name] = metrics

        print(f"筛选条件: 夏普≥{min_sharpe}, 收益≥{min_return}%, 回撤≥{max_drawdown}%")
        print(f"找到 {len(self.all_metrics)} 个策略，通过筛选: {len(filtered)} 个\n")

        if len(filtered) == 0:
            print("警告: 没有策略通过筛选条件")
            return []

        # 2. 计算质量评分
        scored = []
        for name, metrics in filtered.items():
            score = self.calculate_quality_score(metrics)
            metrics['quality_score'] = score
            scored.append((name, metrics))

        # 3. 排序并返回前N个
        scored.sort(key=lambda x: x[1]['quality_score'], reverse=True)
        top_strategies = [name for name, _ in scored[:top_n]]

        return top_strategies

    def filter_strategies_by_return(
        self,
        min_return: float = 1.0,
        max_drawdown: float = -20.0,
        top_n: int = 20
    ) -> List[str]:
        """
        纯收益率筛选（无夏普要求，用于激进型投资者）

        与 filter_strategies() 的区别：
        - 不要求最小夏普比率
        - 按收益率排序（而非质量评分）
        - 放宽回撤要求（-20% vs -10%）

        Args:
            min_return: 最小收益率 %
            max_drawdown: 最大允许回撤 %（负数）
            top_n: 返回收益率最高的N个策略

        Returns:
            策略名称列表，按总收益率降序排列
        """
        if not self.all_metrics:
            self.load_all_strategies()

        # 硬性筛选条件（无夏普要求）
        filtered = {}
        for name, metrics in self.all_metrics.items():
            if (metrics['total_return'] >= min_return and
                metrics['max_drawdown'] >= max_drawdown):
                filtered[name] = metrics

        print(f"筛选条件: 收益≥{min_return}%, 回撤≥{max_drawdown}% (无夏普要求)")
        print(f"找到 {len(self.all_metrics)} 个策略，通过筛选: {len(filtered)} 个\n")

        if len(filtered) == 0:
            print("警告: 没有策略通过筛选条件")
            return []

        # 按总收益率降序排序
        sorted_strategies = sorted(
            filtered.items(),
            key=lambda x: x[1]['total_return'],
            reverse=True
        )

        # 返回前N个
        top_strategies = [name for name, _ in sorted_strategies[:top_n]]

        return top_strategies

    def print_return_quality_report(self, strategies: List[str]):
        """
        打印收益率筛选报告（按收益率降序）

        Args:
            strategies: 策略名称列表（应该是 filter_strategies_by_return 的结果）
        """
        if not self.all_metrics:
            self.load_all_strategies()

        print("=" * 140)
        print(f"策略收益率分析报告 - 激进模式（按收益率降序） ({self.start_date} 至 {self.end_date}, {self.timeframe})")
        print("=" * 140)

        # 打印买入并持有基准（简版）
        if self.buy_hold_metrics:
            bh = self.buy_hold_metrics
            print(f"\n【买入并持有基准】收益: {bh['total_return']:>7.2f}%  |  夏普: {bh['sharpe_ratio']:>5.2f}  |  "
                  f"波动率: {bh['volatility']:>6.2f}%  |  回撤: {bh['max_drawdown']:>7.2f}%\n")

        # 主表格：按收益率排序
        print(f"{'排名':<6} {'策略名称':<30} {'收益%':<10} {'夏普':<8} {'回撤%':<10} {'波动率%':<10}")
        print("-" * 140)

        for rank, name in enumerate(strategies, 1):
            if name in self.all_metrics:
                m = self.all_metrics[name]
                print(f"{rank:<6} {name:<30} {m['total_return']:<10.2f} {m['sharpe_ratio']:<8.2f} "
                      f"{m['max_drawdown']:<10.2f} {m['volatility']:<10.2f}")

        print("=" * 140)
        print()

    def print_quality_report(self, strategies: List[str], show_details: bool = True):
        """
        打印质量分析报告

        Args:
            strategies: 策略名称列表（应该是 filter_strategies 的结果）
            show_details: 是否显示详细指标
        """
        if not self.all_metrics:
            self.load_all_strategies()

        print("=" * 140)
        print(f"策略质量分析报告 ({self.start_date} 至 {self.end_date}, {self.timeframe})")
        print("=" * 140)

        # 打印买入并持有基准（详细版）
        if self.buy_hold_metrics:
            bh = self.buy_hold_metrics
            print(f"\n【买入并持有基准 - NQ期货】")
            print(f"  收益: {bh['total_return']:>7.2f}%  |  夏普: {bh['sharpe_ratio']:>5.2f}  |  "
                  f"波动率: {bh['volatility']:>6.2f}%  |  回撤: {bh['max_drawdown']:>7.2f}%")
            print(f"  胜率: {bh['win_rate']*100:>7.1f}%  |  盈亏比: {bh['profit_loss_ratio']:>4.2f}  |  "
                  f"最大连赢: {bh['max_consecutive_wins']:>3}天  |  最大连亏: {bh['max_consecutive_losses']:>3}天")
            print()

        # 主表格：核心指标
        print(f"{'排名':<6} {'策略名称':<30} {'夏普':<8} {'收益%':<10} {'波动率%':<10} {'胜率%':<8} {'回撤%':<10} {'质量分':<8}")
        print("-" * 140)

        for rank, name in enumerate(strategies, 1):
            if name in self.all_metrics:
                m = self.all_metrics[name]
                score = m.get('quality_score', self.calculate_quality_score(m))
                print(f"{rank:<6} {name:<30} {m['sharpe_ratio']:<8.2f} {m['total_return']:<10.2f} "
                      f"{m['volatility']:<10.2f} {m['win_rate']*100:<8.1f} {m['max_drawdown']:<10.2f} {score:<8.2f}")

        print("=" * 140)

        # 详细指标（可选）
        if show_details and len(strategies) > 0:
            print(f"\n详细风险指标（前5个策略）")
            print("-" * 140)
            print(f"{'策略名称':<30} {'盈亏比':<8} {'平均盈利%':<12} {'平均亏损%':<12} {'最大连赢':<10} {'最大连亏':<10}")
            print("-" * 140)

            for name in strategies[:5]:
                if name in self.all_metrics:
                    m = self.all_metrics[name]
                    print(f"{name:<30} {m['profit_loss_ratio']:<8.2f} {m['avg_win']:<12.3f} "
                          f"{m['avg_loss']:<12.3f} {m['max_consecutive_wins']:<10} {m['max_consecutive_losses']:<10}")

            print("=" * 140)

        print()

    def get_metrics(self, strategy_name: str) -> Dict:
        """
        获取指定策略的指标

        Args:
            strategy_name: 策略名称

        Returns:
            指标字典，如果策略不存在则返回 None
        """
        if not self.all_metrics:
            self.load_all_strategies()

        return self.all_metrics.get(strategy_name)

    def _calculate_buy_hold_benchmark(self) -> Dict:
        """
        计算买入并持有基准的表现

        Returns:
            包含买入并持有指标的字典
        """
        try:
            # 解析日期
            start_dt = pd.to_datetime(self.start_date, format='%Y%m%d')
            end_dt = pd.to_datetime(self.end_date, format='%Y%m%d')

            # 读取价格数据
            df = pd.read_csv(self.price_data_file)
            df['datetime'] = pd.to_datetime(df['ts_event'])
            df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]
            df = df.sort_values('datetime')

            if len(df) == 0:
                print("  警告: 买入并持有基准计算失败（无价格数据）")
                return None

            # 根据 timeframe 进行 resample
            df.set_index('datetime', inplace=True)

            # 映射 timeframe 到 pandas resample 规则
            resample_map = {
                'm1': '1T', 'm5': '5T', 'm15': '15T', 'm30': '30T',
                'h1': '1H', 'h4': '4H', 'd1': '1D'
            }

            resample_rule = resample_map.get(self.timeframe, '1D')
            df_resampled = df['close'].resample(resample_rule).last().dropna()

            if len(df_resampled) < 2:
                print("  警告: 买入并持有基准计算失败（数据点不足）")
                return None

            # 计算买入并持有的收益率
            prices = df_resampled.values
            initial_price = prices[0]
            final_price = prices[-1]

            total_return = (final_price / initial_price - 1) * 100  # 百分比

            # 计算每日收益率
            daily_returns = np.diff(prices) / prices[:-1]

            # 夏普比率（使用统一方法）
            sharpe_ratio = calculate_sharpe_ratio(daily_returns, annualize=True, periods_per_year=252)

            # 波动率
            volatility = daily_returns.std() * np.sqrt(252) * 100  # 百分比

            # 最大回撤
            cumulative = np.cumprod(1 + daily_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0.0

            # 胜率
            positive_days = (daily_returns > 0).sum()
            win_rate = positive_days / len(daily_returns) if len(daily_returns) > 0 else 0.0

            # 连续盈亏（需要创建临时 StrategyMetrics 实例来调用方法）
            # 直接在这里实现简单版本
            max_win_streak = 0
            max_loss_streak = 0
            current_win_streak = 0
            current_loss_streak = 0

            for ret in daily_returns:
                if ret > 0:
                    current_win_streak += 1
                    current_loss_streak = 0
                    max_win_streak = max(max_win_streak, current_win_streak)
                elif ret < 0:
                    current_loss_streak += 1
                    current_win_streak = 0
                    max_loss_streak = max(max_loss_streak, current_loss_streak)

            # 盈亏比
            win_returns = daily_returns[daily_returns > 0]
            loss_returns = daily_returns[daily_returns < 0]
            avg_win = win_returns.mean() if len(win_returns) > 0 else 0.0
            avg_loss = abs(loss_returns.mean()) if len(loss_returns) > 0 else 0.0
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

            print(f"  买入并持有基准: 收益 {total_return:.2f}%, 夏普 {sharpe_ratio:.2f}, 波动率 {volatility:.2f}%, "
                  f"胜率 {win_rate*100:.1f}%, 盈亏比 {profit_loss_ratio:.2f}\n")

            return {
                'name': 'Buy & Hold',
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'win_rate': win_rate,
                'max_consecutive_wins': max_win_streak,
                'max_consecutive_losses': max_loss_streak,
                'profit_loss_ratio': profit_loss_ratio,
                'avg_win': avg_win * 100,
                'avg_loss': avg_loss * 100,
                'num_days': len(daily_returns)
            }

        except Exception as e:
            print(f"  警告: 买入并持有基准计算失败: {e}\n")
            return None


def main():
    """测试用例"""
    import argparse

    parser = argparse.ArgumentParser(description='策略质量筛选器')
    parser.add_argument('--start', required=True, help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end', required=True, help='结束日期 (YYYYMMDD)')
    parser.add_argument('--timeframe', default='d1', help='时间周期')
    parser.add_argument('--min-sharpe', type=float, default=0.5, help='最小夏普比率')
    parser.add_argument('--min-return', type=float, default=1.0, help='最小收益率%')
    parser.add_argument('--max-drawdown', type=float, default=-10.0, help='最大回撤%')
    parser.add_argument('--top-n', type=int, default=20, help='选择质量评分前N个')
    parser.add_argument('--results-dir', default='backtest_results', help='结果目录')

    args = parser.parse_args()

    # 创建筛选器
    filter = QualityFilter(
        results_dir=args.results_dir,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end
    )

    # 筛选高质量策略
    high_quality = filter.filter_strategies(
        min_sharpe=args.min_sharpe,
        min_return=args.min_return,
        max_drawdown=args.max_drawdown,
        top_n=args.top_n
    )

    # 打印报告
    filter.print_quality_report(high_quality)

    print(f"高质量策略列表: {high_quality}")


if __name__ == '__main__':
    main()
