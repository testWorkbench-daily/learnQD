#!/usr/bin/env python
"""
策略组合回测工具

用法:
    # 回测推荐的策略组合
    python portfolio_backtest.py --portfolio-file backtest_results/recommended_portfolios_d1_20240101_20241231.csv

    # 指定时间范围和数据
    python portfolio_backtest.py --portfolio-file xxx.csv --start 2024-01-01 --end 2024-12-31

    # 指定要回测的组合ID
    python portfolio_backtest.py --portfolio-file xxx.csv --portfolio-id 1
"""
import argparse
import datetime
import time
import pandas as pd
import backtrader as bt
import numpy as np
from typing import List, Dict, Type
from bt_base import StrategyAtom, BaseStrategy
from bt_runner import Runner
from bt_main import ATOMS


class PortfolioStrategyAtom(StrategyAtom):
    """
    策略组合原子

    同时运行多个策略，按权重分配资金
    """

    def __init__(self, strategies: List[str], weights: List[float] = None, name: str = None):
        """
        初始化组合策略

        Args:
            strategies: 策略名称列表
            weights: 权重列表（如果为None则等权重）
            name: 组合名称
        """
        self.strategies = strategies
        if weights is None:
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            self.weights = weights

        # 验证权重和为1
        total_weight = sum(self.weights)
        if abs(total_weight - 1.0) > 0.001:
            # 归一化权重
            self.weights = [w / total_weight for w in self.weights]

        # 设置组合名称
        if name:
            self.name = name
        else:
            self.name = f"portfolio_{'+'.join(strategies)}"

        self.params = {}

        # 获取子策略的Atom实例
        self.sub_atoms = []
        for strategy_name in strategies:
            if strategy_name not in ATOMS:
                raise ValueError(f"策略 '{strategy_name}' 不存在于ATOMS注册表中")
            atom_cls = ATOMS[strategy_name]
            self.sub_atoms.append(atom_cls())

    def strategy_cls(self) -> Type[bt.Strategy]:
        """返回组合策略类"""
        sub_atoms = self.sub_atoms
        weights = self.weights

        class PortfolioStrategy(BaseStrategy):
            """
            组合策略：同时运行多个子策略，按权重分配资金
            """
            params = (
                ('printlog', False),
            )

            def __init__(self):
                super().__init__()
                self.sub_strategies = []
                self.sub_strategy_data = []

                # 为每个子策略创建实例
                for atom in sub_atoms:
                    strategy_cls = atom.strategy_cls()
                    # 创建子策略实例（使用相同的数据）
                    sub_strat = strategy_cls()
                    # 将子策略的数据属性复制过来
                    sub_strat.datas = self.datas
                    sub_strat.data = self.data
                    sub_strat.broker = self.broker
                    sub_strat.env = self.env

                    self.sub_strategies.append(sub_strat)

                    # 存储子策略的信号
                    self.sub_strategy_data.append({
                        'name': atom.name,
                        'weight': 0.0,
                        'signal': 0,  # 1=买入, -1=卖出, 0=无信号
                        'position': 0,  # 子策略的虚拟持仓
                    })

            def next(self):
                if self.order:
                    return

                # 收集所有子策略的信号
                current_portfolio_value = self.broker.getvalue()

                for i, sub_strat in enumerate(self.sub_strategies):
                    # 调用子策略的next方法（但不真正下单）
                    # 我们通过检查子策略的内部状态来判断信号

                    # 简化处理：根据子策略的position判断
                    # 这是一个简化的实现，实际上应该让子策略输出信号
                    pass

                # 组合策略的简化逻辑：
                # 按权重计算每个子策略应该持有的仓位
                # 然后汇总计算整体仓位

                # 目前采用简化方案：
                # 1. 让第一个子策略作为主导策略（权重最大的）
                # 2. 其他策略作为过滤条件

                # 为了正确实现，我们需要让每个子策略独立运行
                # 但这需要更复杂的架构

                # 简化实现：使用加权投票
                buy_votes = 0
                sell_votes = 0

                for i, (sub_strat, sub_data) in enumerate(zip(self.sub_strategies, self.sub_strategy_data)):
                    weight = weights[i]

                    # 检查子策略的指标状态
                    # 这里需要根据具体的策略类型来判断
                    # 简化处理：如果子策略没有持仓且满足买入条件，则投买入票

                    # 由于backtrader的限制，子策略无法直接访问
                    # 我们采用另一种方案：直接复制每个子策略的指标逻辑

                    # TODO: 这需要更好的设计
                    pass

                # 暂时使用简单的等权重策略
                # 如果多数子策略看涨则买入，多数看跌则卖出

        return PortfolioStrategy

    def sizer_cls(self):
        """使用固定1手"""
        return None


class SimplePortfolioAtom(StrategyAtom):
    """
    简单组合策略原子

    使用简单的方法：对每个子策略的daily_values加权平均，而不是真正运行多个策略
    这样可以避免backtrader多策略的复杂性
    """

    def __init__(self, strategies: List[str], weights: List[float] = None, name: str = None):
        self.strategies = strategies
        if weights is None:
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            self.weights = weights

        total_weight = sum(self.weights)
        if abs(total_weight - 1.0) > 0.001:
            self.weights = [w / total_weight for w in self.weights]

        if name:
            self.name = name
        else:
            self.name = f"portfolio_{'_'.join(strategies[:2])}_etc"

        self.params = {}

    def strategy_cls(self) -> Type[bt.Strategy]:
        """返回一个不交易的策略（用于占位）"""

        class DummyStrategy(BaseStrategy):
            def __init__(self):
                super().__init__()

            def next(self):
                # 不进行任何交易
                pass

        return DummyStrategy


def load_portfolio_combinations(csv_file: str) -> pd.DataFrame:
    """
    加载策略组合CSV文件

    Args:
        csv_file: CSV文件路径

    Returns:
        DataFrame包含组合信息
    """
    df = pd.read_csv(csv_file)
    return df


def backtest_portfolio_from_daily_values(
    strategies: List[str],
    weights: List[float],
    start_date: str,
    end_date: str,
    timeframe: str,
    results_dir: str = 'backtest_results'
) -> Dict:
    """
    基于已有的daily_values数据计算组合绩效

    这个方法不需要重新运行回测，只需要加载各策略的daily_values并加权平均

    Args:
        strategies: 策略名称列表
        weights: 权重列表
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        timeframe: 时间周期
        results_dir: 回测结果目录

    Returns:
        包含组合绩效的字典
    """
    load_start = time.perf_counter()

    # 加载各策略的daily_values
    daily_values_list = []
    for strategy in strategies:
        filename = f'{results_dir}/daily_values_{strategy}_{timeframe}_{start_date}_{end_date}.csv'
        try:
            df = pd.read_csv(filename)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            daily_values_list.append(df)
        except FileNotFoundError:
            raise FileNotFoundError(f"未找到策略 {strategy} 的数据文件: {filename}\n请先运行该策略的回测")

    load_time = time.perf_counter() - load_start

    # 确保所有策略的日期对齐
    # 使用第一个策略的日期作为基准
    calc_start = time.perf_counter()

    base_dates = daily_values_list[0]['datetime'].values

    # 提取每个策略的cumulative_return
    returns_matrix = []
    for df in daily_values_list:
        # 确保日期对齐
        df_aligned = df[df['datetime'].isin(base_dates)].copy()
        df_aligned = df_aligned.sort_values('datetime')
        returns_matrix.append(df_aligned['cumulative_return'].values)

    returns_matrix = np.array(returns_matrix)

    # 计算组合的累积收益率（加权平均）
    portfolio_cum_returns = np.dot(weights, returns_matrix)

    # 计算每日收益率
    portfolio_values = (1 + portfolio_cum_returns) * 100000  # 假设初始资金10万
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)  # 第一天收益率为0

    # 计算绩效指标
    initial_value = 100000
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_value - 1) * 100

    # 夏普比率（年化）
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # 最大回撤
    cumulative = portfolio_values
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # 由于是组合策略，没有真实交易记录
    # 但我们可以估算：如果每个子策略都有交易，组合的交易次数约为各策略之和
    total_trades = 0
    for strategy in strategies:
        trades_file = f'{results_dir}/trades_{strategy}_{timeframe}_{start_date}_{end_date}.csv'
        try:
            trades_df = pd.read_csv(trades_file)
            total_trades += len(trades_df)
        except:
            pass

    calc_time = time.perf_counter() - calc_start

    # 打印性能日志（仅在耗时较长时）
    total_time = load_time + calc_time
    if total_time > 0.5:
        print(f'      ├─ 数据加载: {load_time:.3f}秒')
        print(f'      └─ 指标计算: {calc_time:.3f}秒')

    result = {
        'name': f"portfolio_{len(strategies)}strats",
        'final_value': final_value,
        'return_pct': total_return,
        'sharpe': sharpe_ratio,
        'max_dd': max_drawdown,
        'total_trades': total_trades,
        'win_rate': 0.0,  # 组合策略无法直接计算胜率
        'total_pnl': final_value - initial_value,
        'strategies': strategies,
        'weights': weights,
    }

    return result


def save_portfolio_daily_values(
    strategies: List[str],
    weights: List[float],
    start_date: str,
    end_date: str,
    timeframe: str,
    portfolio_name: str,
    results_dir: str = 'backtest_results'
):
    """
    计算并保存组合的每日价值数据
    """
    # 加载各策略的daily_values
    daily_values_list = []
    for strategy in strategies:
        filename = f'{results_dir}/daily_values_{strategy}_{timeframe}_{start_date}_{end_date}.csv'
        df = pd.read_csv(filename)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        daily_values_list.append(df)

    # 对齐日期
    base_df = daily_values_list[0][['datetime']].copy()

    # 计算组合的portfolio_value
    portfolio_values = np.zeros(len(base_df))
    for i, (df, weight) in enumerate(zip(daily_values_list, weights)):
        df_aligned = df[df['datetime'].isin(base_df['datetime'])].copy()
        df_aligned = df_aligned.sort_values('datetime')
        portfolio_values += df_aligned['portfolio_value'].values * weight

    # 计算daily_return和cumulative_return
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)

    initial_value = portfolio_values[0]
    cumulative_returns = (portfolio_values - initial_value) / initial_value

    # 创建DataFrame
    result_df = base_df.copy()
    result_df['portfolio_value'] = portfolio_values
    result_df['daily_return'] = daily_returns
    result_df['cumulative_return'] = cumulative_returns

    # 保存
    output_file = f'{results_dir}/daily_values_{portfolio_name}_{timeframe}_{start_date}_{end_date}.csv'
    result_df.to_csv(output_file, index=False)
    print(f'每日价值已保存: {output_file}')


def main():
    parser = argparse.ArgumentParser(description='策略组合回测工具')
    parser.add_argument('--portfolio-file', required=True, help='策略组合CSV文件路径')
    parser.add_argument('--portfolio-id', type=int, default=None, help='指定要回测的组合ID（不指定则回测所有）')
    parser.add_argument('--start', default='2024-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--timeframe', default='d1', help='时间周期 (默认: d1)')
    parser.add_argument('--results-dir', default='backtest_results', help='回测结果目录')

    args = parser.parse_args()

    # 转换日期格式
    start_dt = datetime.datetime.strptime(args.start, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(args.end, '%Y-%m-%d')
    start_str = start_dt.strftime('%Y%m%d')
    end_str = end_dt.strftime('%Y%m%d')

    # 加载组合配置
    print(f"正在加载策略组合配置: {args.portfolio_file}")
    portfolios_df = load_portfolio_combinations(args.portfolio_file)

    if args.portfolio_id:
        portfolios_df = portfolios_df[portfolios_df['portfolio_id'] == args.portfolio_id]
        if portfolios_df.empty:
            print(f"错误: 未找到ID为 {args.portfolio_id} 的组合")
            return

    print(f"共 {len(portfolios_df)} 个组合待回测\n")

    all_results = []

    for idx, row in portfolios_df.iterrows():
        portfolio_id = row['portfolio_id']
        strategies = row['strategies'].split(',')
        num_strategies = len(strategies)
        equal_weight = 1.0 / num_strategies
        weights = [equal_weight] * num_strategies

        portfolio_name = f"portfolio_{portfolio_id}"

        print("=" * 80)
        print(f"回测组合 {portfolio_id}: {strategies}")
        print(f"权重: {weights}")
        print("-" * 80)

        try:
            # 基于daily_values计算组合绩效
            result = backtest_portfolio_from_daily_values(
                strategies=strategies,
                weights=weights,
                start_date=start_str,
                end_date=end_str,
                timeframe=args.timeframe,
                results_dir=args.results_dir
            )

            # 打印结果（格式与bt_runner.py一致）
            print(f'\n回测结果:')
            print(f'  最终资金: ${result["final_value"]:,.2f}')
            print(f'  收益率: {result["return_pct"]:.2f}%')
            print(f'  夏普比率: {result["sharpe"]:.2f}')
            print(f'  最大回撤: {result["max_dd"]:.2f}%')
            print(f'  交易次数: {result["total_trades"]} (所有子策略之和)')
            print(f'  胜率: N/A (组合策略)')

            # 保存组合的每日价值数据
            save_portfolio_daily_values(
                strategies=strategies,
                weights=weights,
                start_date=start_str,
                end_date=end_str,
                timeframe=args.timeframe,
                portfolio_name=portfolio_name,
                results_dir=args.results_dir
            )

            all_results.append(result)

        except Exception as e:
            print(f"错误: 组合 {portfolio_id} 回测失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 打印汇总对比
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("组合策略对比")
        print("=" * 80)
        print(f"{'组合名称':<25} {'收益率':>10} {'夏普':>8} {'回撤':>8} {'策略数':>8}")
        print("-" * 80)
        for r in all_results:
            strat_count = len(r['strategies'])
            print(f"{r['name']:<25} {r['return_pct']:>9.2f}% {r['sharpe']:>8.2f} {r['max_dd']:>7.2f}% {strat_count:>8}")

    print("\n回测完成!")


if __name__ == '__main__':
    main()
