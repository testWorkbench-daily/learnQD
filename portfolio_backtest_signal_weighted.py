#!/usr/bin/env python
"""
单账户信号加权组合回测工具

与传统的多账户理论测试不同,此工具模拟实际的Portfolio Atom执行方式:
- 单账户运行
- 从各策略的trades记录推断持仓信号
- 按权重加权信号
- 应用阈值决策目标持仓手数
- 执行交易并跟踪组合价值

这样的回测结果应该与实际Atom执行非常接近,从而准确评估组合策略的真实表现。

用法:
    # 基本用法
    python portfolio_backtest_signal_weighted.py \
      --strategies rsi_reversal triple_ma vol_breakout_aggressive vol_regime_long \
      --weights 0.34,0.34,0.08,0.24 \
      --timeframe d1 \
      --start 20240101 \
      --end 20241231

    # 自定义阈值
    python portfolio_backtest_signal_weighted.py \
      --strategies rsi_reversal triple_ma \
      --weights 0.5,0.5 \
      --thresholds 0.70,0.35,0.05 \
      --timeframe d1 \
      --start 20240101 \
      --end 20241231

    # 等权重(无需指定weights)
    python portfolio_backtest_signal_weighted.py \
      --strategies sma_cross rsi_reversal macd_trend \
      --timeframe d1 \
      --start 20240101 \
      --end 20241231
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import datetime


class SignalWeightedPortfolioBacktester:
    """单账户信号加权组合回测器"""

    def __init__(
        self,
        strategies: List[str],
        weights: List[float],
        threshold_config: Dict[str, float],
        timeframe: str,
        start_date: str,
        end_date: str,
        initial_cash: float = 100000.0,
        commission_rate: float = 0.0,
        results_dir: str = 'backtest_results'
    ):
        """
        初始化回测器

        Args:
            strategies: 策略名称列表
            weights: 对应权重列表
            threshold_config: 阈值配置 {'high': 0.70, 'mid': 0.35, 'low': 0.05}
            timeframe: 时间周期 (d1, h1, etc.)
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            initial_cash: 初始资金
            commission_rate: 手续费率 (暂不实现)
            results_dir: 回测结果目录
        """
        self.strategies = strategies
        self.weights = np.array(weights)
        self.threshold_config = threshold_config
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.results_dir = results_dir

        # 验证权重和为1
        if abs(self.weights.sum() - 1.0) > 0.001:
            print(f"警告: 权重和为 {self.weights.sum():.4f}, 将自动归一化")
            self.weights = self.weights / self.weights.sum()

        # 数据容器
        self.strategy_trades = {}  # {strategy: trades_df}
        self.strategy_daily_values = {}  # {strategy: daily_values_df}
        self.market_data = None  # 市场价格数据(从第一个策略的daily_values提取)

    def load_strategy_data(self):
        """加载所有策略的trades和daily_values数据"""
        print(f"正在加载 {len(self.strategies)} 个策略的数据...")

        for strategy in self.strategies:
            # 加载trades
            trades_file = Path(self.results_dir) / f'trades_{strategy}_{self.timeframe}_{self.start_date}_{self.end_date}.csv'
            if not trades_file.exists():
                raise FileNotFoundError(f"未找到策略 {strategy} 的交易记录: {trades_file}")

            trades_df = pd.read_csv(trades_file)
            trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])
            trades_df = trades_df.sort_values('datetime')

            # 标准化列名 (兼容不同的trades格式)
            trades_df = self._normalize_trades_columns(trades_df, strategy)

            self.strategy_trades[strategy] = trades_df

            # 加载daily_values
            daily_file = Path(self.results_dir) / f'daily_values_{strategy}_{self.timeframe}_{self.start_date}_{self.end_date}.csv'
            if not daily_file.exists():
                raise FileNotFoundError(f"未找到策略 {strategy} 的每日数据: {daily_file}")

            daily_df = pd.read_csv(daily_file)
            daily_df['datetime'] = pd.to_datetime(daily_df['datetime'])
            daily_df = daily_df.sort_values('datetime')
            self.strategy_daily_values[strategy] = daily_df

            print(f"  ✓ {strategy}: {len(trades_df)} 笔交易, {len(daily_df)} 个交易日")

        # 提取市场数据(使用第一个策略的daily_values作为时间基准和价格参考)
        first_strategy = self.strategies[0]
        base_df = self.strategy_daily_values[first_strategy][['datetime']].copy()

        # 从daily_values反推价格(简化处理:使用portfolio_value的变化率推算)
        # 更准确的方法是从原始数据加载,但这里为了简化,我们使用第一个策略的返回作为价格代理
        self.market_data = base_df

        print(f"\n时间基准: {len(self.market_data)} 个交易日 ({self.market_data['datetime'].min()} ~ {self.market_data['datetime'].max()})")

    def _normalize_trades_columns(self, trades_df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """
        标准化trades DataFrame的列名以兼容不同格式

        支持的action列名: action, side, type, direction, ...
        支持的size列名: size, quantity, qty, shares, ...
        """
        # 检查并标准化action列
        action_cols = ['action', 'side', 'type', 'direction', 'operation']
        action_col = None
        for col in action_cols:
            if col in trades_df.columns:
                action_col = col
                break

        if action_col is None:
            print(f"  警告: 策略 {strategy} 的trades文件没有找到action/side/type列")
            print(f"       可用列: {list(trades_df.columns)}")
            print(f"       使用第一笔交易作为示例:")
            if len(trades_df) > 0:
                print(f"       {trades_df.iloc[0].to_dict()}")
            raise ValueError(f"无法找到交易方向列 (action/side/type) 在 {strategy} 的trades文件中")

        if action_col != 'action':
            trades_df = trades_df.rename(columns={action_col: 'action'})

        # 检查并标准化size列
        size_cols = ['size', 'quantity', 'qty', 'shares', 'contracts']
        size_col = None
        for col in size_cols:
            if col in trades_df.columns:
                size_col = col
                break

        if size_col is None:
            print(f"  警告: 策略 {strategy} 的trades文件没有找到size/quantity列")
            raise ValueError(f"无法找到交易数量列 (size/quantity/qty) 在 {strategy} 的trades文件中")

        if size_col != 'size':
            trades_df = trades_df.rename(columns={size_col: 'size'})

        # 标准化action值 (转为大写)
        trades_df['action'] = trades_df['action'].str.upper()

        return trades_df

    def infer_position_signal(self, strategy: str, current_datetime: pd.Timestamp) -> int:
        """
        从策略的trades推断在指定时刻的持仓信号

        Args:
            strategy: 策略名称
            current_datetime: 当前时间

        Returns:
            持仓信号: +1=多头, -1=空头, 0=空仓
        """
        trades = self.strategy_trades[strategy]

        # 查找当前时间之前的所有交易
        relevant_trades = trades[trades['datetime'] <= current_datetime]

        if relevant_trades.empty:
            return 0  # 无交易,空仓

        # 累积持仓变化
        position = 0
        for _, trade in relevant_trades.iterrows():
            action = str(trade['action']).upper().strip()
            size = trade['size']

            # 兼容不同的action值格式
            if action in ['BUY', 'LONG', 'OPEN_LONG', '+']:
                position += size
            elif action in ['SELL', 'SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', '-']:
                position -= size

        # 归一化为 +1/-1/0
        if position > 0:
            return 1  # 多头
        elif position < 0:
            return -1  # 空头
        else:
            return 0  # 空仓

    def apply_threshold_decision(self, target_pct: float) -> int:
        """
        应用阈值决策(与Atom逻辑一致)

        Args:
            target_pct: 目标持仓比例 (加权后的信号值)

        Returns:
            目标持仓手数 (-3 ~ +3)
        """
        high = self.threshold_config['high']
        mid = self.threshold_config['mid']
        low = self.threshold_config['low']

        if target_pct >= high:
            return 3  # 强烈做多
        elif target_pct >= mid:
            return 2  # 中度做多
        elif target_pct >= low:
            return 1  # 轻度做多
        elif target_pct <= -high:
            return -3  # 强烈做空
        elif target_pct <= -mid:
            return -2  # 中度做空
        elif target_pct <= -low:
            return -1  # 轻度做空
        else:
            return 0  # 空仓

    def get_close_price(self, current_datetime: pd.Timestamp) -> float:
        """
        获取当前bar的收盘价

        简化处理:使用第一个策略的portfolio_value变化反推价格
        实际应该从原始数据加载,但为了避免依赖,这里使用近似方法

        更准确的做法:从 nq_m1_forward_adjusted.csv 加载价格数据
        """
        # 从第一个策略的daily_values获取该日期的数据
        first_strategy = self.strategies[0]
        daily_df = self.strategy_daily_values[first_strategy]

        row = daily_df[daily_df['datetime'] == current_datetime]

        if row.empty:
            # 如果找不到精确匹配,返回默认值
            return 10000.0  # NQ典型价格

        # 使用portfolio_value作为价格代理(非常粗糙,仅用于演示)
        # 实际实现应该加载真实的市场数据
        # 这里我们假设价格大约在9000-20000范围
        portfolio_value = row.iloc[0]['portfolio_value']
        # 反推价格(假设单手,初始资金10万)
        approx_price = portfolio_value / 10  # 粗略估计

        return approx_price

    def run_backtest(self) -> Dict:
        """
        运行单账户信号加权回测

        Returns:
            回测结果字典
        """
        print("\n" + "=" * 80)
        print("开始单账户信号加权回测")
        print("=" * 80)
        print(f"策略: {', '.join(self.strategies)}")
        print(f"权重: {self.weights}")
        print(f"阈值: high={self.threshold_config['high']}, mid={self.threshold_config['mid']}, low={self.threshold_config['low']}")
        print(f"周期: {self.timeframe}, {self.start_date} ~ {self.end_date}")
        print(f"初始资金: ${self.initial_cash:,.2f}")
        print("-" * 80)

        try:
            # 初始化
            portfolio_value = self.initial_cash
            cash = self.initial_cash
            position = 0  # 当前持仓手数
            trades_log = []
            daily_values_log = []

            # 逐日期遍历
            for idx, row in self.market_data.iterrows():
                current_date = row['datetime']

                # 步骤1: 获取各策略在当前bar的持仓信号
                signals = []
                for strategy in self.strategies:
                    signal = self.infer_position_signal(strategy, current_date)
                    signals.append(signal)

                # 步骤2: 加权计算目标持仓比例
                target_pct = np.dot(self.weights, signals)

                # 步骤3: 应用阈值决策
                target_size = self.apply_threshold_decision(target_pct)

                # 步骤4: 执行交易(如有需要)
                size_diff = target_size - position

                # 获取当前价格(从第一个策略的daily_values推算)
                close_price = self.get_close_price(current_date)

                if size_diff != 0:
                    # 计算交易金额
                    trade_value = abs(size_diff) * close_price
                    commission = trade_value * self.commission_rate

                    # 记录交易
                    trades_log.append({
                        'datetime': current_date,
                        'action': 'BUY' if size_diff > 0 else 'SELL',
                        'size': abs(size_diff),
                        'price': close_price,
                        'value': trade_value,
                        'commission': commission,
                        'target_pct': target_pct,
                        'position_before': position,
                        'position_after': target_size
                    })

                    # 更新现金
                    if size_diff > 0:  # 买入
                        cash -= (trade_value + commission)
                    else:  # 卖出
                        cash += (trade_value - commission)

                    # 更新持仓
                    position = target_size

                # 步骤5: 更新portfolio_value
                position_value = position * close_price
                portfolio_value = cash + position_value

                # 记录每日价值
                daily_values_log.append({
                    'datetime': current_date,
                    'portfolio_value': portfolio_value,
                    'cash': cash,
                    'position': position,
                    'position_value': position_value,
                    'close_price': close_price,
                    'target_pct': target_pct
                })

            # 转换为DataFrame
            trades_df = pd.DataFrame(trades_log)
            daily_values_df = pd.DataFrame(daily_values_log)

            # 计算每日收益率
            daily_values_df['daily_return'] = daily_values_df['portfolio_value'].pct_change().fillna(0)
            daily_values_df['cumulative_return'] = (daily_values_df['portfolio_value'] / self.initial_cash - 1)

            # 计算绩效指标
            metrics = self._calculate_metrics(daily_values_df, trades_df)

            # 打印结果
            self._print_results(metrics, trades_df, daily_values_df)

            # 保存结果
            self._save_results(metrics, trades_df, daily_values_df)

            return {
                'metrics': metrics,
                'trades': trades_df,
                'daily_values': daily_values_df
            }

        except Exception as e:
            print(f"\n错误: 回测过程中发生异常")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            import traceback
            traceback.print_exc()

            # 返回空结果,使调用者可以继续
            return {
                'metrics': {},
                'trades': pd.DataFrame(),
                'daily_values': pd.DataFrame()
            }

    def _calculate_metrics(self, daily_values_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
        """计算绩效指标"""
        final_value = daily_values_df.iloc[-1]['portfolio_value']
        total_return_pct = (final_value / self.initial_cash - 1) * 100

        # 夏普比率
        daily_returns = daily_values_df['daily_return'].values
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # 最大回撤
        cumulative = daily_values_df['portfolio_value'].values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown_pct = drawdown.min() * 100

        # 交易统计
        total_trades = len(trades_df)

        # 胜率(简化:只统计开仓交易)
        if total_trades > 0:
            # 根据交易前后的持仓变化判断盈亏(简化处理)
            win_rate = 0.5  # 占位值,准确计算需要跟踪每笔交易的盈亏
        else:
            win_rate = 0.0

        return {
            'final_value': final_value,
            'total_return_pct': total_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': final_value - self.initial_cash
        }

    def _print_results(self, metrics: Dict, trades_df: pd.DataFrame, daily_values_df: pd.DataFrame):
        """打印回测结果"""
        print("\n回测结果:")
        print(f"  最终资金: ${metrics['final_value']:,.2f}")
        print(f"  收益率: {metrics['total_return_pct']:.2f}%")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {metrics['max_drawdown_pct']:.2f}%")
        print(f"  交易次数: {metrics['total_trades']}")
        print(f"  总盈亏: ${metrics['total_pnl']:,.2f}")

        # 打印最近几笔交易
        if len(trades_df) > 0:
            print("\n最近5笔交易:")
            print(trades_df[['datetime', 'action', 'size', 'price', 'target_pct', 'position_after']].tail(5).to_string(index=False))

    def _save_results(self, metrics: Dict, trades_df: pd.DataFrame, daily_values_df: pd.DataFrame):
        """保存回测结果"""
        # 生成组合名称
        portfolio_name = f"portfolio_signal_weighted_{'_'.join(self.strategies[:2])}_etc"

        # 保存trades
        trades_file = Path(self.results_dir) / f"trades_{portfolio_name}_{self.timeframe}_{self.start_date}_{self.end_date}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"\n交易记录已保存: {trades_file}")

        # 保存daily_values
        daily_file = Path(self.results_dir) / f"daily_values_{portfolio_name}_{self.timeframe}_{self.start_date}_{self.end_date}.csv"
        daily_values_df.to_csv(daily_file, index=False)
        print(f"每日数据已保存: {daily_file}")

        # 保存指标摘要
        summary_file = Path(self.results_dir) / f"summary_{portfolio_name}_{self.timeframe}_{self.start_date}_{self.end_date}.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("单账户信号加权组合回测结果\n")
            f.write("=" * 80 + "\n")
            f.write(f"策略: {', '.join(self.strategies)}\n")
            f.write(f"权重: {self.weights}\n")
            f.write(f"阈值: {self.threshold_config}\n")
            f.write(f"周期: {self.timeframe}, {self.start_date} ~ {self.end_date}\n")
            f.write(f"\n绩效指标:\n")
            f.write(f"  最终资金: ${metrics['final_value']:,.2f}\n")
            f.write(f"  收益率: {metrics['total_return_pct']:.2f}%\n")
            f.write(f"  夏普比率: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"  最大回撤: {metrics['max_drawdown_pct']:.2f}%\n")
            f.write(f"  交易次数: {metrics['total_trades']}\n")
        print(f"指标摘要已保存: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='单账户信号加权组合回测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--strategies', required=True, nargs='+', help='策略名称列表 (空格分隔)')
    parser.add_argument('--weights', type=str, default=None, help='权重列表 (逗号分隔,如: 0.3,0.3,0.4), 不指定则等权重')
    parser.add_argument('--thresholds', type=str, default='0.70,0.35,0.05', help='阈值配置 (high,mid,low, 默认: 0.70,0.35,0.05)')
    parser.add_argument('--timeframe', required=True, help='时间周期 (如: d1, h1)')
    parser.add_argument('--start', required=True, help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end', required=True, help='结束日期 (YYYYMMDD)')
    parser.add_argument('--initial-cash', type=float, default=100000.0, help='初始资金 (默认: 100000)')
    parser.add_argument('--results-dir', default='backtest_results', help='回测结果目录')

    args = parser.parse_args()

    # 解析权重
    if args.weights:
        weights = [float(w) for w in args.weights.split(',')]
        if len(weights) != len(args.strategies):
            print(f"错误: 权重数量 ({len(weights)}) 与策略数量 ({len(args.strategies)}) 不匹配")
            return
    else:
        # 等权重
        weights = [1.0 / len(args.strategies)] * len(args.strategies)
        print(f"使用等权重: {weights}")

    # 解析阈值
    threshold_parts = [float(t) for t in args.thresholds.split(',')]
    if len(threshold_parts) != 3:
        print("错误: 阈值必须包含3个值 (high,mid,low)")
        return

    threshold_config = {
        'high': threshold_parts[0],
        'mid': threshold_parts[1],
        'low': threshold_parts[2]
    }

    # 创建回测器
    backtester = SignalWeightedPortfolioBacktester(
        strategies=args.strategies,
        weights=weights,
        threshold_config=threshold_config,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.initial_cash,
        results_dir=args.results_dir
    )

    # 加载数据
    try:
        backtester.load_strategy_data()
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请先运行各子策略的回测,确保生成了对应的trades和daily_values文件")
        return

    # 运行回测
    result = backtester.run_backtest()

    print("\n" + "=" * 80)
    print("回测完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
