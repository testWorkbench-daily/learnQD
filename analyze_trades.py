#!/usr/bin/env python
"""
交易盈亏分析工具

从trades CSV文件中分析交易盈亏比及相关指标。

用法:
    # 方式1: 直接指定文件
    python analyze_trades.py --file backtest_results/trades_sma_cross_d1_20240101_20241231.csv

    # 方式2: 通过参数组合自动定位文件
    python analyze_trades.py --strategy sma_cross --timeframe d1 --start 20240101 --end 20241231

    # 方式3: 指定results目录
    python analyze_trades.py --strategy sma_cross --timeframe d1 --start 20240101 --end 20241231 --results-dir backtest_results
"""
import argparse
import os
import pandas as pd
from typing import Dict, Optional, Tuple


class TradeAnalyzer:
    """交易分析器"""

    def __init__(self, trades_file: str):
        """
        初始化分析器

        Args:
            trades_file: 交易CSV文件路径
        """
        self.trades_file = trades_file
        self.df = None
        self.metrics = {}

    def load_trades(self) -> bool:
        """
        加载交易数据

        Returns:
            是否成功加载
        """
        try:
            if not os.path.exists(self.trades_file):
                print(f"错误: 文件不存在 - {self.trades_file}")
                return False

            self.df = pd.read_csv(self.trades_file)

            # 验证必要列
            required_cols = ['type', 'pnl']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                print(f"错误: 缺少必要列 - {missing_cols}")
                return False

            print(f"✓ 已加载 {len(self.df)} 条交易记录")
            return True
        except Exception as e:
            print(f"错误: 加载交易数据失败 - {e}")
            return False

    def calculate_metrics(self) -> Dict[str, float]:
        """
        计算盈亏比相关指标

        Returns:
            包含各种指标的字典
        """
        if self.df is None or len(self.df) == 0:
            return {}

        # 筛选出SELL订单（完成的交易）
        trades = self.df[self.df['type'] == 'SELL'].copy()

        if len(trades) == 0:
            print("警告: 没有找到已完成的交易（SELL订单）")
            return {}

        # 分离盈利和亏损交易
        winning_trades = trades[trades['pnl'] > 0]['pnl'].values
        losing_trades = trades[trades['pnl'] < 0]['pnl'].values
        break_even_trades = trades[trades['pnl'] == 0]['pnl'].values

        total_trades = len(trades)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        num_break_even = len(break_even_trades)

        # 计算基础统计
        avg_win = winning_trades.mean() if num_wins > 0 else 0.0
        avg_loss = abs(losing_trades.mean()) if num_losses > 0 else 0.0
        total_profit = winning_trades.sum() if num_wins > 0 else 0.0
        total_loss = abs(losing_trades.sum()) if num_losses > 0 else 0.0
        net_profit = trades['pnl'].sum()

        # 计算比率
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0.0

        # 计算期望值
        win_prob = num_wins / total_trades if total_trades > 0 else 0.0
        loss_prob = num_losses / total_trades if total_trades > 0 else 0.0
        expectancy = (win_prob * avg_win) - (loss_prob * avg_loss)

        # 最大单笔
        max_win = winning_trades.max() if num_wins > 0 else 0.0
        max_loss = abs(losing_trades.min()) if num_losses > 0 else 0.0

        # 连续盈亏统计
        consecutive = self._calculate_consecutive_streaks(trades['pnl'].values)

        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': num_wins,
            'losing_trades': num_losses,
            'break_even_trades': num_break_even,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor,
            'net_profit': net_profit,
            'expectancy': expectancy,
            'max_win': max_win,
            'max_loss': max_loss,
            'max_win_streak': consecutive['max_win_streak'],
            'max_loss_streak': consecutive['max_loss_streak'],
        }

        return self.metrics

    def _calculate_consecutive_streaks(self, pnl_values) -> Dict[str, int]:
        """
        计算最大连续盈利和亏损次数

        Args:
            pnl_values: PnL值列表

        Returns:
            包含max_win_streak和max_loss_streak的字典
        """
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for pnl in pnl_values:
            if pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif pnl < 0:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                current_win_streak = 0
                current_loss_streak = 0

        return {
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
        }

    def print_report(self) -> None:
        """打印分析报告"""
        if not self.metrics:
            print("无法生成报告: 没有有效的交易数据")
            return

        m = self.metrics

        # 提取文件名信息
        filename = os.path.basename(self.trades_file)
        # 文件格式: trades_{strategy}_{timeframe}_{start}_{end}.csv
        # 倒序分割以获取日期，避免策略名中的下划线问题
        if filename.startswith('trades_') and filename.endswith('.csv'):
            content = filename.replace('trades_', '').replace('.csv', '')
            # 从右侧分割，获取最后两个日期字段
            parts = content.rsplit('_', 2)
            if len(parts) == 3:
                strategy_timeframe = parts[0]
                start_date = parts[1]
                end_date = parts[2]
                # 再分割策略和周期（最后一个下划线分隔）
                strategy_parts = strategy_timeframe.rsplit('_', 1)
                if len(strategy_parts) == 2:
                    strategy_name = strategy_parts[0]
                    timeframe = strategy_parts[1]
                else:
                    strategy_name = strategy_timeframe
                    timeframe = 'unknown'
                print(f'策略: {strategy_name} | 周期: {timeframe} | 时间: {start_date} 至 {end_date}')
            else:
                print(f'文件: {filename}')
        else:
            print(f'文件: {filename}')
        print('=' * 80)

        print('\n【交易统计】')
        print(f'  总交易次数: {m["total_trades"]}')
        print(f'  盈利交易: {m["winning_trades"]} ({m["win_rate"]:.2f}%)')
        print(f'  亏损交易: {m["losing_trades"]} ({100 - m["win_rate"]:.2f}%)')
        if m["break_even_trades"] > 0:
            print(f'  打平交易: {m["break_even_trades"]}')

        print('\n【盈亏指标】')
        print(f'  平均盈利: ${m["avg_win"]:.2f}')
        print(f'  平均亏损: $-{m["avg_loss"]:.2f}')
        print(f'  盈亏比: {m["profit_loss_ratio"]:.2f}')
        print(f'  最大单笔盈利: ${m["max_win"]:.2f}')
        print(f'  最大单笔亏损: $-{m["max_loss"]:.2f}')

        print('\n【整体表现】')
        print(f'  总盈利: ${m["total_profit"]:.2f}')
        print(f'  总亏损: $-{m["total_loss"]:.2f}')
        print(f'  净利润: ${m["net_profit"]:.2f}')
        print(f'  盈利因子: {m["profit_factor"]:.2f}')
        print(f'  期望值: ${m["expectancy"]:.2f} (每笔交易期望盈利)')

        print('\n【连续性分析】')
        print(f'  最大连续盈利: {m["max_win_streak"]} 笔')
        print(f'  最大连续亏损: {m["max_loss_streak"]} 笔')

        print('\n【风险评估】')
        if m["max_loss"] > 0:
            initial_capital = 100000  # 默认初始资金
            max_loss_pct = (m["max_loss"] / initial_capital) * 100
            avg_loss_pct = (m["avg_loss"] / initial_capital) * 100
            print(f'  最大单笔亏损占初始资金: -{max_loss_pct:.2f}%')
            print(f'  平均亏损占初始资金: -{avg_loss_pct:.2f}%')

        print('=' * 80 + '\n')


def find_trades_file(
    strategy: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    results_dir: str = 'backtest_results',
) -> Optional[str]:
    """
    根据参数自动定位trades文件

    Args:
        strategy: 策略名称
        timeframe: 时间周期
        start_date: 开始日期（yyyymmdd格式）
        end_date: 结束日期（yyyymmdd格式）
        results_dir: 结果目录

    Returns:
        找到的文件路径，如果没找到则返回None
    """
    expected_filename = f'trades_{strategy}_{timeframe}_{start_date}_{end_date}.csv'
    filepath = os.path.join(results_dir, expected_filename)

    if os.path.exists(filepath):
        return filepath

    # 如果精确匹配失败，尝试模糊匹配
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.startswith(f'trades_{strategy}_{timeframe}'):
                return os.path.join(results_dir, filename)

    return None


def main():
    parser = argparse.ArgumentParser(description='交易盈亏分析工具')
    parser.add_argument(
        '--file',
        help='直接指定交易CSV文件路径',
    )
    parser.add_argument(
        '--strategy',
        help='策略名称',
    )
    parser.add_argument(
        '--timeframe',
        default='d1',
        help='时间周期 (m1/m5/m15/m30/h1/h4/d1)',
    )
    parser.add_argument(
        '--start',
        help='开始日期 (yyyymmdd格式)',
    )
    parser.add_argument(
        '--end',
        help='结束日期 (yyyymmdd格式)',
    )
    parser.add_argument(
        '--results-dir',
        default='backtest_results',
        help='结果文件目录',
    )

    args = parser.parse_args()

    # 确定交易文件
    trades_file = None

    if args.file:
        # 直接指定文件
        trades_file = args.file
    elif args.strategy and args.start and args.end:
        # 通过参数自动定位
        trades_file = find_trades_file(
            args.strategy,
            args.timeframe,
            args.start,
            args.end,
            args.results_dir,
        )
    else:
        print('错误: 必须提供 --file 或 (--strategy --start --end)')
        print('用法示例:')
        print('  python analyze_trades.py --file backtest_results/trades_sma_cross_d1_20240101_20241231.csv')
        print('  python analyze_trades.py --strategy sma_cross --timeframe d1 --start 20240101 --end 20241231')
        return

    if not trades_file:
        print(f'错误: 无法找到交易文件')
        if args.strategy:
            print(f'  尝试查找: trades_{args.strategy}_{args.timeframe}_{args.start}_{args.end}.csv')
        return

    # 执行分析
    analyzer = TradeAnalyzer(trades_file)
    if analyzer.load_trades():
        analyzer.calculate_metrics()
        analyzer.print_report()


if __name__ == '__main__':
    main()
