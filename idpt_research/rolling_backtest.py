#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
滚动窗口回测器 (Rolling Window Backtester)

功能:
- 使用固定6个月的数据窗口进行回测
- 每次滑动1个月,遍历整个数据集
- 汇总所有滚动窗口的结果
- 输出统计和可视化

示例:
    python rolling_backtest.py --timeframe h1 --strategy standard
    python rolling_backtest.py --timeframe d1 --strategy conservative --window-months 6 --step-months 1
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import argparse
import os
from multiprocessing import Pool, cpu_count
from functools import partial

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from idpt_research.run_martingale import MartingaleRunner
from idpt_research.martingale_strategy import (
    MartingaleStrategy,
    ConservativeMartingale,
    AggressiveMartingale,
    AntiMartingaleStrategy,
)


# 全局函数，用于多进程调用
def _worker_run_backtest(data_path, timeframe, strategy_class, strategy_name,
                        start_date, end_date, initial_cash, commission, window_idx, total_windows):
    """工作进程函数：运行单个窗口的回测"""
    import io
    import contextlib
    
    # 静默执行
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            # 创建运行器
            runner = MartingaleRunner(
                data_path=data_path,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash,
                commission=commission,
            )
            
            # 运行回测
            result = runner.run(strategy_class, strategy_name)
            
            # 提取关键指标
            window_result = {
                'window_idx': window_idx,
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': (end_date - start_date).days,
                'start_value': result['start_value'],
                'end_value': result['end_value'],
                'pnl': result['pnl'],
                'pnl_pct': result['pnl_pct'],
            }
            
            # 提取分析器数据
            try:
                sharpe = result['analyzers'].sharpe.get_analysis()
                window_result['sharpe_ratio'] = sharpe.get('sharperatio', np.nan)
            except Exception:
                window_result['sharpe_ratio'] = np.nan

            try:
                dd = result['analyzers'].drawdown.get_analysis()
                # Backtrader返回的是正数，转成负数表示回撤
                window_result['max_drawdown'] = -abs(dd.max.drawdown)
                window_result['max_drawdown_len'] = dd.max.len
            except Exception:
                window_result['max_drawdown'] = np.nan
                window_result['max_drawdown_len'] = np.nan

            try:
                returns = result['analyzers'].returns.get_analysis()
                window_result['total_return'] = returns.rtot * 100
                window_result['annual_return'] = returns.rnorm * 100
            except Exception:
                window_result['total_return'] = np.nan
                window_result['annual_return'] = np.nan

            try:
                trades = result['analyzers'].trades.get_analysis()
                window_result['total_trades'] = trades.total.closed if hasattr(trades, 'total') else 0
                window_result['won_trades'] = trades.won.total if hasattr(trades, 'won') else 0
                window_result['lost_trades'] = trades.lost.total if hasattr(trades, 'lost') else 0

                if window_result['total_trades'] > 0:
                    window_result['win_rate'] = (window_result['won_trades'] / window_result['total_trades']) * 100
                else:
                    window_result['win_rate'] = 0

                if hasattr(trades.won, 'pnl'):
                    window_result['avg_win'] = trades.won.pnl.average
                else:
                    window_result['avg_win'] = np.nan

                if hasattr(trades.lost, 'pnl'):
                    window_result['avg_loss'] = trades.lost.pnl.average
                else:
                    window_result['avg_loss'] = np.nan
            except Exception:
                window_result['total_trades'] = 0
                window_result['won_trades'] = 0
                window_result['lost_trades'] = 0
                window_result['win_rate'] = 0
                window_result['avg_win'] = np.nan
                window_result['avg_loss'] = np.nan

            # 计算买入持有基准 (多进程模式需要单独读取数据)
            try:
                df = pd.read_csv(data_path)
                df['ts_event'] = pd.to_datetime(df['ts_event'])
                df_window = df[(df['ts_event'] >= start_date) & (df['ts_event'] <= end_date)]

                if len(df_window) > 0:
                    first_price = df_window.iloc[0]['close']
                    last_price = df_window.iloc[-1]['close']
                    window_result['buy_hold_return_pct'] = ((last_price - first_price) / first_price) * 100

                    # 计算买入持有的最大回撤
                    cummax = df_window['close'].cummax()
                    drawdown = ((df_window['close'] - cummax) / cummax) * 100
                    window_result['buy_hold_max_dd'] = drawdown.min()
                else:
                    window_result['buy_hold_return_pct'] = np.nan
                    window_result['buy_hold_max_dd'] = np.nan
            except Exception:
                window_result['buy_hold_return_pct'] = np.nan
                window_result['buy_hold_max_dd'] = np.nan

            return window_result

        except Exception as e:
            return {
                'window_idx': window_idx,
                'start_date': start_date,
                'end_date': end_date,
                'error': str(e),
            }


class RollingBacktester:
    """滚动窗口回测器"""
    
    def __init__(self, data_path, timeframe, strategy_class, strategy_name,
                 window_months=6, step_months=1, initial_cash=100000.0, commission=0.0002,
                 quiet=False, n_workers=1):
        """
        初始化
        
        Args:
            data_path: 数据文件路径
            timeframe: 时间周期 (m1, m5, m15, m30, h1, h4, d1, w1)
            strategy_class: 策略类
            strategy_name: 策略名称
            window_months: 每次回测的窗口长度(月)
            step_months: 滚动步长(月)
            initial_cash: 初始资金
            commission: 手续费率
            quiet: 是否静默模式(不打印详细日志)
            n_workers: 并发进程数 (1=串行, >1=并行)
        """
        self.data_path = data_path
        self.timeframe = timeframe
        self.strategy_class = strategy_class
        self.strategy_name = strategy_name
        self.window_months = window_months
        self.step_months = step_months
        self.initial_cash = initial_cash
        self.commission = commission
        self.quiet = quiet
        self.n_workers = n_workers
        
        # 读取数据获取时间范围
        print(f"\n{'='*80}")
        print(f"滚动窗口回测器初始化")
        print(f"{'='*80}")
        print(f"数据文件: {data_path}")
        print(f"时间周期: {timeframe}")
        print(f"策略: {strategy_name}")
        print(f"窗口长度: {window_months} 个月")
        print(f"滚动步长: {step_months} 个月")
        print(f"并发进程数: {n_workers}")

        # 获取数据时间范围 (缓存数据避免重复读取)
        self._cached_df = pd.read_csv(data_path)
        self._cached_df['ts_event'] = pd.to_datetime(self._cached_df['ts_event'])
        self.data_start = self._cached_df['ts_event'].min()
        self.data_end = self._cached_df['ts_event'].max()

        print(f"数据范围: {self.data_start.date()} ~ {self.data_end.date()}")
        print(f"数据总行数: {len(self._cached_df)}")
        print(f"{'='*80}\n")
    
    def generate_windows(self):
        """生成滚动窗口的开始和结束日期"""
        windows = []
        
        current_start = self.data_start
        
        while True:
            # 计算窗口结束日期
            current_end = current_start + relativedelta(months=self.window_months)
            
            # 如果窗口结束日期超过数据范围,停止
            if current_end > self.data_end:
                # 最后一个窗口:使用剩余所有数据
                if current_start < self.data_end:
                    current_end = self.data_end
                    windows.append((current_start, current_end))
                break
            
            windows.append((current_start, current_end))
            
            # 滑动窗口
            current_start = current_start + relativedelta(months=self.step_months)
        
        return windows
    
    def _execute_backtest(self, start_date, end_date, window_idx, total_windows):
        """执行单个窗口的回测（核心逻辑）"""
        try:
            # 临时重定向标准输出(静默模式)
            if self.quiet:
                import io
                import contextlib
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    # 创建运行器
                    runner = MartingaleRunner(
                        data_path=self.data_path,
                        timeframe=self.timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        initial_cash=self.initial_cash,
                        commission=self.commission,
                    )
                    # 运行回测
                    result = runner.run(self.strategy_class, self.strategy_name)
            else:
                # 创建运行器
                runner = MartingaleRunner(
                    data_path=self.data_path,
                    timeframe=self.timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    initial_cash=self.initial_cash,
                    commission=self.commission,
                )
                # 运行回测
                result = runner.run(self.strategy_class, self.strategy_name)
            
            # 提取关键指标
            window_result = {
                'window_idx': window_idx,
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': (end_date - start_date).days,
                'start_value': result['start_value'],
                'end_value': result['end_value'],
                'pnl': result['pnl'],
                'pnl_pct': result['pnl_pct'],
            }
            
            # 提取分析器数据
            try:
                sharpe = result['analyzers'].sharpe.get_analysis()
                window_result['sharpe_ratio'] = sharpe.get('sharperatio', np.nan)
            except Exception:
                window_result['sharpe_ratio'] = np.nan

            try:
                dd = result['analyzers'].drawdown.get_analysis()
                # Backtrader返回的是正数，转成负数表示回撤
                window_result['max_drawdown'] = -abs(dd.max.drawdown)
                window_result['max_drawdown_len'] = dd.max.len
            except Exception:
                window_result['max_drawdown'] = np.nan
                window_result['max_drawdown_len'] = np.nan

            try:
                returns = result['analyzers'].returns.get_analysis()
                window_result['total_return'] = returns.rtot * 100
                window_result['annual_return'] = returns.rnorm * 100
            except Exception:
                window_result['total_return'] = np.nan
                window_result['annual_return'] = np.nan

            try:
                trades = result['analyzers'].trades.get_analysis()
                window_result['total_trades'] = trades.total.closed if hasattr(trades, 'total') else 0
                window_result['won_trades'] = trades.won.total if hasattr(trades, 'won') else 0
                window_result['lost_trades'] = trades.lost.total if hasattr(trades, 'lost') else 0

                if window_result['total_trades'] > 0:
                    window_result['win_rate'] = (window_result['won_trades'] / window_result['total_trades']) * 100
                else:
                    window_result['win_rate'] = 0

                if hasattr(trades.won, 'pnl'):
                    window_result['avg_win'] = trades.won.pnl.average
                else:
                    window_result['avg_win'] = np.nan

                if hasattr(trades.lost, 'pnl'):
                    window_result['avg_loss'] = trades.lost.pnl.average
                else:
                    window_result['avg_loss'] = np.nan
            except Exception:
                window_result['total_trades'] = 0
                window_result['won_trades'] = 0
                window_result['lost_trades'] = 0
                window_result['win_rate'] = 0
                window_result['avg_win'] = np.nan
                window_result['avg_loss'] = np.nan

            # 计算买入持有基准 (使用缓存的数据)
            try:
                df_window = self._cached_df[
                    (self._cached_df['ts_event'] >= start_date) &
                    (self._cached_df['ts_event'] <= end_date)
                ]

                if len(df_window) > 0:
                    first_price = df_window.iloc[0]['close']
                    last_price = df_window.iloc[-1]['close']
                    window_result['buy_hold_return_pct'] = ((last_price - first_price) / first_price) * 100

                    # 计算买入持有的最大回撤
                    cummax = df_window['close'].cummax()
                    drawdown = ((df_window['close'] - cummax) / cummax) * 100
                    window_result['buy_hold_max_dd'] = drawdown.min()
                else:
                    window_result['buy_hold_return_pct'] = np.nan
                    window_result['buy_hold_max_dd'] = np.nan
            except Exception as e:
                window_result['buy_hold_return_pct'] = np.nan
                window_result['buy_hold_max_dd'] = np.nan

            return window_result

        except Exception as e:
            return {
                'window_idx': window_idx,
                'start_date': start_date,
                'end_date': end_date,
                'error': str(e),
            }
    
    def run_single_window(self, start_date, end_date, window_idx, total_windows):
        """运行单个窗口的回测（串行模式）"""
        if not self.quiet:
            print(f"\n{'='*80}")
            print(f"窗口 {window_idx}/{total_windows}: {start_date.date()} ~ {end_date.date()}")
            print(f"{'='*80}")
        else:
            # 静默模式只显示进度
            print(f"\r处理窗口: {window_idx}/{total_windows} [{start_date.date()} ~ {end_date.date()}]", end='', flush=True)
        
        result = self._execute_backtest(start_date, end_date, window_idx, total_windows)
        
        if 'error' not in result:
            if not self.quiet:
                print(f"\n✓ 窗口 {window_idx} 完成: 收益率 {result['pnl_pct']:.2f}%")
        else:
            if not self.quiet:
                print(f"\n✗ 窗口 {window_idx} 失败: {result['error']}")
        
        return result
    
    def run(self):
        """运行滚动窗口回测"""
        # 生成窗口
        windows = self.generate_windows()
        total_windows = len(windows)
        
        print(f"\n{'='*80}")
        print(f"开始滚动回测")
        print(f"{'='*80}")
        print(f"总窗口数: {total_windows}")
        print(f"{'='*80}\n")
        
        # 根据并发数选择执行方式
        if self.n_workers <= 1:
            # 串行执行
            results = []
            for idx, (start_date, end_date) in enumerate(windows, 1):
                result = self.run_single_window(start_date, end_date, idx, total_windows)
                results.append(result)
        else:
            # 并行执行
            results = self._run_parallel(windows)
        
        # 汇总结果
        self.summarize_results(results)
        
        return results
    
    def _run_parallel(self, windows):
        """并行执行多个窗口"""
        total_windows = len(windows)
        
        print(f"使用 {self.n_workers} 个进程并行处理...\n")
        
        # 准备参数列表
        tasks = [
            (self.data_path, self.timeframe, self.strategy_class, self.strategy_name,
             start_date, end_date, self.initial_cash, self.commission, idx, total_windows)
            for idx, (start_date, end_date) in enumerate(windows, 1)
        ]
        
        # 使用进程池
        with Pool(processes=self.n_workers) as pool:
            # 使用 starmap 并行执行
            results = pool.starmap(
                _worker_run_backtest,
                tasks
            )
        
        print(f"\n所有窗口处理完成！\n")
        return results
    
    def summarize_results(self, results):
        """汇总和展示结果"""
        print(f"\n{'='*80}")
        print(f"滚动回测汇总报告")
        print(f"{'='*80}\n")
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 过滤掉失败的窗口
        successful_df = df[~df['pnl_pct'].isna()].copy()
        failed_count = len(df) - len(successful_df)
        
        if len(successful_df) == 0:
            print("⚠️  所有窗口都失败了!")
            return
        
        print(f"策略: {self.strategy_name}")
        print(f"时间周期: {self.timeframe}")
        print(f"总窗口数: {len(df)}")
        print(f"成功窗口: {len(successful_df)}")
        if failed_count > 0:
            print(f"失败窗口: {failed_count}")
        
        # 整体统计
        print(f"\n{'-'*80}")
        print(f"整体表现")
        print(f"{'-'*80}")
        
        total_pnl = successful_df['pnl'].sum()
        avg_pnl_pct = successful_df['pnl_pct'].mean()
        median_pnl_pct = successful_df['pnl_pct'].median()
        std_pnl_pct = successful_df['pnl_pct'].std()
        
        print(f"总盈亏: {total_pnl:,.2f}")
        print(f"平均收益率: {avg_pnl_pct:.2f}% (中位数: {median_pnl_pct:.2f}%)")
        print(f"收益率标准差: {std_pnl_pct:.2f}%")
        
        # 胜率统计
        winning_windows = len(successful_df[successful_df['pnl'] > 0])
        losing_windows = len(successful_df[successful_df['pnl'] <= 0])
        win_rate = (winning_windows / len(successful_df)) * 100 if len(successful_df) > 0 else 0
        
        print(f"\n窗口胜率: {win_rate:.2f}% (盈利: {winning_windows}, 亏损: {losing_windows})")
        
        # 最好和最差的窗口
        best_idx = successful_df['pnl_pct'].idxmax()
        worst_idx = successful_df['pnl_pct'].idxmin()
        
        best = successful_df.loc[best_idx]
        worst = successful_df.loc[worst_idx]
        
        print(f"\n最佳窗口: {best['start_date'].date()} ~ {best['end_date'].date()}")
        print(f"  收益率: {best['pnl_pct']:.2f}%")
        
        print(f"\n最差窗口: {worst['start_date'].date()} ~ {worst['end_date'].date()}")
        print(f"  收益率: {worst['pnl_pct']:.2f}%")
        
        # 风险指标
        print(f"\n{'-'*80}")
        print(f"风险指标")
        print(f"{'-'*80}")
        
        avg_sharpe = successful_df['sharpe_ratio'].mean()
        avg_dd = successful_df['max_drawdown'].mean()
        # 回撤是负数，最差回撤是最小值
        worst_dd = successful_df['max_drawdown'].min()
        
        print(f"平均夏普比率: {avg_sharpe:.3f}")
        print(f"平均最大回撤: {avg_dd:.2f}%")
        print(f"最差回撤(所有窗口): {worst_dd:.2f}%")
        
        # 交易统计
        print(f"\n{'-'*80}")
        print(f"交易统计")
        print(f"{'-'*80}")
        
        total_trades = successful_df['total_trades'].sum()
        avg_win_rate = successful_df['win_rate'].mean()
        
        print(f"总交易次数: {total_trades:.0f}")
        print(f"平均胜率: {avg_win_rate:.2f}%")
        
        # 买入持有基准对比
        print(f"\n{'-'*80}")
        print(f"买入持有基准对比")
        print(f"{'-'*80}")
        
        avg_buy_hold_return = successful_df['buy_hold_return_pct'].mean()
        avg_buy_hold_dd = successful_df['buy_hold_max_dd'].mean()
        
        print(f"平均买入持有收益: {avg_buy_hold_return:.2f}%")
        print(f"平均买入持有回撤: {avg_buy_hold_dd:.2f}%")
        print(f"策略超额收益: {avg_pnl_pct - avg_buy_hold_return:.2f}%")
        
        # 分位数统计
        print(f"\n{'-'*80}")
        print(f"收益率分布")
        print(f"{'-'*80}")
        
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            val = successful_df['pnl_pct'].quantile(p/100)
            print(f"P{p}: {val:.2f}%")
        
        # 详细窗口结果表
        print(f"\n{'-'*80}")
        print(f"各窗口详细结果")
        print(f"{'-'*80}")
        print(f"{'窗口':<6} {'开始日期':<12} {'结束日期':<12} {'策略收益':<10} {'买入持有':<10} {'策略回撤':<10} {'持有回撤':<10} {'交易次数':<8}")
        print(f"{'-'*80}")
        
        for _, row in successful_df.iterrows():
            print(
                f"{int(row['window_idx']):<6} "
                f"{row['start_date'].date()!s:<12} "
                f"{row['end_date'].date()!s:<12} "
                f"{row['pnl_pct']:>8.2f}% "
                f"{row.get('buy_hold_return_pct', 0):>8.2f}% "
                f"{row['max_drawdown']:>8.2f}% "
                f"{row.get('buy_hold_max_dd', 0):>8.2f}% "
                f"{int(row['total_trades']):>8}"
            )
        
        print(f"{'='*80}\n")
        
        # 保存结果到CSV
        output_file = f"rolling_results_{self.strategy_name.replace(' ', '_')}_{self.timeframe}.csv"
        df.to_csv(output_file, index=False)
        print(f"✓ 结果已保存到: {output_file}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='滚动窗口回测器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 串行执行
  python rolling_backtest.py --timeframe h1 --strategy conservative --quiet
  
  # 8进程并行
  python rolling_backtest.py --timeframe d1 --strategy standard --workers 8 --quiet
  
  # 自动检测CPU核心数
  python rolling_backtest.py --timeframe h4 --strategy aggressive --workers auto --quiet
        """
    )
    
    parser.add_argument('--data', default='../data/nq_m1_forward_adjusted.csv', help='数据文件路径')
    parser.add_argument('--timeframe', default='d1',
                       choices=['m1', 'm5', 'm10', 'm15', 'm30', 'h1', 'h2', 'h4', 'd1', 'w1'],
                       help='时间周期')
    parser.add_argument('--strategy', default='standard',
                       choices=['standard', 'conservative', 'aggressive', 'anti'],
                       help='策略类型')
    parser.add_argument('--window', type=int, default=6, help='窗口长度(月)')
    parser.add_argument('--step', type=int, default=1, help='滚动步长(月)')
    parser.add_argument('--cash', type=float, default=100000.0, help='初始资金')
    parser.add_argument('--commission', type=float, default=0.0002, help='手续费率')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式(不显示详细日志)')
    parser.add_argument('--workers', '-w', default='1', help='并发进程数 (1-N或auto, 默认1=串行)')
    
    args = parser.parse_args()
    
    # 处理并发进程数
    if args.workers.lower() == 'auto':
        n_workers = max(1, cpu_count() - 1)  # 留一个核心给系统
        print(f"检测到 {cpu_count()} 个CPU核心，使用 {n_workers} 个进程")
    else:
        n_workers = int(args.workers)
        if n_workers < 1:
            n_workers = 1
            print("警告: 进程数至少为1，已自动调整")
    
    # 策略映射
    strategy_map = {
        'standard': (MartingaleStrategy, '标准马丁格尔'),
        'conservative': (ConservativeMartingale, '保守型马丁格尔'),
        'aggressive': (AggressiveMartingale, '激进型马丁格尔'),
        'anti': (AntiMartingaleStrategy, '反马丁格尔'),
    }
    
    strategy_class, strategy_name = strategy_map[args.strategy]
    
    # 创建滚动回测器
    backtester = RollingBacktester(
        data_path=args.data,
        timeframe=args.timeframe,
        strategy_class=strategy_class,
        strategy_name=strategy_name,
        window_months=args.window,
        step_months=args.step,
        initial_cash=args.cash,
        commission=args.commission,
        quiet=args.quiet,
        n_workers=n_workers,
    )
    
    # 运行回测
    results = backtester.run()


if __name__ == '__main__':
    main()
