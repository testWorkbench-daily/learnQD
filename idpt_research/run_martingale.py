#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
马丁格尔策略回测主程序

用法:
    # 基础回测
    python run_martingale.py --start 2024-01-01 --end 2024-12-31
    
    # 指定时间周期
    python run_martingale.py --timeframe h1 --start 2024-01-01 --end 2024-12-31
    
    # 使用保守型策略
    python run_martingale.py --strategy conservative --start 2024-01-01 --end 2024-12-31
    
    # 对比所有策略
    python run_martingale.py --compare --start 2024-01-01 --end 2024-12-31
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

import backtrader as bt
import pandas as pd
import numpy as np

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from martingale_strategy import (
    MartingaleStrategy,
    ConservativeMartingale,
    AggressiveMartingale,
    AntiMartingaleStrategy,
)


class MartingaleRunner:
    """马丁格尔策略回测运行器"""
    
    def __init__(
        self,
        data_path='../data/nq_m1_forward_adjusted.csv',
        timeframe='d1',
        start_date=None,
        end_date=None,
        initial_cash=100000.0,
        commission=0.0002,
    ):
        """
        初始化
        
        Args:
            data_path: 数据文件路径
            timeframe: 时间周期 (m1/m5/m15/m30/h1/h4/d1)
            start_date: 开始日期
            end_date: 结束日期
            initial_cash: 初始资金
            commission: 手续费率
        """
        self.data_path = data_path
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.commission = commission
        
        # 时间周期映射
        self.timeframe_map = {
            'm1': bt.TimeFrame.Minutes,
            'm5': bt.TimeFrame.Minutes,
            'm10': bt.TimeFrame.Minutes,
            'm15': bt.TimeFrame.Minutes,
            'm30': bt.TimeFrame.Minutes,
            'h1': bt.TimeFrame.Minutes,
            'h2': bt.TimeFrame.Minutes,
            'h4': bt.TimeFrame.Minutes,
            'd1': bt.TimeFrame.Days,
            'w1': bt.TimeFrame.Weeks,
        }
        
        self.compression_map = {
            'm1': 1,
            'm5': 5,
            'm10': 10,
            'm15': 15,
            'm30': 30,
            'h1': 60,
            'h2': 120,
            'h4': 240,
            'd1': 1,
            'w1': 1,
        }
    
    def load_data(self):
        """加载和预处理数据"""
        print(f"\n{'='*60}")
        print(f"加载数据: {self.data_path}")
        
        # 读取CSV
        df = pd.read_csv(self.data_path)
        
        # 确保有datetime列
        if 'ts_event' in df.columns:
            df['datetime'] = pd.to_datetime(df['ts_event'])
        elif 'datetime' not in df.columns:
            raise ValueError("数据缺少时间列 (ts_event 或 datetime)")
        
        # 设置索引
        df.set_index('datetime', inplace=True)
        
        # 时间范围过滤
        if self.start_date:
            df = df[df.index >= self.start_date]
        if self.end_date:
            df = df[df.index <= self.end_date]
        
        print(f"数据范围: {df.index[0]} ~ {df.index[-1]}")
        print(f"原始数据行数: {len(df)} (M1级别)")
        print(f"目标时间周期: {self.timeframe}")
        if self.timeframe != 'm1':
            estimated_bars = len(df) // self.compression_map[self.timeframe]
            print(f"重采样后预计: ~{estimated_bars} 根K线")
        print(f"{'='*60}\n")
        
        return df
    
    def create_cerebro(self, strategy_class, **strategy_params):
        """创建backtrader引擎"""
        cerebro = bt.Cerebro()
        
        # 加载数据
        df = self.load_data()
        
        # 创建数据源
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # 使用index
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1,
        )
        
        # 重采样到指定时间周期
        if self.timeframe != 'm1':
            cerebro.resampledata(
                data,
                timeframe=self.timeframe_map[self.timeframe],
                compression=self.compression_map[self.timeframe],
            )
        else:
            cerebro.adddata(data)
        
        # 添加策略
        cerebro.addstrategy(strategy_class, **strategy_params)
        
        # 设置初始资金
        cerebro.broker.setcash(self.initial_cash)
        
        # 设置手续费
        cerebro.broker.setcommission(commission=self.commission)
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        return cerebro
    
    def run(self, strategy_class, strategy_name=None, **strategy_params):
        """运行单个策略"""
        strategy_name = strategy_name or strategy_class.__name__
        
        print(f"\n{'='*60}")
        print(f"运行策略: {strategy_name}")
        print(f"{'='*60}")
        
        # 创建引擎
        cerebro = self.create_cerebro(strategy_class, **strategy_params)
        
        # 记录初始资金
        start_value = cerebro.broker.getvalue()
        print(f'初始资金: {start_value:.2f}')
        
        # 运行回测
        results = cerebro.run()
        strat = results[0]
        
        # 记录最终资金
        end_value = cerebro.broker.getvalue()
        pnl = end_value - start_value
        pnl_pct = (pnl / start_value) * 100
        
        print(f'\n最终资金: {end_value:.2f}')
        print(f'净利润: {pnl:.2f}')
        print(f'收益率: {pnl_pct:.2f}%')
        
        # 输出分析结果
        self.print_analysis(strat)
        
        return {
            'strategy': strategy_name,
            'start_value': start_value,
            'end_value': end_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'analyzers': strat.analyzers,
        }
    
    def print_analysis(self, strat):
        """打印分析结果"""
        print(f"\n{'='*60}")
        print("详细分析")
        print(f"{'='*60}")
        
        # Sharpe Ratio
        try:
            sharpe = strat.analyzers.sharpe.get_analysis()
            print(f"夏普比率: {sharpe.get('sharperatio', 'N/A')}")
        except Exception as e:
            print(f"夏普比率: N/A ({e})")

        # DrawDown
        try:
            dd = strat.analyzers.drawdown.get_analysis()
            print(f"最大回撤: {dd.max.drawdown:.2f}%")
            print(f"最长回撤期: {dd.max.len} 天")
        except Exception as e:
            print(f"回撤信息: N/A ({e})")

        # Returns
        try:
            returns = strat.analyzers.returns.get_analysis()
            print(f"总收益率: {returns.rtot * 100:.2f}%")
            print(f"年化收益率: {returns.rnorm * 100:.2f}%")
        except Exception as e:
            print(f"收益率信息: N/A ({e})")

        # Trade Analysis
        try:
            trades = strat.analyzers.trades.get_analysis()
            total_trades = trades.total.closed if hasattr(trades, 'total') else 0
            won_trades = trades.won.total if hasattr(trades, 'won') else 0
            lost_trades = trades.lost.total if hasattr(trades, 'lost') else 0

            print(f"\n交易统计:")
            print(f"  总交易次数: {total_trades}")
            print(f"  盈利次数: {won_trades}")
            print(f"  亏损次数: {lost_trades}")
            if total_trades > 0:
                win_rate = (won_trades / total_trades) * 100
                print(f"  胜率: {win_rate:.2f}%")

                if hasattr(trades.won, 'pnl'):
                    avg_win = trades.won.pnl.average
                    print(f"  平均盈利: {avg_win:.2f}")

                if hasattr(trades.lost, 'pnl'):
                    avg_loss = trades.lost.pnl.average
                    print(f"  平均亏损: {avg_loss:.2f}")
        except Exception as e:
            print(f"交易统计: 无法获取 ({e})")
        
        print(f"{'='*60}\n")
    
    def compare_strategies(self, strategies):
        """对比多个策略"""
        print(f"\n{'='*60}")
        print("策略对比模式")
        print(f"{'='*60}\n")
        
        results = []
        
        for strategy_class, strategy_name in strategies:
            result = self.run(strategy_class, strategy_name)
            results.append(result)
        
        # 打印对比表
        print(f"\n{'='*60}")
        print("策略对比结果")
        print(f"{'='*60}")
        print(f"{'策略名称':<30} {'收益率':<12} {'最终资金':<12} {'净利润':<12}")
        print(f"{'-'*60}")
        
        for result in results:
            print(
                f"{result['strategy']:<30} "
                f"{result['pnl_pct']:>10.2f}% "
                f"{result['end_value']:>12.2f} "
                f"{result['pnl']:>12.2f}"
            )
        
        print(f"{'='*60}\n")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='马丁格尔策略回测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_martingale.py --start 2024-01-01 --end 2024-12-31
  python run_martingale.py --timeframe h1 --strategy aggressive --start 2024-01-01 --end 2024-12-31
  python run_martingale.py --compare --start 2024-01-01 --end 2024-12-31
        """
    )
    
    parser.add_argument('--data', default='../data/nq_m1_forward_adjusted.csv', help='数据文件路径')
    parser.add_argument('--timeframe', default='d1', 
                       choices=['m1', 'm5', 'm10', 'm15', 'm30', 'h1', 'h2', 'h4', 'd1', 'w1'],
                       help='时间周期')
    parser.add_argument('--start', default='2024-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--cash', type=float, default=100000.0, help='初始资金')
    parser.add_argument('--commission', type=float, default=0.0002, help='手续费率')
    parser.add_argument('--strategy', default='standard',
                       choices=['standard', 'conservative', 'aggressive', 'anti'],
                       help='策略类型')
    parser.add_argument('--compare', action='store_true', help='对比所有策略')
    
    args = parser.parse_args()
    
    # 解析日期
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    # 创建运行器
    runner = MartingaleRunner(
        data_path=args.data,
        timeframe=args.timeframe,
        start_date=start_date,
        end_date=end_date,
        initial_cash=args.cash,
        commission=args.commission,
    )
    
    # 策略映射
    strategy_map = {
        'standard': (MartingaleStrategy, '标准马丁格尔'),
        'conservative': (ConservativeMartingale, '保守型马丁格尔'),
        'aggressive': (AggressiveMartingale, '激进型马丁格尔'),
        'anti': (AntiMartingaleStrategy, '反马丁格尔'),
    }
    
    if args.compare:
        # 对比模式
        strategies = [
            (MartingaleStrategy, '标准马丁格尔'),
            (ConservativeMartingale, '保守型马丁格尔'),
            (AggressiveMartingale, '激进型马丁格尔'),
            (AntiMartingaleStrategy, '反马丁格尔'),
        ]
        runner.compare_strategies(strategies)
    else:
        # 单策略模式
        strategy_class, strategy_name = strategy_map[args.strategy]
        runner.run(strategy_class, strategy_name)


if __name__ == '__main__':
    main()
