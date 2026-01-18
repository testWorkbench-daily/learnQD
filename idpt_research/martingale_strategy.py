#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
马丁格尔策略 (Martingale Strategy) 回测

策略逻辑:
1. 初始仓位开多单
2. 如果亏损,加倍仓位继续开多单(摊低成本)
3. 盈利后平仓,重新开始
4. 设置最大加仓次数和止损保护

适用场景:
- 长期看涨的市场(如NQ指数)
- 有足够资金抗波动
- 严格风控下使用
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime


class MartingaleStrategy(bt.Strategy):
    """马丁格尔策略"""
    
    params = (
        ('initial_units', 1),          # 初始手数
        ('profit_target_pct', 0.5),    # 盈利目标百分比 (0.5%)
        ('loss_trigger_pct', -0.3),    # 亏损触发加仓百分比 (-0.3%)
        ('max_pyramid_levels', 5),     # 最大加仓次数
        ('multiplier', 2.0),           # 加仓倍数
        ('stop_loss_pct', -10.0),      # 全局止损百分比 (-10%)
        ('use_trailing_stop', False),  # 是否使用移动止损
        ('trailing_stop_pct', 2.0),    # 移动止损百分比
    )
    
    def __init__(self):
        """初始化"""
        self.dataclose = self.datas[0].close
        self.order = None
        self.entry_price = 0
        self.pyramid_level = 0
        self.total_units = 0
        self.avg_entry_price = 0
        self.peak_value = 0
        self.last_pyramid_price = 0  # 上次加仓价格，防止重复触发

        # 记录交易
        self.trades_log = []
        self.daily_log = []
        
    def log(self, txt, dt=None):
        """日志输出"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'买入执行 价格: {order.executed.price:.2f}, '
                    f'手数: {order.executed.size:.0f}, '
                    f'费用: {order.executed.comm:.2f}'
                )
            else:
                self.log(
                    f'卖出执行 价格: {order.executed.price:.2f}, '
                    f'手数: {order.executed.size:.0f}, '
                    f'费用: {order.executed.comm:.2f}, '
                    f'净利润: {order.executed.pnl:.2f}'
                )
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
        
        self.order = None
    
    def notify_trade(self, trade):
        """交易通知"""
        if not trade.isclosed:
            return
        
        self.log(f'交易盈亏, 毛利润: {trade.pnl:.2f}, 净利润: {trade.pnlcomm:.2f}')
        
        # 记录交易
        self.trades_log.append({
            'date': self.datas[0].datetime.date(0),
            'pnl': trade.pnl,
            'pnlcomm': trade.pnlcomm,
            'pyramid_level': self.pyramid_level,
        })
    
    def next(self):
        """策略主逻辑"""
        # 等待订单完成
        if self.order:
            return
        
        current_price = self.dataclose[0]
        portfolio_value = self.broker.getvalue()
        
        # 更新峰值
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        # 检查是否有持仓
        if not self.position:
            # 没有持仓,开新仓
            self.entry_price = current_price
            self.pyramid_level = 0
            self.total_units = self.params.initial_units
            self.avg_entry_price = current_price
            self.last_pyramid_price = current_price  # 初始化加仓基准价格

            # 买入
            self.order = self.buy(size=self.total_units)
            self.log(f'开多单 价格: {current_price:.2f}, 手数: {self.total_units}')
            
        else:
            # 有持仓,计算收益率
            pnl_pct = (current_price - self.avg_entry_price) / self.avg_entry_price * 100
            
            # 检查全局止损
            if pnl_pct <= self.params.stop_loss_pct:
                self.order = self.close()
                self.log(f'触发止损 当前价: {current_price:.2f}, 亏损: {pnl_pct:.2f}%')
                self.pyramid_level = 0
                self.total_units = 0
                self.last_pyramid_price = 0
                return
            
            # 检查移动止损
            if self.params.use_trailing_stop:
                drawdown_from_peak = (self.peak_value - portfolio_value) / self.peak_value * 100
                if drawdown_from_peak >= self.params.trailing_stop_pct:
                    self.order = self.close()
                    self.log(f'触发移动止损 回撤: {drawdown_from_peak:.2f}%')
                    self.pyramid_level = 0
                    self.total_units = 0
                    self.last_pyramid_price = 0
                    return

            # 检查盈利目标
            if pnl_pct >= self.params.profit_target_pct:
                self.order = self.close()
                self.log(f'止盈平仓 当前价: {current_price:.2f}, 盈利: {pnl_pct:.2f}%')
                self.pyramid_level = 0
                self.total_units = 0
                self.last_pyramid_price = 0
                return
            
            # 检查是否需要加仓(亏损时)
            # 使用 last_pyramid_price 防止在同一价格水平重复触发加仓
            price_drop_from_last = 0
            if hasattr(self, 'last_pyramid_price') and self.last_pyramid_price > 0:
                price_drop_from_last = (current_price - self.last_pyramid_price) / self.last_pyramid_price * 100
            else:
                price_drop_from_last = pnl_pct  # 第一次加仓使用整体亏损

            if pnl_pct <= self.params.loss_trigger_pct and price_drop_from_last <= self.params.loss_trigger_pct:
                if self.pyramid_level < self.params.max_pyramid_levels:
                    # 标准马丁格尔: 每次加仓数量是初始仓位的倍数
                    # Level 1: initial * multiplier^1
                    # Level 2: initial * multiplier^2
                    # Level 3: initial * multiplier^3
                    new_units = int(self.params.initial_units * (self.params.multiplier ** (self.pyramid_level + 1)))

                    # 检查资金是否足够
                    required_cash = new_units * current_price
                    if self.broker.getcash() >= required_cash:
                        self.order = self.buy(size=new_units)
                        self.pyramid_level += 1
                        self.last_pyramid_price = current_price  # 记录本次加仓价格

                        # 更新平均成本
                        total_cost = self.avg_entry_price * self.total_units + current_price * new_units
                        self.total_units += new_units
                        self.avg_entry_price = total_cost / self.total_units

                        self.log(
                            f'加仓 Level {self.pyramid_level}: '
                            f'价格: {current_price:.2f}, '
                            f'新增手数: {new_units}, '
                            f'总手数: {self.total_units}, '
                            f'平均成本: {self.avg_entry_price:.2f}'
                        )
                    else:
                        self.log(f'资金不足,无法加仓 需要: {required_cash:.2f}, 可用: {self.broker.getcash():.2f}')
                else:
                    self.log(f'达到最大加仓次数 {self.params.max_pyramid_levels}')
    
    def stop(self):
        """策略结束时调用"""
        self.log(f'策略结束 总资产: {self.broker.getvalue():.2f}')


class ConservativeMartingale(MartingaleStrategy):
    """保守型马丁格尔"""
    params = (
        ('initial_units', 1),
        ('profit_target_pct', 0.3),
        ('loss_trigger_pct', -0.5),
        ('max_pyramid_levels', 3),
        ('multiplier', 1.5),
        ('stop_loss_pct', -5.0),
    )


class AggressiveMartingale(MartingaleStrategy):
    """激进型马丁格尔"""
    params = (
        ('initial_units', 2),
        ('profit_target_pct', 1.0),
        ('loss_trigger_pct', -0.2),
        ('max_pyramid_levels', 7),
        ('multiplier', 2.5),
        ('stop_loss_pct', -15.0),
    )


class AntiMartingaleStrategy(bt.Strategy):
    """反马丁格尔策略 (盈利时加仓)"""

    params = (
        ('initial_units', 1),
        ('profit_trigger_pct', 0.5),   # 盈利触发加仓百分比
        ('max_pyramid_levels', 5),
        ('multiplier', 1.5),
        ('stop_loss_pct', -2.0),       # 单次止损
        ('take_profit_pct', 3.0),      # 总止盈
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.entry_price = 0
        self.pyramid_level = 0
        self.total_units = 0
        self.last_pyramid_price = 0  # 上次加仓价格，防止重复触发
        self.trades_log = []

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入 价格: {order.executed.price:.2f}, 手数: {order.executed.size:.0f}')
            else:
                self.log(f'卖出 价格: {order.executed.price:.2f}, 盈亏: {order.executed.pnl:.2f}')
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'交易净利润: {trade.pnlcomm:.2f}')

    def next(self):
        if self.order:
            return

        current_price = self.dataclose[0]

        if not self.position:
            # 开仓
            self.entry_price = current_price
            self.pyramid_level = 0
            self.total_units = self.params.initial_units
            self.last_pyramid_price = current_price  # 初始化加仓基准价格
            self.order = self.buy(size=self.total_units)
            self.log(f'开多单 价格: {current_price:.2f}')
        else:
            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100

            # 止损
            if pnl_pct <= self.params.stop_loss_pct:
                self.order = self.close()
                self.log(f'止损 亏损: {pnl_pct:.2f}%')
                self.last_pyramid_price = 0
                return

            # 止盈
            if pnl_pct >= self.params.take_profit_pct:
                self.order = self.close()
                self.log(f'止盈 盈利: {pnl_pct:.2f}%')
                self.last_pyramid_price = 0
                return

            # 盈利时加仓
            # 检查从上次加仓价格的涨幅，防止重复触发
            price_rise_from_last = 0
            if self.last_pyramid_price > 0:
                price_rise_from_last = (current_price - self.last_pyramid_price) / self.last_pyramid_price * 100
            else:
                price_rise_from_last = pnl_pct

            if pnl_pct >= self.params.profit_trigger_pct * (self.pyramid_level + 1) and price_rise_from_last >= self.params.profit_trigger_pct:
                if self.pyramid_level < self.params.max_pyramid_levels:
                    # 标准反马丁格尔: 每次加仓数量是初始仓位的倍数
                    new_units = int(self.params.initial_units * (self.params.multiplier ** (self.pyramid_level + 1)))
                    if self.broker.getcash() >= new_units * current_price:
                        self.order = self.buy(size=new_units)
                        self.pyramid_level += 1
                        self.last_pyramid_price = current_price  # 记录本次加仓价格
                        self.total_units += new_units
                        self.log(f'盈利加仓 Level {self.pyramid_level}, 新增: {new_units}, 总手数: {self.total_units}')

    def stop(self):
        """策略结束时调用"""
        self.log(f'策略结束 总资产: {self.broker.getvalue():.2f}')
