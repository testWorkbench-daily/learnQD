"""
ATR通道突破策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer


class ATRBreakoutAtom(StrategyAtom):
    """
    ATR通道突破策略
    
    用ATR（真实波幅）构建动态通道，突破通道时顺势入场。
    通道宽度随波动自适应调整，适合趋势市场。
    
    核心原理：
    - 中轨 = MA(ma_period)
    - ATR = 真实波幅的移动平均
    - 上轨 = 中轨 + multiplier × ATR
    - 下轨 = 中轨 - multiplier × ATR
    - 通道宽度自适应波动，波动大时通道宽，波动小时通道窄
    
    信号规则：
    - 做多：价格突破上轨（趋势向上）
    - 做空：价格跌破下轨（趋势向下）
    - 跟踪止损：入场后止损设在中轨或1倍ATR处
    - 离场：价格回破通道或触及止损
    """
    
    name = "atr_breakout"
    params = {'ma_period': 20, 'atr_period': 14, 'multiplier': 2.0, 'risk_pct': 0.1}
    
    def strategy_cls(self):
        ma_period = self.params['ma_period']
        atr_period = self.params['atr_period']
        multiplier = self.params['multiplier']
        
        class Strategy(BaseStrategy):
            params = (
                ('ma_period', ma_period),
                ('atr_period', atr_period),
                ('multiplier', multiplier),
            )
            
            def __init__(self):
                super().__init__()
                # 中轨：移动平均
                self.ma = bt.ind.SMA(period=self.p.ma_period)
                
                # ATR指标
                self.atr = bt.ind.ATR(period=self.p.atr_period)
                
                # 上轨和下轨
                self.upper_band = self.ma + self.p.multiplier * self.atr
                self.lower_band = self.ma - self.p.multiplier * self.atr
                
                # 记录入场价格和止损价格
                self.entry_price = None
                self.stop_loss = None
            
            def next(self):
                if self.order:
                    return
                
                # 当前价格和通道值
                price = self.data.close[0]
                mid_line = self.ma[0]
                upper_line = self.upper_band[0]
                lower_line = self.lower_band[0]
                atr_value = self.atr[0]
                
                if not self.position:
                    # 做多信号：价格突破上轨（趋势向上）
                    if price > upper_line:
                        self.order = self.buy()
                        self.entry_price = price
                        # 止损设在中轨或入场价 - 1倍ATR（取较高者）
                        self.stop_loss = max(mid_line, price - atr_value)
                    
                    # 做空信号：价格跌破下轨（趋势向下）
                    elif price < lower_line:
                        self.order = self.sell()
                        self.entry_price = price
                        # 止损设在中轨或入场价 + 1倍ATR（取较低者）
                        self.stop_loss = min(mid_line, price + atr_value)
                
                else:
                    # 平多仓条件
                    if self.position.size > 0:
                        # 更新跟踪止损：使用中轨或动态止损（取较高者）
                        trailing_stop = max(mid_line, price - atr_value)
                        if trailing_stop > self.stop_loss:
                            self.stop_loss = trailing_stop
                        
                        # 1. 止损：价格跌破止损线
                        if price <= self.stop_loss:
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
                        
                        # 2. 离场：价格跌破下轨（反向突破）
                        elif price < lower_line:
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
                    
                    # 平空仓条件
                    elif self.position.size < 0:
                        # 更新跟踪止损：使用中轨或动态止损（取较低者）
                        trailing_stop = min(mid_line, price + atr_value)
                        if trailing_stop < self.stop_loss:
                            self.stop_loss = trailing_stop
                        
                        # 1. 止损：价格突破止损线
                        if price >= self.stop_loss:
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
                        
                        # 2. 离场：价格突破上轨（反向突破）
                        elif price > upper_line:
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
        
        return Strategy
    
    def sizer_cls(self):
        """使用固定1手"""
        return None  # 使用Runner默认的FixedSize


# 预定义的参数变体
class ATRBreakout_20_14_2(ATRBreakoutAtom):
    """标准ATR突破：20周期MA，14周期ATR，2倍"""
    name = "atr_breakout_20_14_2"
    params = {'ma_period': 20, 'atr_period': 14, 'multiplier': 2.0, 'risk_pct': 0.1}


class ATRBreakout_20_14_3(ATRBreakoutAtom):
    """宽幅ATR突破：20周期MA，14周期ATR，3倍（更宽容）"""
    name = "atr_breakout_20_14_3"
    params = {'ma_period': 20, 'atr_period': 14, 'multiplier': 3.0, 'risk_pct': 0.1}


class ATRBreakout_20_14_1_5(ATRBreakoutAtom):
    """窄幅ATR突破：20周期MA，14周期ATR，1.5倍（更敏感）"""
    name = "atr_breakout_20_14_1_5"
    params = {'ma_period': 20, 'atr_period': 14, 'multiplier': 1.5, 'risk_pct': 0.1}


class ATRBreakout_20_10_2(ATRBreakoutAtom):
    """快速ATR突破：20周期MA，10周期ATR，2倍（更快响应）"""
    name = "atr_breakout_20_10_2"
    params = {'ma_period': 20, 'atr_period': 10, 'multiplier': 2.0, 'risk_pct': 0.1}


class ATRBreakout_50_14_2(ATRBreakoutAtom):
    """长期ATR突破：50周期MA，14周期ATR，2倍（更长期）"""
    name = "atr_breakout_50_14_2"
    params = {'ma_period': 50, 'atr_period': 14, 'multiplier': 2.0, 'risk_pct': 0.1}


class ATRBreakout_10_14_2(ATRBreakoutAtom):
    """短期ATR突破：10周期MA，14周期ATR，2倍（更短期）"""
    name = "atr_breakout_10_14_2"
    params = {'ma_period': 10, 'atr_period': 14, 'multiplier': 2.0, 'risk_pct': 0.1}


class ATRBreakout_Aggressive(ATRBreakoutAtom):
    """激进ATR突破：10周期MA，10周期ATR，1.5倍"""
    name = "atr_breakout_aggressive"
    params = {'ma_period': 10, 'atr_period': 10, 'multiplier': 1.5, 'risk_pct': 0.1}


class ATRBreakout_Conservative(ATRBreakoutAtom):
    """保守ATR突破：50周期MA，20周期ATR，3倍"""
    name = "atr_breakout_conservative"
    params = {'ma_period': 50, 'atr_period': 20, 'multiplier': 3.0, 'risk_pct': 0.1}
