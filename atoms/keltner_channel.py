"""
Keltner通道回归策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer


class KeltnerChannelAtom(StrategyAtom):
    """
    Keltner通道回归策略
    
    价格突破Keltner通道后回归中轨。与布林带类似，但使用ATR而非标准差，
    对波动更敏感，更适合趋势市场。
    
    核心原理：
    - 中轨 = EMA(period)
    - 上轨 = 中轨 + multiplier × ATR
    - 下轨 = 中轨 - multiplier × ATR
    - 通道基于真实波动率（ATR），更贴近实际波动
    
    信号规则：
    - 做多：价格跌破下轨后回升入轨（超卖反弹）
    - 做空：价格突破上轨后回落入轨（超买回落）
    - 平仓：价格回归中轨
    - 止损：价格继续远离通道
    """
    
    name = "keltner_channel"
    params = {'ema_period': 20, 'atr_period': 10, 'multiplier': 1.5, 'risk_pct': 0.1}
    
    def strategy_cls(self):
        ema_period = self.params['ema_period']
        atr_period = self.params['atr_period']
        multiplier = self.params['multiplier']
        
        class Strategy(BaseStrategy):
            params = (
                ('ema_period', ema_period),
                ('atr_period', atr_period),
                ('multiplier', multiplier),
            )
            
            def __init__(self):
                super().__init__()
                # 中轨：EMA
                self.ema = bt.ind.EMA(period=self.p.ema_period)
                
                # ATR指标
                self.atr = bt.ind.ATR(period=self.p.atr_period)
                
                # 上轨和下轨
                self.upper_band = self.ema + self.p.multiplier * self.atr
                self.lower_band = self.ema - self.p.multiplier * self.atr
                
                # 记录是否在通道外
                self.outside_channel = False
                self.outside_direction = None  # 'above' or 'below'
            
            def next(self):
                if self.order:
                    return
                
                # 当前价格和通道值
                price = self.data.close[0]
                mid_line = self.ema[0]
                upper_line = self.upper_band[0]
                lower_line = self.lower_band[0]
                
                # 检测价格是否在通道外
                is_above_upper = price > upper_line
                is_below_lower = price < lower_line
                is_inside = lower_line <= price <= upper_line
                
                if not self.position:
                    # 检测价格跌破下轨
                    if is_below_lower:
                        self.outside_channel = True
                        self.outside_direction = 'below'
                    # 检测价格突破上轨
                    elif is_above_upper:
                        self.outside_channel = True
                        self.outside_direction = 'above'
                    
                    # 做多信号：价格跌破下轨后回升入轨
                    if self.outside_channel and self.outside_direction == 'below' and is_inside:
                        self.order = self.buy()
                        self.outside_channel = False
                        self.outside_direction = None
                    
                    # 做空信号：价格突破上轨后回落入轨
                    elif self.outside_channel and self.outside_direction == 'above' and is_inside:
                        self.order = self.sell()
                        self.outside_channel = False
                        self.outside_direction = None
                
                else:
                    # 平多仓条件
                    if self.position.size > 0:
                        # 1. 获利：价格回归中轨附近
                        if price >= mid_line:
                            self.order = self.close()
                        
                        # 2. 止损：价格继续下跌，远离下轨（2倍ATR）
                        elif price < (lower_line - self.atr[0]):
                            self.order = self.close()
                        
                        # 3. 反向信号：价格突破上轨
                        elif is_above_upper:
                            self.order = self.close()
                    
                    # 平空仓条件
                    elif self.position.size < 0:
                        # 1. 获利：价格回归中轨附近
                        if price <= mid_line:
                            self.order = self.close()
                        
                        # 2. 止损：价格继续上涨，远离上轨（2倍ATR）
                        elif price > (upper_line + self.atr[0]):
                            self.order = self.close()
                        
                        # 3. 反向信号：价格跌破下轨
                        elif is_below_lower:
                            self.order = self.close()
        
        return Strategy
    
    def sizer_cls(self):
        """使用固定1手"""
        return None  # 使用Runner默认的FixedSize


# 预定义的参数变体
class KeltnerChannel_20_10_1_5(KeltnerChannelAtom):
    """标准Keltner通道：20周期EMA，10周期ATR，1.5倍"""
    name = "keltner_20_10_1_5"
    params = {'ema_period': 20, 'atr_period': 10, 'multiplier': 1.5, 'risk_pct': 0.1}


class KeltnerChannel_20_10_2_0(KeltnerChannelAtom):
    """宽幅Keltner通道：20周期EMA，10周期ATR，2.0倍（更宽容）"""
    name = "keltner_20_10_2_0"
    params = {'ema_period': 20, 'atr_period': 10, 'multiplier': 2.0, 'risk_pct': 0.1}


class KeltnerChannel_20_10_1_0(KeltnerChannelAtom):
    """窄幅Keltner通道：20周期EMA，10周期ATR，1.0倍（更敏感）"""
    name = "keltner_20_10_1_0"
    params = {'ema_period': 20, 'atr_period': 10, 'multiplier': 1.0, 'risk_pct': 0.1}


class KeltnerChannel_20_14_1_5(KeltnerChannelAtom):
    """标准ATR Keltner通道：20周期EMA，14周期ATR，1.5倍"""
    name = "keltner_20_14_1_5"
    params = {'ema_period': 20, 'atr_period': 14, 'multiplier': 1.5, 'risk_pct': 0.1}


class KeltnerChannel_30_10_1_5(KeltnerChannelAtom):
    """平滑Keltner通道：30周期EMA，10周期ATR，1.5倍（更平滑）"""
    name = "keltner_30_10_1_5"
    params = {'ema_period': 30, 'atr_period': 10, 'multiplier': 1.5, 'risk_pct': 0.1}


class KeltnerChannel_10_10_1_5(KeltnerChannelAtom):
    """快速Keltner通道：10周期EMA，10周期ATR，1.5倍（更快）"""
    name = "keltner_10_10_1_5"
    params = {'ema_period': 10, 'atr_period': 10, 'multiplier': 1.5, 'risk_pct': 0.1}


class KeltnerChannel_Tight(KeltnerChannelAtom):
    """紧凑Keltner通道：20周期EMA，10周期ATR，1.2倍（紧凑通道）"""
    name = "keltner_tight"
    params = {'ema_period': 20, 'atr_period': 10, 'multiplier': 1.2, 'risk_pct': 0.1}
