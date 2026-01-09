"""
三重均线策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer


class TripleMAAtom(StrategyAtom):
    """
    三重均线策略
    
    利用三条不同周期的均线判断趋势方向和入场时机：
    - 长均线定方向
    - 中均线确认趋势
    - 短均线找入场点
    
    做多条件：短均线 > 中均线 > 长均线（多头排列）
    做空条件：短均线 < 中均线 < 长均线（空头排列）
    入场时机：排列形成后，价格回踩中均线时入场
    离场：排列被打破时离场
    """
    
    name = "triple_ma"
    params = {'short': 10, 'medium': 30, 'long': 60, 'risk_pct': 0.1}
    
    def strategy_cls(self):
        short_period = self.params['short']
        medium_period = self.params['medium']
        long_period = self.params['long']
        
        class Strategy(BaseStrategy):
            params = (
                ('short', short_period),
                ('medium', medium_period),
                ('long', long_period),
            )
            
            def __init__(self):
                super().__init__()
                # 三条均线
                self.sma_short = bt.ind.SMA(period=self.p.short)
                self.sma_medium = bt.ind.SMA(period=self.p.medium)
                self.sma_long = bt.ind.SMA(period=self.p.long)
                
                # 用于检测回踩中均线
                self.prev_close = None
            
            def next(self):
                if self.order:
                    return
                
                # 当前价格和均线位置
                price = self.data.close[0]
                short_ma = self.sma_short[0]
                medium_ma = self.sma_medium[0]
                long_ma = self.sma_long[0]
                
                # 检查多头排列：短 > 中 > 长
                is_bullish_alignment = short_ma > medium_ma > long_ma
                # 检查空头排列：短 < 中 < 长
                is_bearish_alignment = short_ma < medium_ma < long_ma
                
                if not self.position:
                    # 做多信号：多头排列 + 价格回踩中均线
                    if is_bullish_alignment:
                        # 价格接近中均线（在中均线上下1%范围内）
                        if abs(price - medium_ma) / medium_ma < 0.01:
                            self.order = self.buy()
                    
                    # 做空信号：空头排列 + 价格反弹到中均线
                    elif is_bearish_alignment:
                        # 价格接近中均线（在中均线上下1%范围内）
                        if abs(price - medium_ma) / medium_ma < 0.01:
                            self.order = self.sell()
                
                else:
                    # 平多仓：多头排列被打破
                    if self.position.size > 0:
                        if not (short_ma > medium_ma > long_ma):
                            self.order = self.close()
                    
                    # 平空仓：空头排列被打破
                    elif self.position.size < 0:
                        if not (short_ma < medium_ma < long_ma):
                            self.order = self.close()
        
        return Strategy
    
    def sizer_cls(self):
        """使用固定1手"""
        return None  # 使用Runner默认的FixedSize


# 预定义的参数变体
class TripleMA_5_20_50(TripleMAAtom):
    """快速三重均线：5-20-50"""
    name = "triple_ma_5_20_50"
    params = {'short': 5, 'medium': 20, 'long': 50, 'risk_pct': 0.1}


class TripleMA_10_30_60(TripleMAAtom):
    """标准三重均线：10-30-60"""
    name = "triple_ma_10_30_60"
    params = {'short': 10, 'medium': 30, 'long': 60, 'risk_pct': 0.1}


class TripleMA_8_21_55(TripleMAAtom):
    """斐波那契三重均线：8-21-55"""
    name = "triple_ma_8_21_55"
    params = {'short': 8, 'medium': 21, 'long': 55, 'risk_pct': 0.1}


class TripleMA_12_26_52(TripleMAAtom):
    """保守三重均线：12-26-52"""
    name = "triple_ma_12_26_52"
    params = {'short': 12, 'medium': 26, 'long': 52, 'risk_pct': 0.1}
