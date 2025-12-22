"""
布林带突破策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy


class BollingerBreakoutAtom(StrategyAtom):
    """
    布林带突破策略
    
    突破上轨买入，突破下轨卖出
    """
    
    name = "boll_breakout"
    params = {'period': 20, 'devfactor': 2.0}
    
    def strategy_cls(self):
        p = self.params
        
        class Strategy(BaseStrategy):
            params = (
                ('period', p['period']),
                ('devfactor', p['devfactor']),
            )
            
            def __init__(self):
                super().__init__()
                self.boll = bt.ind.BollingerBands(
                    period=self.p.period,
                    devfactor=self.p.devfactor
                )
            
            def next(self):
                if self.order:
                    return
                
                if not self.position:
                    # 突破上轨买入
                    if self.data.close[0] > self.boll.top[0]:
                        self.order = self.buy()
                else:
                    # 跌破下轨卖出
                    if self.data.close[0] < self.boll.bot[0]:
                        self.order = self.sell()
        
        return Strategy


class BollingerMeanReversion(StrategyAtom):
    """
    布林带均值回归策略
    
    触及下轨买入，触及上轨卖出
    """
    
    name = "boll_mean_revert"
    params = {'period': 20, 'devfactor': 2.0}
    
    def strategy_cls(self):
        p = self.params
        
        class Strategy(BaseStrategy):
            params = (
                ('period', p['period']),
                ('devfactor', p['devfactor']),
            )
            
            def __init__(self):
                super().__init__()
                self.boll = bt.ind.BollingerBands(
                    period=self.p.period,
                    devfactor=self.p.devfactor
                )
            
            def next(self):
                if self.order:
                    return
                
                if not self.position:
                    # 触及下轨买入（均值回归）
                    if self.data.close[0] < self.boll.bot[0]:
                        self.order = self.buy()
                else:
                    # 触及上轨卖出
                    if self.data.close[0] > self.boll.top[0]:
                        self.order = self.sell()
        
        return Strategy

