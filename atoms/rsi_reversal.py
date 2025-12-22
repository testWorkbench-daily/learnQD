"""
RSI反转策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy


class RSIReversalAtom(StrategyAtom):
    """
    RSI反转策略
    
    RSI超卖买入，RSI超买卖出
    """
    
    name = "rsi_reversal"
    params = {'period': 14, 'oversold': 30, 'overbought': 70}
    
    def strategy_cls(self):
        p = self.params
        
        class Strategy(BaseStrategy):
            params = (
                ('period', p['period']),
                ('oversold', p['oversold']),
                ('overbought', p['overbought']),
            )
            
            def __init__(self):
                super().__init__()
                self.rsi = bt.ind.RSI(period=self.p.period)
            
            def next(self):
                if self.order:
                    return
                
                if not self.position:
                    if self.rsi < self.p.oversold:
                        self.order = self.buy()
                else:
                    if self.rsi > self.p.overbought:
                        self.order = self.sell()
        
        return Strategy


# 参数变体
class RSI_14_30_70(RSIReversalAtom):
    name = "rsi_14_30_70"
    params = {'period': 14, 'oversold': 30, 'overbought': 70}


class RSI_7_20_80(RSIReversalAtom):
    name = "rsi_7_20_80"
    params = {'period': 7, 'oversold': 20, 'overbought': 80}

