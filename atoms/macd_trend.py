"""
MACD趋势策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy


class MACDTrendAtom(StrategyAtom):
    """
    MACD趋势策略
    
    MACD金叉买入，MACD死叉卖出
    """
    
    name = "macd_trend"
    params = {'fast': 12, 'slow': 26, 'signal': 9}
    
    def strategy_cls(self):
        p = self.params
        
        class Strategy(BaseStrategy):
            params = (
                ('fast', p['fast']),
                ('slow', p['slow']),
                ('signal', p['signal']),
            )
            
            def __init__(self):
                super().__init__()
                self.macd = bt.ind.MACD(
                    period_me1=self.p.fast,
                    period_me2=self.p.slow,
                    period_signal=self.p.signal
                )
                self.crossover = bt.ind.CrossOver(self.macd.macd, self.macd.signal)
            
            def next(self):
                if self.order:
                    return
                
                if not self.position:
                    if self.crossover > 0:
                        self.order = self.buy()
                else:
                    if self.crossover < 0:
                        self.order = self.sell()
        
        return Strategy


# 参数变体
class MACD_12_26_9(MACDTrendAtom):
    name = "macd_12_26_9"
    params = {'fast': 12, 'slow': 26, 'signal': 9}


class MACD_8_17_9(MACDTrendAtom):
    name = "macd_8_17_9"
    params = {'fast': 8, 'slow': 17, 'signal': 9}

