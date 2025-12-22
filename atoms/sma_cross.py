"""
双均线交叉策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer


class SMACrossAtom(StrategyAtom):
    """
    双均线交叉策略
    
    金叉买入，死叉卖出
    """
    
    name = "sma_cross"
    params = {'fast': 10, 'slow': 30, 'risk_pct': 0.1}
    
    def strategy_cls(self):
        fast = self.params['fast']
        slow = self.params['slow']
        
        class Strategy(BaseStrategy):
            params = (
                ('fast', fast),
                ('slow', slow),
            )
            
            def __init__(self):
                super().__init__()
                self.sma_fast = bt.ind.SMA(period=self.p.fast)
                self.sma_slow = bt.ind.SMA(period=self.p.slow)
                self.crossover = bt.ind.CrossOver(self.sma_fast, self.sma_slow)
            
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
    
    def sizer_cls(self):
        """使用固定1手"""
        return None  # 使用Runner默认的FixedSize


# 预定义的参数变体
class SMACross_5_20(SMACrossAtom):
    name = "sma_5_20"
    params = {'fast': 5, 'slow': 20, 'risk_pct': 0.1}


class SMACross_10_30(SMACrossAtom):
    name = "sma_10_30"
    params = {'fast': 10, 'slow': 30, 'risk_pct': 0.1}


class SMACross_20_60(SMACrossAtom):
    name = "sma_20_60"
    params = {'fast': 20, 'slow': 60, 'risk_pct': 0.1}

