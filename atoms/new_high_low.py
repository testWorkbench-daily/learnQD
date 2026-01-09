"""
N日新高新低策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer


class NewHighLowAtom(StrategyAtom):
    """
    N日新高新低策略
    
    创N日新高做多，创N日新低做空。
    最简单直接的动量突破策略，捕捉趋势启动。
    
    核心原理：
    - 新高 = 收盘价 > 过去N日最高收盘价
    - 新低 = 收盘价 < 过去N日最低收盘价
    - 新高新低表示强劲动量，趋势启动信号
    
    信号规则：
    - 做多：收盘价创N日新高
    - 做空：收盘价创N日新低
    - 离场：价格回落到N/2日低点（多）或反弹到N/2日高点（空）
    
    特点：
    - 纯粹的价格动量策略
    - 简单有效，易于执行
    - 适合趋势明显的市场
    - 可作为其他策略的过滤条件
    """
    
    name = "new_high_low"
    params = {'lookback_period': 20, 'risk_pct': 0.1}
    
    def strategy_cls(self):
        lookback_period = self.params['lookback_period']
        
        class Strategy(BaseStrategy):
            params = (
                ('lookback_period', lookback_period),
            )
            
            def __init__(self):
                super().__init__()
                # N日最高收盘价
                self.high_close = bt.ind.Highest(self.data.close, period=self.p.lookback_period)
                # N日最低收盘价
                self.low_close = bt.ind.Lowest(self.data.close, period=self.p.lookback_period)
                
                # N/2日最高/最低（用于离场）
                exit_period = max(1, self.p.lookback_period // 2)
                self.exit_high = bt.ind.Highest(self.data.close, period=exit_period)
                self.exit_low = bt.ind.Lowest(self.data.close, period=exit_period)
                
                # 记录是否是新高新低
                self.is_new_high = False
                self.is_new_low = False
            
            def next(self):
                if self.order:
                    return
                
                # 当前价格
                current_close = self.data.close[0]
                prev_close = self.data.close[-1] if len(self.data) > 1 else current_close
                
                # N日最高/最低（不包括当前bar）
                high_n = self.high_close[-1] if len(self.high_close) > 0 else current_close
                low_n = self.low_close[-1] if len(self.low_close) > 0 else current_close
                
                # 离场参考价格
                exit_high_price = self.exit_high[0]
                exit_low_price = self.exit_low[0]
                
                if not self.position:
                    # 做多信号：收盘价创N日新高
                    if current_close > high_n:
                        self.order = self.buy()
                        self.is_new_high = True
                        self.is_new_low = False
                    
                    # 做空信号：收盘价创N日新低
                    elif current_close < low_n:
                        self.order = self.sell()
                        self.is_new_low = True
                        self.is_new_high = False
                
                else:
                    # 平多仓条件
                    if self.position.size > 0:
                        # 1. 离场：价格回落到N/2日低点
                        if current_close <= exit_low_price:
                            self.order = self.close()
                            self.is_new_high = False
                        
                        # 2. 反向信号：创N日新低
                        elif current_close < low_n:
                            self.order = self.close()
                            self.is_new_high = False
                    
                    # 平空仓条件
                    elif self.position.size < 0:
                        # 1. 离场：价格反弹到N/2日高点
                        if current_close >= exit_high_price:
                            self.order = self.close()
                            self.is_new_low = False
                        
                        # 2. 反向信号：创N日新高
                        elif current_close > high_n:
                            self.order = self.close()
                            self.is_new_low = False
        
        return Strategy
    
    def sizer_cls(self):
        """使用固定1手"""
        return None  # 使用Runner默认的FixedSize


# 预定义的参数变体
class NewHighLow_20(NewHighLowAtom):
    """20日新高新低策略（月度新高新低）"""
    name = "new_hl_20"
    params = {'lookback_period': 20, 'risk_pct': 0.1}


class NewHighLow_50(NewHighLowAtom):
    """50日新高新低策略（季度新高新低）"""
    name = "new_hl_50"
    params = {'lookback_period': 50, 'risk_pct': 0.1}


class NewHighLow_100(NewHighLowAtom):
    """100日新高新低策略（半年新高新低）"""
    name = "new_hl_100"
    params = {'lookback_period': 100, 'risk_pct': 0.1}


class NewHighLow_250(NewHighLowAtom):
    """250日新高新低策略（年度新高新低）"""
    name = "new_hl_250"
    params = {'lookback_period': 250, 'risk_pct': 0.1}


class NewHighLow_10(NewHighLowAtom):
    """10日新高新低策略（短期动量）"""
    name = "new_hl_10"
    params = {'lookback_period': 10, 'risk_pct': 0.1}


class NewHighLow_5(NewHighLowAtom):
    """5日新高新低策略（超短期动量）"""
    name = "new_hl_5"
    params = {'lookback_period': 5, 'risk_pct': 0.1}


class NewHighLow_Aggressive(NewHighLowAtom):
    """激进新高新低策略：5日周期"""
    name = "new_hl_aggressive"
    params = {'lookback_period': 5, 'risk_pct': 0.1}


class NewHighLow_Conservative(NewHighLowAtom):
    """保守新高新低策略：100日周期"""
    name = "new_hl_conservative"
    params = {'lookback_period': 100, 'risk_pct': 0.1}
