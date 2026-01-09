"""
ADX趋势强度策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer


class ADXTrendAtom(StrategyAtom):
    """
    ADX趋势强度策略
    
    ADX（Average Directional Index）衡量趋势强度，配合+DI/-DI判断方向。
    只在趋势明确时交易，避免震荡市。
    
    核心原理：
    - ADX > 阈值：表示有明确趋势
    - +DI > -DI：多头趋势
    - -DI > +DI：空头趋势
    - ADX < 20：无趋势，不交易
    
    信号规则：
    - 做多：ADX > 阈值 且 +DI > -DI
    - 做空：ADX > 阈值 且 -DI > +DI
    - 离场：ADX拐头向下或DI交叉
    """
    
    name = "adx_trend"
    params = {'period': 14, 'adx_threshold': 25, 'risk_pct': 0.1}
    
    def strategy_cls(self):
        period = self.params['period']
        adx_threshold = self.params['adx_threshold']
        
        class Strategy(BaseStrategy):
            params = (
                ('period', period),
                ('adx_threshold', adx_threshold),
            )
            
            def __init__(self):
                super().__init__()
                # 先创建 DI 指标（DirectionalIndicator）
                self.di = bt.ind.DirectionalIndicator(period=self.p.period)
                self.plusDI = self.di.plusDI
                self.minusDI = self.di.minusDI
                
                # ADX指标（使用 DI 指标）
                self.adx = bt.ind.AverageDirectionalMovementIndex(period=self.p.period)
                
                # 记录上一个ADX值，用于检测拐点
                self.prev_adx = None
            
            def next(self):
                if self.order:
                    return
                
                # 当前ADX值和方向指标
                adx_value = self.adx[0]
                plus_di = self.plusDI[0]
                minus_di = self.minusDI[0]
                
                # 检测ADX是否拐头向下
                adx_declining = False
                if self.prev_adx is not None and len(self.adx) > 1:
                    adx_declining = adx_value < self.adx[-1] < self.adx[-2]
                
                if not self.position:
                    # 做多条件：ADX > 阈值 且 +DI > -DI
                    if adx_value > self.p.adx_threshold and plus_di > minus_di:
                        self.order = self.buy()
                    
                    # 做空条件：ADX > 阈值 且 -DI > +DI
                    elif adx_value > self.p.adx_threshold and minus_di > plus_di:
                        self.order = self.sell()
                
                else:
                    # 平多仓条件
                    if self.position.size > 0:
                        # 1. ADX拐头向下
                        # 2. DI交叉：-DI上穿+DI
                        # 3. ADX跌破20（失去趋势）
                        if adx_declining or minus_di > plus_di or adx_value < 20:
                            self.order = self.close()
                    
                    # 平空仓条件
                    elif self.position.size < 0:
                        # 1. ADX拐头向下
                        # 2. DI交叉：+DI上穿-DI
                        # 3. ADX跌破20（失去趋势）
                        if adx_declining or plus_di > minus_di or adx_value < 20:
                            self.order = self.close()
                
                # 更新前一个ADX值
                self.prev_adx = adx_value
        
        return Strategy
    
    def sizer_cls(self):
        """使用固定1手"""
        return None  # 使用Runner默认的FixedSize


# 预定义的参数变体
class ADXTrend_14_25(ADXTrendAtom):
    """标准ADX策略：14周期，阈值25"""
    name = "adx_14_25"
    params = {'period': 14, 'adx_threshold': 25, 'risk_pct': 0.1}


class ADXTrend_14_30(ADXTrendAtom):
    """保守ADX策略：14周期，阈值30（更强趋势）"""
    name = "adx_14_30"
    params = {'period': 14, 'adx_threshold': 30, 'risk_pct': 0.1}


class ADXTrend_14_20(ADXTrendAtom):
    """激进ADX策略：14周期，阈值20（更早入场）"""
    name = "adx_14_20"
    params = {'period': 14, 'adx_threshold': 20, 'risk_pct': 0.1}


class ADXTrend_21_25(ADXTrendAtom):
    """平滑ADX策略：21周期，阈值25（更平滑信号）"""
    name = "adx_21_25"
    params = {'period': 21, 'adx_threshold': 25, 'risk_pct': 0.1}


class ADXTrend_10_25(ADXTrendAtom):
    """快速ADX策略：10周期，阈值25（更敏感）"""
    name = "adx_10_25"
    params = {'period': 10, 'adx_threshold': 25, 'risk_pct': 0.1}
