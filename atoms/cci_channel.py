"""
CCI通道回归策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer


class CCIChannelAtom(StrategyAtom):
    """
    CCI通道回归策略
    
    CCI（Commodity Channel Index，商品通道指数）衡量价格偏离统计均值的程度。
    极端值后预期回归正常范围。
    
    核心原理：
    - CCI > +100：超买区域
    - CCI < -100：超卖区域
    - CCI > +200 或 < -200：极端超买/超卖
    - CCI回归0轴：价格回归正常
    
    计算方法：
    - 典型价格 TP = (最高 + 最低 + 收盘) / 3
    - CCI = (TP - MA(TP)) / (0.015 × 平均绝对偏差)
    
    信号规则：
    - 做多：CCI从-100下方回升穿越-100（超卖反弹）
    - 做空：CCI从+100上方回落穿越+100（超买回落）
    - 平仓：CCI回归0轴附近
    - 止损：CCI继续极端化，超过±200
    """
    
    name = "cci_channel"
    params = {'period': 20, 'overbought': 100, 'oversold': -100, 'risk_pct': 0.1}
    
    def strategy_cls(self):
        period = self.params['period']
        overbought = self.params['overbought']
        oversold = self.params['oversold']
        
        class Strategy(BaseStrategy):
            params = (
                ('period', period),
                ('overbought', overbought),
                ('oversold', oversold),
            )
            
            def __init__(self):
                super().__init__()
                # CCI指标（backtrader内置）
                self.cci = bt.ind.CommodityChannelIndex(period=self.p.period)
                
                # 记录前一个CCI值，用于检测穿越
                self.prev_cci = None
            
            def next(self):
                if self.order:
                    return
                
                # 当前CCI值
                cci_value = self.cci[0]
                
                # 获取前一个CCI值
                prev_cci_value = self.prev_cci if self.prev_cci is not None else cci_value
                
                if not self.position:
                    # 做多信号：CCI从-100下方回升穿越-100（超卖反弹）
                    if prev_cci_value < self.p.oversold and cci_value >= self.p.oversold:
                        self.order = self.buy()
                    
                    # 做空信号：CCI从+100上方回落穿越+100（超买回落）
                    elif prev_cci_value > self.p.overbought and cci_value <= self.p.overbought:
                        self.order = self.sell()
                
                else:
                    # 平多仓条件
                    if self.position.size > 0:
                        # 1. 获利：CCI回归0轴附近（在-20到+20之间）
                        if -20 <= cci_value <= 20:
                            self.order = self.close()
                        
                        # 2. 止损：CCI继续下跌，超过-200（极端超卖）
                        elif cci_value < -200:
                            self.order = self.close()
                        
                        # 3. 反向信号：CCI突破+100进入超买
                        elif cci_value > self.p.overbought:
                            self.order = self.close()
                    
                    # 平空仓条件
                    elif self.position.size < 0:
                        # 1. 获利：CCI回归0轴附近（在-20到+20之间）
                        if -20 <= cci_value <= 20:
                            self.order = self.close()
                        
                        # 2. 止损：CCI继续上涨，超过+200（极端超买）
                        elif cci_value > 200:
                            self.order = self.close()
                        
                        # 3. 反向信号：CCI跌破-100进入超卖
                        elif cci_value < self.p.oversold:
                            self.order = self.close()
                
                # 更新前一个CCI值
                self.prev_cci = cci_value
        
        return Strategy
    
    def sizer_cls(self):
        """使用固定1手"""
        return None  # 使用Runner默认的FixedSize


# 预定义的参数变体
class CCIChannel_20_100(CCIChannelAtom):
    """标准CCI策略：20周期，±100阈值"""
    name = "cci_20_100"
    params = {'period': 20, 'overbought': 100, 'oversold': -100, 'risk_pct': 0.1}


class CCIChannel_20_150(CCIChannelAtom):
    """保守CCI策略：20周期，±150阈值（更极端才交易）"""
    name = "cci_20_150"
    params = {'period': 20, 'overbought': 150, 'oversold': -150, 'risk_pct': 0.1}


class CCIChannel_20_80(CCIChannelAtom):
    """激进CCI策略：20周期，±80阈值（更早入场）"""
    name = "cci_20_80"
    params = {'period': 20, 'overbought': 80, 'oversold': -80, 'risk_pct': 0.1}


class CCIChannel_14_100(CCIChannelAtom):
    """快速CCI策略：14周期，±100阈值（更敏感）"""
    name = "cci_14_100"
    params = {'period': 14, 'overbought': 100, 'oversold': -100, 'risk_pct': 0.1}


class CCIChannel_30_100(CCIChannelAtom):
    """平滑CCI策略：30周期，±100阈值（更平滑）"""
    name = "cci_30_100"
    params = {'period': 30, 'overbought': 100, 'oversold': -100, 'risk_pct': 0.1}


class CCIChannel_Strict(CCIChannelAtom):
    """严格CCI策略：20周期，±120阈值（更严格条件）"""
    name = "cci_strict"
    params = {'period': 20, 'overbought': 120, 'oversold': -120, 'risk_pct': 0.1}
