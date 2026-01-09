"""
唐奇安通道策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer


class DonchianChannelAtom(StrategyAtom):
    """
    唐奇安通道策略（Donchian Channel）
    
    突破N日最高/最低价入场，是海龟法则的基础。
    最纯粹的突破系统，简单而有效。
    
    核心原理：
    - 上轨 = 过去N日最高价
    - 下轨 = 过去N日最低价
    - 中轨 = (上轨 + 下轨) / 2
    - 通道反映价格历史区间，突破表示趋势启动
    
    信号规则：
    - 做多：价格突破上轨（创N日新高）
    - 做空：价格跌破下轨（创N日新低）
    - 离场方式一：反向突破时平仓反手
    - 离场方式二：跌破exit_period日低点（多）或突破exit_period日高点（空）
    
    特点：
    - 纯粹的价格行为策略，不依赖指标
    - 海龟交易法则的核心组成部分
    - 适合趋势明显的市场
    """
    
    name = "donchian_channel"
    params = {'entry_period': 20, 'exit_period': 10, 'risk_pct': 0.1}
    
    def strategy_cls(self):
        entry_period = self.params['entry_period']
        exit_period = self.params['exit_period']
        
        class Strategy(BaseStrategy):
            params = (
                ('entry_period', entry_period),
                ('exit_period', exit_period),
            )
            
            def __init__(self):
                super().__init__()
                # 入场通道：N日最高/最低
                self.entry_high = bt.ind.Highest(self.data.high, period=self.p.entry_period)
                self.entry_low = bt.ind.Lowest(self.data.low, period=self.p.entry_period)
                
                # 离场通道：M日最高/最低（通常较短）
                self.exit_high = bt.ind.Highest(self.data.high, period=self.p.exit_period)
                self.exit_low = bt.ind.Lowest(self.data.low, period=self.p.exit_period)
                
                # 中轨
                self.mid_line = (self.entry_high + self.entry_low) / 2.0
            
            def next(self):
                if self.order:
                    return
                
                # 当前价格和通道值
                price = self.data.close[0]
                high = self.data.high[0]
                low = self.data.low[0]
                
                entry_upper = self.entry_high[0]
                entry_lower = self.entry_low[0]
                exit_upper = self.exit_high[0]
                exit_lower = self.exit_low[0]
                
                if not self.position:
                    # 做多信号：价格突破上轨（创N日新高）
                    if high > entry_upper:
                        self.order = self.buy()
                    
                    # 做空信号：价格跌破下轨（创N日新低）
                    elif low < entry_lower:
                        self.order = self.sell()
                
                else:
                    # 平多仓条件
                    if self.position.size > 0:
                        # 1. 离场方式一：反向突破（跌破入场下轨）
                        if low < entry_lower:
                            self.order = self.close()
                        
                        # 2. 离场方式二：跌破离场通道（短周期低点）
                        elif low < exit_lower:
                            self.order = self.close()
                    
                    # 平空仓条件
                    elif self.position.size < 0:
                        # 1. 离场方式一：反向突破（突破入场上轨）
                        if high > entry_upper:
                            self.order = self.close()
                        
                        # 2. 离场方式二：突破离场通道（短周期高点）
                        elif high > exit_upper:
                            self.order = self.close()
        
        return Strategy
    
    def sizer_cls(self):
        """使用固定1手"""
        return None  # 使用Runner默认的FixedSize


# 预定义的参数变体
class DonchianChannel_20_10(DonchianChannelAtom):
    """标准唐奇安通道：20日入场，10日离场"""
    name = "donchian_20_10"
    params = {'entry_period': 20, 'exit_period': 10, 'risk_pct': 0.1}


class DonchianChannel_55_20(DonchianChannelAtom):
    """海龟法则原版：55日入场，20日离场"""
    name = "donchian_55_20"
    params = {'entry_period': 55, 'exit_period': 20, 'risk_pct': 0.1}


class DonchianChannel_20_20(DonchianChannelAtom):
    """简化唐奇安通道：20日入场，反向突破离场"""
    name = "donchian_20_20"
    params = {'entry_period': 20, 'exit_period': 20, 'risk_pct': 0.1}


class DonchianChannel_10_5(DonchianChannelAtom):
    """短期唐奇安通道：10日入场，5日离场（适合短线）"""
    name = "donchian_10_5"
    params = {'entry_period': 10, 'exit_period': 5, 'risk_pct': 0.1}


class DonchianChannel_5_3(DonchianChannelAtom):
    """超短期唐奇安通道：5日入场，3日离场（日内/短线）"""
    name = "donchian_5_3"
    params = {'entry_period': 5, 'exit_period': 3, 'risk_pct': 0.1}


class DonchianChannel_40_15(DonchianChannelAtom):
    """中期唐奇安通道：40日入场，15日离场"""
    name = "donchian_40_15"
    params = {'entry_period': 40, 'exit_period': 15, 'risk_pct': 0.1}


class DonchianChannel_TurtleSystem1(DonchianChannelAtom):
    """海龟法则系统一：20日入场，10日离场"""
    name = "donchian_turtle_sys1"
    params = {'entry_period': 20, 'exit_period': 10, 'risk_pct': 0.1}


class DonchianChannel_TurtleSystem2(DonchianChannelAtom):
    """海龟法则系统二：55日入场，20日离场"""
    name = "donchian_turtle_sys2"
    params = {'entry_period': 55, 'exit_period': 20, 'risk_pct': 0.1}


class DonchianChannel_Aggressive(DonchianChannelAtom):
    """激进唐奇安通道：10日入场，10日离场（更快进出）"""
    name = "donchian_aggressive"
    params = {'entry_period': 10, 'exit_period': 10, 'risk_pct': 0.1}


class DonchianChannel_Conservative(DonchianChannelAtom):
    """保守唐奇安通道：60日入场，30日离场（更稳健）"""
    name = "donchian_conservative"
    params = {'entry_period': 60, 'exit_period': 30, 'risk_pct': 0.1}
