"""
波动率突破策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer


class VolatilityBreakoutAtom(StrategyAtom):
    """
    波动率突破策略（Volatility Breakout）
    
    当价格变动超过前期波动率的某个倍数时，认为趋势启动。
    基于波动率的动态突破系统，适应市场波动变化。
    
    核心原理：
    - 计算前N日平均波幅（使用ATR）
    - 做多：今日涨幅 > K × ATR（异常上涨）
    - 做空：今日跌幅 > K × ATR（异常下跌）
    - 止损：固定2倍ATR或当日开盘价
    
    特点：
    - 捕捉异常价格波动
    - 自适应市场波动率
    - 过滤正常波动，只捕捉突破性行情
    - 适合波动较大的市场
    """
    
    name = "volatility_breakout"
    params = {'volatility_period': 14, 'breakout_multiplier': 2.0, 'risk_pct': 0.1}
    
    def strategy_cls(self):
        volatility_period = self.params['volatility_period']
        breakout_multiplier = self.params['breakout_multiplier']
        
        class Strategy(BaseStrategy):
            params = (
                ('volatility_period', volatility_period),
                ('breakout_multiplier', breakout_multiplier),
            )
            
            def __init__(self):
                super().__init__()
                # ATR作为波动率指标
                self.atr = bt.ind.ATR(period=self.p.volatility_period)
                
                # 记录开盘价和止损价
                self.entry_price = None
                self.stop_loss = None
                self.entry_open = None
            
            def next(self):
                if self.order:
                    return
                
                # 当前价格数据
                current_open = self.data.open[0]
                current_close = self.data.close[0]
                current_high = self.data.high[0]
                current_low = self.data.low[0]
                
                # 当前波动率
                atr_value = self.atr[0]
                
                # 计算当日价格变动
                daily_change = current_close - current_open
                
                if not self.position:
                    # 做多信号：今日涨幅 > K × ATR（异常上涨突破）
                    if daily_change > self.p.breakout_multiplier * atr_value:
                        self.order = self.buy()
                        self.entry_price = current_close
                        self.entry_open = current_open
                        # 止损设在当日开盘价或入场价 - 2倍ATR
                        self.stop_loss = max(current_open, current_close - 2 * atr_value)
                    
                    # 做空信号：今日跌幅 > K × ATR（异常下跌突破）
                    elif daily_change < -self.p.breakout_multiplier * atr_value:
                        self.order = self.sell()
                        self.entry_price = current_close
                        self.entry_open = current_open
                        # 止损设在当日开盘价或入场价 + 2倍ATR
                        self.stop_loss = min(current_open, current_close + 2 * atr_value)
                
                else:
                    # 平多仓条件
                    if self.position.size > 0:
                        # 1. 止损：价格跌破止损线
                        if current_close <= self.stop_loss:
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
                            self.entry_open = None
                        
                        # 2. 获利：价格回落超过1倍ATR
                        elif current_close < (self.entry_price - atr_value):
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
                            self.entry_open = None
                        
                        # 3. 更新跟踪止损（向上移动）
                        else:
                            trailing_stop = current_close - 2 * atr_value
                            if trailing_stop > self.stop_loss:
                                self.stop_loss = trailing_stop
                    
                    # 平空仓条件
                    elif self.position.size < 0:
                        # 1. 止损：价格突破止损线
                        if current_close >= self.stop_loss:
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
                            self.entry_open = None
                        
                        # 2. 获利：价格反弹超过1倍ATR
                        elif current_close > (self.entry_price + atr_value):
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
                            self.entry_open = None
                        
                        # 3. 更新跟踪止损（向下移动）
                        else:
                            trailing_stop = current_close + 2 * atr_value
                            if trailing_stop < self.stop_loss:
                                self.stop_loss = trailing_stop
        
        return Strategy
    
    def sizer_cls(self):
        """使用固定1手"""
        return None  # 使用Runner默认的FixedSize


# 预定义的参数变体
class VolatilityBreakout_14_2(VolatilityBreakoutAtom):
    """标准波动率突破：14周期，2倍ATR"""
    name = "vol_breakout_14_2"
    params = {'volatility_period': 14, 'breakout_multiplier': 2.0, 'risk_pct': 0.1}


class VolatilityBreakout_14_2_5(VolatilityBreakoutAtom):
    """保守波动率突破：14周期，2.5倍ATR（更大突破才入场）"""
    name = "vol_breakout_14_2_5"
    params = {'volatility_period': 14, 'breakout_multiplier': 2.5, 'risk_pct': 0.1}


class VolatilityBreakout_14_1_5(VolatilityBreakoutAtom):
    """激进波动率突破：14周期，1.5倍ATR（更敏感）"""
    name = "vol_breakout_14_1_5"
    params = {'volatility_period': 14, 'breakout_multiplier': 1.5, 'risk_pct': 0.1}


class VolatilityBreakout_20_2(VolatilityBreakoutAtom):
    """平滑波动率突破：20周期，2倍ATR（更平滑）"""
    name = "vol_breakout_20_2"
    params = {'volatility_period': 20, 'breakout_multiplier': 2.0, 'risk_pct': 0.1}


class VolatilityBreakout_10_2(VolatilityBreakoutAtom):
    """快速波动率突破：10周期，2倍ATR（更快响应）"""
    name = "vol_breakout_10_2"
    params = {'volatility_period': 10, 'breakout_multiplier': 2.0, 'risk_pct': 0.1}


class VolatilityBreakout_10_3(VolatilityBreakoutAtom):
    """极端波动率突破：10周期，3倍ATR（只捕捉极端行情）"""
    name = "vol_breakout_10_3"
    params = {'volatility_period': 10, 'breakout_multiplier': 3.0, 'risk_pct': 0.1}


class VolatilityBreakout_Aggressive(VolatilityBreakoutAtom):
    """超激进波动率突破：10周期，1.5倍ATR"""
    name = "vol_breakout_aggressive"
    params = {'volatility_period': 10, 'breakout_multiplier': 1.5, 'risk_pct': 0.1}


class VolatilityBreakout_Conservative(VolatilityBreakoutAtom):
    """超保守波动率突破：20周期，3倍ATR"""
    name = "vol_breakout_conservative"
    params = {'volatility_period': 20, 'breakout_multiplier': 3.0, 'risk_pct': 0.1}
