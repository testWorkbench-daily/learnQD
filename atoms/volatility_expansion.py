"""
波动率突破入场策略（Volatility Expansion Breakout）
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer


class VolatilityExpansionAtom(StrategyAtom):
    """
    波动率突破入场策略

    核心思想：
    - 波动率从低位突然放大，往往预示着大行情启动
    - 市场在低波动期处于盘整状态
    - 波动率扩张 + 价格突破 = 强烈趋势信号

    策略逻辑：
    1. 识别低波动期：ATR 处于 N 日最低水平或布林带收窄
    2. 等待波动率扩张：ATR 突然放大超过均值的阈值倍数
    3. 价格突破确认：价格同时突破关键价格水平
    4. 入场方向：跟随突破方向

    关键指标：
    - ATR（波动率测量）
    - ATR 移动平均（识别波动率趋势）
    - 布林带宽度（识别收缩期）
    - 价格突破（确认方向）

    适用场景：
    - 适合捕捉盘整后的突破行情
    - 适合趋势启动初期入场
    - 避免在高波动期追涨杀跌
    """

    name = "volatility_expansion"
    params = {
        'atr_period': 14,              # ATR 周期
        'atr_lookback': 50,            # ATR 低点回顾期
        'atr_expansion_mult': 1.5,     # ATR 扩张倍数阈值
        'bb_period': 20,               # 布林带周期
        'bb_dev': 2.0,                 # 布林带标准差
        'price_breakout_period': 20,   # 价格突破周期
        'risk_pct': 0.1,
    }

    def strategy_cls(self):
        atr_period = self.params['atr_period']
        atr_lookback = self.params['atr_lookback']
        atr_expansion_mult = self.params['atr_expansion_mult']
        bb_period = self.params['bb_period']
        bb_dev = self.params['bb_dev']
        price_breakout_period = self.params['price_breakout_period']

        class Strategy(BaseStrategy):
            params = (
                ('atr_period', atr_period),
                ('atr_lookback', atr_lookback),
                ('atr_expansion_mult', atr_expansion_mult),
                ('bb_period', bb_period),
                ('bb_dev', bb_dev),
                ('price_breakout_period', price_breakout_period),
            )

            def __init__(self):
                super().__init__()
                
                # ATR 及其统计指标
                self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
                self.atr_ma = bt.ind.SMA(self.atr, period=self.p.atr_period)
                self.atr_lowest = bt.ind.Lowest(self.atr, period=self.p.atr_lookback)
                
                # 布林带（用于识别收缩期）
                self.bb = bt.ind.BollingerBands(
                    self.data.close,
                    period=self.p.bb_period,
                    devfactor=self.p.bb_dev
                )
                # 布林带宽度（归一化）
                self.bb_width = (self.bb.top - self.bb.bot) / self.bb.mid
                self.bb_width_ma = bt.ind.SMA(self.bb_width, period=self.p.bb_period)
                
                # 价格突破指标
                self.price_high = bt.ind.Highest(self.data.high, period=self.p.price_breakout_period)
                self.price_low = bt.ind.Lowest(self.data.low, period=self.p.price_breakout_period)
                
                # 状态标记
                self.low_vol_detected = False
                self.entry_price = None

            def _is_low_volatility(self):
                """判断是否处于低波动期"""
                if len(self.atr) < self.p.atr_lookback:
                    return False
                
                # 条件1: ATR 接近 N 日最低
                atr_near_low = self.atr[0] <= self.atr_lowest[0] * 1.1
                
                # 条件2: 布林带收窄（相对均值）
                if len(self.bb_width_ma) > 0:
                    bb_contracted = self.bb_width[0] < self.bb_width_ma[0] * 0.8
                else:
                    bb_contracted = False
                
                return atr_near_low or bb_contracted

            def _is_volatility_expanding(self):
                """判断波动率是否正在扩张"""
                if len(self.atr_ma) == 0:
                    return False
                
                # ATR 突破其移动平均线的阈值倍数
                return self.atr[0] > self.atr_ma[0] * self.p.atr_expansion_mult

            def next(self):
                if self.order:
                    return
                
                # 确保指标有足够数据
                if len(self.atr) < self.p.atr_period:
                    return
                
                # 获取当前价格
                close = self.data.close[0]
                high = self.data.high[0]
                low = self.data.low[0]
                
                # 获取突破参考价
                if len(self.price_high) > 1 and len(self.price_low) > 1:
                    breakout_high = self.price_high[-1]
                    breakout_low = self.price_low[-1]
                else:
                    return
                
                # 1. 首先检测低波动期
                if not self.position and not self.low_vol_detected:
                    if self._is_low_volatility():
                        self.low_vol_detected = True
                
                # 2. 在低波动期后，等待波动率扩张 + 价格突破
                if not self.position and self.low_vol_detected:
                    # 检查波动率是否扩张
                    vol_expanding = self._is_volatility_expanding()
                    
                    if vol_expanding:
                        # 向上突破
                        if high > breakout_high:
                            self.order = self.buy()
                            self.entry_price = close
                            self.low_vol_detected = False  # 重置状态
                        
                        # 向下突破
                        elif low < breakout_low:
                            self.order = self.sell()
                            self.entry_price = close
                            self.low_vol_detected = False  # 重置状态
                
                # 3. 持仓管理
                elif self.position:
                    atr_value = self.atr[0]
                    
                    if self.position.size > 0:
                        # 多头止损：跌破布林带中轨或ATR止损
                        stop_loss = self.entry_price - 2 * atr_value if self.entry_price else close - 2 * atr_value
                        
                        if close < stop_loss or close < self.bb.mid[0]:
                            self.order = self.close()
                            self.entry_price = None
                        
                        # 波动率重新收缩，获利了结
                        elif not self._is_volatility_expanding() and close > self.entry_price:
                            self.order = self.close()
                            self.entry_price = None
                    
                    elif self.position.size < 0:
                        # 空头止损：突破布林带中轨或ATR止损
                        stop_loss = self.entry_price + 2 * atr_value if self.entry_price else close + 2 * atr_value
                        
                        if close > stop_loss or close > self.bb.mid[0]:
                            self.order = self.close()
                            self.entry_price = None
                        
                        # 波动率重新收缩，获利了结
                        elif not self._is_volatility_expanding() and close < self.entry_price:
                            self.order = self.close()
                            self.entry_price = None

        return Strategy

    def sizer_cls(self):
        """使用固定手数"""
        return None


# 预定义的参数变体
class VolatilityExpansion_Standard(VolatilityExpansionAtom):
    """波动率突破：标准配置（1.5倍扩张）"""
    name = "vol_expansion_standard"
    params = {
        'atr_period': 14,
        'atr_lookback': 50,
        'atr_expansion_mult': 1.5,
        'bb_period': 20,
        'bb_dev': 2.0,
        'price_breakout_period': 20,
        'risk_pct': 0.1,
    }


class VolatilityExpansion_Sensitive(VolatilityExpansionAtom):
    """波动率突破：敏感配置（1.3倍扩张，更早入场）"""
    name = "vol_expansion_sensitive"
    params = {
        'atr_period': 14,
        'atr_lookback': 50,
        'atr_expansion_mult': 1.3,
        'bb_period': 20,
        'bb_dev': 2.0,
        'price_breakout_period': 20,
        'risk_pct': 0.1,
    }


class VolatilityExpansion_Conservative(VolatilityExpansionAtom):
    """波动率突破：保守配置（2.0倍扩张，只做极端突破）"""
    name = "vol_expansion_conservative"
    params = {
        'atr_period': 14,
        'atr_lookback': 50,
        'atr_expansion_mult': 2.0,
        'bb_period': 20,
        'bb_dev': 2.0,
        'price_breakout_period': 20,
        'risk_pct': 0.1,
    }


class VolatilityExpansion_ShortTerm(VolatilityExpansionAtom):
    """波动率突破：短期配置（10日周期）"""
    name = "vol_expansion_short"
    params = {
        'atr_period': 10,
        'atr_lookback': 30,
        'atr_expansion_mult': 1.5,
        'bb_period': 10,
        'bb_dev': 2.0,
        'price_breakout_period': 10,
        'risk_pct': 0.1,
    }


class VolatilityExpansion_LongTerm(VolatilityExpansionAtom):
    """波动率突破：长期配置（50日周期）"""
    name = "vol_expansion_long"
    params = {
        'atr_period': 20,
        'atr_lookback': 100,
        'atr_expansion_mult': 1.5,
        'bb_period': 50,
        'bb_dev': 2.0,
        'price_breakout_period': 50,
        'risk_pct': 0.1,
    }
