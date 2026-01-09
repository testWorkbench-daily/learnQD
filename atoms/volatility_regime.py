"""
波动率择时策略（Volatility Regime Strategy）
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer
import collections


class VolatilityRegimeAtom(StrategyAtom):
    """
    波动率择时策略

    核心思想：
    - 根据当前波动率在历史分位数中的位置来调整交易行为
    - 低波动期：市场处于平静期，往往预示着即将突破，适合趋势跟踪
    - 高波动期：市场剧烈波动，风险较大，适合减仓或均值回归
    - 中等波动期：正常交易

    波动率测量：
    - 使用 ATR（Average True Range）作为波动率指标
    - 计算 ATR 在过去 N 日的分位数位置

    仓位调整逻辑：
    - 低波动（< 30% 分位）：加大仓位（1.5x），准备捕捉突破
    - 中等波动（30%-70% 分位）：正常仓位（1.0x）
    - 高波动（> 70% 分位）：减小仓位（0.5x），防范风险

    交易策略：
    - 低波动：使用突破策略（价格创新高/新低）
    - 高波动：使用均值回归策略（价格偏离均线过多时反向）
    - 中等波动：两种策略都可以

    特点：
    - 动态调整仓位和策略类型
    - 适应不同市场环境
    - 风险管理
    """

    name = "volatility_regime"
    params = {
        'atr_period': 14,           # ATR 计算周期
        'lookback_period': 252,     # 历史分位数回顾期（约一年交易日）
        'low_vol_percentile': 30,   # 低波动分位数阈值
        'high_vol_percentile': 70,  # 高波动分位数阈值
        'ma_period': 20,            # 均线周期（用于交易信号）
        'breakout_period': 20,      # 突破周期
        'risk_pct': 0.1,
    }

    def strategy_cls(self):
        atr_period = self.params['atr_period']
        lookback_period = self.params['lookback_period']
        low_vol_pct = self.params['low_vol_percentile']
        high_vol_pct = self.params['high_vol_percentile']
        ma_period = self.params['ma_period']
        breakout_period = self.params['breakout_period']

        class Strategy(BaseStrategy):
            params = (
                ('atr_period', atr_period),
                ('lookback_period', lookback_period),
                ('low_vol_percentile', low_vol_pct),
                ('high_vol_percentile', high_vol_pct),
                ('ma_period', ma_period),
                ('breakout_period', breakout_period),
            )

            def __init__(self):
                super().__init__()
                # ATR 指标
                self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
                
                # 移动平均线（用于均值回归信号）
                self.sma = bt.ind.SMA(self.data.close, period=self.p.ma_period)
                
                # 突破指标（用于突破信号）
                self.highest = bt.ind.Highest(self.data.high, period=self.p.breakout_period)
                self.lowest = bt.ind.Lowest(self.data.low, period=self.p.breakout_period)
                
                # 存储历史 ATR 值，用于计算分位数
                self.atr_history = collections.deque(maxlen=self.p.lookback_period)
                
                # 当前波动率状态
                self.vol_regime = None  # 'low', 'medium', 'high'
                self.position_multiplier = 1.0

            def _calculate_percentile(self, value, data_list):
                """计算值在数据列表中的分位数（0-100）"""
                if not data_list or len(data_list) < 10:
                    return 50.0  # 数据不足时返回中位数
                
                sorted_data = sorted(data_list)
                count_below = sum(1 for x in sorted_data if x < value)
                percentile = (count_below / len(sorted_data)) * 100.0
                return percentile

            def _update_vol_regime(self):
                """更新波动率状态"""
                if len(self.atr) == 0:
                    return
                
                current_atr = self.atr[0]
                
                # 更新 ATR 历史记录
                self.atr_history.append(current_atr)
                
                # 计算当前 ATR 的分位数
                if len(self.atr_history) < 30:
                    # 数据不足，使用中等波动状态
                    self.vol_regime = 'medium'
                    self.position_multiplier = 1.0
                    return
                
                percentile = self._calculate_percentile(current_atr, list(self.atr_history))
                
                # 判断波动率状态
                if percentile < self.p.low_vol_percentile:
                    self.vol_regime = 'low'
                    self.position_multiplier = 1.5  # 低波动，加大仓位
                elif percentile > self.p.high_vol_percentile:
                    self.vol_regime = 'high'
                    self.position_multiplier = 0.5  # 高波动，减小仓位
                else:
                    self.vol_regime = 'medium'
                    self.position_multiplier = 1.0  # 中等波动，正常仓位

            def next(self):
                if self.order:
                    return
                
                # 更新波动率状态
                self._update_vol_regime()
                
                if self.vol_regime is None:
                    return
                
                # 获取当前价格和指标值
                close = self.data.close[0]
                high = self.data.high[0]
                low = self.data.low[0]
                
                # 确保指标数据足够
                if len(self.sma) == 0 or len(self.highest) == 0 or len(self.lowest) == 0:
                    return
                
                sma_value = self.sma[0]
                high_breakout = self.highest[-1] if len(self.highest) > 0 else high
                low_breakout = self.lowest[-1] if len(self.lowest) > 0 else low
                
                # 根据波动率状态选择交易策略
                if not self.position:
                    if self.vol_regime == 'low':
                        # 低波动：使用突破策略
                        # 向上突破
                        if high > high_breakout:
                            self.order = self.buy()
                        # 向下突破
                        elif low < low_breakout:
                            self.order = self.sell()
                    
                    elif self.vol_regime == 'high':
                        # 高波动：使用均值回归策略
                        atr_value = self.atr[0]
                        deviation = close - sma_value
                        
                        # 价格显著低于均线，做多
                        if deviation < -2 * atr_value:
                            self.order = self.buy()
                        # 价格显著高于均线，做空
                        elif deviation > 2 * atr_value:
                            self.order = self.sell()
                    
                    else:  # medium volatility
                        # 中等波动：结合突破和均值回归
                        # 简化为突破策略
                        if high > high_breakout:
                            self.order = self.buy()
                        elif low < low_breakout:
                            self.order = self.sell()
                
                else:
                    # 持仓管理
                    atr_value = self.atr[0] if len(self.atr) > 0 else 0
                    
                    if self.position.size > 0:
                        # 多头止损/止盈
                        if self.vol_regime == 'low':
                            # 低波动期：等待价格回到均线下方
                            if close < sma_value:
                                self.order = self.close()
                        elif self.vol_regime == 'high':
                            # 高波动期：价格回到均线附近即获利了结
                            if close >= sma_value:
                                self.order = self.close()
                        else:
                            # 中等波动：跌破支撑位
                            if low < low_breakout:
                                self.order = self.close()
                    
                    elif self.position.size < 0:
                        # 空头止损/止盈
                        if self.vol_regime == 'low':
                            # 低波动期：等待价格回到均线上方
                            if close > sma_value:
                                self.order = self.close()
                        elif self.vol_regime == 'high':
                            # 高波动期：价格回到均线附近即获利了结
                            if close <= sma_value:
                                self.order = self.close()
                        else:
                            # 中等波动：突破阻力位
                            if high > high_breakout:
                                self.order = self.close()

        return Strategy

    def sizer_cls(self):
        """使用固定手数，波动率调整通过策略内部实现"""
        return None


# 预定义的参数变体
class VolatilityRegime_Standard(VolatilityRegimeAtom):
    """波动率择时：标准配置（30-70分位）"""
    name = "vol_regime_standard"
    params = {
        'atr_period': 14,
        'lookback_period': 252,
        'low_vol_percentile': 30,
        'high_vol_percentile': 70,
        'ma_period': 20,
        'breakout_period': 20,
        'risk_pct': 0.1,
    }


class VolatilityRegime_Sensitive(VolatilityRegimeAtom):
    """波动率择时：敏感配置（20-80分位）"""
    name = "vol_regime_sensitive"
    params = {
        'atr_period': 14,
        'lookback_period': 252,
        'low_vol_percentile': 20,
        'high_vol_percentile': 80,
        'ma_period': 20,
        'breakout_period': 20,
        'risk_pct': 0.1,
    }


class VolatilityRegime_Conservative(VolatilityRegimeAtom):
    """波动率择时：保守配置（25-75分位）"""
    name = "vol_regime_conservative"
    params = {
        'atr_period': 14,
        'lookback_period': 252,
        'low_vol_percentile': 25,
        'high_vol_percentile': 75,
        'ma_period': 20,
        'breakout_period': 20,
        'risk_pct': 0.1,
    }


class VolatilityRegime_ShortTerm(VolatilityRegimeAtom):
    """波动率择时：短期配置（126日回顾）"""
    name = "vol_regime_short"
    params = {
        'atr_period': 10,
        'lookback_period': 126,  # 约半年
        'low_vol_percentile': 30,
        'high_vol_percentile': 70,
        'ma_period': 10,
        'breakout_period': 10,
        'risk_pct': 0.1,
    }


class VolatilityRegime_LongTerm(VolatilityRegimeAtom):
    """波动率择时：长期配置（504日回顾）"""
    name = "vol_regime_long"
    params = {
        'atr_period': 20,
        'lookback_period': 504,  # 约两年
        'low_vol_percentile': 30,
        'high_vol_percentile': 70,
        'ma_period': 50,
        'breakout_period': 50,
        'risk_pct': 0.1,
    }
