"""
组合策略: 稳健排名 #3

组成策略: vol_breakout_aggressive, vol_regime_long, triple_ma, rsi_reversal
权重配置: 8.43%, 23.90%, 33.66%, 34.01%

稳健性指标:
- 推荐频率: 25.6%
- 平均夏普: 2.998
- 夏普范围: 2.541 ~ 3.301
- 稳健评分: 1.175

最佳配置表现:
- 预期夏普: 3.301
- 预期收益: 4.17%
- 预期最大回撤: -0.35%

实现方式: 虚拟持仓模拟法
- 为每个子策略维护独立的虚拟持仓状态（+1/-1/0）
- 完全按照原始策略的逻辑更新虚拟持仓
- 按权重加权虚拟持仓，账户持有加权后的仓位
- 数学上等价于理论加权法
"""

import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy
import collections


class PortfolioRank3ComboAtom(StrategyAtom):
    """稳健组合策略 - 排名#3

    组合4个低相关性策略：
    1. vol_breakout_aggressive (8.43%): 超激进波动率突破
    2. vol_regime_long (23.90%): 长期波动率择时
    3. triple_ma (33.66%): 三重均线趋势跟踪
    4. rsi_reversal (34.01%): RSI反转策略
    """

    name = "portfolio_rank3_combo"

    def strategy_cls(self):
        # 权重配置
        weights = {
            'vol_breakout_aggressive': 0.0843,
            'vol_regime_long': 0.2390,
            'triple_ma': 0.3366,
            'rsi_reversal': 0.3401
        }

        class Strategy(BaseStrategy):
            """组合策略实现 - 虚拟持仓模拟法"""

            def __init__(self):
                super().__init__()

                # ========== 1. RSI反转策略指标 (34.01%) ==========
                self.rsi = bt.ind.RSI(period=14)

                # ========== 2. 三重均线策略指标 (33.66%) ==========
                self.sma_short = bt.ind.SMA(period=10)
                self.sma_medium = bt.ind.SMA(period=30)
                self.sma_long = bt.ind.SMA(period=60)

                # ========== 3. 波动率突破策略指标 (8.43%) ==========
                self.atr_breakout = bt.ind.ATR(period=10)

                # ========== 4. 波动率择时策略指标 (23.90%) ==========
                self.atr_regime = bt.ind.ATR(period=20)
                self.sma_regime = bt.ind.SMA(period=50)
                self.highest_regime = bt.ind.Highest(self.data.high, period=50)
                self.lowest_regime = bt.ind.Lowest(self.data.low, period=50)

                # 波动率历史记录（用于计算分位数）
                self.atr_history = collections.deque(maxlen=504)  # 约两年
                self.vol_regime_state = None

                # ========== 虚拟持仓状态（模拟各子策略独立运行） ==========
                # +1=做多, -1=做空, 0=空仓
                self.virtual_pos_rsi = 0
                self.virtual_pos_triple_ma = 0
                self.virtual_pos_vol_breakout = 0
                self.virtual_pos_vol_regime = 0

                # vol_breakout需要的额外状态
                self.virtual_vol_breakout_entry_price = 0
                self.virtual_vol_breakout_stop_loss = 0

            def _calculate_percentile(self, value, data_list):
                """计算分位数"""
                if not data_list or len(data_list) < 10:
                    return 50.0
                sorted_data = sorted(data_list)
                count_below = sum(1 for x in sorted_data if x < value)
                return (count_below / len(sorted_data)) * 100.0

            def _update_vol_regime(self):
                """更新波动率状态"""
                if len(self.atr_regime) == 0:
                    return 'medium'

                current_atr = self.atr_regime[0]
                self.atr_history.append(current_atr)

                if len(self.atr_history) < 30:
                    return 'medium'

                percentile = self._calculate_percentile(current_atr, list(self.atr_history))

                if percentile < 30:
                    return 'low'
                elif percentile > 70:
                    return 'high'
                else:
                    return 'medium'

            def _simulate_rsi_reversal(self):
                """模拟RSI反转策略的虚拟持仓

                改进逻辑（支持做空）：
                - 空仓时：RSI<30买入，RSI>70卖空
                - 多头持仓时：RSI>70平仓
                - 空头持仓时：RSI<30平仓
                """
                if len(self.rsi) == 0:
                    return self.virtual_pos_rsi

                rsi_value = self.rsi[0]

                if self.virtual_pos_rsi == 0:  # 虚拟空仓
                    if rsi_value < 30:
                        self.virtual_pos_rsi = 1  # 虚拟买入
                    elif rsi_value > 70:
                        self.virtual_pos_rsi = -1  # 虚拟卖空（新增）

                elif self.virtual_pos_rsi > 0:  # 虚拟多头持仓
                    if rsi_value > 70:
                        self.virtual_pos_rsi = 0  # 虚拟平多

                elif self.virtual_pos_rsi < 0:  # 虚拟空头持仓（新增）
                    if rsi_value < 30:
                        self.virtual_pos_rsi = 0  # 虚拟平空

                return self.virtual_pos_rsi

            def _simulate_triple_ma(self):
                """模拟三重均线策略的虚拟持仓

                原始逻辑：
                - 空仓时：多头排列+价格接近中均线→买入，空头排列+价格接近中均线→卖出
                - 持仓时：排列打破→平仓
                """
                if len(self.sma_short) == 0 or len(self.sma_medium) == 0 or len(self.sma_long) == 0:
                    return self.virtual_pos_triple_ma

                price = self.data.close[0]
                short_ma = self.sma_short[0]
                medium_ma = self.sma_medium[0]
                long_ma = self.sma_long[0]

                is_bullish = short_ma > medium_ma > long_ma
                is_bearish = short_ma < medium_ma < long_ma

                if self.virtual_pos_triple_ma == 0:  # 虚拟空仓
                    if is_bullish and abs(price - medium_ma) / medium_ma < 0.01:
                        self.virtual_pos_triple_ma = 1  # 虚拟买入
                    elif is_bearish and abs(price - medium_ma) / medium_ma < 0.01:
                        self.virtual_pos_triple_ma = -1  # 虚拟卖出

                elif self.virtual_pos_triple_ma > 0:  # 虚拟多头持仓
                    if not is_bullish:
                        self.virtual_pos_triple_ma = 0  # 虚拟平仓

                elif self.virtual_pos_triple_ma < 0:  # 虚拟空头持仓
                    if not is_bearish:
                        self.virtual_pos_triple_ma = 0  # 虚拟平仓

                return self.virtual_pos_triple_ma

            def _simulate_vol_breakout(self):
                """模拟波动率突破策略的虚拟持仓

                原始逻辑：
                - 空仓时：日内变动>1.5×ATR→买入/卖出，设置止损
                - 持仓时：触及止损或获利了结→平仓
                """
                if len(self.atr_breakout) == 0:
                    return self.virtual_pos_vol_breakout

                current_open = self.data.open[0]
                current_close = self.data.close[0]
                atr_value = self.atr_breakout[0]
                daily_change = current_close - current_open

                if self.virtual_pos_vol_breakout == 0:  # 虚拟空仓
                    # 异常上涨突破
                    if daily_change > 1.5 * atr_value:
                        self.virtual_pos_vol_breakout = 1
                        self.virtual_vol_breakout_entry_price = current_close
                        self.virtual_vol_breakout_stop_loss = max(current_open, current_close - 2 * atr_value)

                    # 异常下跌突破
                    elif daily_change < -1.5 * atr_value:
                        self.virtual_pos_vol_breakout = -1
                        self.virtual_vol_breakout_entry_price = current_close
                        self.virtual_vol_breakout_stop_loss = min(current_open, current_close + 2 * atr_value)

                elif self.virtual_pos_vol_breakout > 0:  # 虚拟多头持仓
                    # 止损
                    if current_close <= self.virtual_vol_breakout_stop_loss:
                        self.virtual_pos_vol_breakout = 0
                    # 获利了结
                    elif current_close < (self.virtual_vol_breakout_entry_price - atr_value):
                        self.virtual_pos_vol_breakout = 0
                    # 追踪止损
                    else:
                        trailing_stop = current_close - 2 * atr_value
                        if trailing_stop > self.virtual_vol_breakout_stop_loss:
                            self.virtual_vol_breakout_stop_loss = trailing_stop

                elif self.virtual_pos_vol_breakout < 0:  # 虚拟空头持仓
                    # 止损
                    if current_close >= self.virtual_vol_breakout_stop_loss:
                        self.virtual_pos_vol_breakout = 0
                    # 获利了结
                    elif current_close > (self.virtual_vol_breakout_entry_price + atr_value):
                        self.virtual_pos_vol_breakout = 0
                    # 追踪止损
                    else:
                        trailing_stop = current_close + 2 * atr_value
                        if trailing_stop < self.virtual_vol_breakout_stop_loss:
                            self.virtual_vol_breakout_stop_loss = trailing_stop

                return self.virtual_pos_vol_breakout

            def _simulate_vol_regime(self):
                """模拟波动率择时策略的虚拟持仓

                原始逻辑：
                - 根据波动率状态选择策略
                - 低波动：突破策略
                - 高波动：均值回归策略
                - 中波动：突破策略
                """
                if len(self.sma_regime) == 0 or len(self.highest_regime) == 0 or len(self.lowest_regime) == 0:
                    return self.virtual_pos_vol_regime

                self.vol_regime_state = self._update_vol_regime()

                close = self.data.close[0]
                high = self.data.high[0]
                low = self.data.low[0]
                sma_value = self.sma_regime[0]
                high_breakout = self.highest_regime[-1] if len(self.highest_regime) > 0 else high
                low_breakout = self.lowest_regime[-1] if len(self.lowest_regime) > 0 else low

                # 开仓逻辑
                if self.virtual_pos_vol_regime == 0:  # 虚拟空仓
                    if self.vol_regime_state == 'low':  # 低波动：突破策略
                        if high > high_breakout:
                            self.virtual_pos_vol_regime = 1
                        elif low < low_breakout:
                            self.virtual_pos_vol_regime = -1

                    elif self.vol_regime_state == 'high':  # 高波动：均值回归
                        atr_value = self.atr_regime[0] if len(self.atr_regime) > 0 else 0
                        deviation = close - sma_value
                        if deviation < -2 * atr_value:
                            self.virtual_pos_vol_regime = 1
                        elif deviation > 2 * atr_value:
                            self.virtual_pos_vol_regime = -1

                    else:  # 中等波动：突破策略
                        if high > high_breakout:
                            self.virtual_pos_vol_regime = 1
                        elif low < low_breakout:
                            self.virtual_pos_vol_regime = -1

                # 平仓逻辑
                elif self.virtual_pos_vol_regime > 0:  # 虚拟多头持仓
                    if self.vol_regime_state == 'low' and close < sma_value:
                        self.virtual_pos_vol_regime = 0
                    elif self.vol_regime_state == 'high' and close >= sma_value:
                        self.virtual_pos_vol_regime = 0
                    elif low < low_breakout:
                        self.virtual_pos_vol_regime = 0

                elif self.virtual_pos_vol_regime < 0:  # 虚拟空头持仓
                    if self.vol_regime_state == 'low' and close > sma_value:
                        self.virtual_pos_vol_regime = 0
                    elif self.vol_regime_state == 'high' and close <= sma_value:
                        self.virtual_pos_vol_regime = 0
                    elif high > high_breakout:
                        self.virtual_pos_vol_regime = 0

                return self.virtual_pos_vol_regime

            def next(self):
                if self.order:
                    return

                # 1. 模拟各子策略的虚拟持仓（完全独立运行）
                vpos_rsi = self._simulate_rsi_reversal()
                vpos_triple_ma = self._simulate_triple_ma()
                vpos_vol_breakout = self._simulate_vol_breakout()
                vpos_vol_regime = self._simulate_vol_regime()

                # 2. 加权计算目标持仓比例
                target_position_pct = (
                    vpos_rsi * weights['rsi_reversal'] +
                    vpos_triple_ma * weights['triple_ma'] +
                    vpos_vol_breakout * weights['vol_breakout_aggressive'] +
                    vpos_vol_regime * weights['vol_regime_long']
                )

                # 3. 计算目标手数（多手仓位支持）
                # 降低阈值并支持1-3手仓位
                if target_position_pct >= 0.70:
                    target_size = 3  # 强烈做多
                elif target_position_pct >= 0.35:
                    target_size = 2  # 中度做多
                elif target_position_pct >= 0.05:
                    target_size = 1  # 轻度做多
                elif target_position_pct <= -0.70:
                    target_size = -3  # 强烈做空
                elif target_position_pct <= -0.35:
                    target_size = -2  # 中度做空
                elif target_position_pct <= -0.05:
                    target_size = -1  # 轻度做空
                else:
                    target_size = 0  # 空仓

                # 4. 调整实际持仓
                current_size = self.position.size if self.position else 0
                size_diff = target_size - current_size

                if size_diff > 0:
                    self.order = self.buy(size=abs(size_diff))
                elif size_diff < 0:
                    self.order = self.sell(size=abs(size_diff))

        return Strategy
