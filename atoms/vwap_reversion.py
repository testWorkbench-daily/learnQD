"""
VWAP回归策略（VWAP Mean Reversion）
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer
import datetime
import statistics


class VWAPReversionAtom(StrategyAtom):
    """
    VWAP回归策略

    核心思想：
    - 价格围绕当日 VWAP（成交量加权平均价）上下波动
    - 当价格偏离 VWAP 过多时，倾向于向 VWAP 回归
    - 适合震荡或均值回归特征明显的日内行情

    关键点：
    - VWAP 起算时点：当日开盘
    - VWAP = 累计成交额 / 累计成交量
      这里用 典型价 * 成交量 近似成交额：
      典型价 = (高 + 低 + 收) / 3
    - 偏离阈值：1–2 倍当日价格标准差（std）
    - 止盈：价格回归至 VWAP 附近
    - 止损：继续朝同方向偏离，超过 3 倍标准差

    信号规则（单品种版）：
    - 做多：价格 < VWAP - k * std
    - 做空：价格 > VWAP + k * std
    - 目标：价格回到 VWAP（交叉或接近）
    - 止损：价格相对 VWAP 偏离 > 3 * std（继续走极端）

    数据要求：
    - 分钟级别数据（m1/m5/m15/m30）
    - 包含 volume 字段，用于计算 VWAP
    """

    name = "vwap_reversion"
    params = {
        'dev_threshold': 1.5,     # 入场偏离阈值（标准差倍数）
        'stop_multiplier': 3.0,   # 止损偏离倍数（标准差倍数）
        'min_bars': 10,           # 当日最少 bar 数，避免早盘太少数据
        'risk_pct': 0.1,
    }

    def strategy_cls(self):
        dev_threshold = self.params['dev_threshold']
        stop_multiplier = self.params['stop_multiplier']
        min_bars = self.params['min_bars']

        class Strategy(BaseStrategy):
            params = (
                ('dev_threshold', dev_threshold),
                ('stop_multiplier', stop_multiplier),
                ('min_bars', min_bars),
            )

            def __init__(self):
                super().__init__()
                # 当前交易日
                self.current_date = None

                # 当日 VWAP 累计量
                self.cum_pv = 0.0    # 累计 典型价 * 成交量
                self.cum_vol = 0.0   # 累计 成交量
                self.vwap = None

                # 当日价格序列，用于计算当日标准差
                self.day_prices = []

                # 入场相关
                self.entry_price = None

            def _reset_day(self, today):
                self.current_date = today
                self.cum_pv = 0.0
                self.cum_vol = 0.0
                self.vwap = None
                self.day_prices = []
                self.entry_price = None

            def next(self):
                if self.order:
                    return

                # 当前时间与日期
                current_dt = self.data.datetime.datetime(0)
                today = current_dt.date()

                # 新的一天，重置所有当日状态
                if self.current_date != today:
                    self._reset_day(today)

                # 价格与成交量
                close = self.data.close[0]
                high = self.data.high[0]
                low = self.data.low[0]
                volume = float(self.data.volume[0]) if len(self.data.volume) > 0 else 0.0

                # 典型价
                typical_price = (high + low + close) / 3.0

                # 更新当日价格列表
                self.day_prices.append(close)

                # 更新 VWAP 累计
                if volume > 0:
                    self.cum_pv += typical_price * volume
                    self.cum_vol += volume

                # 计算 VWAP
                if self.cum_vol > 0:
                    self.vwap = self.cum_pv / self.cum_vol
                else:
                    self.vwap = None

                # 如果 VWAP 不可用或当日数据太少，暂不交易
                if self.vwap is None or len(self.day_prices) < self.p.min_bars:
                    return

                # 计算当日标准差（基于 close）
                if len(self.day_prices) >= 2:
                    std = statistics.pstdev(self.day_prices)
                else:
                    std = 0.0

                # 标准差太小（如极度窄幅震荡）时，避免产生噪音信号
                if std <= 0:
                    return

                # 当前相对 VWAP 的偏离
                deviation = close - self.vwap

                # 入场逻辑：均值回归
                if not self.position:
                    # 做多：价格显著低于 VWAP
                    if deviation < -self.p.dev_threshold * std:
                        self.order = self.buy()
                        self.entry_price = close

                    # 做空：价格显著高于 VWAP
                    elif deviation > self.p.dev_threshold * std:
                        self.order = self.sell()
                        self.entry_price = close

                else:
                    # 多头仓位管理
                    if self.position.size > 0:
                        # 目标：价格回归 VWAP（收盘价 >= VWAP）
                        if close >= self.vwap:
                            self.order = self.close()
                            self.entry_price = None
                            return

                        # 止损：价格继续偏离，低于 VWAP - 3*std
                        if deviation < -self.p.stop_multiplier * std:
                            self.order = self.close()
                            self.entry_price = None
                            return

                    # 空头仓位管理
                    elif self.position.size < 0:
                        # 目标：价格回归 VWAP（收盘价 <= VWAP）
                        if close <= self.vwap:
                            self.order = self.close()
                            self.entry_price = None
                            return

                        # 止损：价格继续偏离，高于 VWAP + 3*std
                        if deviation > self.p.stop_multiplier * std:
                            self.order = self.close()
                            self.entry_price = None
                            return

        return Strategy

    def sizer_cls(self):
        """使用默认的固定手数"""
        return None  # 使用 Runner 默认 sizer


# 预定义参数变体
class VWAPReversion_1_0(VWAPReversionAtom):
    """VWAP回归：1.0 标准差阈值（更频繁交易）"""
    name = "vwap_rev_1_0"
    params = {
        'dev_threshold': 1.0,
        'stop_multiplier': 3.0,
        'min_bars': 10,
        'risk_pct': 0.1,
    }


class VWAPReversion_1_5(VWAPReversionAtom):
    """VWAP回归：1.5 标准差阈值（标准）"""
    name = "vwap_rev_1_5"
    params = {
        'dev_threshold': 1.5,
        'stop_multiplier': 3.0,
        'min_bars': 10,
        'risk_pct': 0.1,
    }


class VWAPReversion_2_0(VWAPReversionAtom):
    """VWAP回归：2.0 标准差阈值（更严格，仅极端偏离才入场）"""
    name = "vwap_rev_2_0"
    params = {
        'dev_threshold': 2.0,
        'stop_multiplier': 3.0,
        'min_bars': 10,
        'risk_pct': 0.1,
    }


class VWAPReversion_Aggressive(VWAPReversionAtom):
    """激进 VWAP 回归：1.0 标准差阈值"""
    name = "vwap_rev_aggressive"
    params = {
        'dev_threshold': 1.0,
        'stop_multiplier': 2.5,
        'min_bars': 10,
        'risk_pct': 0.1,
    }


class VWAPReversion_Conservative(VWAPReversionAtom):
    """保守 VWAP 回归：1.5 标准差阈值 + 更宽止损"""
    name = "vwap_rev_conservative"
    params = {
        'dev_threshold': 1.5,
        'stop_multiplier': 3.5,
        'min_bars': 10,
        'risk_pct': 0.1,
    }
