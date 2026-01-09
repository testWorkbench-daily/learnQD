"""
日内反转策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer
import datetime


class IntradayReversalAtom(StrategyAtom):
    """
    日内反转策略

    上午大涨的品种下午可能回调，上午大跌的品种下午可能反弹。
    利用日内价格的均值回归特性，在震荡日表现更好。

    核心原理：
    - 上午涨幅过大，可能超涨，下午存在回调压力
    - 上午跌幅过大，可能超跌，下午存在反弹动力
    - 与日内动量策略相反，更适合震荡市、无明显趋势的交易日

    信号规则（单品种版）：
    - 观察期：开盘至 11:30（上午收盘）
    - 交易期：13:00 至 14:55（下午交易）
    - 反转阈值：上午涨跌幅绝对值 > 阈值（如 1.5%）
    - 上午大涨（> 阈值）：下午做空
    - 上午大跌（< -阈值）：下午做多
    - 中等涨跌幅：不交易
    - 尾盘平仓：14:55 之后必须平仓

    趋势日过滤（可选 ATR 过滤）：
    - ATR 相对自身均值显著放大，视为趋势日
    - 趋势日不做反转，以避免"逆势抄顶/抄底"

    特点：
    - 纯日内策略，不隔夜
    - 需要分钟级数据
    - 适合震荡日、无明显趋势的环境
    """

    name = "intraday_reversal"
    params = {
        'reversal_threshold': 1.5,  # 反转阈值（%）
        'use_atr_filter': False,    # 是否启用 ATR 趋势过滤
        'atr_period': 14,
        'atr_multiplier': 1.5,
        'risk_pct': 0.1,
    }

    def strategy_cls(self):
        reversal_threshold = self.params['reversal_threshold']
        use_atr_filter = self.params.get('use_atr_filter', False)
        atr_period = self.params.get('atr_period', 14)
        atr_multiplier = self.params.get('atr_multiplier', 1.5)

        class Strategy(BaseStrategy):
            params = (
                ('reversal_threshold', reversal_threshold),
                ('use_atr_filter', use_atr_filter),
                ('atr_period', atr_period),
                ('atr_multiplier', atr_multiplier),
            )

            def __init__(self):
                super().__init__()
                # 记录当前日期
                self.current_date = None
                # 上午开盘价和收盘价
                self.morning_open = None
                self.morning_close = None
                # 上午收益率（%）
                self.morning_return = None
                # 状态标记
                self.in_observation = False
                self.in_trading = False
                self.traded_today = False

                # ATR 过滤：用于识别趋势日
                if self.p.use_atr_filter:
                    self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
                    self.atr_sma = bt.ind.SMA(self.atr, period=self.p.atr_period)
                else:
                    self.atr = None
                    self.atr_sma = None

            def _is_trend_day(self):
                """
                使用 ATR 判断是否为趋势日：
                当前 ATR 明显高于自身均值 -> 趋势日（不做反转）。
                """
                if not self.p.use_atr_filter or self.atr is None or self.atr_sma is None:
                    return False

                if len(self.atr_sma) == 0:
                    return False

                return self.atr[0] > self.p.atr_multiplier * self.atr_sma[0]

            def next(self):
                if self.order:
                    return

                # 当前时间
                current_dt = self.data.datetime.datetime(0)
                today = current_dt.date()
                current_time = current_dt.time()

                # 新的一天，重置日内状态
                if self.current_date != today:
                    self.current_date = today
                    self.morning_open = None
                    self.morning_close = None
                    self.morning_return = None
                    self.in_observation = False
                    self.in_trading = False
                    self.traded_today = False

                # 时间段（中国 A 股 / 期货常规日盘）
                morning_start = datetime.time(9, 30)
                morning_end = datetime.time(11, 30)
                afternoon_start = datetime.time(13, 0)
                afternoon_end = datetime.time(14, 55)

                # 1）观察期：记录上午开盘价 & 最新收盘价
                if morning_start <= current_time < morning_end:
                    if not self.in_observation:
                        self.in_observation = True
                        if self.morning_open is None:
                            self.morning_open = self.data.open[0]

                    # 每个 bar 更新一次"上午收盘价"
                    self.morning_close = self.data.close[0]

                # 2）上午结束后：计算上午收益率
                elif morning_end <= current_time < afternoon_start:
                    if self.in_observation and self.morning_return is None:
                        if self.morning_open and self.morning_close:
                            self.morning_return = (
                                self.morning_close - self.morning_open
                            ) / self.morning_open * 100.0

                # 3）交易期：下午根据上午涨跌幅做"反转"交易
                elif afternoon_start <= current_time < afternoon_end:
                    if not self.in_trading:
                        self.in_trading = True

                    if not self.traded_today and self.morning_return is not None:
                        # 趋势日过滤：ATR 明显放大 -> 不做反转
                        if self._is_trend_day():
                            return

                        if not self.position:
                            # 上午大涨：> 阈值 -> 下午做空
                            if self.morning_return > self.p.reversal_threshold:
                                self.order = self.sell()
                                self.traded_today = True

                            # 上午大跌：< -阈值 -> 下午做多
                            elif self.morning_return < -self.p.reversal_threshold:
                                self.order = self.buy()
                                self.traded_today = True

                # 4）尾盘平仓：14:55 之后必须平掉所有仓位
                elif current_time >= afternoon_end:
                    if self.position:
                        self.order = self.close()

        return Strategy

    def sizer_cls(self):
        """使用固定1手（单品种回测环境下，仓位控制主要交给 Runner）"""
        return None  # 使用 Runner 默认的 FixedSize


# 预定义的参数变体
class IntradayReversal_1_5(IntradayReversalAtom):
    """日内反转策略：1.5% 阈值（标准，默认不启用 ATR 过滤）"""
    name = "intraday_rev_1_5"
    params = {
        'reversal_threshold': 1.5,
        'use_atr_filter': False,
        'atr_period': 14,
        'atr_multiplier': 1.5,
        'risk_pct': 0.1,
    }


class IntradayReversal_1_0(IntradayReversalAtom):
    """日内反转策略：1.0% 阈值（更敏感）"""
    name = "intraday_rev_1_0"
    params = {
        'reversal_threshold': 1.0,
        'use_atr_filter': False,
        'atr_period': 14,
        'atr_multiplier': 1.5,
        'risk_pct': 0.1,
    }


class IntradayReversal_2_0(IntradayReversalAtom):
    """日内反转策略：2.0% 阈值（更严格，仅极端涨跌才交易）"""
    name = "intraday_rev_2_0"
    params = {
        'reversal_threshold': 2.0,
        'use_atr_filter': False,
        'atr_period': 14,
        'atr_multiplier': 1.5,
        'risk_pct': 0.1,
    }


class IntradayReversal_Aggressive(IntradayReversalAtom):
    """激进日内反转：1.0% 阈值"""
    name = "intraday_rev_aggressive"
    params = {
        'reversal_threshold': 1.0,
        'use_atr_filter': False,
        'atr_period': 14,
        'atr_multiplier': 1.5,
        'risk_pct': 0.1,
    }


class IntradayReversal_Conservative(IntradayReversalAtom):
    """保守日内反转：1.5% 阈值 + ATR 趋势过滤"""
    name = "intraday_rev_conservative"
    params = {
        'reversal_threshold': 1.5,
        'use_atr_filter': True,     # 启用 ATR 过滤，趋势日不做反转
        'atr_period': 14,
        'atr_multiplier': 1.5,
        'risk_pct': 0.1,
    }
