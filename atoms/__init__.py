"""策略原子模块"""

# ========== 趋势跟踪策略 ==========
from atoms.sma_cross import (
    SMACrossAtom,
    SMACross_5_20,
    SMACross_10_30,
    SMACross_20_60,
)
from atoms.macd_trend import (
    MACDTrendAtom,
    MACD_12_26_9,
    MACD_8_17_9,
)
from atoms.adx_trend import (
    ADXTrendAtom,
    ADXTrend_14_25,
    ADXTrend_14_30,
    ADXTrend_14_20,
    ADXTrend_21_25,
    ADXTrend_10_25,
)
from atoms.triple_ma import (
    TripleMAAtom,
    TripleMA_5_20_50,
    TripleMA_10_30_60,
    TripleMA_8_21_55,
    TripleMA_12_26_52,
)

# ========== 突破策略 ==========
from atoms.bollinger_breakout import (
    BollingerBreakoutAtom,
    BollingerMeanReversion,
)
from atoms.donchian_channel import (
    DonchianChannelAtom,
    DonchianChannel_20_10,
    DonchianChannel_55_20,
    DonchianChannel_20_20,
    DonchianChannel_10_5,
    DonchianChannel_5_3,
    DonchianChannel_40_15,
    DonchianChannel_TurtleSystem1,
    DonchianChannel_TurtleSystem2,
    DonchianChannel_Aggressive,
    DonchianChannel_Conservative,
)
from atoms.keltner_channel import (
    KeltnerChannelAtom,
    KeltnerChannel_20_10_1_5,
    KeltnerChannel_20_10_2_0,
    KeltnerChannel_20_10_1_0,
    KeltnerChannel_20_14_1_5,
    KeltnerChannel_30_10_1_5,
    KeltnerChannel_10_10_1_5,
    KeltnerChannel_Tight,
)
from atoms.atr_breakout import (
    ATRBreakoutAtom,
    ATRBreakout_20_14_2,
    ATRBreakout_20_14_3,
    ATRBreakout_20_14_1_5,
    ATRBreakout_20_10_2,
    ATRBreakout_50_14_2,
    ATRBreakout_10_14_2,
    ATRBreakout_Aggressive,
    ATRBreakout_Conservative,
)
from atoms.volatility_breakout import (
    VolatilityBreakoutAtom,
    VolatilityBreakout_14_2,
    VolatilityBreakout_14_2_5,
    VolatilityBreakout_14_1_5,
    VolatilityBreakout_20_2,
    VolatilityBreakout_10_2,
    VolatilityBreakout_10_3,
    VolatilityBreakout_Aggressive,
    VolatilityBreakout_Conservative,
)
from atoms.new_high_low import (
    NewHighLowAtom,
    NewHighLow_20,
    NewHighLow_50,
    NewHighLow_100,
    NewHighLow_250,
    NewHighLow_10,
    NewHighLow_5,
    NewHighLow_Aggressive,
    NewHighLow_Conservative,
)
from atoms.opening_range_breakout import (
    OpeningRangeBreakoutAtom,
    ORB_15min,
    ORB_30min,
    ORB_60min,
    ORB_30min_NoClose,
    ORB_45min,
    ORB_Aggressive,
    ORB_Conservative,
)

# ========== 均值回归策略 ==========
from atoms.rsi_reversal import (
    RSIReversalAtom,
    RSI_14_30_70,
    RSI_7_20_80,
)
from atoms.bollinger_mean_reversion import (
    BollingerMeanReversionAtom,
    BollingerMR_20_2,
    BollingerMR_20_2_5,
    BollingerMR_20_1_5,
    BollingerMR_30_2,
    BollingerMR_10_2,
    BollingerMR_Strict,
)
from atoms.vwap_reversion import (
    VWAPReversionAtom,
    VWAPReversion_1_0,
    VWAPReversion_1_5,
    VWAPReversion_2_0,
    VWAPReversion_Aggressive,
    VWAPReversion_Conservative,
)
from atoms.cci_channel import (
    CCIChannelAtom,
    CCIChannel_20_100,
    CCIChannel_20_150,
    CCIChannel_20_80,
    CCIChannel_14_100,
    CCIChannel_30_100,
    CCIChannel_Strict,
)

# ========== 波动率策略 ==========
from atoms.constant_volatility import (
    ConstantVolatilityAtom,
    ConstantVolatility_10pct,
    ConstantVolatility_15pct,
    ConstantVolatility_20pct,
    ConstantVolatility_Conservative,
    ConstantVolatility_Aggressive,
)
from atoms.volatility_expansion import (
    VolatilityExpansionAtom,
    VolatilityExpansion_Standard,
    VolatilityExpansion_Sensitive,
    VolatilityExpansion_Conservative,
    VolatilityExpansion_ShortTerm,
    VolatilityExpansion_LongTerm,
)
from atoms.volatility_regime import (
    VolatilityRegimeAtom,
    VolatilityRegime_Standard,
    VolatilityRegime_Sensitive,
    VolatilityRegime_Conservative,
    VolatilityRegime_ShortTerm,
    VolatilityRegime_LongTerm,
)

# ========== 日内交易策略 ==========
from atoms.intraday_momentum import (
    IntradayMomentumAtom,
    IntradayMomentum_0_5,
    IntradayMomentum_1_0,
    IntradayMomentum_1_5,
    IntradayMomentum_2_0,
    IntradayMomentum_0_3,
    IntradayMomentum_Aggressive,
    IntradayMomentum_Conservative,
    IntradayMomentum_Moderate,
)
from atoms.intraday_reversal import (
    IntradayReversalAtom,
    IntradayReversal_1_5,
    IntradayReversal_1_0,
    IntradayReversal_2_0,
    IntradayReversal_Aggressive,
    IntradayReversal_Conservative,
)

# ========== 海龟交易系统 ==========
from atoms.turtle_trading import (
    TurtleTradingAtom,
    Turtle_System1_Standard,
    Turtle_System1_Conservative,
    Turtle_System1_Aggressive,
    Turtle_System2_Standard,
    Turtle_ES_Futures,
    Turtle_MNQ_Micro,
)

# ========== 基准策略 ==========
from atoms.buy_and_hold import BuyAndHoldAtom

__all__ = [
    # ========== 趋势跟踪策略 ==========
    'SMACrossAtom',
    'SMACross_5_20',
    'SMACross_10_30',
    'SMACross_20_60',

    'MACDTrendAtom',
    'MACD_12_26_9',
    'MACD_8_17_9',

    'ADXTrendAtom',
    'ADXTrend_14_25',
    'ADXTrend_14_30',
    'ADXTrend_14_20',
    'ADXTrend_21_25',
    'ADXTrend_10_25',

    'TripleMAAtom',
    'TripleMA_5_20_50',
    'TripleMA_10_30_60',
    'TripleMA_8_21_55',
    'TripleMA_12_26_52',

    # ========== 突破策略 ==========
    'BollingerBreakoutAtom',
    'BollingerMeanReversion',

    'DonchianChannelAtom',
    'DonchianChannel_20_10',
    'DonchianChannel_55_20',
    'DonchianChannel_20_20',
    'DonchianChannel_10_5',
    'DonchianChannel_5_3',
    'DonchianChannel_40_15',
    'DonchianChannel_TurtleSystem1',
    'DonchianChannel_TurtleSystem2',
    'DonchianChannel_Aggressive',
    'DonchianChannel_Conservative',

    'KeltnerChannelAtom',
    'KeltnerChannel_20_10_1_5',
    'KeltnerChannel_20_10_2_0',
    'KeltnerChannel_20_10_1_0',
    'KeltnerChannel_20_14_1_5',
    'KeltnerChannel_30_10_1_5',
    'KeltnerChannel_10_10_1_5',
    'KeltnerChannel_Tight',

    'ATRBreakoutAtom',
    'ATRBreakout_20_14_2',
    'ATRBreakout_20_14_3',
    'ATRBreakout_20_14_1_5',
    'ATRBreakout_20_10_2',
    'ATRBreakout_50_14_2',
    'ATRBreakout_10_14_2',
    'ATRBreakout_Aggressive',
    'ATRBreakout_Conservative',

    'VolatilityBreakoutAtom',
    'VolatilityBreakout_14_2',
    'VolatilityBreakout_14_2_5',
    'VolatilityBreakout_14_1_5',
    'VolatilityBreakout_20_2',
    'VolatilityBreakout_10_2',
    'VolatilityBreakout_10_3',
    'VolatilityBreakout_Aggressive',
    'VolatilityBreakout_Conservative',

    'NewHighLowAtom',
    'NewHighLow_20',
    'NewHighLow_50',
    'NewHighLow_100',
    'NewHighLow_250',
    'NewHighLow_10',
    'NewHighLow_5',
    'NewHighLow_Aggressive',
    'NewHighLow_Conservative',

    'OpeningRangeBreakoutAtom',
    'ORB_15min',
    'ORB_30min',
    'ORB_60min',
    'ORB_30min_NoClose',
    'ORB_45min',
    'ORB_Aggressive',
    'ORB_Conservative',

    # ========== 均值回归策略 ==========
    'RSIReversalAtom',
    'RSI_14_30_70',
    'RSI_7_20_80',

    'BollingerMeanReversionAtom',
    'BollingerMR_20_2',
    'BollingerMR_20_2_5',
    'BollingerMR_20_1_5',
    'BollingerMR_30_2',
    'BollingerMR_10_2',
    'BollingerMR_Strict',

    'VWAPReversionAtom',
    'VWAPReversion_1_0',
    'VWAPReversion_1_5',
    'VWAPReversion_2_0',
    'VWAPReversion_Aggressive',
    'VWAPReversion_Conservative',

    'CCIChannelAtom',
    'CCIChannel_20_100',
    'CCIChannel_20_150',
    'CCIChannel_20_80',
    'CCIChannel_14_100',
    'CCIChannel_30_100',
    'CCIChannel_Strict',

    # ========== 波动率策略 ==========
    'ConstantVolatilityAtom',
    'ConstantVolatility_10pct',
    'ConstantVolatility_15pct',
    'ConstantVolatility_20pct',
    'ConstantVolatility_Conservative',
    'ConstantVolatility_Aggressive',

    'VolatilityExpansionAtom',
    'VolatilityExpansion_Standard',
    'VolatilityExpansion_Sensitive',
    'VolatilityExpansion_Conservative',
    'VolatilityExpansion_ShortTerm',
    'VolatilityExpansion_LongTerm',

    'VolatilityRegimeAtom',
    'VolatilityRegime_Standard',
    'VolatilityRegime_Sensitive',
    'VolatilityRegime_Conservative',
    'VolatilityRegime_ShortTerm',
    'VolatilityRegime_LongTerm',

    # ========== 日内交易策略 ==========
    'IntradayMomentumAtom',
    'IntradayMomentum_0_5',
    'IntradayMomentum_1_0',
    'IntradayMomentum_1_5',
    'IntradayMomentum_2_0',
    'IntradayMomentum_0_3',
    'IntradayMomentum_Aggressive',
    'IntradayMomentum_Conservative',
    'IntradayMomentum_Moderate',

    'IntradayReversalAtom',
    'IntradayReversal_1_5',
    'IntradayReversal_1_0',
    'IntradayReversal_2_0',
    'IntradayReversal_Aggressive',
    'IntradayReversal_Conservative',

    # ========== 海龟交易系统 ==========
    'TurtleTradingAtom',
    'Turtle_System1_Standard',
    'Turtle_System1_Conservative',
    'Turtle_System1_Aggressive',
    'Turtle_System2_Standard',
    'Turtle_ES_Futures',
    'Turtle_MNQ_Micro',

    # ========== 基准策略 ==========
    'BuyAndHoldAtom',
]
