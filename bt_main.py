#!/usr/bin/env python
"""
NQ期货回测系统 - 主入口

用法:
    # 单策略回测
    python bt_main.py
    
    # 指定策略
    python bt_main.py --atom sma_cross

    # 多策略对比
    python bt_main.py --compare
    
    # 指定时间范围
    python bt_main.py --start 2024-01-01 --end 2024-12-31 --compare
"""
import argparse
import datetime
from bt_runner import Runner
from atoms import SMACrossAtom, RSIReversalAtom, MACDTrendAtom, BollingerBreakoutAtom
from atoms.sma_cross import SMACross_5_20, SMACross_10_30, SMACross_20_60
from atoms.triple_ma import (
    TripleMAAtom,
    TripleMA_5_20_50,
    TripleMA_10_30_60,
    TripleMA_8_21_55,
    TripleMA_12_26_52,
)
from atoms.adx_trend import (
    ADXTrendAtom,
    ADXTrend_14_25,
    ADXTrend_14_30,
    ADXTrend_14_20,
    ADXTrend_21_25,
    ADXTrend_10_25,
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
from atoms.cci_channel import (
    CCIChannelAtom,
    CCIChannel_20_100,
    CCIChannel_20_150,
    CCIChannel_20_80,
    CCIChannel_14_100,
    CCIChannel_30_100,
    CCIChannel_Strict,
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
from atoms.vwap_reversion import (
    VWAPReversionAtom,
    VWAPReversion_1_0,
    VWAPReversion_1_5,
    VWAPReversion_2_0,
    VWAPReversion_Aggressive,
    VWAPReversion_Conservative,
)
from atoms.volatility_regime import (
    VolatilityRegimeAtom,
    VolatilityRegime_Standard,
    VolatilityRegime_Sensitive,
    VolatilityRegime_Conservative,
    VolatilityRegime_ShortTerm,
    VolatilityRegime_LongTerm,
)
from atoms.volatility_expansion import (
    VolatilityExpansionAtom,
    VolatilityExpansion_Standard,
    VolatilityExpansion_Sensitive,
    VolatilityExpansion_Conservative,
    VolatilityExpansion_ShortTerm,
    VolatilityExpansion_LongTerm,
)
from atoms.constant_volatility import (
    ConstantVolatilityAtom,
    ConstantVolatility_10pct,
    ConstantVolatility_15pct,
    ConstantVolatility_20pct,
    ConstantVolatility_Conservative,
    ConstantVolatility_Aggressive,
)
from atoms.turtle_trading import (
    TurtleTradingAtom,
    Turtle_System1_Standard,
    Turtle_System1_Conservative,
    Turtle_System1_Aggressive,
    Turtle_System2_Standard,
    Turtle_ES_Futures,
    Turtle_MNQ_Micro,
)


# 可用策略注册表
ATOMS = {
    'sma_cross': SMACrossAtom,
    'sma_5_20': SMACross_5_20,
    'sma_10_30': SMACross_10_30,
    'sma_20_60': SMACross_20_60,
    'rsi_reversal': RSIReversalAtom,
    'macd_trend': MACDTrendAtom,
    'boll_breakout': BollingerBreakoutAtom,
    # 三重均线策略
    'triple_ma': TripleMAAtom,
    'triple_ma_5_20_50': TripleMA_5_20_50,
    'triple_ma_10_30_60': TripleMA_10_30_60,
    'triple_ma_8_21_55': TripleMA_8_21_55,
    'triple_ma_12_26_52': TripleMA_12_26_52,
    # ADX趋势强度策略
    'adx_trend': ADXTrendAtom,
    'adx_14_25': ADXTrend_14_25,
    'adx_14_30': ADXTrend_14_30,
    'adx_14_20': ADXTrend_14_20,
    'adx_21_25': ADXTrend_21_25,
    'adx_10_25': ADXTrend_10_25,
    # 布林带回归策略
    'boll_mr': BollingerMeanReversionAtom,
    'boll_mr_20_2': BollingerMR_20_2,
    'boll_mr_20_2_5': BollingerMR_20_2_5,
    'boll_mr_20_1_5': BollingerMR_20_1_5,
    'boll_mr_30_2': BollingerMR_30_2,
    'boll_mr_10_2': BollingerMR_10_2,
    'boll_mr_strict': BollingerMR_Strict,
    # CCI通道回归策略
    'cci_channel': CCIChannelAtom,
    'cci_20_100': CCIChannel_20_100,
    'cci_20_150': CCIChannel_20_150,
    'cci_20_80': CCIChannel_20_80,
    'cci_14_100': CCIChannel_14_100,
    'cci_30_100': CCIChannel_30_100,
    'cci_strict': CCIChannel_Strict,
    # Keltner通道回归策略
    'keltner_channel': KeltnerChannelAtom,
    'keltner_20_10_1_5': KeltnerChannel_20_10_1_5,
    'keltner_20_10_2_0': KeltnerChannel_20_10_2_0,
    'keltner_20_10_1_0': KeltnerChannel_20_10_1_0,
    'keltner_20_14_1_5': KeltnerChannel_20_14_1_5,
    'keltner_30_10_1_5': KeltnerChannel_30_10_1_5,
    'keltner_10_10_1_5': KeltnerChannel_10_10_1_5,
    'keltner_tight': KeltnerChannel_Tight,
    # ATR通道突破策略
    'atr_breakout': ATRBreakoutAtom,
    'atr_breakout_20_14_2': ATRBreakout_20_14_2,
    'atr_breakout_20_14_3': ATRBreakout_20_14_3,
    'atr_breakout_20_14_1_5': ATRBreakout_20_14_1_5,
    'atr_breakout_20_10_2': ATRBreakout_20_10_2,
    'atr_breakout_50_14_2': ATRBreakout_50_14_2,
    'atr_breakout_10_14_2': ATRBreakout_10_14_2,
    'atr_breakout_aggressive': ATRBreakout_Aggressive,
    'atr_breakout_conservative': ATRBreakout_Conservative,
    # 唐奇安通道突破策略
    'donchian_channel': DonchianChannelAtom,
    'donchian_20_10': DonchianChannel_20_10,
    'donchian_55_20': DonchianChannel_55_20,
    'donchian_20_20': DonchianChannel_20_20,
    'donchian_10_5': DonchianChannel_10_5,
    'donchian_5_3': DonchianChannel_5_3,
    'donchian_40_15': DonchianChannel_40_15,
    'donchian_turtle_sys1': DonchianChannel_TurtleSystem1,
    'donchian_turtle_sys2': DonchianChannel_TurtleSystem2,
    'donchian_aggressive': DonchianChannel_Aggressive,
    'donchian_conservative': DonchianChannel_Conservative,
    # 波动率突破策略
    'vol_breakout': VolatilityBreakoutAtom,
    'vol_breakout_14_2': VolatilityBreakout_14_2,
    'vol_breakout_14_2_5': VolatilityBreakout_14_2_5,
    'vol_breakout_14_1_5': VolatilityBreakout_14_1_5,
    'vol_breakout_20_2': VolatilityBreakout_20_2,
    'vol_breakout_10_2': VolatilityBreakout_10_2,
    'vol_breakout_10_3': VolatilityBreakout_10_3,
    'vol_breakout_aggressive': VolatilityBreakout_Aggressive,
    'vol_breakout_conservative': VolatilityBreakout_Conservative,
    # N日新高新低策略
    'new_hl': NewHighLowAtom,
    'new_hl_20': NewHighLow_20,
    'new_hl_50': NewHighLow_50,
    'new_hl_100': NewHighLow_100,
    'new_hl_250': NewHighLow_250,
    'new_hl_10': NewHighLow_10,
    'new_hl_5': NewHighLow_5,
    'new_hl_aggressive': NewHighLow_Aggressive,
    'new_hl_conservative': NewHighLow_Conservative,
    # 开盘区间突破策略
    'orb': OpeningRangeBreakoutAtom,
    'orb_15min': ORB_15min,
    'orb_30min': ORB_30min,
    'orb_60min': ORB_60min,
    'orb_30min_no_close': ORB_30min_NoClose,
    'orb_45min': ORB_45min,
    'orb_aggressive': ORB_Aggressive,
    'orb_conservative': ORB_Conservative,
    # 日内动量策略
    'intraday_mom': IntradayMomentumAtom,
    'intraday_mom_0_5': IntradayMomentum_0_5,
    'intraday_mom_1_0': IntradayMomentum_1_0,
    'intraday_mom_1_5': IntradayMomentum_1_5,
    'intraday_mom_2_0': IntradayMomentum_2_0,
    'intraday_mom_0_3': IntradayMomentum_0_3,
    'intraday_mom_aggressive': IntradayMomentum_Aggressive,
    'intraday_mom_conservative': IntradayMomentum_Conservative,
    'intraday_mom_moderate': IntradayMomentum_Moderate,
    # 日内反转策略
    'intraday_reversal': IntradayReversalAtom,
    'intraday_rev_1_5': IntradayReversal_1_5,
    'intraday_rev_1_0': IntradayReversal_1_0,
    'intraday_rev_2_0': IntradayReversal_2_0,
    'intraday_rev_aggressive': IntradayReversal_Aggressive,
    'intraday_rev_conservative': IntradayReversal_Conservative,
    # VWAP回归策略
    'vwap_reversion': VWAPReversionAtom,
    'vwap_rev_1_0': VWAPReversion_1_0,
    'vwap_rev_1_5': VWAPReversion_1_5,
    'vwap_rev_2_0': VWAPReversion_2_0,
    'vwap_rev_aggressive': VWAPReversion_Aggressive,
    'vwap_rev_conservative': VWAPReversion_Conservative,
    # 波动率择时策略
    'vol_regime': VolatilityRegimeAtom,
    'vol_regime_standard': VolatilityRegime_Standard,
    'vol_regime_sensitive': VolatilityRegime_Sensitive,
    'vol_regime_conservative': VolatilityRegime_Conservative,
    'vol_regime_short': VolatilityRegime_ShortTerm,
    'vol_regime_long': VolatilityRegime_LongTerm,
    # 波动率突破入场策略
    'vol_expansion': VolatilityExpansionAtom,
    'vol_expansion_standard': VolatilityExpansion_Standard,
    'vol_expansion_sensitive': VolatilityExpansion_Sensitive,
    'vol_expansion_conservative': VolatilityExpansion_Conservative,
    'vol_expansion_short': VolatilityExpansion_ShortTerm,
    'vol_expansion_long': VolatilityExpansion_LongTerm,
    # 恒定波动率目标策略
    'const_vol': ConstantVolatilityAtom,
    'const_vol_10': ConstantVolatility_10pct,
    'const_vol_15': ConstantVolatility_15pct,
    'const_vol_20': ConstantVolatility_20pct,
    'const_vol_conservative': ConstantVolatility_Conservative,
    'const_vol_aggressive': ConstantVolatility_Aggressive,
    # 海龟交易策略
    'turtle': TurtleTradingAtom,
    'turtle_sys1': Turtle_System1_Standard,
    'turtle_sys1_conservative': Turtle_System1_Conservative,
    'turtle_sys1_aggressive': Turtle_System1_Aggressive,
    'turtle_sys2': Turtle_System2_Standard,
    'turtle_es': Turtle_ES_Futures,
    'turtle_mnq': Turtle_MNQ_Micro,
}


def main():
    parser = argparse.ArgumentParser(description='NQ期货回测系统')
    parser.add_argument('--start', default='1900-01-01', help='开始日期')
    parser.add_argument('--end', default=datetime.datetime.now().strftime('%Y-%m-%d'), help='结束日期')
    parser.add_argument('--data', default='./data/nq_m1_forward_adjusted.csv', help='数据文件')
    parser.add_argument('--timeframe', default='d1', choices=['m1', 'm5', 'm15', 'm30', 'h1', 'h4', 'd1'], help='K线周期')
    parser.add_argument('--atom', default='sma_cross', choices=list(ATOMS.keys()), help='策略名称')
    parser.add_argument('--compare', action='store_true', help='多策略对比模式')
    parser.add_argument('--no-save', dest='save', action='store_false', help='不保存交易记录')
    parser.add_argument('--no-plot', dest='plot', action='store_false', help='不生成图表')
    parser.add_argument('--list', action='store_true', help='列出可用策略')
    parser.set_defaults(save=True, plot=True)
    
    args = parser.parse_args()
    
    # 列出策略
    if args.list:
        print('可用策略:')
        for name, cls in ATOMS.items():
            print(f'  {name:<15} - {cls().name}')
        return
    
    # 解析日期
    start = datetime.datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.datetime.strptime(args.end, '%Y-%m-%d')
    
    # 创建Runner
    runner = Runner(
        data_path=args.data,
        timeframe=args.timeframe,
        start_date=start,
        end_date=end,
    )
    
    if args.compare:
        # 多策略对比
        atoms = [cls() for cls in ATOMS.values()]
        runner.run_multiple(atoms, save_trades=False, plot=False)
    else:
        # 单策略回测
        atom_cls = ATOMS[args.atom]
        result = runner.run(atom_cls(), save_trades=args.save, plot=args.plot)


if __name__ == '__main__':
    main()